# -*- coding: utf-8 -*-
"""本地降AIGC改写模型：在改写语料(rewrite_corpus.jsonl)上微调 千问 Qwen 基座(causal decoder)。
基座：Qwen2.5-1.5B-Instruct(默认, 8GB VRAM 可全参)。config 可换 Qwen3-1.7B / LoRA。
用途：把AI句改写成更像人写的自然表达(降AIGC)，保持原意，本地零token。
用法：
  python scripts/train_rewrite_qwen.py                 # 默认 1.5B 全参微调 + 评估
  python scripts/train_rewrite_qwen.py --epochs 3 --batch 2 --grad-accum 8
评估：生成改写 → 用检测器(aigc-detector 双流融合)验证降AIGC + 语义相似度(cos)。

⚠️ 注意(本机实测)：必须「先 import torch，再 from transformers import ...」；
   且不能用 transformers.Trainer/Seq2SeqTrainer（导入即触发 WinError 0xC0000005/栈溢出）。
   所以这里用「手动训练循环(纯torch)」。
"""
import os, sys, json, re, argparse
sys.path.insert(0, r"C:\Users\woshi\.dsh\aigc-detector")
import numpy as np
import torch  # 必须最先 import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments  # 不含 Trainer

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
D = r"C:\Users\woshi\.dsh\aigc-detector\data"
# 优先用净化后的语料(剔除了节标题/垃圾句, 且含保意sim_ref)，无则回退原语料
_clean = os.path.join(D, "rewrite_corpus_clean.jsonl")
CORPUS = _clean if os.path.exists(_clean) else os.path.join(D, "rewrite_corpus.jsonl")
OUT = r"C:\Users\woshi\.dsh\aigc-detector\models\qwen_rewrite"
DEFAULT_MODEL = r"C:\Users\woshi\.dsh\aigc-detector\models\Qwen2.5-1.5B-Instruct"

# 指令模板(降重/人化指令 + AI句)
SYS = "你是论文降重助手，把下面这句话改写成更自然、更像人写的学术表达，保持原意不变，不要增删信息，只输出改写后的句子。"
def build_prompt(ai_sentence):
    return f"{SYS}\n原句：{ai_sentence.strip()}\n改写："

def zhh(t): return len(re.findall(r"[\u4e00-\u9fff]", t))


def make_dataset(pairs, tok, max_len):
    """返回 list[{'input_ids':[..], 'labels':[..]}], loss只在target段。"""
    enc = []
    for p in pairs:
        inp = build_prompt(p["src_ai"])
        out = p["tgt_human"].strip()
        t = tok(inp, max_length=max_len, truncation=True)
        tgt = tok(out, max_length=max(max_len - len(t["input_ids"]), 8), truncation=True)
        full_ids = t["input_ids"] + tgt["input_ids"]
        if tgt["input_ids"] and tgt["input_ids"][-1] != tok.eos_token_id:
            full_ids.append(tok.eos_token_id)
        labels = [-100] * len(t["input_ids"]) + list(tgt["input_ids"])
        if len(labels) < len(full_ids):
            labels = labels + [tok.eos_token_id]
        enc.append({"input_ids": full_ids, "labels": labels[:len(full_ids)]})
    return enc


def collate_fn(batch, pad_id):
    input_ids = [b["input_ids"] for b in batch]
    labels = [b["labels"] for b in batch]
    m = max(len(x) for x in input_ids)
    ii = [x + [pad_id] * (m - len(x)) for x in input_ids]
    ll = [x + [-100] * (m - len(x)) for x in labels]
    am = [[0 if z == pad_id else 1 for z in row] for row in ii]
    return {"input_ids": torch.tensor(ii), "attention_mask": torch.tensor(am), "labels": torch.tensor(ll)}


def main():
    global args
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--max-len", type=int, default=128)
    ap.add_argument("--eval-split", type=float, default=0.1)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--lora", action="store_true")
    ap.add_argument("--skip-train", action="store_true")
    ap.add_argument("--max-train-steps", type=int, default=0, help=">0 则限制训练步数(快速冒烟)")
    ap.add_argument("--real-only", action="store_true", help="只用真·忠实对训练(隔离实验：验证合成数据稀释假说)")
    args = ap.parse_args()

    torch.manual_seed(42); np.random.seed(42)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print("[train_rewrite_qwen] device:", dev, "| base:", args.model, flush=True)
    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id

    rows = [json.loads(l) for l in open(CORPUS, encoding="utf-8")]
    if args.real_only:
        rows = [p for p in rows if p.get("src") in ("real", "wechat_real")]
    real = sum(1 for p in rows if p.get("src") in ("real", "wechat_real"))
    print(f"语料: 总 {len(rows)} | 真实人化 {real} | 自助合成 {len(rows)-real}", flush=True)
    import random
    random.shuffle(rows)

    n_eval = max(1, int(len(rows) * args.eval_split))
    if args.real_only:
        # 保意评估: 独立从清洗语料的真实对里抽专用测试集, 不掺训练
        clean = [json.loads(l) for l in open(os.path.join(D, "rewrite_corpus_clean.jsonl"), encoding="utf-8")]
        clean_real = [p for p in clean if p.get("src") in ("real", "wechat_real")]
        random.shuffle(clean_real)
        test_real = clean_real[: max(1, int(len(clean_real) * 0.2))]
        ev = test_real
        tr = [p for p in rows if p not in ev]
    else:
        ev, tr = rows[:n_eval], rows[n_eval:]

    model = None
    if not args.skip_train:
        ds_tr = make_dataset(tr, tok, args.max_len)
        ds_ev = make_dataset(ev, tok, args.max_len)
        tr_dl = DataLoader(ds_tr, batch_size=args.batch, shuffle=True, collate_fn=lambda b: collate_fn(b, tok.pad_token_id))
        ev_dl = DataLoader(ds_ev, batch_size=args.batch, shuffle=False, collate_fn=lambda b: collate_fn(b, tok.pad_token_id))

        model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16)
        if args.lora:
            try:
                from peft import LoraConfig, get_peft_model
                cfg = LoraConfig(r=8, lora_alpha=16, lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
                                 target_modules=["q_proj", "k_proj", "v_proj", "o_proj"])
                model = get_peft_model(model, cfg)
                print("[train_rewrite_qwen] LoRA 模式", flush=True)
            except Exception as e:
                print("[train_rewrite_qwen] peft 不可用，全参微调:", e, flush=True)
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()
        model.train()
        model.to(dev)
        opt = AdamW(model.parameters(), lr=args.lr)
        bf = torch.cuda.is_bf16_supported()
        print("[train_rewrite_qwen] 开始微调(千问 1.5B)...", flush=True)

        step = 0
        for epoch in range(args.epochs):
            total_loss, nb = 0.0, 0
            for bi, batch in enumerate(tr_dl):
                mbatch = {k: v.to(dev) for k, v in batch.items()}
                with torch.autocast(device_type=dev, dtype=torch.bfloat16 if bf else torch.float16):
                    out = model(**mbatch)
                loss = out.loss / args.grad_accum
                loss.backward()
                total_loss += float(out.loss.detach()); nb += 1
                if (bi + 1) % args.grad_accum == 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    opt.step(); opt.zero_grad(); step += 1
                    if step % 20 == 0:
                        print(f"  epoch{epoch+1} step{step} loss={total_loss/nb:.4f}", flush=True)
                    if args.max_train_steps > 0 and step >= args.max_train_steps:
                        print(f"  达到 max_train_steps={args.max_train_steps}, 提前结束", flush=True)
                        break
                if args.max_train_steps > 0 and step >= args.max_train_steps:
                    break
            print(f"[epoch{epoch+1}] 平均 loss={total_loss/max(nb,1):.4f}", flush=True)
            if args.max_train_steps > 0 and step >= args.max_train_steps:
                break
        model.save_pretrained(OUT); tok.save_pretrained(OUT)
        print(f"[train_rewrite_qwen] 模型保存到 {OUT}", flush=True)
    else:
        model = AutoModelForCausalLM.from_pretrained(OUT, torch_dtype=torch.bfloat16).to(dev)

    model.eval()

    # ---- 推理改写 + 评估 ----
    from detector.dual_stream import load_bert, fuse as ds_fuse, bert_score_per_sentence
    from scripts.cross_validate import stat_probs, load_cls
    stat = load_cls()
    bmtok = bmmod = bmdev = None
    try:
        bm = load_bert(device=dev); bmtok, bmmod, bmdev = bm
    except Exception:
        pass
    st = None
    try:
        from sentence_transformers import SentenceTransformer, util as stutil
        st = SentenceTransformer("shibing624/text2vec-base-chinese", device=dev)
    except Exception:
        pass

    def aiscore(s):
        pt = stat_probs(stat, [s])
        pb = bert_score_per_sentence(bmtok, bmmod, bmdev, [s], batch=4) if bmdev is not None else [0.5]
        return float(ds_fuse(float(pt[0]), float(pb[0])))

    def rewrite(src):
        inp = build_prompt(src)
        enc = tok(inp, return_tensors="pt").to(dev)
        with torch.no_grad():
            g = model.generate(**enc, max_new_tokens=args.max_len, do_sample=False, num_beams=4,
                               no_repeat_ngram_size=2, early_stopping=True)
        return tok.decode(g[0][enc["input_ids"].shape[1]:], skip_special_tokens=True).strip()

    print("\n=== 改写效果评估 (真实人化对, 抽60) ===", flush=True)
    ev_real = [p for p in ev if p.get("src") in ("real", "wechat_real")] or ev[:60]
    res = []
    for p in ev_real[:60]:
        src = p["src_ai"].strip(); tgt = p["tgt_human"].strip()
        s_src = aiscore(src)
        try:
            gen = rewrite(src)
        except Exception as e:
            gen = ""
        s_gen = aiscore(gen) if gen else s_src
        sim = None
        if st is not None and gen:
            try:
                sim = float(stutil.cos_sim(st.encode([src])[0], st.encode([gen])[0]).item())
            except Exception: sim = None
        res.append({"src_ai": src, "tgt_human": tgt, "gen": gen,
                    "src_ai_prob": round(s_src, 3), "gen_ai_prob": round(s_gen, 3),
                    "delta": round(s_src - s_gen, 3), "sim": round(sim, 3) if sim is not None else None})
    deltas = [r["delta"] for r in res]
    imps = sum(1 for d in deltas if d > 0); big = sum(1 for d in deltas if d > 0.15)
    print(f"已评估 {len(res)} | 降分成功 {imps}/{len(res)} ({100*imps/max(len(res),1):.0f}%) | 降>0.15 {big} | 平均降分 {np.mean(deltas):.3f}", flush=True)
    sims = [r["sim"] for r in res if r["sim"] is not None]
    if sims: print(f"语义相似度(与原文) 平均 {np.mean(sims):.3f}", flush=True)
    for r in res[:5]:
        print(f"  原:{r['src_ai'][:46]}... ({r['src_ai_prob']})", flush=True)
        print(f"  改:{r['gen'][:46]}... ({r['gen_ai_prob']}, 降{r['delta']})", flush=True)
    out_eval = os.path.join(D, "rewrite_qwen_eval.jsonl")
    with open(out_eval, "w", encoding="utf-8") as f:
        for r in res: f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"评估结果保存 {out_eval}", flush=True)


if __name__ == "__main__":
    main()
