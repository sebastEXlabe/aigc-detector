# -*- coding: utf-8 -*-
"""段落级降重改写训练：输入=整段(带上下文)，输出=人化改写后的整段。
关键改进：给模型语义锚点(段落上下文)，让它能"保意+降重"，而不是编通用学术话。
数据：data/rewrite_para_pairs.jsonl (src_para -> tgt_para)。
用法：python scripts/train_rewrite_para.py --lora --epochs 5
"""
import os, sys, json, re, argparse
sys.path.insert(0, r"C:\Users\woshi\.dsh\aigc-detector")
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import AutoTokenizer, AutoModelForCausalLM

def zhh(t): return len(re.findall(r"[\u4e00-\u9fff]", t or ""))

def ref_clean_ok(src, tgt):
    """参考质量门槛：目标段落不得引入'序号漂移'或'明显中英混杂'。"""
    if not src or not tgt: return False
    def nummark(s): return len(re.findall(r"[（(]\s*[0-9一二三四五]+\s*[）)]|^\s*\d+[\.、]", s))
    def enwords(s): return len(re.findall(r"[A-Za-z]{2,}", s))
    if nummark(tgt) > 0 and nummark(src) == 0: return False  # 原无序号改后有
    if enwords(tgt) > enwords(src) + 1: return False  # 目标英文词明显多
    return True

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
D = r"C:\Users\woshi\.dsh\aigc-detector\data"
PARA_CORPUS = os.path.join(D, "rewrite_para_pairs.jsonl")
OUT = r"C:\Users\woshi\.dsh\aigc-detector\models\qwen_rewrite_para"
BASE = r"C:\Users\woshi\.dsh\aigc-detector\models\Qwen2.5-1.5B-Instruct"

SYS = ("你是论文降重助手，把下面这一段改写成更自然、更像人写的学术表达。"
       "硬性要求：保留原文全部信息与主旨，不增删事实，不改专业术语，"
       "保持原文的序号结构（如①②、1.2、第一/其一等）不变，"
       "中文术语保持中文、不要混入英文单词，只输出改写后的整段。")
def build_prompt(para): return f"{SYS}\n原文段落：{para.strip()}\n改写："

def make_dataset(pairs, tok, max_len):
    enc = []
    for p in pairs:
        inp = build_prompt(p["src_para"])
        out = p["tgt_para"].strip()
        t = tok(inp, max_length=max_len, truncation=True)
        tgt = tok(out, max_length=max(max_len - len(t["input_ids"]), 8), truncation=True)
        full = t["input_ids"] + tgt["input_ids"]
        if tgt["input_ids"] and tgt["input_ids"][-1] != tok.eos_token_id:
            full.append(tok.eos_token_id)
        labels = [-100]*len(t["input_ids"]) + list(tgt["input_ids"])
        if len(labels) < len(full):
            labels = labels + [tok.eos_token_id]
        enc.append({"input_ids": full, "labels": labels[:len(full)]})
    return enc

def collate_fn(batch, pad):
    ii = [b["input_ids"] for b in batch]; ll = [b["labels"] for b in batch]
    m = max(len(x) for x in ii)
    ii = [x + [pad]*(m-len(x)) for x in ii]; ll = [x + [-100]*(m-len(x)) for x in ll]
    am = [[0 if z == pad else 1 for z in r] for r in ii]
    return {"input_ids": torch.tensor(ii), "attention_mask": torch.tensor(am), "labels": torch.tensor(ll)}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=BASE)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--max-len", type=int, default=384)
    ap.add_argument("--eval-split", type=float, default=0.15)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--grad-accum", type=int, default=4)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--lora", action="store_true")
    ap.add_argument("--max-train-steps", type=int, default=0)
    ap.add_argument("--skip-train", action="store_true")
    args = ap.parse_args()

    torch.manual_seed(42); np.random.seed(42)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print("[para] device:", dev, flush=True)
    tok = AutoTokenizer.from_pretrained(args.base)
    if tok.pad_token_id is None: tok.pad_token_id = tok.eos_token_id

    rows = [json.loads(l) for l in open(PARA_CORPUS, encoding="utf-8")]
    rows = [p for p in rows if zhh(p.get("src_para")) >= 30 and zhh(p.get("tgt_para")) >= 30]
    rows = [p for p in rows if ref_clean_ok(p.get("src_para"), p.get("tgt_para"))]
    print("[para] 段落配对(净化后):", len(rows), flush=True)
    import random; random.shuffle(rows)
    n_eval = max(1, int(len(rows)*args.eval_split))
    ev, tr = rows[:n_eval], rows[n_eval:]
    print(f"[para] 训练 {len(tr)} | 评估 {len(ev)}", flush=True)

    model = None
    if not args.skip_train:
        ds_tr = make_dataset(tr, tok, args.max_len)
        tr_dl = DataLoader(ds_tr, batch_size=args.batch, shuffle=True, collate_fn=lambda b: collate_fn(b, tok.pad_token_id))
        model = AutoModelForCausalLM.from_pretrained(args.base, torch_dtype=torch.bfloat16)
        if args.lora:
            from peft import LoraConfig, get_peft_model
            cfg = LoraConfig(r=8, lora_alpha=16, lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
                             target_modules=["q_proj","k_proj","v_proj","o_proj"])
            model = get_peft_model(model, cfg)
        model.gradient_checkpointing_enable(); model.enable_input_require_grads(); model.train(); model.to(dev)
        opt = AdamW(model.parameters(), lr=args.lr)
        bf = torch.cuda.is_bf16_supported()
        print("[para] 开始训练...", flush=True)
        step = 0
        for epoch in range(args.epochs):
            tot, nb = 0.0, 0
            for bi, batch in enumerate(tr_dl):
                mb = {k: v.to(dev) for k, v in batch.items()}
                with torch.autocast(device_type=dev, dtype=torch.bfloat16 if bf else torch.float16):
                    out = model(**mb)
                loss = out.loss / args.grad_accum; loss.backward()
                tot += float(out.loss.detach()); nb += 1
                if (bi+1) % args.grad_accum == 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); opt.zero_grad(); step += 1
                    if step % 20 == 0: print(f"  epoch{epoch+1} step{step} loss={tot/nb:.4f}", flush=True)
                    if args.max_train_steps > 0 and step >= args.max_train_steps: break
                if args.max_train_steps > 0 and step >= args.max_train_steps: break
            print(f"[epoch{epoch+1}] 平均 loss={tot/max(nb,1):.4f}", flush=True)
            if args.max_train_steps > 0 and step >= args.max_train_steps: break
        model.save_pretrained(OUT); tok.save_pretrained(OUT)
        print(f"[para] 模型保存到 {OUT}", flush=True)
    else:
        model = AutoModelForCausalLM.from_pretrained(OUT, torch_dtype=torch.bfloat16).to(dev)
    model.eval()

    # ---- 推理改写 + 诚实评估(段落级 sim + 降分) ----
    from detector.dual_stream import load_bert, fuse as ds_fuse, bert_score_per_sentence
    from scripts.cross_validate import stat_probs, load_cls
    stat = load_cls(); bmtok=bmmod=bmdev=None
    try:
        bm = load_bert(device=dev); bmtok, bmmod, bmdev = bm
    except Exception: pass
    def embed(s):
        inp = bmtok(s, return_tensors="pt", truncation=True, max_length=380).to(bmdev)
        with torch.no_grad():
            out = bmmod(**inp, output_hidden_states=True)
        h = out.hidden_states[-1]; m = inp["attention_mask"].unsqueeze(-1).float()
        return ((h*m).sum(1)/m.sum(1).clamp(min=1e-9))[0].float().cpu()
    def aiscore(s):
        pt = stat_probs(stat, [s]); pb = bert_score_per_sentence(bmtok, bmmod, bmdev, [s], batch=4)
        return float(ds_fuse(float(pt[0]), float(pb[0])))
    def rewrite(para):
        enc = tok(build_prompt(para), return_tensors="pt").to(dev)
        with torch.no_grad():
            g = model.generate(**enc, max_new_tokens=args.max_len, do_sample=False, num_beams=4,
                               no_repeat_ngram_size=2, early_stopping=True)
        return tok.decode(g[0][enc["input_ids"].shape[1]:], skip_special_tokens=True).strip()

    res = []
    for p in ev[: max(1, int(len(ev)*0.5))]:
        s = p["src_para"].strip()
        s_src = aiscore(s); gen = rewrite(s); s_gen = aiscore(gen) if gen else s_src
        sim = float(torch.nn.functional.cosine_similarity(embed(s).unsqueeze(0), embed(gen).unsqueeze(0)).item()) if gen else None
        # 参考相似度(真实人化 vs 原段)
        sim_ref = float(torch.nn.functional.cosine_similarity(embed(s).unsqueeze(0), embed(p["tgt_para"].strip()).unsqueeze(0)).item())
        res.append({"src": s, "tgt": p["tgt_para"], "gen": gen,
                    "src_prob": round(s_src,3), "gen_prob": round(s_gen,3),
                    "delta": round(s_src-s_gen,3),
                    "sim": round(sim,3) if sim else None, "sim_ref": round(sim_ref,3)})
    ds = [r["delta"] for r in res]; sg = [r["sim"] for r in res if r["sim"] is not None]
    print("\n=== 段落级诚实评估 ===", flush=True)
    print(f"样本 {len(res)} | 降分 {sum(1 for d in ds if d>0)}/{len(ds)} ({100*sum(1 for d in ds if d>0)/max(len(ds),1):.0f}%) 平均降分 {np.mean(ds):.3f}", flush=True)
    if sg: print(f"模型段落相似度 平均 {np.mean(sg):.3f} | 参考相似度 平均 {np.mean([r['sim_ref'] for r in res]):.3f}", flush=True)
    good = [r for r in res if r["sim"] and r["sim"]>=0.7 and r["delta"]>0.1]
    game = [r for r in res if r["sim"] and r["sim"]<0.4 and r["delta"]>0.1]
    print(f"真降重(相似度>=0.7 且 降分>0.1): {len(good)} | 跑题(相似度<0.4 且 降分>0.1): {len(game)}", flush=True)
    for r in res[:5]:
        print(f"  原段:{r['src'][:50]}... ({r['src_prob']})", flush=True)
        print(f"  改后:{r['gen'][:50]}... ({r['gen_prob']}, 降{r['delta']}, 相似{r['sim']})", flush=True)
    with open(os.path.join(D, "rewrite_para_eval.jsonl"), "w", encoding="utf-8") as f:
        for r in res: f.write(json.dumps(r, ensure_ascii=False)+"\n")
    print("保存 data/rewrite_para_eval.jsonl", flush=True)

if __name__ == "__main__":
    main()
