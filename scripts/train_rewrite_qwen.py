# -*- coding: utf-8 -*-
"""本地降AIGC改写模型：在改写语料(rewrite_corpus.jsonl)上微调 千问 Qwen 基座(causal decoder)。
基座：Qwen2.5-1.5B-Instruct(默认, 8GB VRAM 可全参/慢速) 或 config 可换 Qwen3-1.7B / LoRA。
用途：把AI句改写成更像人写的自然表达(降AIGC)，保持原意，本地零token。
用法：
  python scripts/train_rewrite_qwen.py                 # 默认 1.5B 全参微调 + 评估
  python scripts/train_rewrite_qwen.py --model Qwen/Qwen2.5-1.5B-Instruct --loara --epochs 3
评估：生成改写 → 用检测器(aigc-detector 双流融合)验证降AIGC + 语义相似度(cos)。
注意：训练需独占GPU(与挖矿/检测服务冲突)；先停其他CUDA任务再跑。
"""
import os, sys, json, re, argparse
sys.path.insert(0, r"C:\Users\woshi\.dsh\aigc-detector")
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (AutoTokenizer, AutoModelForCausalLM,
                          Trainer, TrainingArguments, DataCollatorForSeq2Seq)

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
D = r"C:\Users\woshi\.dsh\aigc-detector\data"
CORPUS = os.path.join(D, "rewrite_corpus.jsonl")
OUT = r"C:\Users\woshi\.dsh\aigc-detector\models\qwen_rewrite"
DEFAULT_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"

# 指令模板(降重/人化指令 + AI句)
SYS = "你是论文降重助手，把下面这句话改写成更自然、更像人写的学术表达，保持原意不变，不要增删信息，只输出改写后的句子。"
def build_prompt(ai_sentence):
    return f"{SYS}\n原句：{ai_sentence.strip()}\n改写："

def zhh(t): return len(re.findall(r"[\u4e00-\u9fff]", t))

class RewriteDS(Dataset):
    def __init__(self, pairs, tok, max_len):
        self.tok, self.max_len = tok, max_len
        self.enc = []
        for p in pairs:
            inp = build_prompt(p["src_ai"])
            out = p["tgt_human"].strip()
            t = self.tok(inp, max_length=max_len, truncation=True)
            tgt = self.tok(out, max_length=max_len - len(t["input_ids"]), truncation=True)
            full_ids = t["input_ids"] + tgt["input_ids"]
            if tgt["input_ids"] and tgt["input_ids"][-1] != self.tok.eos_token_id:
                full_ids.append(self.tok.eos_token_id)
            labels = [-100] * len(t["input_ids"]) + list(tgt["input_ids"])
            if len(labels) < len(full_ids):
                labels = labels + [tok.eos_token_id]
            self.enc.append({"input_ids": full_ids, "labels": labels[:len(full_ids)]})
    def __len__(self): return len(self.enc)
    def __getitem__(self, i):
        return dict(self.enc[i])

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
    ap.add_argument("--lora", action="store_true", help="用LoRA省显存")
    ap.add_argument("--skip-train", action="store_true")
    args = ap.parse_args()

    torch.manual_seed(42); np.random.seed(42)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print("[train_rewrite_qwen] device:", dev, "| base:", args.model)
    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id

    rows = [json.loads(l) for l in open(CORPUS, encoding="utf-8")]
    real = [p for p in rows if p.get("src") in ("real", "wechat_real")]
    synth = [p for p in rows if p.get("src") == "synth"]
    print(f"语料: 总 {len(rows)} | 真实人化 {len(real)} | 自助合成 {len(synth)}")
    random.shuffle(rows)

    n_eval = max(1, int(len(rows) * args.eval_split))
    ev, tr = rows[:n_eval], rows[n_eval:]

    if not args.skip_train:
        ds_tr = RewriteDS(tr, tok, args.max_len)
        ds_ev = RewriteDS(ev, tok, args.max_len)

        def collate(batch):
            input_ids = [b["input_ids"] for b in batch]
            labels = [b["labels"] for b in batch]
            m = max(len(x) for x in input_ids)
            pad = tok.pad_token_id
            ii = [x + [pad] * (m - len(x)) for x in input_ids]
            ll = [x + [-100] * (m - len(x)) for x in labels]
            return {"input_ids": torch.tensor(ii), "attention_mask": torch.tensor([[0 if z == pad else 1 for z in row] for row in ii]), "labels": torch.tensor(ll)}

        tr_dl = DataLoader(ds_tr, batch_size=args.batch, shuffle=True, collate_fn=collate)
        ev_dl = DataLoader(ds_ev, batch_size=args.batch, shuffle=False, collate_fn=collate)

        model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16)
        if args.lora:
            try:
                from peft import LoraConfig, get_peft_model
                cfg = LoraConfig(r=8, lora_alpha=16, lora_dropout=0.05, bias="none",
                                 task_type="CAUSAL_LM", target_modules=["q_proj", "k_proj", "v_proj", "o_proj"])
                model = get_peft_model(model, cfg)
                print("[train_rewrite_qwen] LoRA 训练模式")
            except ModuleNotFoundError:
                print("[train_rewrite_qwen] 未安装 peft，回退为全参微调(1.5B 在8GB下配合梯度检查点可行)")
        model.enable_input_require_grads()
        model.gradient_checkpointing_enable()
        model = model.to(dev)
        bf = torch.cuda.is_bf16_supported()

        targs = TrainingArguments(
            output_dir=OUT, num_train_epochs=args.epochs,
            per_device_train_batch_size=args.batch, gradient_accumulation_steps=args.grad_accum,
            per_device_eval_batch_size=1, learning_rate=args.lr,
            evaluation_strategy="no", save_strategy="no", logging_steps=10,
            bf16=bf, fp16=(not bf), seed=42, gradient_checkpointing=True,
        )
        trainer = Trainer(model=model, args=targs, train_dataset=tr_dl, eval_dataset=ev_dl)
        print("[train_rewrite_qwen] 开始微调(千问)...")
        trainer.train()
        model.save_pretrained(OUT); tok.save_pretrained(OUT)
        print(f"[train_rewrite_qwen] 模型保存到 {OUT}")
    else:
        model = AutoModelForCausalLM.from_pretrained(OUT, torch_dtype=torch.bfloat16).to(dev)

    # ---- 推理改写 + 评估 ----
    from detector.dual_stream import load_bert, fuse as ds_fuse
    from scripts.cross_validate import stat_probs, load_cls
    stat = load_cls()
    bmtok = bmmod = bmdev = None
    try:
        bm = load_bert(device=dev); bmtok, bmmod, bmdev = bm
    except Exception:
        pass
    try:
        from sentence_transformers import SentenceTransformer, util as stutil
        st = SentenceTransformer("shibing624/text2vec-base-chinese", device=dev)
    except Exception:
        st = None
    from detector.dual_stream import bert_score_per_sentence

    def aiscore(s):
        pt = stat_probs(stat, [s])
        if bmdev is not None:
            pb = bert_score_per_sentence(bmtok, bmmod, bmdev, [s], batch=4)
        else:
            pb = [0.5]
        return float(ds_fuse(float(pt[0]), float(pb[0])))

    def rewrite(src):
        inp = build_prompt(src)
        enc = tok(inp, return_tensors="pt").to(dev)
        with torch.no_grad():
            g = model.generate(**enc, max_new_tokens=args.max_len, do_sample=False,
                               num_beams=4, no_repeat_ngram_size=2, early_stopping=True)
        return tok.decode(g[0][enc["input_ids"].shape[1]:], skip_special_tokens=True).strip()

    print("\n=== 改写效果评估 (真实人化对, 抽60) ===")
    ev_real = [p for p in ev if p.get("src") in ("real", "wechat_real")] or ev[:60]
    rows = []
    for p in ev_real[:60]:
        src = p["src_ai"].strip(); tgt = p["tgt_human"].strip()
        s_src = aiscore(src)
        gen = rewrite(src)
        s_gen = aiscore(gen)
        sim = None
        if st is not None:
            try:
                sim = float(stutil.cos_sim(st.encode([src])[0], st.encode([gen])[0]).item())
            except Exception: sim = None
        rows.append({"src_ai": src, "tgt_human": tgt, "gen": gen,
                     "src_ai_prob": round(s_src, 3), "gen_ai_prob": round(s_gen, 3),
                     "delta": round(s_src - s_gen, 3), "sim": round(sim, 3) if sim is not None else None})
    deltas = [r["delta"] for r in rows]
    imps = sum(1 for d in deltas if d > 0); big = sum(1 for d in deltas if d > 0.15)
    print(f"已评估 {len(rows)} | 降分成功 {imps}/{len(rows)} ({100*imps/max(len(rows),1):.0f}%) | 降>0.15 {big} | 平均降分 {np.mean(deltas):.3f}")
    sims = [r["sim"] for r in rows if r["sim"] is not None]
    if sims: print(f"语义相似度 平均 {np.mean(sims):.3f}")
    for r in rows[:5]:
        print(f"  原:{r['src_ai'][:46]}... ({r['src_ai_prob']})")
        print(f"  改:{r['gen'][:46]}... ({r['gen_ai_prob']}, 降{r['delta']})")
    out_eval = os.path.join(D, "rewrite_qwen_eval.jsonl")
    with open(out_eval, "w", encoding="utf-8") as f:
        for r in rows: f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"评估结果保存 {out_eval}")

import random
if __name__ == "__main__":
    main()
