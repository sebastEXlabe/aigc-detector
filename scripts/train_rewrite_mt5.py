# -*- coding: utf-8 -*-
"""本地降AIGC改写模型：在改写语料(rewrite_corpus.jsonl)上微调 mT5-small。
训练在 Windows python + CUDA torch (Python312, torch 2.6+cu124, 8GB VRAM) 执行。
用法：python scripts/train_rewrite_mt5.py [--epochs 3] [--max-len 128] [--eval-split 0.1]
步骤：载语料 → 划分train/eval → 微调mT5-small → 推理改写 → 用检测器验证降AIGC + 语义相似度。
"""
import os, sys, json, re, argparse, random
sys.path.insert(0, r"C:\Users\woshi\.dsh\aigc-detector")
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (AutoTokenizer, AutoModelForSeq2SeqLM,
                          Seq2SeqTrainer, Seq2SeqTrainingArguments,
                          DataCollatorForSeq2Seq, TrainerCallback)

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
MODEL = "google/mt5-small"
D = r"C:\Users\woshi\.dsh\aigc-detector\data"
CORPUS = os.path.join(D, "rewrite_corpus.jsonl")
OUT = r"C:\Users\woshi\.dsh\aigc-detector\models\mt5_rewrite"
PREFIX = "降到人话："

def zhh(t): return len(re.findall(r"[\u4e00-\u9fff]", t))
def sss(t): return [s for s in re.split(r"(?<=[。！？；])", t) if s.strip()]

def load_pairs():
    rows = [json.loads(l) for l in open(CORPUS, encoding="utf-8")]
    # 过滤太短/太长，保留中英文本
    out = []
    for r in rows:
        s = (r.get("src_ai") or "").strip()
        t = (r.get("tgt_human") or "").strip()
        if zhh(s) < 8 or zhh(t) < 8: continue
        if len(s) > args.max_len * 3 or len(t) > args.max_len * 3: continue
        out.append({"src_ai": s, "tgt_human": t, "src": r.get("src", "?")})
    return out

class RewriteDS(Dataset):
    def __init__(self, pairs, tok):
        self.pairs, self.tok = pairs, tok
    def __len__(self): return len(self.pairs)
    def __getitem__(self, i):
        p = self.pairs[i]
        inp = PREFIX + p["src_ai"]
        return {"inp": inp, "out": p["tgt_human"],
                "src_ai": p["src_ai"], "src_tag": p.get("src")}

def main():
    global args
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--max-len", type=int, default=128)
    ap.add_argument("--eval-split", type=float, default=0.1)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--skip-train", action="store_true")
    args = ap.parse_args()

    torch.manual_seed(42); random.seed(42); np.random.seed(42)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print("[train_rewrite_mt5] device:", dev, "| model:", MODEL)
    tok = AutoTokenizer.from_pretrained(MODEL)
    pairs = load_pairs()
    real = [p for p in pairs if p.get("src") in ("real", "wechat_real")]
    synth = [p for p in pairs if p.get("src") == "synth"]
    print(f"语料: 总 {len(pairs)} | 真实人化 {len(real)} | 自助合成 {len(synth)}")

    random.shuffle(pairs)
    n_eval = max(1, int(len(pairs) * args.eval_split))
    ev, tr = pairs[:n_eval], pairs[n_eval:]

    if not args.skip_train:
        ds_tr = RewriteDS(tr, tok); ds_ev = RewriteDS(ev, tok)
        coll = DataCollatorForSeq2Seq(tok, model=None, padding=True)
        # 手动 batch collate
        def batch_collate(batch):
            inps = [b["inp"] for b in batch]; outs = [b["out"] for b in batch]
            enc = tok(inps, max_length=args.max_len, truncation=True, padding=True, return_tensors="pt")
            dec = tok(outs, max_length=args.max_len, truncation=True, padding=True, return_tensors="pt")
            enc["labels"] = dec["input_ids"].clone()
            enc["labels"][enc["labels"] == tok.pad_token_id] = -100
            enc["src_ai"] = [b["src_ai"] for b in batch]
            enc["src_tag"] = [b["src_tag"] for b in batch]
            return enc
        tr_dl = DataLoader(ds_tr, batch_size=args.batch, shuffle=True, collate_fn=batch_collate)
        ev_dl = DataLoader(ds_ev, batch_size=args.batch, shuffle=False, collate_fn=batch_collate)

        model = AutoModelForSeq2SeqLM.from_pretrained(MODEL).to(dev)
        # bf16 若支持，否则 fp16
        dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
        targs = Seq2SeqTrainingArguments(
            output_dir=OUT, num_train_epochs=args.epochs,
            per_device_train_batch_size=args.batch, per_device_eval_batch_size=args.batch,
            learning_rate=args.lr, evaluation_strategy="no", save_strategy="no",
            logging_steps=20, fp16=(dtype == torch.float16), bf16=(dtype == torch.bfloat16),
            predict_with_generate=True, seed=42,
        )
        trainer = Seq2SeqTrainer(model=model, args=targs,
                                 train_dataset=tr_dl, eval_dataset=ev_dl)
        print("[train_rewrite_mt5] 开始微调...")
        trainer.train()
        model.save_pretrained(OUT)
        tok.save_pretrained(OUT)
        print(f"[train_rewrite_mt5] 模型保存到 {OUT}")
    else:
        from transformers import AutoModelForSeq2SeqLM
        model = AutoModelForSeq2SeqLM.from_pretrained(OUT).to(dev)

    # ---- 推理改写 ----
    from detector.dual_stream import load_bert, fuse as ds_fuse
    from scripts.cross_validate import stat_probs, load_cls
    stat = load_cls()
    try:
        bm = load_bert(device=dev); bmtok, bmmod, bmdev = bm
    except Exception:
        bmtok = bmmod = bmdev = None
    from sentence_transformers import SentenceTransformer, util as stutil  # 可选
    st = None
    try:
        st = SentenceTransformer("shibing624/text2vec-base-chinese", device=dev)
    except Exception:
        pass

    def aiscore(s):
        sents = [s] if isinstance(s, str) else s
        pt = stat_probs(stat, sents)
        if bmdev is not None:
            pb = __import__("detector.dual_stream", fromlist=["bert_score_per_sentence"]).bert_score_per_sentence(bmtok, bmmod, bmdev, sents, batch=4)
        else:
            pb = [0.5] * len(sents)
        return float(np.mean([ds_fuse(float(a), float(b)) for a, b in zip(pt, pb)]))

    def rewrite(src):
        inp = PREFIX + src
        with torch.no_grad():
            ids = tok(inp, return_tensors="pt").to(dev)
            g = model.generate(**ids, max_new_tokens=args.max_len, num_beams=4,
                               no_repeat_ngram_size=2, early_stopping=True)
        return tok.decode(g[0], skip_special_tokens=True)

    print("\n=== 改写效果评估 (对 eval 中真实人化对) ===")
    ev_real = [p for p in ev if p.get("src") in ("real", "wechat_real")] or ev[:max(1, int(len(ev)*0.3))]
    rows = []
    i = 0
    for p in ev_real[:60]:
        src = p["src_ai"].strip(); tgt = p["tgt_human"].strip()
        s_src = aiscore(src)
        gen = rewrite(src).strip()
        s_gen = aiscore(gen)
        sim = None
        if st is not None:
            try:
                sim = float(stutil.cos_sim(st.encode([src])[0], st.encode([gen])[0]).item())
            except Exception:
                sim = None
        rows.append({"src_ai": src, "tgt_human": tgt, "gen": gen,
                     "src_ai_prob": round(s_src, 3), "gen_ai_prob": round(s_gen, 3),
                     "delta": round(s_src - s_gen, 3), "sim": round(sim, 3) if sim is not None else None})
        i += 1
    if rows:
        deltas = [r["delta"] for r in rows]
        imps = sum(1 for d in deltas if d > 0)
        big = sum(1 for d in deltas if d > 0.15)
        print(f"已评估 {len(rows)} 条 | 降分成功 {imps}/{len(rows)} ({100*imps/len(rows):.0f}%) | 降>0.15 {big} | 平均降分 {np.mean(deltas):.3f}")
        sims = [r["sim"] for r in rows if r["sim"] is not None]
        if sims: print(f"语义相似度(与原文) 平均 {np.mean(sims):.3f} (越高越保真)")
        print("\n示例:")
        for r in rows[:5]:
            print(f"  原:{r['src_ai'][:50]}... ({r['src_ai_prob']})")
            print(f"  改:{r['gen'][:50]}... ({r['gen_ai_prob']}, 降{r['delta']})")
    out_eval = os.path.join(D, "rewrite_mt5_eval.jsonl")
    with open(out_eval, "w", encoding="utf-8") as f:
        for r in rows: f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"评估结果保存 {out_eval}")

if __name__ == "__main__":
    main()
