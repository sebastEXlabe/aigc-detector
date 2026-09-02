# -*- coding: utf-8 -*-
"""验证增强版深流 RoBERTa 的误报水平（WSL 内直接加载，Linux 路径）。
用法：在WSL: python verify_deep_aug.py
"""
import os, sys, io, json, re, random
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
import torch, torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification

MODEL_PATH = "/mnt/c/Users/woshi/.dsh/aigc-detector/models/roberta_ft"
DATA = "/mnt/c/Users/woshi/.dsh/aigc-detector/data"

def zh_len(t): return len(re.findall(r"[\u4e00-\u9fff]", t))

def read_jsonl(path):
    out = []
    for l in open(path, encoding="utf-8"):
        if l.strip():
            try: out.append(json.loads(l))
            except: pass
    return out

def sample_sents(recs, min_zh=20, n=150):
    pool = [r["text"] for r in recs if zh_len(r.get("text","") or "") >= min_zh]
    random.seed(1); random.shuffle(pool)
    return pool[:n]

human_sents = sample_sents(read_jsonl(os.path.join(DATA,"human_corpus.jsonl")))
ai_pub = [r for r in read_jsonl(os.path.join(DATA,"ai_pub_samples.jsonl")) if r.get("source")=="C-ReD-paper"]
ai_sents = sample_sents(ai_pub)
print(f"真人句:{len(human_sents)} AI句:{len(ai_sents)}", flush=True)

tok = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(dev); model.eval()
print("模型加载OK device=", dev, flush=True)

def score(sents):
    probs = []
    with torch.no_grad():
        for i in range(0, len(sents), 32):
            ch = sents[i:i+32]
            inp = tok(ch, truncation=True, max_length=200, padding="max_length", return_tensors="pt")
            inp = {k: v.to(dev) for k, v in inp.items()}
            out = model(**inp).logits
            p = F.softmax(out, -1)[:, 1].cpu().numpy()
            probs.extend(p.tolist())
            del inp, out, p
            if torch.cuda.is_available(): torch.cuda.empty_cache()
    return probs

p_h = score(human_sents)
p_a = score(ai_sents)
hr = float(sum(1 for p in p_h if p >= 0.5)/len(p_h))
ar = float(sum(1 for p in p_a if p >= 0.5)/len(p_a))
print(f"\n真人学术句: 均值={sum(p_h)/len(p_h):.3f} 误报率(>=0.5)={hr:.3f}", flush=True)
print(f"AI学术句:   均值={sum(p_a)/len(p_a):.3f} 检出率(>=0.5)={ar:.3f}", flush=True)
