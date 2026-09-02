# -*- coding: utf-8 -*-
"""深流守卫：过滤 human 负样本池里被深流判为高AI的"污染样本"（真实是AI却标成human）。
用深流 RoBERTa（语义最强）交叉验证 human_pub_samples，剔除 deep score>=0.5 的样本，
生成干净的 human_pub_clean.jsonl，供统计流训练使用（避免污染标注）。

在 WSL 运行。
用法：python human_guard_filter.py [--thr 0.5]
"""
import os, sys, io, json, random
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
import torch, torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification

MODEL_PATH = "/mnt/c/Users/woshi/.dsh/aigc-detector/models/roberta_ft"
DATA = "/mnt/c/Users/woshi/.dsh/aigc-detector/data"

def zh_len(t): return len(re.findall(r"[\u4e00-\u9fff]", t))

def main():
    import argparse
    ap = argparse.ArgumentParser(); ap.add_argument("--thr", type=float, default=0.5); a = ap.parse_args()
    src = os.path.join(DATA,"human_pub_samples.jsonl")
    out_path = os.path.join(DATA,"human_pub_clean.jsonl")
    recs = [json.loads(l) for l in open(src, encoding="utf-8") if l.strip()]
    print("输入 human_pub_samples:", len(recs), flush=True)

    tok = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(dev); model.eval()
    print("模型加载OK", flush=True)

    texts = [r.get("text","") for r in recs]
    probs = []
    with torch.no_grad():
        for i in range(0, len(texts), 48):
            ch = texts[i:i+48]
            ch = [t[:200] for t in ch]
            inp = tok(ch, truncation=True, max_length=200, padding=True, return_tensors="pt")
            inp = {k: v.to(dev) for k, v in inp.items()}
            out = model(**inp).logits
            p = F.softmax(out,-1)[:,1].cpu().numpy()
            probs.extend(p.tolist())
            del inp,out,p
            if torch.cuda.is_available(): torch.cuda.empty_cache()
            if (i//48) % 100 == 0:
                print(f"  进度 {i}/{len(texts)}", flush=True)

    # 过滤：C-ReD-paper(中文学术论文human) 先验真学术，强制保留（深流对规范学术句有误报，不作剔除依据）；
    # 其余(HC3/M4问答百科类) 用深流判AI高(<thr)的剔除污染。
    keep = []
    dropped = 0
    for r, p in zip(recs, probs):
        if r.get("source") == "C-ReD-paper":
            keep.append(r)
        elif p < a.thr:
            keep.append(r)
        else:
            dropped += 1
    print(f"保留: {len(keep)} 剔除(判AI>=thr): {dropped}  (保留率={len(keep)/max(len(recs),1):.3f})", flush=True)
    from collections import Counter
    before = Counter(r.get("source") for r in recs)
    after = Counter(r.get("source") for r in keep)
    print("--- 各来源 保留/原量 ---")
    for k in sorted(before):
        print(f"  {k}: {after.get(k,0)} / {before[k]}")
    with open(out_path, "w", encoding="utf-8") as f:
        for r in keep:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print("已保存:", out_path, flush=True)

if __name__ == "__main__":
    main()
