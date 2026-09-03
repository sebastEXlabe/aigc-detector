# -*- coding: utf-8 -*-
"""英文检测融合策略评估：en-classifier(0.891) + 中文RoBERTa(英文≈噪音) 如何融合最优。
对比：esclass-only / esclass+bertsc(中文) / esclass+avg / esclass+bertsc乘0.2
结论指导：英文文档应切英文分类器，中文RoBERTa深流对英文是否该弃用/降权。
用法：python scripts/eval_en_fusion.py [--n 500]
"""
import os, sys, re, json, random, pickle
sys.path.insert(0, r"C:\Users\woshi\.dsh\aigc-detector")
import numpy as np
from sklearn.metrics import roc_auc_score
from scripts.cross_validate import read_recs
from detector.dual_stream import load_bert, bert_score_per_sentence

def is_en(s):
    s = (s or "").strip()
    return bool(re.fullmatch(r"[ -~]{30,}", s)) and len(re.findall(r"[A-Za-z]", s)) > 40

def main():
    import argparse
    ap = argparse.ArgumentParser(); ap.add_argument("--n", type=int, default=500)
    a = ap.parse_args()
    D = r"C:\Users\woshi\.dsh\aigc-detector\data"
    ai = [r["text"] for r in read_recs(os.path.join(D, "ai_pub_samples.jsonl")) if is_en(r.get("text"))]
    hu = [r["text"] for r in read_recs(os.path.join(D, "human_pub_samples.jsonl")) if is_en(r.get("text"))]
    random.seed(0); random.shuffle(ai); random.shuffle(hu)
    ai = ai[:a.n]; hu = hu[:a.n]
    y = [1]*len(ai) + [0]*len(hu)
    texts = ai + hu

    en_cls = pickle.load(open(r"C:\Users\woshi\.dsh\aigc-detector\models\en_classifier.pkl", "rb"))
    evec, emodel = en_cls["vec"], en_cls["model"]
    ep = emodel.predict_proba(evec.transform(texts))[:, 1]
    bm = load_bert(device="cuda"); tok, model, dev = bm
    bp = np.array(bert_score_per_sentence(tok, model, dev, texts, batch=32))

    print("\n=== 英文-融合策略对比 ===", flush=True)
    configs = {
        "en-classifier only": ep,
        "en-cls + bertsc(中文RoBERTa) 等权": (ep+bp)/2,
        "en-cls + 0.3*bertsc": ep*0.7 + bp*0.3,
        "en-cls(stat) 与 bertsc 取 max": np.maximum(ep, bp),
        "仅 bertsc(中文)": bp,
    }
    for name, p in configs.items():
        auc = roc_auc_score(y, p)
        m_ai, m_hu = np.mean(p[:len(ai)]), np.mean(p[len(ai):])
        print(f"  {name}: AUC={auc:.3f} | AI={m_ai:.3f} 人类={m_hu:.3f}", flush=True)

if __name__ == "__main__":
    main()
