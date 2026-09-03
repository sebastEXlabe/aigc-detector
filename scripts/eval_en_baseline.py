# -*- coding: utf-8 -*-
"""英文AIGC检测基线评估：用真实英文 AI/人类样本，量化当前(中文)检测管线的英文分辨力。
指标：stat流/bert流/fused 各自的 AUC、以及 AI/human 分数分布。
用法：python scripts/eval_en_baseline.py [--n 800]
"""
import os, sys, re, json, random
sys.path.insert(0, r"C:\Users\woshi\.dsh\aigc-detector")
import numpy as np
from scripts.cross_validate import stat_probs, load_cls, read_recs
from detector.dual_stream import load_bert, bert_score_per_sentence, fuse as ds_fuse

def is_en(s):
    s = (s or "").strip()
    return bool(re.fullmatch(r"[ -~]{20,}", s)) and len(re.findall(r"[A-Za-z]", s)) > 20

def main():
    import argparse
    ap = argparse.ArgumentParser(); ap.add_argument("--n", type=int, default=600)
    a = ap.parse_args()
    ai = [r["text"] for r in read_recs(r"C:\Users\woshi\.dsh\aigc-detector\data\ai_pub_samples.jsonl") if is_en(r.get("text"))]
    hu = [r["text"] for r in read_recs(r"C:\Users\woshi\.dsh\aigc-detector\data\human_pub_samples.jsonl") if is_en(r.get("text"))]
    random.seed(0); random.shuffle(ai); random.shuffle(hu)
    ai = ai[:a.n]; hu = hu[:a.n]
    print("英文AI", len(ai), "| 英文人类", len(hu), flush=True)

    cls = load_cls(); bm = load_bert(device="cuda"); tok, model, dev = bm
    def score(texts):
        stat = stat_probs(cls, texts)
        bert = bert_score_per_sentence(tok, model, dev, texts, batch=32)
        fused = [ds_fuse(float(s), float(b)) for s, b in zip(stat, bert)]
        return np.array(stat), np.array(bert), np.array(fused)

    sta, ber, fus = score(ai)
    hsta, hber, hfus = score(hu)

    from sklearn.metrics import roc_auc_score
    y = [1]*len(ai) + [0]*len(hu)
    print("\n=== 英文AIGC检测基线(当前中文管线) ===", flush=True)
    for name, a_s, h_s in [("stat(TFIDF中文)", sta, hsta), ("bert(中文RoBERTa)", ber, hber), ("fused", fus, hfus)]:
        auc = roc_auc_score(y, np.concatenate([a_s, h_s]))
        m_ai, m_hu = np.mean(a_s), np.mean(h_s)
        print(f"  {name}: AUC={auc:.3f} | AI均值={m_ai:.3f} | 人类均值={m_hu:.3f} | 区分度={m_ai-m_hu:+.3f}", flush=True)

if __name__ == "__main__":
    main()
