# -*- coding: utf-8 -*-
"""成本敏感阈值校准：在真实数据上按「不对称代价」找最优 AI 判定阈值。
误判代价：把真学术(人类)判成 AI 的代价 > 漏检 AI 的代价（用户核心诉求：别冤枉真论文）。
用法：python scripts/cost_sensitive_threshold.py [--fp-cost 3.0] [--fn-cost 1.0]
"""
import os, sys, numpy as np
sys.path.insert(0, r"C:\Users\woshi\.dsh\aigc-detector")
from detector.dual_stream import load_bert, bert_score_per_sentence, fuse as ds_fuse
from scripts.cross_validate import build_aigc_test, build_cnki_test, label_to_ai, read_recs, stat_probs, load_cls

def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--fp-cost", type=float, default=3.0, help="冤枉真论文(误报)的相对代价")
    ap.add_argument("--fn-cost", type=float, default=1.0, help="漏检AI的相对代价")
    a = ap.parse_args()
    cls = load_cls()
    # 真实 AIGC-AI 句
    aigc_test, _, ht = build_aigc_test(0.2, 42)
    ai_recs = [r for r in aigc_test if label_to_ai(r.get("label"))]
    ai_texts = [r["text"] for r in ai_recs]
    # 真实 CNKI 人类句
    tt = set()
    for r in read_recs(os.path.join(r"C:\Users\woshi\.dsh\aigc-detector\data","train_unified.jsonl")):
        tt.add(r.get("text"))
    cnki = build_cnki_test(2000, 42, tt)
    bm = load_bert(device="cuda")
    def fused(texts):
        pt = stat_probs(cls, texts)
        pb = bert_score_per_sentence(bm[0], bm[1], bm[2], texts, batch=32)
        return np.array([ds_fuse(float(x), float(y)) for x, y in zip(pt, pb)])
    p_ai = fused(ai_texts)      # 真 AI
    p_hu = fused(cnki)          # 真人类
    print(f"真实AI句={len(p_ai)}  真实CNKI人类句={len(p_hu)}")
    best = None
    print(f"代价: 误报(冤枉真论文)={a.fp_cost}x  漏检AI={a.fn_cost}x")
    for t in np.arange(0.30, 0.70, 0.01):
        fp = int((p_hu >= t).sum())      # 真人类被判AI
        fn = int((p_ai < t).sum())       # 真AI漏检
        cost = a.fp_cost*fp + a.fn_cost*fn
        fpr = fp/len(p_hu); recall = (p_ai >= t).mean()
        if best is None or cost < best[1]:
            best = (round(t,2), cost, fpr, recall)
    t, cost, fpr, recall = best
    print(f"\n成本最优阈值 = {t:.2f}（最低代价 {cost:.0f}）")
    print(f"  该点: 真实CNKI误报={fpr:.3f}  真实AIGC-AI检出={recall:.3f}")
    print("--- 阈值/误报/检出 表 ---")
    for t in (0.4, 0.45, 0.5, 0.55, 0.6):
        fp=int((p_hu>=t).sum()); fn=int((p_ai<t).sum()); c=a.fp_cost*fp+a.fn_cost*fn
        print(f"  阈{t:.2f}: 误报={fp/len(p_hu):.3f} 检出={(p_ai>=t).mean():.3f} 代价={c:.0f}")

if __name__ == "__main__":
    main()
