# -*- coding: utf-8 -*-
"""融合权重(bias)扫描：在真实 AIGC 报告 + 真实 CNKI 上，扫 TFIDF_BIAS_ON_DISAGREE，
找「真实 CNKI 误报最低 / AIGC 检出最高」的融合参数。
"""
import os, sys, io, numpy as np
sys.path.insert(0, r"C:\Users\woshi\.dsh\aigc-detector")
from detector.dual_stream import load_bert, bert_score_per_sentence
from scripts.cross_validate import build_aigc_test, build_cnki_test, label_to_ai, read_recs, stat_probs, evaluate, load_cls

def fuse_bias(pa, pb, bias):
    a = np.asarray(pa, dtype=float); b = np.asarray(pb, dtype=float)
    diff = np.abs(a - b)
    agree = diff <= 0.25
    out = np.empty_like(a)
    out[agree] = (a[agree] + b[agree]) / 2
    out[~agree] = bias * a[~agree] + (1 - bias) * b[~agree]
    return out

def main():
    cls = load_cls()
    aigc_test, _, ht = build_aigc_test(0.2, 42)
    aigc_labels = [1 if label_to_ai(r.get("label")) else 0 for r in aigc_test]
    texts = [r["text"] for r in aigc_test]
    p_tf = stat_probs(cls, texts)
    bm = load_bert(device="cuda")
    bp = bert_score_per_sentence(bm[0], bm[1], bm[2], texts, batch=32)
    thr = cls.get("threshold", 0.345)
    # 真实 CNKI
    train_texts = set()
    for r in read_recs(os.path.join(r"C:\Users\woshi\.dsh\aigc-detector\data","train_unified.jsonl")):
        train_texts.add(r.get("text"))
    cnki = build_cnki_test(800, 42, train_texts)
    c_tf = stat_probs(cls, cnki)
    c_bp = bert_score_per_sentence(bm[0], bm[1], bm[2], cnki, batch=32)

    print("=== 融合权重(bias)扫描（真实数据）===")
    print(f"报告 {len(ht)}份/句{len(aigc_test)}  CNKI {len(cnki)}句")
    best = None
    for bias in (0.0, 0.2, 0.35, 0.5, 0.65, 0.8, 1.0):
        fa = fuse_bias(p_tf, bp, bias)
        fc = fuse_bias(c_tf, c_bp, bias)
        ea = evaluate(fa, aigc_labels, thr)
        ec = evaluate(fc, [0]*len(cnki), thr)
        # 3% 误报预算下的 AIGC 检出
        rec_at_3 = None
        for t in np.arange(0.2, 0.65, 0.02):
            e = evaluate(fc, [0]*len(cnki), t)
            if e['fpr'] <= 0.03:
                rec_at_3 = evaluate(fa, aigc_labels, t)['ai_recall']; break
        print(f"  bias={bias:.2f}: CNKI误报={ec['fpr']:.3f}  AIGC检出={ea['ai_recall']:.3f}  (3%误报预算检出={rec_at_3 if rec_at_3 is None else round(rec_at_3,3)})")
        score = ea['ai_recall'] - 2*ec['fpr']  # 目标：高检出 + 低误报
        if best is None or score > best[0]: best = (score, bias, ea['ai_recall'], ec['fpr'])
    print(f"  最优 bias={best[1]:.2f} (AIGC检出{best[2]:.3f} / CNKI误报{best[3]:.3f})")

if __name__ == "__main__":
    main()
