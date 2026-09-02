# -*- coding: utf-8 -*-
"""验证文档一致性(TTA)校准：在真实 AIGC 报告测试集上，按报告分组应用校准，
对比校准前后 FPR（标为人却被判AI）与 AI 检出。
"""
import os, sys, io, json, numpy as np, collections
sys.path.insert(0, r"C:\Users\woshi\.dsh\aigc-detector")
from detector.dual_stream import load_bert, bert_score_per_sentence, fuse as ds_fuse
from detector.consistency import document_calibrate
from scripts.cross_validate import build_aigc_test, build_cnki_test, label_to_ai, read_recs, stat_probs, evaluate, load_cls

def main():
    cls = load_cls()
    aigc_test, _, ht = build_aigc_test(0.2, 42)
    aigc_labels = [1 if label_to_ai(r.get("label")) else 0 for r in aigc_test]
    texts = [r["text"] for r in aigc_test]
    doc_ids = [r.get("title") or "doc" for r in aigc_test]

    p_tf = stat_probs(cls, texts)
    bm = load_bert(device="cuda")
    bp = bert_score_per_sentence(bm[0], bm[1], bm[2], texts, batch=32)
    fused = np.array([ds_fuse(float(a), float(b)) for a, b in zip(p_tf, bp)])
    thr = cls.get("threshold", 0.345)

    base = evaluate(fused, aigc_labels, thr)
    cal = document_calibrate(fused, doc_ids)
    calR = evaluate(cal, aigc_labels, thr)
    print("=== 文档一致性(TTA)校准 —— 真实 AIGC 报告（按报告分组）===")
    print(f"报告 {len(ht)} 份 / 句 {len(aigc_test)}（AI {sum(aigc_labels)} / human {len(aigc_labels)-sum(aigc_labels)}）")
    print(f"校准前  阈{thr:.3f}: AI检出={base['ai_recall']:.3f}  人类误报(FPR)={base['fpr']:.3f}  AUC={base['auc']:.3f}")
    print(f"校准后  阈{thr:.3f}: AI检出={calR['ai_recall']:.3f}  人类误报(FPR)={calR['fpr']:.3f}  AUC={calR['auc']:.3f}")
    print("--- 各阈值下 ---")
    for t in (0.3,0.4,0.5):
        e1=evaluate(fused,aigc_labels,t); e2=evaluate(cal,aigc_labels,t)
        print(f"  阈{t:.1f}: 校前 检出{e1['ai_recall']:.3f}/误报{e1['fpr']:.3f}  校后 检出{e2['ai_recall']:.3f}/误报{e2['fpr']:.3f}")

if __name__ == "__main__":
    main()
