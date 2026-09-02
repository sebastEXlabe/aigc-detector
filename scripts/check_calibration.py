# -*- coding: utf-8 -*-
"""校准度评估：检查融合概率的 Brier + 可靠性分箱，判断"疑似"边界置信度是否可信。
实测(2026-09): Brier=0.063(良好)；高置信(0.8-1.0)预测0.888 vs 实际0.895 校准好；
但中段 0.2-0.6 过度向AI偏(预测0.286-0.500, 实际仅0.051-0.098) —— 说明部分真人学术句在
"证据不足/疑似"带被过度拉向AI，可用作进一步收紧该段边界的依据。
用法：python scripts/check_calibration.py
"""
import os, sys, json, numpy as np
sys.path.insert(0, r"C:\Users\woshi\.dsh\aigc-detector")
from detector.dual_stream import load_bert, bert_score_per_sentence, fuse as ds_fuse
from scripts.cross_validate import build_aigc_test, build_cnki_test, label_to_ai, read_recs, stat_probs, load_cls

def main():
    stat = load_cls(); bm = load_bert(device="cuda"); tok,model,dev = bm
    aigc_test,_,_ = build_aigc_test(0.2, 42)
    al = [1 if label_to_ai(r.get("label")) else 0 for r in aigc_test]
    ai_txt = [r["text"] for r in aigc_test]
    tt = set()
    for r in read_recs(os.path.join(r"C:\Users\woshi\.dsh\aigc-detector\data","train_unified.jsonl")): tt.add(r.get("text"))
    cnki = build_cnki_test(1500, 42, tt)
    texts = ai_txt + cnki; y = np.array(al + [0]*len(cnki))
    pt = stat_probs(stat, texts); pb = bert_score_per_sentence(tok,model,dev,texts,batch=64)
    fused = np.array([ds_fuse(float(a),float(b)) for a,b in zip(pt,pb)])
    brier = float(np.mean((fused-y)**2))
    print("=== 校准度 (Brier + 可靠性分箱) ===")
    print(f"Brier={brier:.4f} (0=完美, 0.25=随机)")
    print(" 桶   预测均值 | 实际AI比例 | n")
    for lo,hi in [(0,.2),(.2,.4),(.4,.6),(.6,.8),(.8,1)]:
        m=(fused>=lo)&(fused<hi)
        if m.sum()==0: continue
        print(f"  [{lo},{hi})  {float(fused[m].mean()):.3f} | {float(y[m].mean()):.3f} | {int(m.sum())}")

if __name__=="__main__": main()
