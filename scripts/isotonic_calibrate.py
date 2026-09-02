# -*- coding: utf-8 -*-
"""isotonic 校准：改善"证据不足/疑似"中段概率可靠度。
拟合 isotonic 把 fused 概率映射为更可靠概率，留出验证 Brier/可靠性。
用法：python scripts/isotonic_calibrate.py
"""
import os, sys, json, numpy as np
sys.path.insert(0, r"C:\Users\woshi\.dsh\aigc-detector")
from sklearn.isotonic import IsotonicRegression
from detector.dual_stream import load_bert, bert_score_per_sentence, fuse as ds_fuse
from scripts.cross_validate import build_aigc_test, build_cnki_test, label_to_ai, read_recs, stat_probs, load_cls

def main():
    stat=load_cls(); bm=load_bert(device="cuda"); tok,model,dev=bm
    # 真实标注集：真AI(AIGC报告的AI句) + 真人类(CNKI去重)  → 独立划分校准/验证
    aigc_test,_,_ = build_aigc_test(0.2, 42)
    al=[1 if label_to_ai(r.get("label")) else 0 for r in aigc_test]
    ai_txt=[r["text"] for r in aigc_test]
    tt=set()
    for r in read_recs(os.path.join(r"C:\Users\woshi\.dsh\aigc-detector\data","train_unified.jsonl")): tt.add(r.get("text"))
    cnki=build_cnki_test(3000,42,tt)
    texts=ai_txt+cnki; y=np.array(al+[0]*len(cnki))
    pt=stat_probs(stat,texts); pb=bert_score_per_sentence(tok,model,dev,texts,batch=64)
    fused=np.array([ds_fuse(float(a),float(b)) for a,b in zip(pt,pb)])
    # 划分（按 0.5 前 / 后 分层保平衡）
    rng=np.random.RandomState(11); idx=rng.permutation(len(fused))
    ncal=int(len(fused)*0.6); cal,val=idx[:ncal],idx[ncal:]
    # 拟合 isotonic（仅用校准集）
    iso=IsotonicRegression(out_of_bounds="clip")
    iso.fit(fused[cal], y[cal])
    # 验证
    def brier(p,yy): return float(np.mean((p-yy)**2))
    b_raw=brier(fused[val],y[val]); b_iso=brier(iso.predict(fused[val]),y[val])
    print(f"=== isotonic 校准（留出验证 n={len(val)}）===")
    print(f"Brier: 原始 {b_raw:.4f} → isotonic {b_iso:.4f}  (改善 {b_raw-b_iso:+.4f})")
    for lo,hi in [(0,.2),(.2,.4),(.4,.6),(.6,.8),(.8,1)]:
        m=(fused[val]>=lo)&(fused[val]<hi)
        if m.sum()==0: continue
        pr=iso.predict(fused[val][m])
        print(f"  [{lo},{hi}) 原始预测{fused[val][m].mean():.3f}/实际{y[val][m].mean():.3f} | isotonic {pr.mean():.3f}")

if __name__=="__main__": main()
