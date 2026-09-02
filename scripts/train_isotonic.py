# -*- coding: utf-8 -*-
"""训练 isotonic 校准模型并保存 models/isotonic.pkl（供 detect_pipeline 用）。
用法：python scripts/train_isotonic.py
"""
import os, sys, pickle, numpy as np
sys.path.insert(0, r"C:\Users\woshi\.dsh\aigc-detector")
from sklearn.isotonic import IsotonicRegression
from detector.dual_stream import load_bert, bert_score_per_sentence, fuse as ds_fuse
from scripts.cross_validate import build_aigc_test, build_cnki_test, label_to_ai, read_recs, stat_probs, load_cls

def main():
    stat = load_cls(); bm = load_bert(device="cuda"); tok,model,dev = bm
    aigc_test,_,_ = build_aigc_test(0.2, 42)
    al = [1 if label_to_ai(r.get("label")) else 0 for r in aigc_test]
    ai_txt = [r["text"] for r in aigc_test]
    tt = set()
    for r in read_recs(os.path.join(r"C:\Users\woshi\.dsh\aigc-detector\data","train_unified.jsonl")): tt.add(r.get("text"))
    cnki = build_cnki_test(5000, 42, tt)
    texts = ai_txt + cnki; y = np.array(al + [0]*len(cnki))
    pt = stat_probs(stat, texts); pb = bert_score_per_sentence(tok,model,dev,texts,batch=64)
    fused = np.array([ds_fuse(float(a),float(b)) for a,b in zip(pt,pb)])
    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(fused, y)
    out = os.path.join(r"C:\Users\woshi\.dsh\aigc-detector\models", "isotonic.pkl")
    pickle.dump(iso, open(out, "wb"))
    print(f"isotonic 校准模型已保存: {out}")
    print(f"  校准样本 {len(fused)} (AI {int(al.count(1)+0)} / human {len(cnki)})")

if __name__ == "__main__":
    main()
