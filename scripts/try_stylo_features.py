# -*- coding: utf-8 -*-
"""B1: 给统计流补正交文体特征。TF-IDF vs TF-IDF+stylo(scalars) 在真实数据上对比。
（第一性原理：n-gram + 深流都没覆盖"人类写作更多样、AI更平顺"的文体信号）
用法：python scripts/try_stylo_features.py
"""
import os, sys, io, numpy as np, scipy.sparse as sp
sys.path.insert(0, r"C:\Users\woshi\.dsh\aigc-detector")
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import roc_auc_score
from detector.stylometric import stylo_features
from scripts.cross_validate import build_aigc_test, build_cnki_test, label_to_ai, read_recs, zh_len, load_cls

def load_train(guarded=True):
    p = os.path.join(r"C:\Users\woshi\.dsh\aigc-detector\data",
                     "train_unified_guarded.jsonl" if guarded else "train_unified.jsonl")
    recs = read_recs(p)
    # 只取有 title 的报告训练（与 cross_validate 的测试划分一致：留出部分报告）
    aigc_test, _, ht = build_aigc_test(0.2, 42)
    out = []
    for r in recs:
        if r.get("title") and r["title"] not in ht and zh_len((r.get("text") or "")) >= 6:
            lab = r.get("label")
            y = 1 if lab in ("high","medium","low","AI") else (0 if lab=="human" else None)
            if y is None: continue
            out.append((r["text"], y))
    return out

def feats(texts, use_stylo, vec=None):
    if vec is None:
        vec = TfidfVectorizer(analyzer="char", ngram_range=(2,4), max_features=60000,
                              sublinear_tf=True, min_df=2)
        X = vec.fit_transform(texts)
    else:
        X = vec.transform(texts)
    if use_stylo:
        S = np.array([list(stylo_features(t).values()) for t in texts])
        return sp.hstack([X, sp.csr_matrix(S)]).tocsr(), vec
    return X, vec

def fit_clf(feat, ys):
    lr = LogisticRegression(C=1.0, class_weight="balanced", max_iter=3000)
    cal = CalibratedClassifierCV(lr, method="sigmoid", cv=3)
    cal.fit(feat, ys)
    return cal

def main():
    train = load_train(True)
    texts = [t for t,_ in train]; ys = np.array([y for _,y in train])
    print("训练集(守卫后报告句):", len(train))
    aigc_test, _, ht = build_aigc_test(0.2, 42)
    aigc_labels = [1 if label_to_ai(r.get("label")) else 0 for r in aigc_test]
    aigc_texts = [r["text"] for r in aigc_test]
    tt = set()
    for r in read_recs(os.path.join(r"C:\Users\woshi\.dsh\aigc-detector\data","train_unified.jsonl")):
        tt.add(r.get("text"))
    cnki = build_cnki_test(800, 42, tt)
    thr = 0.5

    for tag, use_stylo in [("TF-IDF 纯", False), ("TF-IDF+文体", True)]:
        X, vec = feats(texts, use_stylo)
        clf = fit_clf(X, ys)
        Xa, _ = feats(aigc_texts, use_stylo, vec)
        Xc, _ = feats(cnki, use_stylo, vec)
        pa = clf.predict_proba(Xa)[:,1]; pc = clf.predict_proba(Xc)[:,1]
        recall = float((pa>=thr).mean()); fpr = float((pc>=thr).mean())
        try: auc = roc_auc_score(aigc_labels, pa)
        except: auc = float('nan')
        print(f"[{tag}]  阈{thr}: AIGC检出={recall:.3f}  真实CNKI误报={fpr:.3f}  AUC={auc:.3f}")

if __name__ == "__main__":
    main()
