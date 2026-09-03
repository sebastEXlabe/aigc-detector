# -*- coding: utf-8 -*-
"""英文AIGC统计分类器训练：用真实英文 AI/人类样本，训练 char(2,4)+word 特征 + CalibratedLogistic。
目的：英文文本当前 stat 中文TF-IDF几乎无分辨力(AUC~0.59)，独立训练英文分类器大幅补上。
数据：ai_pub_samples.jsonl(英文AI① 77k) + human_pub_samples.jsonl(英文人类 22k)。
输出：models/en_classifier.pkl {vec, model, threshold, acc, f1, auc, n_train}
用法：python scripts/train_en_classifier.py [--n 20000]
"""
import os, sys, re, json, random, pickle
sys.path.insert(0, r"C:\Users\woshi\.dsh\aigc-detector")
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score
from scripts.cross_validate import read_recs

def is_en(s):
    s = (s or "").strip()
    return bool(re.fullmatch(r"[ -~]{30,}", s)) and len(re.findall(r"[A-Za-z]", s)) > 40

def main():
    import argparse
    ap = argparse.ArgumentParser(); ap.add_argument("--n", type=int, default=24000)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    D = r"C:\Users\woshi\.dsh\aigc-detector\data"
    ai = [r["text"] for r in read_recs(os.path.join(D, "ai_pub_samples.jsonl")) if is_en(r.get("text"))]
    hu = [r["text"] for r in read_recs(os.path.join(D, "human_pub_samples.jsonl")) if is_en(r.get("text"))]
    print("英文AI", len(ai), "| 英文人类", len(hu), flush=True)
    random.seed(a.seed); random.shuffle(ai); random.shuffle(hu)
    # 平衡采样 (AI较多, 人类较少) → 取 min
    n_ai = min(len(ai), a.n); n_hu = min(len(hu), int(n_ai*0.6))
    ai = ai[:n_ai]; hu = hu[:n_hu]
    X = ai + hu; y = [1]*len(ai) + [0]*len(hu)
    print("训练/评估样本:", len(X), "AI", len(ai), "人类", len(hu), flush=True)
    # 按句做5/5划分(避免同源过拟合, 简单随机)
    idx = np.arange(len(X)); np.random.seed(a.seed); np.random.shuffle(idx)
    cut = int(len(idx)*0.85)
    tr, te = idx[:cut], idx[cut:]
    Xtr = [X[i] for i in tr]; ytr = [y[i] for i in tr]
    Xte = [X[i] for i in te]; yte = [y[i] for i in te]

    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), max_features=120000,
                          min_df=2, sublinear_tf=True)
    Xv = vec.fit_transform(Xtr)
    clf = LogisticRegression(C=1.0, max_iter=2000)
    cal = CalibratedClassifierCV(clf, cv=3).fit(Xv, ytr)
    p = cal.predict_proba(vec.transform(Xte))[:, 1]
    auc = roc_auc_score(yte, p); f1 = f1_score(yte, (p > 0.5).astype(int)); acc = accuracy_score(yte, (p > 0.5).astype(int))
    print(f"\n=== 英文分类器(clean char-ngram) ===", flush=True)
    print(f"AUC={auc:.3f} | F1={f1:.3f} | ACC={acc:.3f} | 训练={len(Xtr)} 测试={len(Xte)}", flush=True)
    outp = r"C:\Users\woshi\.dsh\aigc-detector\models\classifier_en.pkl"
    pickle.dump({"vec": vec, "model": cal, "threshold": 0.5, "acc": acc, "f1": f1,
                 "auc": auc, "n_train": len(Xtr)}, open(outp, "wb"))
    print("保存", outp, flush=True)

if __name__ == "__main__":
    main()
