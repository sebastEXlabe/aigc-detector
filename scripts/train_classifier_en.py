# -*- coding: utf-8 -*-
"""英文统计流分类器训练（独立于中文分类器，互不干扰）。
数据：英文公开数据集（M4-en-academic 英文学术 / M4-en-wiki 英文百科）
平衡：human:AI ≈ 2.3:1
模型：char(2,4) TF-IDF + LogisticRegression(C=1.0,class_weight=balanced) + sigmoid校准
输出：models/classifier_en.pkl（含 vec/model/threshold/acc/auc）
用法：python train_classifier_en.py
说明：因为中英文 char n-gram 特征空间不同，英文用独立词表与分类器更准确，
     检测管线按文本语言路由到相应分类器。
"""
import os, sys, io, json, random
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
import numpy as np
BASE = r"C:\Users\woshi\.dsh\aigc-detector"
DATA = os.path.join(BASE, "data")

def read_recs(path):
    recs = []
    if not os.path.exists(path):
        return recs
    for l in open(path, encoding="utf-8"):
        if not l.strip():
            continue
        d = json.loads(l)
        if isinstance(d, list):
            recs.extend(d)
        elif "text" in d:
            recs.append(d)
    return recs

def main():
    random.seed(2)
    # 英文 AI 正样本
    ai_public = [r for r in read_recs(os.path.join(DATA, "ai_pub_samples.jsonl"))
                 if r.get("text") and r.get("lang") == "en"]
    # 英文 human 负样本
    hu_public = [r for r in read_recs(os.path.join(DATA, "human_pub_samples.jsonl"))
                 if r.get("text") and r.get("lang") == "en"]
    print(f"英文公开 AI句={len(ai_public)}  human句={len(hu_public)}")
    # AI 分层采样：按 source 分组，控制总量使 human 能配平（human 池 × 2.3 上限）
    from collections import defaultdict
    agroups = defaultdict(list)
    for r in ai_public:
        agroups[r.get("source","?")].append(r)
    ai_cap = int(len(hu_public) * 2.3)  # human 池能支撑的 AI 最大量
    ai_sel = []
    for src, recs in sorted(agroups.items()):
        cap = 1500 if "academic" in src else 800
        random.shuffle(recs)
        for r in recs[:cap]:
            if len(ai_sel) >= ai_cap:
                break
            ai_sel.append(r)
    if len(ai_sel) > ai_cap:
        random.shuffle(ai_sel); ai_sel = ai_sel[:ai_cap]
    print(f"英文AI采样={len(ai_sel)} (池={len(ai_public)})")
    ai = [(r["text"], r.get("prob", 0.85)) for r in ai_sel]
    human = [(r["text"], r.get("prob", 0.08)) for r in hu_public]
    # 平衡到 2.3:1（英文 human 优先学术文体）
    def hprio(r):
        return 0 if str(r.get("source","")).startswith("M4-en-academic") else 1
    hgroups = defaultdict(list)
    for r in hu_public:
        hgroups[hprio(r)].append((r["text"], r.get("prob", 0.08)))
    human_pool = []
    for k in sorted(hgroups):
        gg = list(hgroups[k]); random.shuffle(gg); human_pool.extend(gg)
    target = int(len(ai) * 2.3)
    human_sel = human_pool[:target] if len(human_pool) >= target else human_pool
    print("AI:", len(ai), " human:", len(human_sel), " ratio:", round(len(human_sel)/max(len(ai),1),2))
    texts = [t for t, _ in ai] + [t for t, _ in human_sel]
    y = np.array([1]*len(ai) + [0]*len(human_sel))
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, precision_recall_curve, f1_score as f1s
    from sklearn.calibration import CalibratedClassifierCV
    Xtr, Xte, ytr, yte = train_test_split(texts, y, test_size=0.2, random_state=42, stratify=y)
    vec = TfidfVectorizer(analyzer="char", ngram_range=(2,4), max_features=60000, sublinear_tf=True, min_df=2)
    Xtr_v = vec.fit_transform(Xtr); Xte_v = vec.transform(Xte)
    lr = LogisticRegression(C=1.0, class_weight="balanced", max_iter=3000); lr.fit(Xtr_v, ytr)
    cal = CalibratedClassifierCV(lr, method="sigmoid", cv=3); cal.fit(Xtr_v, ytr)
    pred = cal.predict(Xte_v); proba = cal.predict_proba(Xte_v)[:,1]
    acc = accuracy_score(yte, pred); f1 = f1_score(yte, pred); auc = roc_auc_score(yte, proba)
    print(f"\n== 英文分类器评估 ==\nacc={acc:.4f} f1={f1:.4f} AUC={auc:.4f}")
    pr, rc, thr = precision_recall_curve(yte, proba)
    best = 0.5; bf = 0
    for t in thr:
        f = f1s(yte, (proba >= t).astype(int))
        if f > bf: bf = f; best = t
    print(f"最优阈值:{best:.3f} (f1={bf:.4f})")
    os.makedirs(os.path.join(BASE, "models"), exist_ok=True)
    with open(os.path.join(BASE, "models", "classifier_en.pkl"), "wb") as f:
        import pickle
        pickle.dump({"vec": vec, "model": cal, "threshold": float(best),
                     "acc": acc, "f1": f1, "auc": auc}, f)
    print("英文模型已保存 models/classifier_en.pkl")

if __name__ == "__main__":
    main()
