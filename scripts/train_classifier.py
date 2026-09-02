# -*- coding: utf-8 -*-
"""路线B 最终训练脚本（最优配置，可复现）。
数据：AI 样本(train_unified prob>=0.4) + 人类样本(cnki 学术语料 + real_thesis + train_unified human)
平衡：human:AI ≈ 2.3:1（避免过平衡导致对AI过度宽容）
模型：char(2,4) TF-IDF + LogisticRegression(C=1.0,class_weight=balanced) + sigmoid校准
用法：python train_classifier.py
"""
import os, sys, io, json, re, math, pickle, random
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
import numpy as np
BASE = r"C:\Users\woshi\.dsh\aigc-detector"
DATA = os.path.join(BASE, "data")

def read_recs(path):
    recs=[]
    if not os.path.exists(path): return recs
    for l in open(path, encoding="utf-8"):
        if not l.strip(): continue
        d=json.loads(l)
        if isinstance(d,list): recs.extend(d)
        elif "text" in d: recs.append(d)
        elif "sentences" in d: recs.extend(d["sentences"])
    return recs

def main():
    # 1. AI 样本
    ai=[]
    for r in read_recs(os.path.join(DATA,"train_unified.jsonl")):
        if r.get("text") and r.get("prob",0.5)>=0.4:
            ai.append((r["text"],r["prob"]))
    # 2. 人类样本（真实学术语料优先）
    human=[]
    for src in ["human_cnki.jsonl","human_positive.jsonl"]:
        for r in read_recs(os.path.join(DATA,src)):
            if r.get("text"): human.append((r["text"],0.08))
    # 补 train_unified 的 human
    for r in read_recs(os.path.join(DATA,"train_unified.jsonl")):
        if r.get("text") and r.get("prob",0.5)<0.4:
            human.append((r["text"],r["prob"]))
    # 3. 平衡到 2.3:1
    random.seed(2)
    target=int(len(ai)*2.3)
    random.shuffle(human)
    human_sel=human[:target] if len(human)>=target else human
    print("AI:",len(ai)," human:",len(human_sel)," human_total:",len(human)," ratio:",round(len(human_sel)/max(len(ai),1),2))
    texts=[t for t,_ in ai]+[t for t,_ in human_sel]
    y=np.array([1]*len(ai)+[0]*len(human_sel))
    # 4. 训练
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score,f1_score,roc_auc_score,precision_recall_curve,f1_score as f1s
    from sklearn.calibration import CalibratedClassifierCV
    Xtr,Xte,ytr,yte=train_test_split(texts,y,test_size=0.2,random_state=42,stratify=y)
    vec=TfidfVectorizer(analyzer="char",ngram_range=(2,4),max_features=60000,sublinear_tf=True,min_df=2)
    Xtr_v=vec.fit_transform(Xtr); Xte_v=vec.transform(Xte)
    lr=LogisticRegression(C=1.0,class_weight="balanced",max_iter=3000)
    lr.fit(Xtr_v,ytr)
    cal=CalibratedClassifierCV(lr,method="sigmoid",cv=3); cal.fit(Xtr_v,ytr)
    pred=cal.predict(Xte_v); proba=cal.predict_proba(Xte_v)[:,1]
    acc=accuracy_score(yte,pred); f1=f1_score(yte,pred); auc=roc_auc_score(yte,proba)
    print(f"\n== 路线B 模型评估 ==\nacc={acc:.4f} f1={f1:.4f} AUC={auc:.4f}")
    pr,rc,thr=precision_recall_curve(yte,proba)
    best=0.5;bf=0
    for t in thr:
        f=f1s(yte,(proba>=t).astype(int))
        if f>bf: bf=f;best=t
    print(f"最优阈值:{best:.3f} (f1={bf:.4f})")
    os.makedirs(os.path.join(BASE,"models"),exist_ok=True)
    with open(os.path.join(BASE,"models","classifier.pkl"),"wb") as f:
        pickle.dump({"vec":vec,"model":cal,"threshold":float(best),"acc":acc,"f1":f1,"auc":auc},f)
    print("模型已保存 models/classifier.pkl")

if __name__=="__main__":
    main()
