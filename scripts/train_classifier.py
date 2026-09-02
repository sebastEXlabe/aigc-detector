# -*- coding: utf-8 -*-
"""路线B 最终训练脚本（最优配置，可复现）。
数据：AI 样本(train_unified prob>=0.4 + 公开数据集) + 人类样本(cnki 学术语料 + real_thesis + train_unified human + 公开数据)
平衡：human:AI ≈ 2.3:1（避免过平衡导致对AI过度宽容）
模型：char(2,4) TF-IDF + LogisticRegression(C=1.0,class_weight=balanced) + sigmoid校准
用法：python train_classifier.py [--no-public]   (--no-public 不加公开数据集，回到纯本地基线)
说明：公开数据集增量更新：ai_pub_samples.jsonl / human_pub_samples.jsonl
     由 scripts/prepare_public_datasets.py 从 C-ReD + HC3 构建。
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

def _is_en(text):
    """粗略判断英文：拉丁字母数大于中文字数。"""
    import re as _re
    lat = len(_re.findall(r"[A-Za-z]", text))
    zh = len(_re.findall(r"[\u4e00-\u9fff]", text))
    return lat > zh

def main():
    use_public = "--no-public" not in sys.argv
    random.seed(2)
    # 1. AI 样本
    ai=[]
    for r in read_recs(os.path.join(DATA,"train_unified.jsonl")):
        if r.get("text") and r.get("prob",0.5)>=0.4:
            ai.append((r["text"],r["prob"]))
    base_ai = len(ai)
    # human 本地池（cnki 学术 + real_thesis + train_unified human）
    human=[]
    for src in ["human_cnki.jsonl","human_positive.jsonl"]:
        for r in read_recs(os.path.join(DATA,src)):
            if r.get("text"): human.append((r["text"],0.08))
    for r in read_recs(os.path.join(DATA,"train_unified.jsonl")):
        if r.get("text") and r.get("prob",0.5)<0.4:
            human.append((r["text"],r["prob"]))
    # 自训练负样本：默认不加（实验证明对统计流反而升高误报）；用 --use-self-train 显式启用
    use_st = "--use-self-train" in sys.argv
    if use_st:
        for r in read_recs(os.path.join(DATA,"human_self_train.jsonl")):
            if r.get("text"): human.append((r["text"], r.get("prob",0.08)))
    base_human_total = len(human)
    pub_ai_groups = {}; pub_human_pool = []
    if use_public:
        # 公开 AI 池按 (source,lang) 分组，保证中英文/不同文体都有代表
        from collections import defaultdict
        ai_pool = [r for r in read_recs(os.path.join(DATA,"ai_pub_samples.jsonl")) if r.get("text")]
        # human 负样本池：优先用深流守卫过滤后的干净版(human_pub_clean)，避免污染标注
        hu_clean = os.path.join(DATA,"human_pub_clean.jsonl")
        hu_src = hu_clean if os.path.exists(hu_clean) else os.path.join(DATA,"human_pub_samples.jsonl")
        pub_human_pool = [r for r in read_recs(hu_src) if r.get("text")]
        # 按 --lang 过滤（默认中文）；英文由 train_classifier_en.py 单独训练
        lang_filter = "zh"
        for i, a in enumerate(sys.argv):
            if a == "--lang" and i+1 < len(sys.argv):
                lang_filter = sys.argv[i+1]
        ai_pool = [r for r in ai_pool if r.get("lang","zh") == lang_filter]
        pub_human_pool = [r for r in pub_human_pool if r.get("lang","zh") == lang_filter]
        groups = defaultdict(list)
        for r in ai_pool:
            key = r.get("source","?")
            groups[key].append(r)
        pub_ai_groups = groups
        print(f"  公开AI池(lang={lang_filter}): {len(ai_pool)}")
    TARGET_RATIO = 2.3
    max_human_avail = base_human_total + len(pub_human_pool)
    inc_cap_from_human = int(max_human_avail / TARGET_RATIO) - base_ai
    inc_target = min(9000, max(inc_cap_from_human, 1000))
    # 按 source 分层采样，更偏重论文文体（C-ReD / M4-zh-qa 等 zh 源），保持中文不同文体代表
    pub_ai = []
    for src, recs in sorted(pub_ai_groups.items()):
        if src == "C-ReD-paper":
            cap = 1600      # 中文学术论文文体（核心）
        elif src == "HC3":
            cap = 900       # 中文开放域问答
        elif src.startswith("M4-zh-qa"):
            cap = 900       # 中文问答
        elif src == "M4-en-academic" or src.startswith("M4-en-wiki"):
            continue        # 英文源由英文分类器负责
        else:
            cap = 300
        random.shuffle(recs)
        for r in recs[:cap]:
            if len(pub_ai) >= inc_target:
                break
            pub_ai.append((r["text"], r.get("prob",0.85)))
    ai.extend(pub_ai)
    print(f"  公开AI增量: 目标={inc_target} 采样={len(pub_ai)}")
    if use_public:
        # 公开 human：优先真实论文文体（C-ReD paper），再补其余
        def hprio(r):
            return 0 if r.get("source","") == "C-ReD-paper" else 1
        spool = sorted(pub_human_pool, key=hprio)
        from itertools import groupby
        prend = []
        for k, grp in groupby(spool, key=hprio):
            gg = list(grp); random.shuffle(gg); prend.extend(gg)
        need = int(len(ai) * TARGET_RATIO) - base_human_total
        take = prend[:max(0, need)]
        human.extend((r["text"], r.get("prob",0.08)) for r in take)
        print(f"  公开human增量: 池={len(pub_human_pool)} 采样={len(take)}")
    # 3. 平衡到 2.3:1（human 过采样则随机截断，不足则全用）
    target = int(len(ai) * TARGET_RATIO)
    random.shuffle(human)
    human_sel = human[:target] if len(human) >= target else human
    print("AI:",len(ai)," human:",len(human_sel)," human_total:",len(human),
          " ratio:",round(len(human_sel)/max(len(ai),1),2),
          " (AI基数:",base_ai," 本地human池:",base_human_total,")")
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
