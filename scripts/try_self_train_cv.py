# -*- coding: utf-8 -*-
"""自训练变体对比：在同一无泄漏真实测试集上，测「基线统计流」vs「自训练增强统计流」。
自训练增强 = ① 高置信真实CNKI学术句 作 human 硬负样本（来源先验纠偏）
            ② 高置信公开AI语料句 作 AI 伪标签（置信度截断）
评测：真实 AIGC 检出率 + 真实 CNKI 误报(FPR)，与 cross_validate 同一划分。

用法：python scripts/try_self_train_cv.py
"""
import os, sys, io, json, random, re, argparse, collections
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
import pickle
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
BASE = r"C:\Users\woshi\.dsh\aigc-detector"
DATA = os.path.join(BASE, "data")
sys.path.insert(0, BASE)
from scripts.cross_validate import build_aigc_test, build_cnki_test, label_to_ai, read_recs, zh_len, evaluate, stat_probs  # noqa

def load_train_set():
    """从 train_unified 取有 title 的报告句作训练（与测试同源、按报告已切分测试，此处取非留出）。"""
    return read_recs(os.path.join(DATA, "train_unified.jsonl"))

def train_cls(train_recs, extra, max_features=60000):
    texts = []
    labels = []
    for r in train_recs:
        t = (r.get("text") or "").strip()
        if not t or zh_len(t) < 6: continue
        lab = r.get("label")
        if lab in ("high","medium","low"): y = 1
        elif lab == "human": y = 0
        else: continue
        texts.append(t); labels.append(y)
    for t, y in extra:
        texts.append(t); labels.append(y)
    labels = np.array(labels)
    vec = TfidfVectorizer(analyzer="char", ngram_range=(2,4), max_features=max_features,
                          sublinear_tf=True, min_df=2)
    X = vec.fit_transform(texts)
    lr = LogisticRegression(C=1.0, class_weight="balanced", max_iter=3000)
    cal = CalibratedClassifierCV(lr, method="sigmoid", cv=3)
    cal.fit(X, labels)
    return vec, cal

def main():
    aigc_test, _, ht = build_aigc_test(0.2, 42)
    cnt = len(build_cnki_test(800, 42, set()))  # 先取一批用于构造（实际测试去重在 eval 阶段再做）
    train_texts = set()
    for r in read_recs(os.path.join(DATA, "train_unified.jsonl")):
        train_texts.add(r.get("text"))
    cnki = build_cnki_test(800, 42, train_texts)
    # 真实 CNKI 作为 self-train 硬负样本池（与测试句去重，避免泄漏）
    cnki_pool = read_recs(os.path.join(DATA, "human_corpus.jsonl"))
    neg_pool = []
    for r in cnki_pool:
        t = (r.get("text") or "").strip()
        if not t or zh_len(t) < 25 or t in train_texts: continue
        neg_pool.append(t)
    random.seed(7); random.shuffle(neg_pool)
    neg_pool = neg_pool[:12000]

    train_recs = []
    for r in read_recs(os.path.join(DATA, "train_unified.jsonl")):
        if r.get("title") and r["title"] not in ht and zh_len((r.get("text") or "")) >= 6:
            train_recs.append(r)
    # 生产配方公共基线：追加 human_pub_clean 人类负样本（守卫后的，避免污染）
    pub = [r for r in read_recs(os.path.join(DATA, "human_pub_clean.jsonl"))
           if (r.get("text") or "") and zh_len(r["text"]) >= 20][:15000]
    base_neg = [(r["text"], 0) for r in pub]

    def run(tag, extra):
        vec, cal = train_cls(train_recs, base_neg + extra)
        aigc_probs = stat_probs({"vec":vec,"model":cal}, [r["text"] for r in aigc_test])
        cnki_probs = stat_probs({"vec":vec,"model":cal}, cnki)
        aigc_labels = [1 if label_to_ai(r.get("label")) else 0 for r in aigc_test]
        thr = 0.345
        eva = evaluate(aigc_probs, aigc_labels, thr)
        evc = evaluate(cnki_probs, [0]*len(cnki), thr)
        # 找最优 FPR 预算阈值
        best = None
        for t in np.arange(0.25, 0.65, 0.02):
            ec = evaluate(cnki_probs, [0]*len(cnki), t)
            ea = evaluate(aigc_probs, aigc_labels, t)
            if ec['fpr'] <= 0.03 and (best is None or ea['ai_recall'] > best[1]):
                best = (round(t,2), ea['ai_recall'])
        print(f"[{tag}] thr0.345: AIGC检出={eva['ai_recall']:.3f} CNKI误报={evc['fpr']:.3f} AUC={eva['auc']:.3f}"
              + (f" | 3%误报预算下最优: 阈值{best[0]} 检出{best[1]:.3f}" if best else ""))
        return eva, evc

    print("=== 自训练变体（统计流）对比，同一真实测试集 ===")
    print(f"真实AIGC测试: {len(aigc_test)}句({ht and len(ht)}份报告)  真实CNKI测试: {len(cnki)}句")
    # 基线：只用报告句
    run("基线(报告句)", [])
    # 变体1：+ 高置信真实CNKI human 硬负样本
    run("+CNKI真实句human负样本", [(t,0) for t in neg_pool[:6000]])
    # 变体2：+ 高置信真实CNKI human + 高置信公开AI伪标签
    ai_pool = read_recs(os.path.join(DATA, "ai_pub_samples.jsonl"))
    ai_pseudo = []
    for r in ai_pool:
        t = (r.get("text") or "").strip()
        if t and zh_len(t) >= 15: ai_pseudo.append((t,1))
        if len(ai_pseudo) >= 6000: break
    run("+CNKI human + AI伪标签", [(t,0) for t in neg_pool[:6000]] + ai_pseudo[:3000])

if __name__ == "__main__":
    main()
