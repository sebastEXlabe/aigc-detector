# -*- coding: utf-8 -*-
"""受控 OOD（域外）泛化实验：检验公开数据集增量更新的真实价值。

方法：
  - 保留 C-ReD 的某个生成器（--holdout）做纯域外测试：训练时完全不让它露面；
  - 基线模型 = 纯本地数据（train_unified AI/human + cnki 学术 human）；
  - 增量模型 = 基线 + 公开数据（C-ReD 其余生成器 + HC3），同样 2.3:1 平衡；
  - 在 OOD 测试集（heldout 生成器的 AI 摘要分句 + C-ReD human 分句）上比较两模型的
    AUC / F1 / 混淆矩阵，判定增量是否提升了对"没见过的新 LLM 风格"的鲁棒性。

用法：python run_ood_experiment.py [--holdout doubao-1.5-pro]
"""
import os, sys, io, json, re, csv, glob, random, argparse
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
import numpy as np
BASE = r"C:\Users\woshi\.dsh\aigc-detector"
CRED = os.path.join(BASE, "data", "datasets_raw", "cred")
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

def is_zh(s):
    return len(re.findall(r"[\u4e00-\u9fff]", s)) >= 6

def split_sentences(text):
    out, buf, in_q = [], "", False
    i, n = 0, len(text)
    while i < n:
        c = text[i]
        if c in "\u201c\u300c\u300e":
            in_q = True
        elif c in "\u201d\u300d\u300f":
            in_q = False
        buf += c
        if not in_q and c in "\u3002\uff1f\uff01":
            j = i + 1
            while j < n and text[j] in "\u3002\uff1f\uff01":
                buf += text[j]; j += 1
            i = j - 1
            if buf.strip() and len(buf.strip()) > 4:
                out.append(buf.strip())
            buf = ""
        elif not in_q and c == "\u2026":
            j = i + 1
            while j < n and text[j] == "\u2026":
                buf += text[j]; j += 1
            i = j - 1
        elif c in ";；":
            if buf.strip() and len(buf.strip()) > 4:
                out.append(buf.strip())
            buf = ""
        i += 1
    if buf.strip() and len(buf.strip()) > 4:
        out.append(buf.strip())
    return [s for s in out if is_zh(s)]

def load_cred_sentences():
    """C-ReD paper 摘要分句，返回 (text, label, generator)。label: 1=AI, 0=human。"""
    items = []
    for fp in sorted(glob.glob(os.path.join(CRED, "CReD_paper_*.csv"))):
        gen = os.path.basename(fp).replace("CReD_paper_", "").replace(".csv", "")
        with open(fp, encoding="utf-8") as f:
            for r in csv.DictReader(f):
                text = (r.get("text") or "").strip()
                if not text:
                    continue
                try:
                    label = int(r.get("label"))  # 0=AI, 1=human
                except (ValueError, TypeError):
                    continue
                for s in split_sentences(text):
                    items.append((s, 1 if label == 0 else 0, gen))  # 1=AI
    return items

def load_hc3_sentences():
    items = []
    hc3 = os.path.join(BASE, "data", "datasets_raw", "hc3chinese_all.jsonl")
    if not os.path.exists(hc3):
        return items
    for l in open(hc3, encoding="utf-8"):
        if not l.strip():
            continue
        d = json.loads(l)
        ca = "".join(x for x in (d.get("chatgpt_answers") or []) if isinstance(x, str))
        ha = "".join(x for x in (d.get("human_answers") or []) if isinstance(x, str))
        for s in split_sentences(ca):
            items.append((s, 1, "chatgpt"))
        for s in split_sentences(ha):
            items.append((s, 0, "hc3human"))
    return items

def train_model(train_texts, train_y):
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.model_selection import train_test_split
    vec = TfidfVectorizer(analyzer="char", ngram_range=(2, 4), max_features=60000,
                          sublinear_tf=True, min_df=2)
    Xtr = vec.fit_transform(train_texts)
    lr = LogisticRegression(C=1.0, class_weight="balanced", max_iter=3000)
    lr.fit(Xtr, train_y)
    cal = CalibratedClassifierCV(lr, method="sigmoid", cv=3)
    cal.fit(Xtr, train_y)
    return vec, cal

def evaluate(vec, cal, ood_texts, ood_y):
    from sklearn.metrics import roc_auc_score, f1_score, accuracy_score
    X = vec.transform(ood_texts)
    p = cal.predict_proba(X)[:, 1]
    pred = (p >= 0.5).astype(int)
    return {
        "auc": roc_auc_score(ood_y, p),
        "f1": f1_score(ood_y, pred),
        "acc": accuracy_score(ood_y, pred),
        "n_ai": int(sum(ood_y)), "n_hu": int(len(ood_y) - sum(ood_y)),
    }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--holdout", default="doubao-1.5-pro")
    a = ap.parse_args()
    random.seed(42); np.random.seed(42)

    # ---- OOD 黄金测试集：heldout 生成器的 AI + C-ReD human 全部 ----
    cred = load_cred_sentences()
    hc3 = load_hc3_sentences()
    ood_ai = [(t, g) for t, l, g in cred if l == 1 and g == a.holdout]
    ood_hu = [(t, g) for t, l, g in cred if l == 0]
    ood_texts = [t for t, _ in ood_ai] + [t for t, _ in ood_hu]
    ood_y = [1] * len(ood_ai) + [0] * len(ood_hu)
    print(f"== OOD 黄金测试集：heldout={a.holdout} ==  AI句={len(ood_ai)} human句={len(ood_hu)}")

    # ---- 本地基线数据 ----
    ai_local, hu_local = [], []
    for r in read_recs(os.path.join(DATA, "train_unified.jsonl")):
        if r.get("text") and r.get("prob", 0.5) >= 0.4:
            ai_local.append(r["text"])
        elif r.get("text"):
            hu_local.append(r["text"])
    for src in ["human_cnki.jsonl", "human_positive.jsonl"]:
        for r in read_recs(os.path.join(DATA, src)):
            if r.get("text"):
                hu_local.append(r["text"])

    # ---- 基线模型：纯本地，2.3:1 ----
    random.shuffle(hu_local)
    hu_sel = hu_local[:int(len(ai_local) * 2.3)]
    base_texts = ai_local + hu_sel
    base_y = [1] * len(ai_local) + [0] * len(hu_sel)
    print(f"\n基线: AI={len(ai_local)} human={len(hu_sel)}")

    # ---- 增量数据：C-ReD（去掉 heldout）+ HC3，AI 采样受控 ----
    inc_ai = [(t, g) for t, l, g in cred if l == 1 and g != a.holdout]
    inc_hu = [(t, g) for t, l, g in cred if l == 0]
    # AI 按生成器分层采样，总量控制在约 1w
    from collections import defaultdict
    grp = defaultdict(list)
    for t, g in inc_ai:
        grp[g].append(t)
    ai_inc = []
    for g, ts in sorted(grp.items()):
        random.shuffle(ts)
        cap = 900 if g != "chatgpt" else 200
        ai_inc.extend(ts[:cap])
    # HC3 补充少量对话域 AI
    hc3_ai = [t for t, _l, _g in hc3 if _l == 1]
    random.shuffle(hc3_ai)
    ai_inc.extend(hc3_ai[:800])
    if len(ai_inc) > 9500:
        random.shuffle(ai_inc); ai_inc = ai_inc[:9500]
    # human 增量：优先 C-ReD human（论文文体），凑足 2.3
    inc_hu_texts = [t for t, _ in inc_hu]
    random.shuffle(inc_hu_texts)
    need = int((len(ai_local) + len(ai_inc)) * 2.3) - len(hu_sel)
    hu_inc = inc_hu_texts[:max(0, need)]
    print(f"增量: AI_inc={len(ai_inc)} hu_inc={len(hu_inc)}")

    # ---- 增量模型：基线 + 增量，2.3:1 ----
    inc_texts = ai_local + ai_inc + hu_sel + hu_inc
    inc_y = [1] * (len(ai_local) + len(ai_inc)) + [0] * (len(hu_sel) + len(hu_inc))
    print(f"增量总: AI={len(ai_local)+len(ai_inc)} human={len(hu_sel)+len(hu_inc)} ratio="
          f"{round((len(hu_sel)+len(hu_inc))/max(len(ai_local)+len(ai_inc),1),2)}")

    print("\n开始训练...")
    vec_b, cal_b = train_model(base_texts, base_y)
    vec_i, cal_i = train_model(inc_texts, inc_y)

    rb = evaluate(vec_b, cal_b, ood_texts, ood_y)
    ri = evaluate(vec_i, cal_i, ood_texts, ood_y)
    print("\n== OOD 泛化对比（heldout 生成器 =%s ）==" % a.holdout)
    for name, r in (("基线(纯本地)", rb), ("增量(公开数据)", ri)):
        print(f"  {name}: AUC={r['auc']:.4f}  F1={r['f1']:.4f}  acc={r['acc']:.4f}  "
              f"(AI={r['n_ai']} human={r['n_hu']})")
    d = ri["auc"] - rb["auc"]
    print(f"\nAUC 变化: {rb['auc']:.4f} → {ri['auc']:.4f}  Δ={d:+.4f}  "
          f"{'提升 ✓ 增量有效' if d > 0.005 else ('下降 ✗ 增量稀释' if d < -0.005 else '持平')}")

if __name__ == "__main__":
    main()
