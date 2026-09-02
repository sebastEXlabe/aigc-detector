# -*- coding: utf-8 -*-
"""真实数据交叉验证：对真实 CNKI 学术句测误报率(FPR)，对真实 AIGC 报告测检出率。

设计（力求无泄漏）：
  · 真实 AIGC 测试 = 按报告(title)留出整份报告的全部句级标注（label 高/中/低→AI，human→human）
  · 真实 CNKI 测试 = 与训练文本去重后的真实学术句（ground truth=human），测误报
  · 基线 = 当前生产 classifier.pkl（离线统计流）+ 深流（可选），对上述测试集打分类别/概率

用法：
  python scripts/cross_validate.py [--aigc-holdout-frac 0.2] [--cnki-test-n 2000]
                                    [--use-deep 1] [--aisim AI]
"""
import os, sys, io, json, random, re, argparse, collections
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
import pickle
import numpy as np
BASE = r"C:\Users\woshi\.dsh\aigc-detector"
DATA = os.path.join(BASE, "data")
MODELS = os.path.join(BASE, "models")
sys.path.insert(0, BASE)

def read_recs(path):
    out = []
    if not os.path.exists(path): return out
    for l in open(path, encoding="utf-8"):
        l = l.strip()
        if not l: continue
        try: d = json.loads(l)
        except: continue
        if isinstance(d, list): out.extend(d)
        else: out.append(d)
    return out

def zh_len(t):
    return len(re.findall(r"[\u4e00-\u9fff]", t))

def label_to_ai(label):
    # AIGC 报告三档：high/medium/low 为 AI 痕迹，human 为真人
    return label in ("high", "medium", "low")

def build_aigc_test(frac, seed):
    """按报告(title)留出整份报告。返回 (test_recs, train_recs)。train 不用于本脚本基线评估，
    但保留供 self-training 变体使用。"""
    recs = read_recs(os.path.join(DATA, "train_unified.jsonl"))
    # 只保留有 title（真实 AIGC 报告）且句子长度的
    by_title = collections.defaultdict(list)
    for r in recs:
        t = r.get("title")
        if not t: continue
        txt = (r.get("text") or "").strip()
        if not txt or zh_len(txt) < 10: continue
        by_title[t].append(r)
    titles = sorted(by_title.keys())
    random.seed(seed); random.shuffle(titles)
    k = max(1, int(len(titles) * frac))
    holdout = set(titles[:k])
    test = [r for t in holdout for r in by_title[t]]
    train = [r for t in titles if t not in holdout for r in by_title[t]]
    return test, train, holdout

def build_cnki_test(n, seed, exclude):
    """真实 CNKI 学术句，排除与训练文本重叠（去重），采样 n 句。"""
    pool = read_recs(os.path.join(DATA, "human_corpus.jsonl"))
    exclude_set = set(exclude or [])
    cand = []
    for r in pool:
        txt = (r.get("text") or "").strip()
        if not txt or zh_len(txt) < 20: continue
        if txt in exclude_set: continue
        cand.append(txt)
    random.seed(seed); random.shuffle(cand)
    return cand[:n]

def load_cls():
    p = os.path.join(MODELS, "classifier.pkl")
    if not os.path.exists(p):
        print("classifier.pkl 不存在"); return None
    return pickle.load(open(p, "rb"))

def stat_probs(cls, texts, batch=2000):
    vec, model = cls["vec"], cls["model"]
    probs = []
    for i in range(0, len(texts), batch):
        X = vec.transform(texts[i:i+batch])
        probs.extend(model.predict_proba(X)[:, 1].tolist())
    return np.array(probs)

def evaluate(probs, labels, threshold):
    """labels: AI=1, human=0。返回 FPR/AI-recall/AUC/ACC。"""
    labels = np.array(labels)
    pred = (probs >= threshold).astype(int)
    ai = labels == 1; hu = labels == 0
    fpr = pred[hu].mean() if hu.any() else 0.0
    recall = pred[ai].mean() if ai.any() else 0.0
    acc = (pred == labels).mean() if len(labels) else 0.0
    # AUC (rank-based)
    order = np.argsort(probs)
    ranks = np.empty(len(probs)); ranks[order] = np.arange(len(probs))
    npos = ai.sum(); nneg = hu.sum()
    if npos == 0 or nneg == 0: auc = float('nan')
    else: auc = (ranks[ai].sum() - npos*(npos-1)/2) / (npos*nneg)
    return {"fpr": fpr, "ai_recall": recall, "acc": acc, "auc": float(auc)}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--aigc-holdout-frac", type=float, default=0.2)
    ap.add_argument("--cnki-test-n", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--use-deep", type=int, default=1)
    ap.add_argument("--min-sim", type=float, default=0.55, help="最小融合概率阈值口径")
    a = ap.parse_args()

    cls = load_cls()
    if cls is None: return
    thr = cls.get("threshold", 0.4)
    print(f"基线模型: thr={thr:.3f} acc={cls.get('acc'):.3f} auc={cls.get('auc'):.3f}")

    # 真实 AIGC 测试（整份报告留出）
    aigc_test, _, holdout_titles = build_aigc_test(a.aigc_holdout_frac, a.seed)
    print(f"\n[真实AIGC] 留出报告 {len(holdout_titles)} 份 / 句子 {len(aigc_test)}")
    aigc_ai = [r for r in aigc_test if label_to_ai(r.get("label"))]
    aigc_hu = [r for r in aigc_test if not label_to_ai(r.get("label"))]
    print(f"  AI 句={len(aigc_ai)}  human 句={len(aigc_hu)}")

    # 真实 CNKI 测试（与当前训练集去重，避免测到训练句）
    train_texts = set()
    for r in read_recs(os.path.join(DATA, "train_unified.jsonl")):
        train_texts.add(r.get("text"))
    cnki = build_cnki_test(a.cnki_test_n, a.seed, train_texts)
    print(f"\n[真实CNKI] 真实学术测试句(去重后)={len(cnki)}")

    # 统计流打分
    aigc_probs = stat_probs(cls, [r["text"] for r in aigc_test])
    cnki_probs = stat_probs(cls, cnki)
    aigc_labels = [1 if label_to_ai(r.get("label")) else 0 for r in aigc_test]

    # 深流融合（若模型可加载；CUDA 不可用则 CPU，慢但能跑）
    bert_model = None
    if a.use_deep:
        try:
            from detector.dual_stream import load_bert, bert_score_per_sentence, fuse as ds_fuse
            bert_model = load_bert(device="cuda")
            print(f"  (深流加载: {'✓' if bert_model else '✗'} device={bert_model[2] if bert_model else '-'})")
        except Exception as e:
            print(f"  (深流未加载: {e})")
            bert_model = None

    def fused_probs(stat_p, texts):
        if bert_model is None: return stat_p
        tok, model, dev = bert_model
        try:
            bp = bert_score_per_sentence(tok, model, dev, texts)
        except Exception as e:
            print(f"  (深流打分失败: {e})"); bp = None
        if bp is None or len(bp) != len(stat_p): return stat_p
        return np.array([ds_fuse(float(a), float(b)) for a, b in zip(stat_p, bp)])

    fused_aigc = fused_probs(aigc_probs, [r["text"] for r in aigc_test])
    fused_cnki = fused_probs(cnki_probs, cnki)
    stream_tag = "统计流+深流" if bert_model is not None else "统计流"

    ev_aigc = evaluate(aigc_probs, aigc_labels, thr)
    ev_cnki = evaluate(cnki_probs, [0]*len(cnki), thr)
    ev_aigc_f = evaluate(fused_aigc, aigc_labels, thr)
    ev_cnki_f = evaluate(fused_cnki, [0]*len(cnki), thr)

    print("\n=========== 真实数据交叉验证 ===========")
    print(f"--- 统计流 ---")
    print(f"[真实AIGC] 检出={ev_aigc['ai_recall']:.3f} 人类误报={ev_aigc['fpr']:.3f} AUC={ev_aigc['auc']:.3f}")
    print(f"[真实CNKI] 人类误报(FPR)={ev_cnki['fpr']:.3f}  ({len(cnki)}句)")
    if bert_model is not None:
        print(f"--- {stream_tag} 融合 ---")
        print(f"[真实AIGC] 检出={ev_aigc_f['ai_recall']:.3f} 人类误报={ev_aigc_f['fpr']:.3f} AUC={ev_aigc_f['auc']:.3f}")
        print(f"[真实CNKI] 人类误报(FPR)={ev_cnki_f['fpr']:.3f}")
    for t in (0.3, 0.4, 0.5, 0.6):
        e1 = evaluate(fused_aigc, aigc_labels, t)
        e2 = evaluate(fused_cnki, [0]*len(cnki), t)
        print(f"  阈值{t:.1f}: AIGC检出={e1['ai_recall']:.3f} CNKI误报={e2['fpr']:.3f}")


if __name__ == "__main__":
    main()
