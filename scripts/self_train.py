# -*- coding: utf-8 -*-
"""自训练（针对误报盲区，不消耗外部API/资金）。

目标：现有模型对真人学术语料（cnki）存在误报（把真人论文的统计结果句/框架句
判成高AI）。自训练利用 cnki 未标注语料，用模型打分 → 找出被误判为AI的真人句
→ 强制标注为 human 负样本 → 重训，降低"真人论文误判"。

做法（对来源先验的纠偏，比纯自训练更可靠，因为 cnki 语料先验全是真人）：
  1. 用当前 classifier.pkl 对 human_corpus.jsonl 未用句打分。
  2. 高分句（prob >= 误报阈值）但确实是真人学术语料 → 判为误报，标 human 强负样本。
  3. 补充高置信 human 长学术句，扩大负样本多样性。
  4. 与既有标注合并，2.3:1 平衡重训，对比误报率变化。

用法：python scripts/self_train.py [--rounds N] [--verbose]
"""
import os, sys, io, json, pickle, shutil, random, re, glob, argparse, time
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
import numpy as np
BASE = r"C:\Users\woshi\.dsh\aigc-detector"
DATA = os.path.join(BASE, "data")
MODELS = os.path.join(BASE, "models")

def read_recs(path):
    recs = []
    if not os.path.exists(path): return recs
    for l in open(path, encoding="utf-8"):
        if not l.strip(): continue
        d = json.loads(l)
        if isinstance(d, list): recs.extend(d)
        elif "text" in d: recs.append(d)
    return recs

def load_cls():
    p = os.path.join(MODELS, "classifier.pkl")
    if not os.path.exists(p):
        print("classifier.pkl 不存在"); return None
    return pickle.load(open(p, "rb"))

def zh_len(t):
    return len(re.findall(r"[\u4e00-\u9fff]", t))

def get_used_texts():
    """已用于训练的文本（train_unified human/ai + cnki 9000 + positive）去重。"""
    used = set()
    for r in read_recs(os.path.join(DATA, "train_unified.jsonl")):
        used.add(r["text"])
    for r in read_recs(os.path.join(DATA, "human_cnki.jsonl")):
        used.add(r["text"])
    for r in read_recs(os.path.join(DATA, "human_positive.jsonl")):
        used.add(r["text"])
    return used

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", type=int, default=1)
    ap.add_argument("--fp-threshold", type=float, default=0.40, help="误报阈值(真人句prob>=此值判为误报)")
    ap.add_argument("--max-fp", type=int, default=8000, help="最多纠偏的误报句数")
    ap.add_argument("--max-conf", type=int, default=20000, help="最多补充的高置信human句数")
    a = ap.parse_args()
    cls = load_cls()
    if cls is None: return
    print(f"当前模型: acc={cls.get('acc'):.3f} auc={cls.get('auc'):.3f} thr={cls.get('threshold'):.3f}")

    used = get_used_texts()
    print("已用文本数:", len(used))

    # 对 cnki 未用句打分
    corpus = read_recs(os.path.join(DATA, "human_corpus.jsonl"))
    unlabeled = [r for r in corpus if r.get("text") and r["text"] not in used and zh_len(r["text"]) >= 20]
    print(f"cnki 未用长学术句: {len(unlabeled)}")
    random.seed(2); random.shuffle(unlabeled)

    vec = cls["vec"]; model = cls["model"]
    texts = [r["text"] for r in unlabeled]
    # 分批打分
    probs = []
    for i in range(0, len(texts), 2000):
        X = vec.transform(texts[i:i+2000])
        probs.extend(model.predict_proba(X)[:, 1].tolist())

    # 高置信 human 句：扩展 human 负样本覆盖（安全：真人句加负样本）
    # 误报句(真人判AI)：仅诊断，不强制标 human 进训练 —— 因为这类句式与 AI 字面高度相似，
    # 强制标 human 会污染（把 AI 风格句当真人），且 TF-IDF 统计流靠加同类难以真正纠偏。
    conf_hu = [(r["text"], p) for r, p in zip(unlabeled, probs) if p <= 0.10]
    print(f"高置信human句(p<=0.10): {len(conf_hu)}")
    random.shuffle(conf_hu)
    conf_hu = conf_hu[:a.max_conf]

    # 生成自训练增量负样本（仅高置信真人句 → human）
    st_human = [(t, 0.08) for t, _ in conf_hu]

    # 输出自训练数据集（追加到 human_st.jsonl）
    st_path = os.path.join(DATA, "human_self_train.jsonl")
    with open(st_path, "w", encoding="utf-8") as f:
        for t, pr in st_human:
            f.write(json.dumps({"text": t, "prob": pr, "label": "human", "source": "self_train"}, ensure_ascii=False) + "\n")
    print(f"自训练负样本(高置信human)已保存: {len(st_human)} 句 -> {st_path}")

    # 误报诊断（不只写入）：打印误报句示例，评估模型对真人学术句的误报程度
    fp = [(r["text"], p) for r, p in zip(unlabeled, probs) if p >= a.fp_threshold]
    print(f"--- 误报诊断（真人句被判AI p>={a.fp_threshold}: {len(fp)}句）---")
    for t, p in fp[:5]:
        print(f"  p={p:.2f} {t[:60]}")

if __name__ == "__main__":
    main()
