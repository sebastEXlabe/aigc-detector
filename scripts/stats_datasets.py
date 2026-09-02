# -*- coding: utf-8 -*-
"""AIGC 检测数据集统计盘点。
统计本地标注集与公开数据集暂存区各来源/语言/标签的样本量，输出汇总。
用法：python scripts/stats_datasets.py
"""
import os, sys, io, json, csv, glob, re
from collections import Counter, defaultdict
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

BASE = r"C:\Users\woshi\.dsh\aigc-detector"
DATA = os.path.join(BASE, "data")
RAW = os.path.join(DATA, "datasets_raw")
CRED = os.path.join(RAW, "cred")
M4 = os.path.join(RAW, "m4")

def nl(path):
    n = 0
    if not os.path.exists(path):
        return 0
    with open(path, encoding="utf-8", errors="ignore") as f:
        for _ in f:
            n += 1
    return n

def zh_count(t):
    return len(re.findall(r"[\u4e00-\u9fff]", t))

def is_en(t):
    return len(re.findall(r"[A-Za-z]", t)) > zh_count(t)

def main():
    print("=" * 60)
    print(" AIGC 检测数据集统计盘点")
    print("=" * 60)

    # 1. 本地训练/标注集
    print("\n[1] 本地标注集 (data/)")
    local_files = ["train_unified.jsonl", "human_cnki.jsonl", "human_positive.jsonl",
                   "human_corpus.jsonl", "word_annotations.jsonl"]
    for fn in local_files:
        p = os.path.join(DATA, fn)
        n = nl(p)
        print(f"   {fn:<28} 行数={n:>7}")

    # train_unified 标签分布
    p = os.path.join(DATA, "train_unified.jsonl")
    lab = Counter(); lang = Counter()
    exact = 0
    if os.path.exists(p):
        for l in open(p, encoding="utf-8"):
            if not l.strip(): continue
            d = json.loads(l)
            lab[d.get("label")] += 1
            t = d.get("text","")
            if zh_count(t) > 0: lang["zh"] += 1
            elif is_en(t): lang["en"] += 1
            exact += 1
    print(f"   train_unified 标签分布: {dict(lab)}")
    print(f"   train_unified 语言分布: {dict(lang)}  (总 {exact})")

    # 2. 公开数据集暂存区
    print("\n[2] 公开数据集暂存区 (data/datasets_raw/)")
    # C-ReD
    cred = []
    for fp in sorted(glob.glob(os.path.join(CRED, "CReD_paper_*.csv"))):
        gen = os.path.basename(fp).replace("CReD_paper_","").replace(".csv","")
        with open(fp, encoding="utf-8") as f:
            n = sum(1 for _ in f) - 1
        cred.append((gen, n))
    print("   C-ReD (paper领域, 摘要级):")
    for gen, n in sorted(cred):
        print(f"      {gen:<20} {n:>5} 篇")
    total_cred = sum(n for _, n in cred)
    print(f"      ── C-ReD paper 合计: {total_cred} 篇")

    # HC3
    hc3 = nl(os.path.join(RAW, "hc3chinese_all.jsonl"))
    hc3_lang = Counter()
    hc3_answers = 0
    if os.path.exists(os.path.join(RAW, "hc3chinese_all.jsonl")):
        for l in open(os.path.join(RAW,"hc3chinese_all.jsonl"), encoding="utf-8"):
            if not l.strip(): continue
            d = json.loads(l)
            hc3_answers += len(d.get("chatgpt_answers") or []) + len(d.get("human_answers") or [])
    print(f"   HC3-Chinese: {hc3} 条问答对 (含 AI/human 答案共约 {hc3_answers} 条)")

    # M4
    m4_files = defaultdict(list)
    for fp in sorted(glob.glob(os.path.join(M4, "*.jsonl"))):
        base = os.path.basename(fp).replace(".jsonl","")
        dom = base.split("_")[0]
        gen = base.split("_",1)[1] if "_" in base else "?"
        m4_files[dom].append((gen, nl(fp)))
    print("   M4 (Multi-generator/Multi-domain/Multi-lingual):")
    for dom, items in sorted(m4_files.items()):
        tot = sum(n for _, n in items)
        print(f"      [{dom}] {len(items)}生成器 共{tot}条 (每条含human+AI配对)")

    # 构建产物
    print("\n[3] 增量构建产物 (data/, 句级)")
    ai_pub = os.path.join(DATA, "ai_pub_samples.jsonl")
    hu_pub = os.path.join(DATA, "human_pub_samples.jsonl")
    ai_lang = Counter(); hu_lang = Counter(); ai_src = Counter(); hu_src = Counter()
    for p, lc, sc in ((ai_pub, ai_lang, ai_src), (hu_pub, hu_lang, hu_src)):
        if os.path.exists(p):
            for l in open(p, encoding="utf-8"):
                if not l.strip(): continue
                d = json.loads(l)
                lc[d.get("lang","zh")] += 1
                sc[d.get("source","?")] += 1
    print(f"   ai_pub_samples.jsonl  AI正样本:  总{sum(ai_lang.values())} (zh={ai_lang.get('zh',0)} en={ai_lang.get('en',0)})")
    print(f"      来源: {dict(ai_src)}")
    print(f"   human_pub_samples.jsonl human负样本: 总{sum(hu_lang.values())} (zh={hu_lang.get('zh',0)} en={hu_lang.get('en',0)})")
    print(f"      来源: {dict(hu_src)}")

    # 模型
    print("\n[4] 模型 (models/)")
    for fn in ["classifier.pkl", "classifier_en.pkl"]:
        p = os.path.join(BASE, "models", fn)
        if os.path.exists(p):
            import pickle
            m = pickle.load(open(p, "rb"))
            print(f"   {fn:<24} acc={m.get('acc'):.3f} auc={m.get('auc'):.3f} thr={m.get('threshold'):.3f}")
        else:
            print(f"   {fn:<24} (缺失)")

if __name__ == "__main__":
    main()
