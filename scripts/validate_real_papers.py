# -*- coding: utf-8 -*-
"""真实论文整篇实测：抽取真实 CNKI 学术论文全文，跑双流融合检测，验证整篇判定。
用户核心诉求：真实学术论文不得被误判为 AI。
用法：python scripts/validate_real_papers.py
"""
import os, sys, io, glob, re
sys.path.insert(0, r"C:\Users\woshi\.dsh\aigc-detector")
import numpy as np
import fitz  # pymupdf
from detector.dual_stream import load_bert, bert_score_per_sentence, fuse as ds_fuse
sys.path.insert(0, r"C:\Users\woshi\.dsh\aigc-detector\scripts")

def split_sentences(text):
    return [s for s in re.split(r"(?<=[。！？；])", text) if s.strip()]

DB = r"C:\Users\woshi\cnki-hub\data\downloads\cnki"
PAPERS = [
    "1011163632.nh_后现代语境下的传媒研究——戴维·莫利传播思想探析.pdf",
    "1012514735.nh_宋代女词人常见意象分析.pdf",
    "1011091582.nh_大学生社会联结、自我分化与心理幸福感的关系研究.pdf",
    "1012271379.nh_论环境协同治理——社会治理演进史视角中的环境问题及其应对.pdf",
    "1012520763.nh_电动往复输送系统的研究与优化设计.pdf",
]

def zh(t): return len(re.findall(r"[\u4e00-\u9fff]", t))

def load_stat():
    import pickle
    return pickle.load(open(r"C:\Users\woshi\.dsh\aigc-detector\models\classifier.pkl", "rb"))

def main():
    stat = load_stat()
    bm = load_bert(device="cuda")
    tok, model, dev = bm
    print("=== 真实学术论文整篇实测（双流融合，成本敏感阈 0.5）===\n")
    for name in PAPERS:
        fp = os.path.join(DB, name)
        if not os.path.exists(fp):
            print(f"[跳过] 缺 {name}"); continue
        doc = fitz.open(fp)
        text = "".join(p.get_text() for p in doc)
        doc.close()
        sents = split_sentences(text)
        sents = [s for s in sents if zh(s) >= 6]
        if not sents:
            print(f"[{name[:30]}] 无中文句"); continue
        # 限制句数避免过长
        use = sents[:400]
        X = stat["vec"].transform(use)
        p_tf = stat["model"].predict_proba(X)[:, 1].tolist()
        p_b = bert_score_per_sentence(tok, model, dev, use, batch=32)
        fused = [ds_fuse(float(a), float(b)) for a, b in zip(p_tf, p_b)]
        overall = sum(p*len(s) for p, s in zip(fused, use))/max(sum(len(s) for s in use),1)
        ai_count = sum(1 for p in fused if p >= 0.5)
        if overall >= 0.5: verdict = "高度疑似AI生成"
        elif overall >= 0.35: verdict = "疑似AI（人工复核）"
        elif overall >= 0.2: verdict = "证据不足（少量AI痕迹）"
        else: verdict = "基本人类撰写"
        print(f"[{name[:38]}]")
        print(f"   句数={len(use)} 整体分数={overall:.3f} AI句(阈0.5)={ai_count} 判定=【{verdict}】")

if __name__ == "__main__":
    main()
