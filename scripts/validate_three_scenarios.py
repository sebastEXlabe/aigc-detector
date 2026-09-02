# -*- coding: utf-8 -*-
"""三场景验证：AI套话 / 知网高度AI段落 / 真实人类论文。
对比 统计流(TF-IDF) vs 深度流(RoBERTa) vs 融合后 的判定。
用法：python validate_three_scenarios.py
"""
import os, sys, io, json, re, pickle
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
BASE = r"C:\Users\woshi\.dsh\aigc-detector"
sys.path.insert(0, os.path.join(BASE))
from detector.dual_stream import load_bert, bert_score_per_sentence, fuse, doc_score

def read_text(path):
    ext = os.path.splitext(path)[1].lower()
    if ext == ".docx":
        from docx import Document
        d = Document(path); return "\n".join(p.text for p in d.paragraphs)
    return open(path, encoding="utf-8", errors="ignore").read()

def split_sentences(t):
    return [s.strip() for s in re.split(r"(?<=[。！？；])", t) if s.strip() and len(s.strip()) > 4]

def load_tfidf():
    p = os.path.join(BASE, "models", "classifier.pkl")
    if not os.path.exists(p): return None
    with open(p, "rb") as f: return pickle.load(f)

def main():
    # 加载两流
    tf_model = load_tfidf()
    bert = load_bert(device="cuda")
    print("统计流(TF-IDF):", "✓" if tf_model else "✗", " 深度流(RoBERTa):", "✓" if bert else "✗(需先微调)")
    if bert:
        tok, model, dev = bert
    else:
        tok = model = dev = None

    scenarios = [
        ("AI套话", os.path.join(BASE, "data", "test_sample.txt")),
        ("知网高度AI段落", os.path.join(BASE, "data", "test_aireal.txt")),
        ("真实人类论文", r"C:\Users\woshi\Downloads\2701821146195009967\轻资产运营模式下企业财务风险管理研究（修改稿）(2).docx"),
    ]
    for name, path in scenarios:
        if not os.path.exists(path): print(f"[{name}] 文件不存在"); continue
        text = read_text(path); sents = split_sentences(text)
        # 统计流
        p_tf = None
        if tf_model:
            X = tf_model["vec"].transform(sents)
            p_tf = tf_model["model"].predict_proba(X)[:, 1].tolist()
        # 深度流
        p_bert = None
        if bert:
            p_bert = bert_score_per_sentence(tok, model, dev, sents)
        # 融合
        if p_bert:
            fused = [fuse(a, b) for a, b in zip(p_tf or [0]*len(sents), p_bert)]
        else:
            fused = p_tf
        ds_tf = doc_score(sents, p_tf) if p_tf else None
        ds_bert = doc_score(sents, p_bert) if p_bert else None
        ds_fused = doc_score(sents, fused) if fused else None
        print(f"\n=== [{name}] 共{len(sents)}句 ===")
        if ds_tf is not None: print(f"  统计流 AI概率: {ds_tf*100:.1f}%")
        if ds_bert is not None: print(f"  深度流 AI概率: {ds_bert*100:.1f}%")
        if ds_fused is not None: print(f"  融合   AI概率: {ds_fused*100:.1f}%")

if __name__ == "__main__":
    main()
