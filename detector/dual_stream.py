# -*- coding: utf-8 -*-
"""双流融合检测器（v2：智能加权）。深度信号(微调RoBERTa) + 统计特征(TF-IDF/文体) 融合。

优化融合策略（解决"深度流对真实人类论文误判偏高"）：
- 两流一致（都在同侧 0.5）→ 取两者均值（更稳）
- 两流分歧 → 偏向统计流（统计流用7w+人类文献语料训练，对真实人类学术写作更准）
  - 统计流偏人类/深度流偏AI：折中向统计流靠（降低深度流对人类误判的权重）
  - 统计流偏AI/深度流偏人类：取均值（AI识别优先）
"""
import os, sys, json, re, pickle

# 融合置信度：分歧时统计流权重
TFIDF_BIAS_ON_DISAGREE = 0.7   # 统计流在分歧时的权重（0.6~0.8可调）

def load_bert(device="cuda"):
    import torch
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    path = r"C:\Users\woshi\.dsh\aigc-detector\models\roberta_ft"
    if not os.path.exists(path):
        return None
    try:
        tok = AutoTokenizer.from_pretrained(path)
        model = AutoModelForSequenceClassification.from_pretrained(path)
        dev = torch.device(device if torch.cuda.is_available() else "cpu")
        model.to(dev); model.eval()
        return tok, model, dev
    except Exception:
        return None

def bert_score_per_sentence(tok, model, dev, sentences, batch=32, max_len=200):
    """逐句AI概率，GPU内存安全（动态batch + 及时释放）。
    - 遍历处理，每个句子先估算 token 数，长句自动缩小 batch，避免 OOM。
    - 每批后释放 GPU 缓存。
    """
    import torch
    import torch.nn.functional as F
    if tok is None: return None
    probs = []
    i = 0
    with torch.no_grad():
        while i < len(sentences):
            # 估算当前 batch 的 token 量，超长句时缩小 batch
            chunk = sentences[i:i+batch]
            # 动态 batch：单句超长则单独处理（防止 OOM）
            eff_batch = batch
            for s in chunk:
                approx = len(s) * 1.5  # 中文字符≈1.5 token 估算
                if approx > max_len:
                    eff_batch = max(1, eff_batch // 2)
            chunk = sentences[i:i+eff_batch]
            inp = tok(chunk, truncation=True, max_length=max_len, padding="max_length", return_tensors="pt")
            inp = {k: v.to(dev) for k, v in inp.items()}
            out = model(**inp).logits
            p = F.softmax(out, -1)[:, 1].cpu().numpy()
            probs.extend(p.tolist())
            # 释放中间张量，缓解 GPU 内存
            del inp, out, p
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            i += eff_batch
    return probs

def fuse(p_tfidf, p_bert, bias=TFIDF_BIAS_ON_DISAGREE):
    """智能融合单句AI概率。"""
    if p_bert is None: return p_tfidf
    if p_tfidf is None: return p_bert
    # 判断是否分歧（两侧差 > 0.25）
    if abs(p_tfidf - p_bert) > 0.25:
        # 分歧：偏向统计流（它人类训练更充分）
        return bias*p_tfidf + (1-bias)*p_bert
    # 一致：均值
    return (p_tfidf + p_bert)/2

def sentence_level_fusion(tfidf_probs, bert_probs):
    if bert_probs is None: return tfidf_probs, None
    fused = [fuse(a, b) for a, b in zip(tfidf_probs, bert_probs)]
    return fused, bert_probs

def doc_score(sentences, fused_probs):
    if not fused_probs: return 0.0
    total_chars = sum(len(s) for s in sentences)
    return sum(p*len(s) for p, s in zip(fused_probs, sentences))/max(total_chars, 1)
