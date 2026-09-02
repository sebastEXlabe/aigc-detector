
# -*- coding: utf-8 -*-
"""路线A：中文AI写作特征计算器。
输入一句话/一段文字，输出一组可解释的AI痕迹特征（数值 + 命中明细）。
"""
import re, math
from collections import Counter
from .features import (
    TEMPLATE_PATTERNS_ZH, HIGH_FREQ_PATTERNS_EN, HIGH_FREQ_WORDS_EN,
    SHORT_SENT_LEN, LONG_SENT_LEN,
    CONNECTIVE_WORDS_ZH, CONNECTIVE_WORDS_EN, THRESHOLDS,
)

def is_chinese(text):
    """粗略判断是否中文为主（中文字符占比>0.3）。"""
    zh = len(re.findall(r"[\u4e00-\u9fff]", text))
    return zh / max(len(text), 1) > 0.3

def count_patterns(text, patterns):
    """统计每条模板正则的命中次数。返回 (总命中, 明细)。"""
    total = 0
    hits = []
    for pat, w, desc in patterns:
        n = len(re.findall(pat, text))
        if n:
            total += n * w
            hits.append({"pattern": pat, "count": n, "weight": w, "desc": desc})
    return total, hits

def per_sentence_features(sent):
    """单句AI痕迹特征。返回 dict。"""
    feats = {}
    feats["len"] = len(sent)
    feats["is_chinese"] = is_chinese(sent)
    # 模板句式命中
    if feats["is_chinese"]:
        t_score, t_hits = count_patterns(sent, TEMPLATE_PATTERNS_ZH)
    else:
        t_score, t_hits = count_patterns(sent, HIGH_FREQ_PATTERNS_EN)
    feats["template_score"] = t_score
    feats["template_hits"] = t_hits
    # 连接词密度
    conn_set = CONNECTIVE_WORDS_ZH if feats["is_chinese"] else CONNECTIVE_WORDS_EN
    conn_count = sum(len(re.findall(r"\b"+re.escape(w)+r"\b", sent)) if not feats["is_chinese"] else sent.count(w) for w in conn_set)
    feats["connective_density"] = conn_count / max(len(sent), 1)
    # AI高频英文词
    hi_en = sum(sent.lower().count(w) for w in HIGH_FREQ_WORDS_EN if w in sent.lower()) if not feats["is_chinese"] else 0
    feats["ai_en_words"] = hi_en
    # 长句/短句判定
    feats["is_long"] = len(sent) > LONG_SENT_LEN
    feats["is_short"] = len(sent) < SHORT_SENT_LEN
    return feats

def document_features(sentences):
    """整篇文档级AI痕迹特征。sentences = 句子字符串列表。"""
    if not sentences:
        return {}
    lens = [len(s) for s in sentences]
    avg_len = sum(lens) / len(lens)
    # burstiness: 句长标准差（AI文本句长均匀→低标准差）
    std = math.sqrt(sum((l - avg_len)**2 for l in lens) / len(lens))
    # 变异系数（burstiness）
    cv = std / avg_len if avg_len else 0
    # 句长分桶：AI常集中在一两个长度档
    len_var = max(lens) - min(lens)
    # 连接词密度
    total_chars = sum(lens)
    conn_total = 0
    template_total = 0
    for s in sentences:
        sf = per_sentence_features(s)
        conn_total += sf["connective_density"] * len(s)
        template_total += sf["template_score"]
    conn_doc = conn_total / max(total_chars, 1)
    # 模板命中率（含模板句占比）
    templated = sum(1 for s in sentences if per_sentence_features(s)["template_score"] > 0)
    return {
        "n_sentences": len(sentences),
        "avg_sent_len": avg_len,
        "sent_len_std": std,
        "burstiness_cv": cv,
        "sent_len_range": len_var,
        "connective_density": conn_doc,
        "template_total": template_total,
        "templated_sent_ratio": templated / len(sentences),
        "share_long": sum(1 for l in lens if l > LONG_SENT_LEN) / len(sentences),
        "share_short": sum(1 for l in lens if l < SHORT_SENT_LEN) / len(sentences),
    }
