# -*- coding: utf-8 -*-
"""可解释文体特征（调研方向1）。
词汇多样性、重复度、标点密度、句长变异、功能词分布等——用于辅助 AI 判定。
"""
import re, math
from collections import Counter

def mattr(text, window=50):
    """移动平均类型-词符比（MATTR），衡量词汇多样性。人类词汇更多样→MATTR高。"""
    w = [x for x in re.findall(r"[\u4e00-\u9fff]{1,4}|[a-zA-Z]+", text)]
    if len(w) < window:
        return len(set(w))/max(len(w),1)
    vals=[]
    for i in range(0, len(w)-window+1, window//2):
        seg=w[i:i+window]
        vals.append(len(set(seg))/len(seg))
    return sum(vals)/len(vals)

def repetition(text):
    """词汇重复度（AI 词汇重复高）。1 - 唯一词比例。"""
    w=[x for x in re.findall(r"[\u4e00-\u9fff]{1,4}|[a-zA-Z]+", text)]
    if not w: return 0.0
    return 1.0 - len(set(w))/len(w)

def punct_density(text):
    """标点密度（AI常标点密集）。"""
    if not text: return 0.0
    punct=len(re.findall(r"[，。、；：！？,.;:!?]", text))
    return punct/len(text)

def clause_len_variability(text):
    """子句长度变异（句长标准差/均值，burstiness 的一种）。"""
    clauses=[c.strip() for c in re.split(r"[，。；、：]", text) if c.strip()]
    lens=[len(c) for c in clauses]
    if len(lens)<2: return 0.0
    avg=sum(lens)/len(lens)
    if avg==0: return 0.0
    return math.sqrt(sum((l-avg)**2 for l in lens)/len(lens))/avg

def function_word_density(text):
    """中文功能词密度（AI/规范写作功能词分布特征）。"""
    fw=["的","了","和","与","及","或","在","对","为","从","以","而","于","是","也","都","就","并","且","但","这","其","之","将"]
    return sum(text.count(w) for w in fw)/max(len(text),1)

def stylo_features(text):
    return {
        "mattr": mattr(text),
        "repetition": repetition(text),
        "punct_density": punct_density(text),
        "clause_var": clause_len_variability(text),
        "fw_density": function_word_density(text),
    }
