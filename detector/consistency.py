# -*- coding: utf-8 -*-
"""文档一致性（TTA 式测试时自适应）校准。

思路（源自"测试时自适应/半监督"：利用推理时未标注样本的同质性）：
  一篇文档/一份报告内的句子通常是同质的（要么整体 AI 生成，要么整体真人写作）。
  因此用文档级信号（句均 AI 概率）作为"上下文先验"，把每句的概率做一次上下文校准，
  抑制孤立误判：
    · 文档整体强 AI（docm 远高于 0.5）→ 句子分数向 AI 端靠拢（降低漏检孤立人类句）
    · 文档整体强人类（docm 远低于 0.5）→ 句子分数向人类端靠拢（降低误报孤立AI句）
    · 文档本身模糊（docm 接近 0.5）→ 文档信息量低，不校准，信任句子自身分数

函数：
  document_calibrate(sent_probs, doc_groups) -> np.ndarray
    其中 doc_groups 为与 sent_probs 等长的"文档分组 id"数组（同一文档同 id）。
"""
import numpy as np

def _doc_mean(group_probs):
    return float(np.mean(group_probs)) if len(group_probs) else 0.5

def _alpha(docm):
    """文档置信度加权。|docm-0.5| 越大 → 文档信号越强 → 校准权重越高。"""
    d = abs(docm - 0.5)          # 0 ~ 0.5
    # 置信度越高，alpha 越大；模糊时 alpha≈0（完全信任句子自身）
    return float(min(0.35, 0.9 * d))   # 上限 0.35，避免过度抹平

def document_calibrate(sent_probs, doc_ids):
    """返回校准后的句级概率（范围保守收敛到 [sent, docm] 之间，不会越过边界）。
    sent_probs: np.ndarray 句级融合概率；doc_ids: 等长数组，同文档同 id。
    """
    sent_probs = np.asarray(sent_probs, dtype=float)
    doc_ids = np.asarray(doc_ids)
    out = np.empty_like(sent_probs)
    # 分组
    seen = {}
    for i, d in enumerate(doc_ids):
        if d not in seen:
            mask = doc_ids == d
            group = sent_probs[mask]
            docm = _doc_mean(group)
            alpha = _alpha(docm)
            seen[d] = (docm, alpha)
        docm, alpha = seen[d]
        # 向文档均值靠拢：c = alpha*docm + (1-alpha)*sent
        out[i] = alpha * docm + (1 - alpha) * sent_probs[i]
    return out

def doc_mean(sent_probs):
    return _doc_mean(list(sent_probs))
