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

# ---- 方向1：文档门控两遍校准 ----
# 只在"文档明显偏真人"时向下软化孤立的高AI句（真实论文里的字面误标）；
# 偏AI/混杂文档不软化，保住检出。修正上一版"对混杂报告无差别拉平"的缺陷。
HUMAN_DOC_GATE = 0.35     # doc 均值 < 此值 → 判为偏真人文档，允许向下软化
SOFTEN_ALPHA_MAX = 0.45   # 偏真人文档里，句级向文档均值靠拢的最大权重

def gated_doc_calibrate(sent_probs, doc_ids):
    """返回 (新句级概率, 是否应用了软化)。
    对 doc 均值 < HUMAN_DOC_GATE 的文档：把孤立高AI句向文档均值靠拢(抑制误标)；
    其余文档：保持不变。
    """
    sent_probs = np.asarray(sent_probs, dtype=float)
    doc_ids = np.asarray(doc_ids)
    out = sent_probs.copy()
    applied = {}
    for d in np.unique(doc_ids):
        mask = doc_ids == d
        group = sent_probs[mask]
        docm = _doc_mean(group)
        if docm < HUMAN_DOC_GATE:
            # 偏真人文档：向下软化高AI句（仅对 > docm 的句子，抑制孤立误标）
            alpha = _alpha(docm) * SOFTEN_ALPHA_MAX / 0.35
            alpha = min(0.45, alpha)
            for i in np.where(mask)[0]:
                s = sent_probs[i]
                if s > docm:
                    out[i] = alpha * docm + (1 - alpha) * s
            applied[d] = True
    return out, applied

# ---- 方向2：AI 密集段识别（解决"AI局部掺入被稀释"盲区）----
def max_ai_window(scores, thr=0.5, window=6):
    """滑动窗口内句级>=thr的最大占比。文档整体低分但某段高AI(如AI摘要)时该值高。
    返回 (max_proportion, 窗口起始索引, 命中窗口句列表)。"""
    scores = np.asarray(scores, dtype=float)
    n = len(scores)
    if n == 0: return 0.0, -1, []
    if n < window: window = n
    best = 0.0; best_start = -1; best_sents = []
    for start in range(0, n - window + 1):
        w = scores[start:start+window]
        prop = float((w >= thr).mean())
        if prop > best:
            best = prop; best_start = start
            best_sents = w.tolist()
    return best, best_start, best_sents

def max_ai_window_mean(scores, window=6):
    """滑动窗口内句级概率的**平均分**最大值。用途：识别"AI密集段"。
    实测：真人论文窗口平均最高~0.80；真实AI/混合AI段 ~0.95+。阈值 0.9 可分离。
    返回 (max_window_mean, 窗口起始索引)。"""
    scores = np.asarray(scores, dtype=float)
    n = len(scores)
    if n == 0: return 0.0, -1
    if n < window: window = n
    best = 0.0; best_start = -1
    for start in range(0, n - window + 1):
        m = float(scores[start:start+window].mean())
        if m > best:
            best = m; best_start = start
    return best, best_start
