
# -*- coding: utf-8 -*-
"""路线A：中文 AI 写作特征库。

基于 humanizer skill（Wikipedia "Signs of AI writing" 中文映射）+ 学术论文 AIGC 检测的
常见语言学特征整理。每类特征给出：正则 / 词表、权重、说明。用于启发式打分。
"""

# ---------- 1. AI 高频写作模板句式（固定搭配，AI 极常用） ----------
# 每条: (正则, 权重, 说明)
TEMPLATE_PATTERNS_ZH = [
    # 学术论文常见 AI 模板
    (r"综上所述[，,]", 1.2, "『综上所述』：AI/Sci论文高频总结词"),
    (r"总而言之[，,]", 1.2, "『总而言之』：AI总结模板"),
    (r"值得注意的是[，,]", 1.3, "『值得注意的是』：AI强调句式"),
    (r"不难发现[，,]", 1.2, "『不难发现』：AI推断模板"),
    (r"进一步而言[，,]", 1.2, "『进一步而言』：AI递进模板"),
    (r"从理论层面看", 1.3, "『从理论层面看』：AI空泛概括"),
    (r"从实践角度看", 1.2, "『从实践角度看』：AI对仗模板"),
    (r"在[^。，]{2,30}的背景下[，,]", 1.0, "『在...的背景下』：AI背景铺垫模板"),
    (r"具有重要的现实意义", 1.2, "『具有重要的现实意义』：AI意义套话"),
    (r"不仅[^。，]{2,20}，更[^。，]{2,20}", 1.0, "『不仅...更...』：AI排比递进"),
    (r"一方面[^。，]{2,30}，另一方面[^。，]{2,30}", 1.0, "『一方面...另一方面』：AI并列模板"),
    (r"总而言之|总的来说|总体而言", 1.1, "『总的来说』：AI概括"),
    (r"随着[^。，]{2,30}的不断[^。，]{2,20}", 1.0, "『随着...的不断...』：AI背景引入"),
    (r"起到了[^。，]{2,20}的作用", 1.0, "『起到了...的作用』：AI功能套话"),
    (r"为[^。，]{2,30}提供[^。，]{2,20}", 0.9, "『为...提供...』：AI价值句式"),
    (r"赋能[^。，]{0,20}", 1.3, "『赋能』：AI高频时髦词"),
    (r"助益|助力[^。，]{0,20}", 1.0, "『助力』：AI时髦动词"),
    (r"新质生产力|范式重构|范式转移", 1.2, "『范式重构/新质生产力』：AI热词"),
    (r"不谋而合|殊途同归", 1.0, "『不谋而合』：AI成语化结论"),
    (r"擘画|绘就|谱写|谱写新篇章", 1.1, "『擘画/绘就』：AI宣传化动词"),
    (r"其核心在于", 1.1, "『其核心在于』：AI点题模板"),
    (r"归根结底[，,]", 1.1, "『归根结底』：AI收束词"),
    (r"毋庸置疑[，,]", 1.2, "『毋庸置疑』：AI断言"),
    (r"不可否认[，,]", 1.1, "『不可否认』：AI让步转折"),
    (r"由此可见[，,]", 1.1, "『由此可见』：AI推导"),
    (r"换言之[，,]", 1.1, "『换言之』：AI改写词"),
    (r"换言之|也就是说|换句话说", 1.0, "『换句话说』：AI解释套话"),
]

# ---------- 2. AI 高频词汇（英文论文 + 学术） ----------
# 每条: (词, 权重)
HIGH_FREQ_WORDS_EN = [
    "delve", "crucial", "moreover", "furthermore", "underscore", "emphasize",
    "pivotal", "essential", "significant", "notably", "meanwhile", "consequently",
    "henceforth", "thus", "thereby", "fostering", "cultivating", "showcase",
    "exemplify", "comprehensive", "holistic", "multifaceted", "interplay",
    "trajectory", "landscape", "paradigm", "synergy", "leverage", "robust",
    "impactful", "insightful", "delve into", "it is worth noting", "in conclusion",
    "this paper explores", "plays a pivotal role", "underscores the importance",
]
# 每条: (正则, 权重, 说明)
HIGH_FREQ_PATTERNS_EN = [
    (r"\bdelve\b", 1.3, "delve"),
    (r"\bcrucial\b", 1.2, "crucial"),
    (r"\bpivotal\b", 1.2, "pivotal"),
    (r"\bmoreover\b", 1.1, "moreover"),
    (r"\bfurthermore\b", 1.1, "furthermore"),
    (r"\bunderscor\w*\b", 1.2, "underscore"),
    (r"\bplays a (pivotal|crucial|vital) role\b", 1.3, "plays a...role"),
    (r"\bit is worth noting\b", 1.3, "it is worth noting"),
    (r"\bin conclusion\b", 1.0, "in conclusion"),
    (r"\bthis paper explores\b", 0.9, "this paper explores"),
]

# ---------- 3. 结构/统计特征 ----------
# 说明：AI 文本通常句长均匀、连接词过多、标点整齐（burstiness 低）。
# 这些在 detector/features.py 里数值计算，此处仅列常量。

# 句长过短异常阈值
SHORT_SENT_LEN = 8
# 句长过长（AI 常写长复合句超过 80 字）
LONG_SENT_LEN = 80
# 并列连接词密度阈值（AI 常用）
CONNECTIVE_WORDS_ZH = ["同时", "此外", "另外", "而且", "进而", "此外", "因此", "然而", "并且", "并且", "但是", "由于", "从而", "综上"]
CONNECTIVE_WORDS_EN = ["however", "moreover", "furthermore", "additionally", "therefore", "thus", "consequently", "meanwhile", "nevertheless", "accordingly"]

# ---------- 4. 判定档位（对照 PaperYY/知网口径） ----------
THRESHOLDS = {
    "human": 0.40,    # <40% 判定人类创作
    "suspect": 0.60,  # 40-60% 疑似AI
    "ai": 0.60,       # >=60% 判定AI
}
