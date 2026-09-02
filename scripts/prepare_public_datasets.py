# -*- coding: utf-8 -*-
"""公开 AIGC 检测数据集 → 本地句级标注增量数据。

来源：
  1. C-ReD (ACL 2026 Findings, 中文论文/学术 AIGC 检测基准)
     论文摘要级：label 0=AI 生成, 1=真人。按摘要整体分句，句子继承摘要 label。
  2. HC3-Chinese (Hello-SimpleAI, CC-BY-SA 4.0)
     问答对：chatgpt_answers → AI；human_answers → 真人。分句标注。
  3. M4 (mbzuai-nlp, EACL 2024, 多生成器/多领域/多语言)
     每条含配对 human_text(真人) + machine_text(AI)。arxiv/peerread=英文学术，
     qazh=中文问答。分句标注，含语言标签。

输出：
  data/ai_pub_samples.jsonl     AI 正样本（句级）   {text, prob, source, generator, lang}
  data/human_pub_samples.jsonl  human 负样本（句级）{text, prob, source, generator, lang}
说明：
  - AI 句 prob 设 0.85，human 句 prob 设 0.08（与现有训练脚本 prob 阈值兼容）。
  - lang: 'zh'(中文) / 'en'(英文)。中文句要求 >=6 个中文字；英文句要求 >=8 个英文字符词数。
  - 中文分句用字符级扫描（处理引号内句号/省略号/连续标点）；英文分句用句号边界（处理缩写/小数点）。
"""
import os, sys, io, json, re, glob, csv

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

BASE = r"C:\Users\woshi\.dsh\aigc-detector"
RAW = os.path.join(BASE, "data", "datasets_raw")
CRED = os.path.join(RAW, "cred")
HC3 = os.path.join(RAW, "hc3chinese_all.jsonl")
M4 = os.path.join(RAW, "m4")
OUT_AI = os.path.join(BASE, "data", "ai_pub_samples.jsonl")
OUT_HUMAN = os.path.join(BASE, "data", "human_pub_samples.jsonl")

MIN_ZH = 6  # 最少中文字数
MIN_EN_WORDS = 8  # 最少英文字数（Word 计数）

def has_zh(t):
    return bool(re.search(r"[\u4e00-\u9fff]", t))

def zh_count(t):
    return len(re.findall(r"[\u4e00-\u9fff]", t))

def is_latin(t):
    # 判断是否主要为英文（拉丁字母占比高且含空格单词）
    letters = len(re.findall(r"[A-Za-z]", t))
    return letters > 0 and zh_count(t) == 0

def split_sentences(text):
    """字符级智能分句（与检测服务一致）：引号内句号不分、省略号不分、连续标点合并。"""
    out = []
    buf = ""
    in_quote = False
    i = 0
    n = len(text)
    while i < n:
        c = text[i]
        if c in "“「『":
            in_quote = True
        elif c in "”」』":
            in_quote = False
        buf += c
        if not in_quote and c in "。？！":
            j = i + 1
            while j < n and text[j] in "。？！":
                buf += text[j]; j += 1
            i = j - 1
            if buf.strip() and len(buf.strip()) > 4:
                out.append(buf.strip())
            buf = ""
        elif not in_quote and c == "…":
            j = i + 1
            while j < n and text[j] == "…":
                buf += text[j]; j += 1
            i = j - 1
        elif c in ";；":
            if buf.strip() and len(buf.strip()) > 4:
                out.append(buf.strip())
            buf = ""
        i += 1
    if buf.strip() and len(buf.strip()) > 4:
        out.append(buf.strip())
    return out

def split_en_sentences(text):
    """英文智能分句：按句号+空格/换行边界切分，避免小数点、缩写(如 e.g./i.e./Dr.)、版本号误切。"""
    # 保护缩写与小数：把 'e.g.' 'i.e.' 'et al.' 'Dr.' 'Mr.' 'Fig.' 及小数保留不切
    protected = re.sub(r"\b(e\.g\.|i\.e\.|et al\.|Dr\.|Mr\.|Ms\.|Fig\.|Eq\.|vs\.|etc\.|Inc\.|Ltd\.|ca\.|approx\.|cf\.|ibid\.)\b", lambda m: m.group(0).replace(".", "\u0001"), text)
    protected = re.sub(r"(\d\.\d)", lambda m: m.group(1).replace(".", "\u0002"), protected)
    parts = re.split(r"(?<=\.)\s+|\n+", protected)
    sents = []
    for p in parts:
        p = p.replace("\u0001", ".").replace("\u0002", ".").strip()
        p = re.sub(r"\s+", " ", p).strip()
        if p and len(p.split()) >= MIN_EN_WORDS:
            sents.append(p)
    return sents

def clean_sent(text, lang=None):
    """清洗：去空白、去 markdown 残留、去纯数字/符号。按语言校验长度。"""
    t = re.sub(r"\s+", " ", text).strip()
    t = t.strip("·—•*#")
    if not t:
        return None
    if lang is None:
        lang = "zh" if has_zh(t) else ("en" if is_latin(t) else "other")
    if lang == "zh":
        if zh_count(t) < MIN_ZH:
            return None
        if not has_zh(t):
            return None
    elif lang == "en":
        if len(t.split()) < MIN_EN_WORDS:
            return None
        if not is_latin(t):
            return None
    else:
        return None
    return t

def load_cred():
    ai = []
    human = []
    for fp in sorted(glob.glob(os.path.join(CRED, "CReD_paper_*.csv"))):
        gen = os.path.basename(fp).replace("CReD_paper_", "").replace(".csv", "")
        with open(fp, encoding="utf-8") as f:
            for r in csv.DictReader(f):
                text = (r.get("text") or "").strip()
                if not text:
                    continue
                try:
                    label = int(r.get("label"))
                except (ValueError, TypeError):
                    continue
                for s in split_sentences(text):
                    c = clean_sent(s)
                    if not c:
                        continue
                    rec = {"text": c, "prob": 0.85 if label == 0 else 0.08,
                           "source": "C-ReD-paper", "generator": gen}
                    (ai if label == 0 else human).append(rec)
    return ai, human

def load_hc3():
    ai = []
    human = []
    if not os.path.exists(HC3):
        print("HC3 文件缺失，跳过:", HC3)
        return ai, human
    for l in open(HC3, encoding="utf-8"):
        if not l.strip():
            continue
        d = json.loads(l)
        ca = d.get("chatgpt_answers") or []
        ha = d.get("human_answers") or []
        ca = "".join(x for x in ca if isinstance(x, str))
        ha = "".join(x for x in ha if isinstance(x, str))
        for s in split_sentences(ca):
            c = clean_sent(s)
            if c:
                ai.append({"text": c, "prob": 0.85, "source": "HC3", "generator": "chatgpt"})
        for s in split_sentences(ha):
            c = clean_sent(s)
            if c:
                human.append({"text": c, "prob": 0.08, "source": "HC3", "generator": "human"})
    return ai, human

def _join_text(v):
    """M4 字段可能是 str 或 list[str]（peerread 为逐条意见列表）。统一成字符串。"""
    if v is None:
        return ""
    if isinstance(v, str):
        return v
    if isinstance(v, list):
        return "\n".join(str(x) for x in v if isinstance(x, str))
    return str(v)

def load_m4():
    """M4：配对 human_text/machine_text。source 前缀判文体与语言。"""
    ai = []
    human = []
    if not os.path.isdir(M4):
        print("M4 目录缺失，跳过:", M4)
        return ai, human
    for fn in sorted(glob.glob(os.path.join(M4, "*.jsonl"))):
        base = os.path.basename(fn).replace(".jsonl", "")
        # 从文件名前缀推断 source 类别
        if base.startswith("arxiv"):
            src = "M4-en-academic"      # 英文学术摘要
        elif base.startswith("peerread"):
            src = "M4-en-academic"      # 英文学术同行评议
        elif base.startswith("qazh"):
            src = "M4-zh-qa"            # 中文问答
        elif base.startswith("wikipedia"):
            src = "M4-en-wiki"          # 英文维基
        else:
            continue
        gen = (base.split("_", 1)[1] if "_" in base else "?")
        try:
            # 每文件句子产出上限，避免单个来源过度膨胀
            cap_ai, cap_hu = 6000, 6000
            n_ai = n_hu = 0
            with open(fn, encoding="utf-8") as f:
                for l in f:
                    if not l.strip():
                        continue
                    d = json.loads(l)
                    hu = _join_text(d.get("human_text"))
                    ma = _join_text(d.get("machine_text"))
                    for txt, is_ai, prob in ((hu, False, 0.08), (ma, True, 0.85)):
                        if not txt:
                            continue
                        lang = "zh" if has_zh(txt) else ("en" if is_latin(txt) else None)
                        if lang not in ("zh", "en"):
                            continue
                        for s in (split_sentences(txt) if lang == "zh" else split_en_sentences(txt)):
                            c = clean_sent(s, lang)
                            if not c:
                                continue
                            if is_ai and n_ai >= cap_ai:
                                continue
                            if not is_ai and n_hu >= cap_hu:
                                continue
                            rec = {"text": c, "prob": prob, "source": src,
                                   "generator": gen, "lang": lang}
                            (ai if is_ai else human).append(rec)
                            if is_ai:
                                n_ai += 1
                            else:
                                n_hu += 1
        except Exception as e:
            print(f"  M4 解析失败 {base}: {e}")
    return ai, human

def dedup(recs):
    seen = set()
    out = []
    for r in recs:
        t = r["text"]
        if t in seen:
            continue
        seen.add(t)
        out.append(r)
    return out

def main():
    cred_ai, cred_hu = load_cred()
    hc3_ai, hc3_hu = load_hc3()
    m4_ai, m4_hu = load_m4()
    # 统一补 lang 字段（C-ReD/HC3 为中文，M4 由 load_m4 已标 lang）
    for r in cred_ai + cred_hu + hc3_ai + hc3_hu:
        r["lang"] = "zh"
    ai = dedup(cred_ai + hc3_ai + m4_ai)
    human = dedup(cred_hu + hc3_hu + m4_hu)
    print("== 增量数据构建 ==")
    print(f"  C-ReD  AI句: {len(cred_ai)}  human句: {len(cred_hu)}")
    print(f"  HC3    AI句: {len(hc3_ai)}  human句: {len(hc3_hu)}")
    print(f"  M4     AI句: {len(m4_ai)}  human句: {len(m4_hu)}")
    # 语言统计
    ai_en = sum(1 for r in ai if r.get("lang") == "en")
    ai_zh = sum(1 for r in ai if r.get("lang") == "zh")
    hu_en = sum(1 for r in human if r.get("lang") == "en")
    hu_zh = sum(1 for r in human if r.get("lang") == "zh")
    print(f"  去重后 AI正样本: {len(ai)} (zh={ai_zh} en={ai_en})")
    print(f"  去重后 Human负样本: {len(human)} (zh={hu_zh} en={hu_en})")
    os.makedirs(os.path.dirname(OUT_AI), exist_ok=True)
    with open(OUT_AI, "w", encoding="utf-8") as f:
        for r in ai:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with open(OUT_HUMAN, "w", encoding="utf-8") as f:
        for r in human:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print("  已保存:", OUT_AI)
    print("  已保存:", OUT_HUMAN)
    # 抽样展示
    if ai:
        print("--- AI 样本示例 ---")
        for r in ai[:2]:
            print("   ", r["text"][:70], "| gen:", r["generator"], "| lang:", r.get("lang"))
    if human:
        print("--- Human 样本示例 ---")
        for r in human[:2]:
            print("   ", r["text"][:70], "| src:", r["source"], "| lang:", r.get("lang"))

if __name__ == "__main__":
    main()
