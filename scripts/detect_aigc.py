# -*- coding: utf-8 -*-
"""统一 AIGC 检测工具（四路线融合 + 可解释特征 + 三态判定）。
路线B(统计流TF-IDF) + 路线D(深度流微调RoBERTa) 双流智能融合主判
+ 路线A(模板)定位 + 路线C(困惑度)佐证 + 文体特征。
用法：python detect_aigc.py <file> [--top-k 20]
"""
import os, sys, json, re, math, pickle
BASE = r"C:\Users\woshi\.dsh\aigc-detector"
sys.path.insert(0, os.path.join(BASE))
from detector.route_c import score_text, NgramLM
from detector.stylometric import stylo_features
from detector.dual_stream import load_bert, bert_score_per_sentence, fuse, doc_score
try:
    from detector.features import TEMPLATE_PATTERNS_ZH
except Exception:
    TEMPLATE_PATTERNS_ZH = []

def load_model():
    p = os.path.join(BASE, "models", "classifier.pkl")
    if not os.path.exists(p):
        print("模型不存在，先运行 scripts/train_classifier.py"); return None
    with open(p, "rb") as f: return pickle.load(f)

def load_lm():
    p = os.path.join(BASE, "models", "n-gram-lm.pkl")
    if not os.path.exists(p): return None
    with open(p, "rb") as f: return pickle.load(f)

BODY_END_MARKERS = ["参考文献", "致谢", "致　谢", "致 谢", "附录", "附　录", "发表论文", "攻读学位"]

def is_body_end(t):
    t = t.strip()
    if not t or len(t) > 25:
        return False
    for m in BODY_END_MARKERS:
        if t.startswith(m) or m in t[:6]:
            return True
    return False

def extract_body(paragraphs):
    body = []
    cut = None
    for t in paragraphs:
        if is_body_end(t):
            cut = 1
            break
        body.append(t)
    text = "\n".join(body).strip()
    if cut is not None and len(text) < 100:
        return "\n".join(paragraphs)
    return text

def docx_ordered_text(path):
    from docx import Document
    from docx.table import Table
    from docx.text.paragraph import Paragraph
    from docx.oxml.ns import qn
    d = Document(path)
    items = []
    for child in d.element.body.iterchildren():
        if child.tag == qn('w:p'):
            t = Paragraph(child, d).text
            if t.strip(): items.append(t)
        elif child.tag == qn('w:tbl'):
            tb = Table(child, d)
            for row in tb.rows:
                row_text = " | ".join(c.text.strip() for c in row.cells if c.text.strip())
                if row_text: items.append(row_text)
    return items

def read_text(path):
    ext = os.path.splitext(path)[1].lower()
    if ext == ".docx":
        paras = docx_ordered_text(path)
        full = "\n".join(paras)
        text = extract_body(paras)
        if len(text.strip()) < max(100, len(full.strip()) * 0.4):
            text = full
        return text
    return open(path, encoding="utf-8", errors="ignore").read()

def split_sentences(text):
    out = []
    buf = ""
    in_quote = False
    i = 0; n = len(text)
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
            if buf.strip() and len(buf.strip()) > 4:
                out.append(buf.strip())
            buf = ""
        i += 1
    if buf.strip() and len(buf.strip()) > 4:
        out.append(buf.strip())
    return out

def main():
    args = sys.argv[1:]
    if not args:
        print("用法: python detect_aigc.py <file> [--top-k 20]"); return
    path = args[0]; top_k = 20
    for i, a in enumerate(args):
        if a == "--top-k" and i + 1 < len(args):
            try: top_k = int(args[i + 1])
            except Exception: pass
    if not os.path.exists(path):
        print("文件不存在"); return
    model = load_model(); lm = load_lm()
    if not model: return
    text = read_text(path); sents = split_sentences(text)
    print(f"读入 {len(sents)} 句 (总字数 {len(text)})")

    # 统计流 (TF-IDF)
    vec = model["vec"]; cal = model["model"]; thr = model.get("threshold", 0.5)
    X = vec.transform(sents); p_tf = cal.predict_proba(X)[:, 1].tolist()
    # 深度流 (微调RoBERTa)
    bert = load_bert(device="cuda")
    p_bert = None
    if bert:
        tok, bmodel, dev = bert
        p_bert = bert_score_per_sentence(tok, bmodel, dev, sents)

    # 双流融合（智能加权）
    if p_bert:
        fused = [fuse(a, b) for a, b in zip(p_tf, p_bert)]
    else:
        fused = p_tf
    overall_b = doc_score(sents, p_tf)
    if p_bert: overall_d = doc_score(sents, p_bert)
    overall = doc_score(sents, fused)

    # 路线C 佐证（弱）
    perp_score = 0
    if lm:
        try:
            cscore, rels, bun, basep = score_text(lm, sents)
            perp_score = cscore
        except Exception:
            pass
    final = 0.9*overall + 0.1*perp_score

    ai_count = sum(1 for p in fused if p >= thr)
    print(f"\n=== 综合AI概率: {final*100:.1f}% ===")
    print(f"  统计流(TF-IDF): {overall_b*100:.1f}%")
    if p_bert: print(f"  深度流(RoBERTa): {overall_d*100:.1f}%")
    print(f"  双流融合: {overall*100:.1f}%  (路线C困惑度{perp_score*100:.1f}%)")
    print(f"  AI疑似句: {ai_count}/{len(sents)} ({ai_count/len(sents)*100:.1f}%)  (阈值 {thr:.2f})")

    doc_style = stylo_features(text)
    print(f"\n=== 文体特征（文档级）===")
    print(f"  词汇多样性MATTR: {doc_style['mattr']:.3f} (越高越像人)")
    print(f"  词汇重复度: {doc_style['repetition']:.3f} (越低越像人)")
    print(f"  标点密度: {doc_style['punct_density']:.3f} (AI常偏高)")
    print(f"  功能词密度: {doc_style['fw_density']:.3f} (AI常偏高)")

    if final >= 0.5:
        state = "⚠️ 高度疑似AI生成"
    elif final >= 0.35:
        state = "🔶 疑似AI（建议人工复核）"
    elif final >= 0.2:
        state = "❓ 证据不足（倾向人类，存在少量AI痕迹）"
    else:
        state = "✅ 基本人类撰写"
    print(f"三态判定: {state}  (综合AI概率 {final*100:.1f}%)")

    hi = sorted(sorted(range(len(sents)), key=lambda i: -fused[i])[:top_k])
    if hi:
        print(f"\n=== AI特征最明显的前 {len(hi)} 句 ===")
        for i in hi: print(f"  [{fused[i]*100:.0f}%] {sents[i][:65]}")

    print("\n=== 修改建议(AI模板) ===")
    found = 0
    for p, s in zip(fused, sents):
        if p < thr: continue
        for pat, w, d in TEMPLATE_PATTERNS_ZH:
            if re.search(pat, s):
                print(f"  · 「{d}」: {s[:48]}"); found += 1; break
    if not found: print("  (未命中已知模板句式)")

if __name__ == "__main__":
    main()
