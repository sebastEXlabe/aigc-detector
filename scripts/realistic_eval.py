# -*- coding: utf-8 -*-
"""跳出报告标签框架，做真实文档级独立验证。
真实人类 (真实CNKI全篇) vs 真实AI (真实LLM生成的学术文本=C-ReD) vs 混合(AI摘要+真人正文)。
跑双流融合检测(含方向1门控)，给整篇判定 + 句级概览。
用法：python scripts/realistic_eval.py
"""
import os, sys, re, csv, glob
sys.path.insert(0, r"C:\Users\woshi\.dsh\aigc-detector")
import numpy as np, fitz
from detector.dual_stream import load_bert, bert_score_per_sentence, fuse as ds_fuse
from detector.consistency import gated_doc_calibrate
from scripts.cross_validate import stat_probs, load_cls

def zh(t): return len(re.findall(r"[\u4e00-\u9fff]", t))
def ss(t): return [s for s in re.split(r"(?<=[。！？；])", t) if s.strip()]

DB = r"C:\Users\woshi\cnki-hub\data\downloads\cnki"
CRED = r"C:\Users\woshi\.dsh\aigc-detector\data\datasets_raw\cred"

def load_human_docs(n=6):
    """取若干真实 CNKI 论文全篇。"""
    import random
    pdfs = [g for g in glob.glob(os.path.join(DB, "*.pdf")) if g.endswith(".pdf")]
    random.seed(3); random.shuffle(pdfs)
    docs = []
    for fp in pdfs:
        try:
            d = fitz.open(fp); txt = "".join(p.get_text() for p in d); d.close()
        except: continue
        if zh(txt) >= 300: docs.append(txt)
        if len(docs) >= n: break
    return docs

def load_ai_docs(n=6):
    """取真实 LLM 生成的学术文本（C-ReD 各生成器摘要），组装成 AI 文档。"""
    import random
    gens = ["gpt-4o","deepseek-r1","deepseek-v3","claude-3.5-haiku","qwen-3","doubao-1.5-pro","qwen-2.5","gemini-2.5-flash"]
    docs = []
    for gen in gens:
        fp = os.path.join(CRED, "CReD_paper_%s.csv" % gen)
        if not os.path.exists(fp): continue
        rows = []
        for r in csv.DictReader(open(fp, encoding="utf-8")):
            try: lab=int(r.get("label"))
            except: continue
            if lab==0 and (r.get("text") or ""): rows.append(r["text"])  # AI
        if not rows: continue
        ai_txt = "".join(rows[:8])  # 拼几段真实 LLM 生成
        if zh(ai_txt) >= 300: docs.append((gen, ai_txt))
        if len(docs) >= n: break
    return docs

def run_detector(text, stat, bm):
    tok,model,dev = bm
    sents = [s for s in ss(text) if zh(s)>=6][:500]
    if not sents: return None
    pt = stat_probs(stat, sents); pb = bert_score_per_sentence(tok,model,dev,sents,batch=32)
    fused_raw = np.array([ds_fuse(float(a),float(b)) for a,b in zip(pt,pb)])
    from detector.consistency import gated_doc_calibrate, max_ai_window_mean
    ai_island,_ = max_ai_window_mean(fused_raw, 6)
    gated,_ = gated_doc_calibrate(fused_raw, [0]*len(sents))
    overall = float((gated*np.array([len(s) for s in sents])).sum()/max(sum(len(s) for s in sents),1))
    ai_cnt = int((gated>=0.5).sum()); ai_rate = float((gated>=0.5).mean())
    if overall>=0.5: verdict="高度疑似AI生成"
    elif ai_island>=0.9: verdict="疑似AI（存在AI密集段）"
    elif overall>=0.2: verdict="证据不足（少量AI痕迹）"
    else: verdict="基本人类撰写"
    return dict(overall=overall, verdict=verdict, ai_rate=ai_rate, ai_cnt=ai_cnt, island=ai_island)

def main():
    stat = load_cls(); bm = load_bert(device="cuda")
    print("=== 真实文档级独立验证（双流融合+方向1门控, 阈0.5）===\n")
    print("【真实人类论文】(应判 基本人类/证据不足，不该判高度疑似)")
    for i,txt in enumerate(load_human_docs(6)):
        r=run_detector(txt,stat,bm)
        if r: print(f"  人类{i+1}: {r['overall']:.3f} [ {r['verdict']} ]  句级AI率={r['ai_rate']:.3f}")
    print("\n【真实 AI 生成学术文本】(应判 疑似/高度疑似)")
    for gen,ai_txt in load_ai_docs(6):
        r=run_detector(ai_txt,stat,bm)
        if r: print(f"  AI[{gen}]: {r['overall']:.3f} [ {r['verdict']} ]  句级AI率={r['ai_rate']:.3f}")
    print("\n【混合: AI摘要 + 真人正文】(应标出AI痕迹)")
    hu=[t for t in load_human_docs(3)]
    ai=load_ai_docs(3)
    if hu and ai:
        gen,aitxt = ai[0]
        mixed = aitxt[:600] + hu[0][:6000]
        r=run_detector(mixed,stat,bm)
        if r: print(f"  混合[AI{gen}+真人]: {r['overall']:.3f} [ {r['verdict']} ]")

if __name__=="__main__": main()
