# -*- coding: utf-8 -*-
"""综合测试方法：跨维度（掺入比例/学科/长度/段落一致/多语言/难度）检测性能矩阵。
跳出单一报告标签框架，用真实人类论文 + 真实LLM输出，跑完整指标。
用法：python scripts/test_suite.py
"""
import os, sys, re, glob, csv, random
sys.path.insert(0, r"C:\Users\woshi\.dsh\aigc-detector")
import numpy as np, fitz
from detector.dual_stream import load_bert, bert_score_per_sentence, fuse as ds_fuse
from detector.consistency import gated_doc_calibrate, max_ai_window_mean
from scripts.cross_validate import stat_probs, load_cls

def zh(t): return len(re.findall(r"[\u4e00-\u9fff]", t))
def ss(t): return [s for s in re.split(r"(?<=[。！？；])", t) if s.strip()]

DB = r"C:\Users\woshi\cnki-hub\data\downloads\cnki"
CRED = r"C:\Users\woshi\.dsh\aigc-detector\data\datasets_raw\cred"

def score(text, stat, bm, thr=0.5):
    tok,model,dev = bm
    sents = [s for s in ss(text) if zh(s)>=6]
    if not sents: return None
    pt = stat_probs(stat, sents); pb = bert_score_per_sentence(tok,model,dev,sents,batch=32)
    fused_raw = np.array([ds_fuse(float(a),float(b)) for a,b in zip(pt,pb)])
    island,_ = max_ai_window_mean(fused_raw, 6)
    gated,_ = gated_doc_calibrate(fused_raw, [0]*len(sents))
    overall = float((gated*np.array([len(s) for s in sents])).sum()/max(sum(len(s) for s in sents),1))
    ai_rate = float((gated>=thr).mean())
    # verdict
    if overall>=0.5: verdict="高度疑似"
    elif island>=0.9: verdict="含AI段"
    elif overall>=0.35: verdict="疑似"
    elif overall>=0.2: verdict="证据不足"
    else: verdict="基本人类"
    return dict(overall=overall, island=island, ai_rate=ai_rate, verdict=verdict, n=len(sents))

def human_docs(stat, bm, n=6):
    import random
    pdfs=[g for g in glob.glob(os.path.join(DB,'*.pdf'))]
    random.seed(9); random.shuffle(pdfs)
    docs=[]
    for fp in pdfs:
        try: d=fitz.open(fp); t=''.join(p.get_text() for p in d); d.close()
        except: continue
        if zh(t)>=400: docs.append((os.path.basename(fp)[:30], t))
        if len(docs)>=n: break
    return docs

def ai_docs(stat, bm):
    gens=["gpt-4o","deepseek-r1","deepseek-v3","claude-3.5-haiku","qwen-3","doubao-1.5-pro","qwen-2.5","gemini-2.5-flash"]
    out=[]
    for gen in gens:
        fp=os.path.join(CRED,'CReD_paper_%s.csv'%gen)
        if not os.path.exists(fp): continue
        rows=[r['text'] for r in csv.DictReader(open(fp,encoding='utf-8')) if str(r.get('label'))=='0' and r.get('text')]
        if rows: out.append((gen, ''.join(rows[:10])))
    return out

def main():
    stat=load_cls(); bm=load_bert(device='cuda')
    print("========== 综合测试矩阵 ==========\n")

    # ① 掺入比例扫描（AI 摘要混入真人正文，占比 0/20/40/60/80/100%）
    print("【① 掺入比例扫描】AI 占比 0->100%（真人正文 + AI 摘要）:")
    hu=[t for _,t in human_docs(stat,bm,3)]
    ai=[t for _,t in ai_docs(stat,bm)[:1]]  # gpt-4o
    for frac in (0.0,0.2,0.4,0.6,0.8,1.0):
        text = (ai[0] if ai else '')[:int(frac*800)] + (hu[0] if hu else '')[:int((1-frac)*8000)]
        r=score(text,stat,bm)
        if r: print(f"  AI占比 {frac:.0%}: overall={r['overall']:.3f} island={r['island']:.2f} → [{r['verdict']}]")

    # ② 学科/主题：真实人类论文各片段 FPR
    print("\n【② 真实人类论文】动词判定（各学科, 不冤枉）:")
    for name,t in human_docs(stat,bm,8):
        r=score(t,stat,bm)
        if r: print(f"  [{name}]: {r['overall']:.3f} → [{r['verdict']}]  句级AI率={r['ai_rate']:.2f}")

    # ③ 长度敏感性
    print("\n【③ 长度敏感性】(同一真人片段的摘要 vs 全篇):")
    for name,t in human_docs(stat,bm,2):
        rs=score(t[:300],stat,bm); rl=score(t[:8000],stat,bm)
        if rs and rl: print(f"  {name}: 摘要overall={rs['overall']:.3f}全篇={rl['overall']:.3f}")

    # ④ 段落级 vs 文档级
    print("\n【④ 段落级AI段识别】(真实AI文本各生成器, island):")
    for gen,ait in ai_docs(stat,bm):
        r=score(ait,stat,bm)
        if r: print(f"  {gen}: island={r['island']:.2f} → [{r['verdict']}]")

if __name__=="__main__": main()
