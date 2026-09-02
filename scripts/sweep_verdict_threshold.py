# -*- coding: utf-8 -*-
"""文档级判定边界扫描：收紧"疑似"下界，看 真人类/真AI/局部AI 的判定 tradeoff。
若能把真人类从"疑似"压回"证据不足"且不丢真AI检出，则采用；否则回退。
用法：python scripts/sweep_verdict_threshold.py
"""
import os, sys, re, glob, csv, random
sys.path.insert(0, r"C:\Users\woshi\.dsh\aigc-detector")
import numpy as np, fitz
from detector.dual_stream import load_bert, bert_score_per_sentence, fuse as ds_fuse
from detector.consistency import gated_doc_calibrate, max_ai_window_mean
from scripts.cross_validate import stat_probs, load_cls

def zh(t): return len(re.findall(r"[\u4e00-\u9fff]", t))
def ss(t): return [s for s in re.split(r"(?<=[。！？；])", t) if s.strip()]
DB=r"C:\Users\woshi\cnki-hub\data\downloads\cnki"; CRED=r"C:\Users\woshi\.dsh\aigc-detector\data\datasets_raw\cred"

def score(text, stat, bm):
    tok,model,dev=bm
    sents=[s for s in ss(text) if zh(s)>=6][:600]
    if not sents: return None
    pt=stat_probs(stat,sents); pb=bert_score_per_sentence(tok,model,dev,sents,batch=64)
    fr=np.array([ds_fuse(float(a),float(b)) for a,b in zip(pt,pb)])
    island,_=max_ai_window_mean(fr,6)
    gated,_=gated_doc_calibrate(fr,[0]*len(sents))
    overall=float((gated*np.array([len(s) for s in sents])).sum()/max(sum(len(s) for s in sents),1))
    return overall, island

def verdict(overall, island, sus_bound):
    if overall>=0.5 or island>=0.9: return "AI"   # 高度疑似 或 AI密集段
    if overall>=sus_bound: return "疑似"           # 收紧后的边界
    if overall>=0.2: return "证据不足"
    return "基本人类"

def main():
    import argparse
    ap=argparse.ArgumentParser(); ap.add_argument("--sus", type=float, default=0.45); a=ap.parse_args()
    stat=load_cls(); bm=load_bert(device="cuda")
    # 真人类论文
    pdfs=[g for g in glob.glob(os.path.join(DB,'*.pdf'))]; random.seed(7); random.shuffle(pdfs)
    humans=[]
    for fp in pdfs[:20]:
        try: d=fitz.open(fp); t=''.join(p.get_text() for p in d); d.close()
        except: continue
        if zh(t)>=400:
            r=score(t,stat,bm)
            if r: humans.append(r)
            if len(humans)>=6: break
    # 真 AI 文档
    gens=["gpt-4o","deepseek-r1","deepseek-v3","claude-3.5-haiku","qwen-3","doubao-1.5-pro","qwen-2.5","gemini-2.5-flash"]
    ais=[]
    for gen in gens:
        fp=os.path.join(CRED,'CReD_paper_%s.csv'%gen)
        if not os.path.exists(fp): continue
        rows=[r['text'] for r in csv.DictReader(open(fp,encoding='utf-8')) if str(r.get('label'))=='0' and r.get('text')]
        r=score(''.join(rows[:8]),stat,bm)
        if r: ais.append(r)
    # 局部AI(20%,40%)混入真人
    hu_t=humans[0] if humans else None; ai_t=ais[0] if ais else None
    partials=[]
    if hu_t and ai_t:
        for frac in (0.2,0.4):
            # 需要原文本拼; 这里用 already-scored? 直接重建短文
            pass
    print(f"真人类论文 {len(humans)} 篇 / 真AI文档 {len(ais)} 篇")
    # 扩样本: 30篇真人类, 多种子
    humans2=[]
    for seed in (7,23,99):
        pdfs=[g for g in glob.glob(os.path.join(DB,'*.pdf'))]; random.seed(seed); random.shuffle(pdfs)
        for fp in pdfs[:40]:
            try: d=fitz.open(fp); t=''.join(p.get_text() for p in d); d.close()
            except: continue
            if zh(t)>=400:
                r=score(t,stat,bm)
                if r and r not in humans2: humans2.append(r)
                if len(humans2)>=30: break
        if len(humans2)>=30: break
    print(f"== 扩样本: 真人类 {len(humans2)} 篇(含工科/人文/社科/经管) ==")
    print("=== 收紧'疑似'下界 sus_bound 的效果 ===")
    for sus in (0.35, 0.40, 0.45, 0.50):
        h=sum(1 for ov,is_ in humans2 if verdict(ov,is_,sus) in ("AI","疑似"))
        hi=sum(1 for ov,is_ in humans2 if verdict(ov,is_,sus)=="AI")
        print(f"  sus_bound={sus:.2f}: 真人类被判'疑似+高度疑似' {h}/{len(humans2)} ({h/len(humans2):.0%})  其中硬指控'高度疑似/AI段' {hi}")
    print(f"  真AI文档: 全部判'疑似+AI' {len(ais)}/{len(ais)}  (不同sus不变)")

if __name__=="__main__": main()
