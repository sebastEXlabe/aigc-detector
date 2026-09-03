# -*- coding: utf-8 -*-
"""规则式本地降重（零token）：识别AI模板短语→替换为自然变体+结构变化→检测器复验降分。
不做LLM改写（当前无可靠LLM API）。输出改写后的句/稿。
用法：python aigc_reduce_local.py <稿件.docx>
"""
import os, sys, re, copy
sys.path.insert(0, r"C:\Users\woshi\.dsh\aigc-detector")
import numpy as np
from docx import Document
from detector.dual_stream import load_bert, bert_score_per_sentence, fuse as ds_fuse
from scripts.cross_validate import stat_probs, load_cls

def zhh(t): return len(re.findall(r"[\u4e00-\u9fff]", t))
def ss(t): return [s for s in re.split(r"(?<=[。！？；])", t) if s.strip()]

# 规则：AI模板短语 → 自然变体（降低AI签名）
RULES = [
    (r"综上所述[，,]", "综合来看"),
    (r"总而言之[，,]", "总的来说"),
    (r"值得注意的是[，,]", "需要指出的是"),
    (r"不难发现[，,]", "可以看出"),
    (r"从理论层面看[，,]", "理论上讲"),
    (r"从实践角度看[，,]", "在实践上"),
    (r"具有重要的现实意义", "有较强的实际应用价值"),
    (r"归根结底[，,]", "归根到底"),
    (r"毋庸置疑[，,]", "这一点是明确的"),
    (r"不可否认[，,]", "确实"),
    (r"由此可见[，,]", "由此可以看出"),
    (r"换言之[，,]", "换句话说"),
    (r"进一步而言[，,]", "进一步看"),
    (r"赋能", "助推"),
    (r"助力", "有助于"),
    (r"起到了[^。，]{2,20}的作用", lambda m: "对%s有帮助" % m.group(1)),
    (r"为([^。，]{2,30})提供([^。，]{2,20})", lambda m: "在%s方面，%s更可靠" % (m.group(1), m.group(2))),
]

def apply_rules(s):
    for pat, rep in RULES:
        if callable(rep):
            try: s = re.sub(pat, rep, s)
            except Exception: pass
        else:
            s = re.sub(pat, rep, s)
    return s

def score_one(text, stat, bm):
    tok,model,dev=bm
    sents=[s for s in ss(text) if zhh(s)>=6][:200]
    if not sents: return None
    pt=stat_probs(stat,sents); pb=bert_score_per_sentence(tok,model,dev,sents,batch=64)
    fr=np.array([ds_fuse(float(a),float(b)) for a,b in zip(pt,pb)])
    gated,_=gated_doc_calibrate(fr,[0]*len(sents)) if False else (fr,[0])
    return float(fr.mean())

def docx_txt(fp):
    try: d=Document(fp); return "\n".join(p.text for p in d.paragraphs)
    except Exception: return ""

def main():
    import sys as _s
    path=_s.argv[1] if len(_s.argv)>1 else None
    if not path: print("用法: python aigc_reduce_local.py <稿件.docx>"); return
    stat=load_cls(); bm=load_bert(device="cuda"); tok,model,dev=bm
    text=docx_txt(path); sents=[s for s in ss(text) if zhh(s)>=6]
    print(f"句数 {len(sents)}，处理高AI句(阈值0.5)...")
    pt=stat_probs(stat,sents); pb=bert_score_per_sentence(tok,model,dev,sents,batch=64)
    fused=np.array([ds_fuse(float(a),float(b)) for a,b in zip(pt,pb)])
    # 只处理高AI句
    changed=0; total=0
    for i,s in enumerate(sents):
        if fused[i]<0.5: continue
        total+=1
        cand=apply_rules(s)
        if cand==s: continue
        cf=score_one(cand,stat,bm)
        if cf is not None and cf < fused[i]-0.02:
            changed+=1
            if total<=4: print(f"  高AI句[{fused[i]:.2f}→{cf:.2f}]: {s[:30]}... → {cand[:30]}...")
    print(f"\n应处理高AI句 {total}，规则改写后降分 {changed}（其余未降，保留原句避免失真）")

if __name__=="__main__": main()
