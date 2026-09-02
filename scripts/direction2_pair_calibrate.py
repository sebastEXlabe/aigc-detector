# -*- coding: utf-8 -*-
"""方向1+2 配对校准：原始稿 vs 降AIGC后稿。
目标：验证检测器能区分"原始高AIGC"(判高)与"降AIGC后通过"(判低)；并看高AI句是否对应实际改写的句。
用法：python scripts/direction2_pair_calibrate.py
"""
import os, sys, re
sys.path.insert(0, r"C:\Users\woshi\.dsh\aigc-detector")
import numpy as np
from docx import Document
from detector.dual_stream import load_bert, bert_score_per_sentence, fuse as ds_fuse
from detector.consistency import gated_doc_calibrate, max_ai_window_mean
from scripts.cross_validate import stat_probs, load_cls

def zhh(t): return len(re.findall(r"[\u4e00-\u9fff]", t))
def sss(t): return [s for s in re.split(r"(?<=[。！？；])", t) if s.strip()]

def dx(txt, stat, bm):
    tok,model,dev=bm
    sents=[s for s in sss(txt) if zhh(s)>=6][:800]
    if not sents: return None
    pt=stat_probs(stat,sents); pb=bert_score_per_sentence(tok,model,dev,sents,batch=64)
    fr=np.array([ds_fuse(float(a),float(b)) for a,b in zip(pt,pb)])
    island,_=max_ai_window_mean(fr,6); gated,_=gated_doc_calibrate(fr,[0]*len(sents))
    overall=float((gated*np.array([len(s) for s in sents])).sum()/max(sum(len(s) for s in sents),1))
    # 高AI句(校准后>=0.5) 句数
    ai_cnt=int((gated>=0.5).sum())
    return overall, island, ai_cnt, len(sents)

def docx_txt(fp):
    try: d=Document(fp); return "\n".join(p.text for p in d.paragraphs)
    except Exception: return ""

# 配对(原始, 改后, 名称)
PAIRS = [
    (r"C:\Users\woshi\Downloads\黄龙-3.docx", r"C:\Users\woshi\Downloads\黄龙-3_降AI率.docx", "黄龙-3"),
    (r"C:\Users\woshi\Downloads\B26052610\中小企业员工激励机制存在的问题及优化对策_已修复.docx",
     r"C:\Users\woshi\Downloads\B26052610\中小企业员工激励机制存在的问题及优化对策_已修复_降AI.docx", "中小企业激励"),
    (r"C:\Users\woshi\Downloads\J-1130\论文初稿_政务服务一网通办V2_fixed.docx",
     r"C:\Users\woshi\Downloads\J-1130\论文初稿_政务服务一网通办V2_fixed_降AI率_降重复率.docx", "政务服务一网通办"),
    (r"C:\Users\woshi\Downloads\D248-CPP\毕业论文_基于混沌的数字指纹系统_V51.docx",
     r"C:\Users\woshi\Downloads\D248-CPP\毕业论文_基于混沌的数字指纹系统_V51_降AI率.docx", "数字指纹系统"),
    (r"C:\Users\woshi\Downloads\KBL26061610\高灿V1.docx", r"C:\Users\woshi\Downloads\KBL26061610\高灿V1_降AIGC.docx", "高灿V1"),
]

def main():
    stat=load_cls(); bm=load_bert(device="cuda")
    print("=== 方向1+2 配对校准：原始 vs 降AIGC后 ===")
    for o, r, name in PAIRS:
        if not (os.path.exists(o) and os.path.exists(r)):
            print(f"[缺] {name}"); continue
        co=dx(docx_txt(o),stat,bm); cr=dx(docx_txt(r),stat,bm)
        if not co or not cr: print(f"[无句] {name}"); continue
        print(f"{name}: 原始 overall={co[0]:.3f}/AI句{co[2]} → 改后 overall={cr[0]:.3f}/AI句{cr[2]}  (Δ={co[0]-cr[0]:+.3f})")

if __name__=="__main__": main()
