# -*- coding: utf-8 -*-
"""真实学生论文/毕设检测验证：抽取真实毕设 docx 全文，跑检测，暴露与"发表论文"不同的新问题。
"""
import os, sys, re, glob
sys.path.insert(0, r"C:\Users\woshi\.dsh\aigc-detector")
import numpy as np
from docx import Document
from detector.dual_stream import load_bert, bert_score_per_sentence, fuse as ds_fuse
from detector.consistency import gated_doc_calibrate, max_ai_window_mean
from scripts.cross_validate import stat_probs, load_cls

def zh(t): return len(re.findall(r"[\u4e00-\u9fff]", t))
def ss(t): return [s for s in re.split(r"(?<=[。！？；])", t) if s.strip()]

def docx_text(fp):
    try:
        d = Document(fp)
        return "\n".join(p.text for p in d.paragraphs)
    except Exception as e:
        return ""

def detect(text, stat, bm):
    tok,model,dev = bm
    sents = [s for s in ss(text) if zh(s)>=6][:700]
    if not sents: return None
    pt = stat_probs(stat, sents); pb = bert_score_per_sentence(tok,model,dev,sents,batch=64)
    fr = np.array([ds_fuse(float(a),float(b)) for a,b in zip(pt,pb)])
    island,_ = max_ai_window_mean(fr, 6)
    gated,_ = gated_doc_calibrate(fr, [0]*len(sents))
    overall = float((gated*np.array([len(s) for s in sents])).sum()/max(sum(len(s) for s in sents),1))
    if overall>=0.5: verdict="高度疑似AI生成"
    elif island>=0.9: verdict="疑似AI（存在AI密集段）"
    elif overall>=0.2: verdict="证据不足（少量AI痕迹）"
    else: verdict="基本人类撰写"
    return dict(overall=overall, island=island, verdict=verdict, n=len(sents))

DOCS = [
    (r"C:\Users\woshi\Downloads\D-256\终稿降AI+降重后_V9.docx", "D-256 毕设(降AI后)"),
    (r"C:\Users\woshi\Downloads\SM26072914\最终版fyn(正文引用)_修正版.docx", "SM26072914 论文"),
    (r"C:\Users\woshi\Downloads\J-1148\卓创资讯数据资产入表的动因及效果研究.docx", "J-1148 数据资产论文"),
    (r"C:\Users\woshi\Downloads\Q26062816\装饰企业商务谈判的策略研究——以唐红装饰工程公司为例.docx", "Q26062816 商务谈判论文"),
    (r"C:\Users\woshi\Downloads\D-253\周建宗智能制造学院毕业设计论文改(2)(1)(2)(1).docx", "D-253 毕设"),
]

def main():
    stat = load_cls(); bm = load_bert(device="cuda")
    print("=== 真实学生论文/毕设检测（不同样本）===\n")
    for fp, name in DOCS:
        if not os.path.exists(fp):
            print(f"[缺] {name}"); continue
        txt = docx_text(fp)
        r = detect(txt, stat, bm)
        if r:
            print(f"[{name}] 句数={r['n']} 整体={r['overall']:.3f} AI岛={r['island']:.2f} → [{r['verdict']}]")
        else:
            print(f"[{name}] 无有效中文句")

if __name__ == "__main__": main()
