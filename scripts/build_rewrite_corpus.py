# -*- coding: utf-8 -*-
"""本地语料清洗整理：把(原始稿, 降重后稿)配对 → 句子级对齐 → 提取(AI句→人化改写句)平行配对。
用检测器标分：只保留 原始句高AI(>=0.5) 且 对应改写句分数更低 的配对。输出清洗后的并行语料。
用法：python scripts/build_rewrite_corpus.py
"""
import os, sys, re, json, collections
sys.path.insert(0, r"C:\Users\woshi\.dsh\aigc-detector")
import numpy as np
from docx import Document
from detector.dual_stream import load_bert, bert_score_per_sentence, fuse as ds_fuse
from detector.consistency import gated_doc_calibrate
from scripts.cross_validate import stat_probs, load_cls

def zhh(t): return len(re.findall(r"[\u4e00-\u9fff]", t))
def sss(t): return [s for s in re.split(r"(?<=[。！？；])", t) if s.strip()]

PAIRS = [
    (r"C:\Users\woshi\Downloads\黄龙-3.docx", r"C:\Users\woshi\Downloads\黄龙-3_降AI率.docx"),
    (r"C:\Users\woshi\Downloads\B26052610\中小企业员工激励机制存在的问题及优化对策_已修复.docx",
     r"C:\Users\woshi\Downloads\B26052610\中小企业员工激励机制存在的问题及优化对策_已修复_降AI.docx"),
    (r"C:\Users\woshi\Downloads\J-1130\论文初稿_政务服务一网通办V2_fixed.docx",
     r"C:\Users\woshi\Downloads\J-1130\论文初稿_政务服务一网通办V2_fixed_降AI率_降重复率.docx"),
    (r"C:\Users\woshi\Downloads\D248-CPP\毕业论文_基于混沌的数字指纹系统_V51.docx",
     r"C:\Users\woshi\Downloads\D248-CPP\毕业论文_基于混沌的数字指纹系统_V51_降AI率.docx"),
    (r"C:\Users\woshi\Downloads\KBL26061610\高灿V1.docx", r"C:\Users\woshi\Downloads\KBL26061610\高灿V1_降AIGC.docx"),
    (r"C:\Users\woshi\Downloads\凝血联合肝功检测评估肝硬化凝血障碍.docx",
     r"C:\Users\woshi\Downloads\凝血联合肝功检测评估肝硬化凝血障碍_降重复率.docx"),
    (r"C:\Users\woshi\Downloads\B26051924\招商银行盈利能力分析8.141.docx",
     r"C:\Users\woshi\Downloads\B26051924\招商银行盈利能力分析8.141_已处理.docx"),
    (r"C:\Users\woshi\Downloads\B26051924\招商银行盈利能力分析8.9.docx",
     r"C:\Users\woshi\Downloads\B26051924\招商银行盈利能力分析8.9_已处理.docx"),
    (r"C:\Users\woshi\Downloads\D-193+1\初稿.docx", r"C:\Users\woshi\Downloads\D-193+1\初稿_改写版.docx"),
]

def docx_txt(fp):
    try: d=Document(fp); return "\n".join(p.text for p in d.paragraphs)
    except Exception: return ""

HEADER_BLACK = re.compile(r"(UNIVERSITY|NANCHANG|届高等学历|摘要|目录|参考文献|致谢|关键词|^[0-9]{3,}|^\s*[A-Za-z ]{10,}\s*$)", re.I)

def similarity(a, b):
    # 字符2-gram Jaccard 相似度
    sa=set(re.findall(r"[\u4e00-\u9fff]{2}", a)); sb=set(re.findall(r"[\u4e00-\u9fff]{2}", b))
    if not sa or not sb: return 0.0
    return len(sa&sb)/len(sa|sb)

def score_sents(sents, stat, bm):
    tok,model,dev=bm
    if not sents: return []
    pt=stat_probs(stat,sents); pb=bert_score_per_sentence(tok,model,dev,sents,batch=64)
    fused=np.array([ds_fuse(float(a),float(b)) for a,b in zip(pt,pb)])
    gated,_=gated_doc_calibrate(fused,[0]*len(sents))
    return gated.tolist()

def main():
    stat=load_cls(); bm=load_bert(device="cuda")
    out=[]; dup=set()
    for o,r in PAIRS:
        if not (os.path.exists(o) and os.path.exists(r)): continue
        ot=docx_txt(o); rt=docx_txt(r)
        os_=[s for s in sss(ot) if zhh(s)>=10]; rs_=[s for s in sss(rt) if zhh(s)>=10]
        if not os_ or not rs_: continue
        o_sc=score_sents(os_,stat,bm); r_sc=score_sents(rs_,stat,bm)
        # 对齐: 每个原始句找最相似的改写句
        for i,oi in enumerate(os_):
            oi_s=oi; s_sim=0; best=-1
            for j,rj in enumerate(rs_):
                s=similarity(oi_s,rj)
                if s>s_sim: s_sim=s; best=j
            if best<0 or s_sim<0.18: continue
            rj=rs_[best]
            o_ai=o_sc[i]; r_ai=r_sc[best]
            # 清洗: 长度比合理 + 无文档头垃圾 + 高AI且改写降分
            lr = len(rj)/max(len(oi),1)
            if lr<0.4 or lr>2.5: continue
            if HEADER_BLACK.search(oi) or HEADER_BLACK.search(rj): continue
            if o_ai>=0.5 and r_ai < o_ai-0.03 and oi!=rj:
                key=(oi,rj)
                if key in dup: continue
                dup.add(key)
                out.append({"src_ai":oi, "tgt_human":rj, "src_ai_prob":round(o_ai,3), "tgt_ai_prob":round(r_ai,3)})
    # 清洗: 去重(按src) + 排序 + 保存
    seen=set(); clean=[]
    for x in out:
        if x["src_ai"] in seen: continue
        seen.add(x["src_ai"]); clean.append(x)
    outp=r"C:\Users\woshi\.dsh\aigc-detector\data\rewrite_pairs.jsonl"
    with open(outp,"w",encoding="utf-8") as f:
        for x in clean: f.write(json.dumps(x,ensure_ascii=False)+"\n")
    print("=== 本地语料清洗整理结果 ===")
    print(f"  配对 {len(PAIRS)} 组 | 提取(AI句→人化改写句)配对 {len(clean)} 对")
    # 平均降分
    drops=[x['src_ai_prob']-x['tgt_ai_prob'] for x in clean]
    if drops: print(f"  平均降分 {np.mean(drops):.3f}  (原始均值 {np.mean([x['src_ai_prob'] for x in clean]):.3f} -> 改写均值 {np.mean([x['tgt_ai_prob'] for x in clean]):.3f})")
    print(f"  保存 {outp}")
    for x in clean[:3]: print(f"    AI: {x['src_ai'][:40]}...\n    人化: {x['tgt_human'][:40]}...\n")

if __name__=="__main__": main()
