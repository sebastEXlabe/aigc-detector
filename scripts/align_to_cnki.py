# -*- coding: utf-8 -*-
"""方向B：检测器 vs 知网/维普AIGC报告 口径对齐测量。
用已知报告(卓创资讯)的CNKI逐句AI/human标注作ground truth，看检测器高AI句与CNKI判AI句的重合度。
重合高 → 检测器能可靠引导降AIGC；重合低 → 需校准。
用法：python scripts/align_to_cnki.py
"""
import os, sys, re, json
sys.path.insert(0, r"C:\Users\woshi\.dsh\aigc-detector")
import numpy as np
from docx import Document
from detector.dual_stream import load_bert, bert_score_per_sentence, fuse as ds_fuse
from scripts.cross_validate import stat_probs, load_cls

def zhh(t): return len(re.findall(r"[\u4e00-\u9fff]", t))
def ss(t): return [s for s in re.split(r"(?<=[。！？；])", t) if s.strip()]

def main():
    stat=load_cls(); bm=load_bert(device="cuda"); tok,model,dev=bm
    # CNKI 报告句级标注（title=卓创资讯）
    TITLE = "免费_Word标红版_AIGC检测报告_[卓创资讯数据资产入表的动因及效果研究（查].docx"
    recs=[]
    for l in open(r"C:\Users\woshi\.dsh\aigc-detector\data\train_unified.jsonl",encoding="utf-8"):
        if TITLE not in l: continue
        try: d=json.loads(l)
        except: continue
        tf=(d.get('text') or '').strip()
        if not tf or zhh(tf)<10: continue
        label=d.get('label')
        yi = 1 if label in ("high","medium","low") else 0
        recs.append((tf,yi))
    print(f"CNKI报告句数(有标注): {len(recs)}  AI句={sum(y for _,y in recs)}")
    # 检测器对报告句打分（逐句）
    texts=[t for t,_ in recs]; y=np.array([yy for _,yy in recs])
    pt=stat_probs(stat,texts); pb=bert_score_per_sentence(tok,model,dev,texts,batch=64)
    fused=np.array([ds_fuse(float(a),float(b)) for a,b in zip(pt,pb)])
    # 阈值0.5下，检测器判AI vs CNKI判AI 的重合
    for thr in (0.5,0.45,0.4):
        det= (fused>=thr).astype(int)
        agree=float((det==y).mean())
        # 检测器判AI中，被CNKI确认的比例(精确率) + CNKI判AI中被检测器抓到的(召回)
        det_ai=(det==1); cnki_ai=(y==1)
        prec=float(det[det_ai].mean()) if det_ai.any() else 0
        rec=float((det[cnki_ai]==1).mean()) if cnki_ai.any() else 0
        print(f"  阈{thr}: 一致率={agree:.3f}  检测器判AI中被CNKI确认={prec:.3f}  CNKI判AI被检测器抓到={rec:.3f}")
    # AUC
    from sklearn.metrics import roc_auc_score
    try: print(f"  检测器vs CNKI标注 AUC={roc_auc_score(y,fused):.3f}")
    except: pass

if __name__=="__main__": main()
