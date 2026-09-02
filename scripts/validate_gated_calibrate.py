# -*- coding: utf-8 -*-
"""方向1 文档门控校准验证：真 AIGC-AI 检出不受损 + 真论文句级误标被压 + 真CNKI句级FPR。
"""
import os, sys, re, glob
sys.path.insert(0, r"C:\Users\woshi\.dsh\aigc-detector")
import numpy as np, fitz
from detector.dual_stream import load_bert, bert_score_per_sentence, fuse as ds_fuse
from detector.consistency import gated_doc_calibrate
from scripts.cross_validate import build_aigc_test, build_cnki_test, label_to_ai, read_recs, stat_probs, load_cls

def zh(t): return len(re.findall(r"[\u4e00-\u9fff]", t))
def ss(t): return [s for s in re.split(r"(?<=[。！？；])", t) if s.strip()]

def main():
    cls = load_cls(); bm = load_bert(device="cuda"); tok,model,dev = bm
    thr = 0.5
    # 1) 真 AIGC 测试（按报告分组）—— AI 检出
    aigc_test, _, ht = build_aigc_test(0.2, 42)
    al = [1 if label_to_ai(r.get("label")) else 0 for r in aigc_test]
    texts = [r["text"] for r in aigc_test]; doc = [r.get("title") or "d" for r in aigc_test]
    pt = stat_probs(cls, texts); pb = bert_score_per_sentence(tok,model,dev,texts,batch=32)
    fused = np.array([ds_fuse(float(a),float(b)) for a,b in zip(pt,pb)])
    rec_b = float((fused[np.array(al)==1] >= thr).mean())
    gated,_ = gated_doc_calibrate(fused, doc)
    rec_a = float((gated[np.array(al)==1] >= thr).mean())
    print("=== 方向1 门控校准 ===")
    print(f"[真AIGC-AI] 检出: 前 {rec_b:.3f} → 后 {rec_a:.3f}  (AI句{sum(al)})")
    # 2) 真 CNKI 句级 FPR
    tt=set()
    for r in read_recs(os.path.join(r"C:\Users\woshi\.dsh\aigc-detector\data","train_unified.jsonl")): tt.add(r.get("text"))
    cnki = build_cnki_test(800,42,tt)
    cpt=stat_probs(cls,cnki); cpb=bert_score_per_sentence(tok,model,dev,cnki,batch=32)
    cf=np.array([ds_fuse(float(a),float(b)) for a,b in zip(cpt,cpb)])
    cg,_=gated_doc_calibrate(cf,[1]*len(cf))  # 单句即一组，docm=自身
    print(f"[真CNKI单句] FPR(>=0.5): 前 {float((cf>=thr).mean()):.3f} → 后 {float((cg>=thr).mean()):.3f}")
    # 3) 真实论文整篇（句级误标数）
    DB=r"C:\Users\woshi\cnki-hub\data\downloads\cnki"
    for name in ["1011163632.nh_后现代语境下的传媒研究——戴维·莫利传播思想探析.pdf",
                 "1012271379.nh_论环境协同治理——社会治理演进史视角中的环境问题及其应对.pdf"]:
        fp=os.path.join(DB,name)
        if not os.path.exists(fp): continue
        docf=fitz.open(fp); text="".join(p.get_text() for p in docf); docf.close()
        sents=[s for s in ss(text) if zh(s)>=6][:400]
        if not sents: continue
        pt=stat_probs(cls,sents); pb=bert_score_per_sentence(tok,model,dev,sents,batch=32)
        fused=np.array([ds_fuse(float(a),float(b)) for a,b in zip(pt,pb)])
        docm=float((fused*[len(s) for s in sents]).sum()/max(sum(len(s) for s in sents),1))
        b=float((fused>=thr).mean()); g,_=gated_doc_calibrate(fused,[0]*len(sents))
        gcnt=float((g>=thr).mean())
        print(f"[真论文 {name[:24]}...] doc均值={docm:.3f} 句级≥0.5: 前{b:.3f} → 后{gcnt:.3f}")

if __name__=="__main__": main()
