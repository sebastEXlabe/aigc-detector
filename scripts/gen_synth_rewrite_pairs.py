# -*- coding: utf-8 -*-
"""自助生成改写语料（挖到底的规模化来源）：海量AI句 → 规则+同义改写 → 检测器复验(降分才保留) → (AI句->人化改写句)。
检测器已对齐知网/维普(AUC 0.925)，通过它复验的改写句即"能降知网/维普"的子。
用法：python gen_synth_rewrite_pairs.py [--n 4000]
"""
import os, sys, re, json, random
sys.path.insert(0, r"C:\Users\woshi\.dsh\aigc-detector")
import numpy as np
from detector.dual_stream import load_bert, bert_score_per_sentence, fuse as ds_fuse
from detector.consistency import gated_doc_calibrate
from scripts.cross_validate import stat_probs, load_cls

def zhh(t): return len(re.findall(r"[\u4e00-\u9fff]", t))

# 同义词（降低AI词频签名）
SYN = [
    ("显著","明显"),("旨在","目的在于"),("本研究","本文"),("研究表明","研究显示"),
    ("结果表明","结果显示"),("具有","拥有"),("提供","带来"),("影响","作用"),("促进","推动"),
    ("提升","提高"),("增强","加强"),("体现","反映"),("表明","显示"),("综上","总体来看"),
    ("赋能","助推"),("助力","有助于"),("推动","带动"),("实现","达成"),("构建","建立"),
    ("探讨","考察"),("深入","进一步"),("重要","关键"),("广泛","普遍"),("有效","切实"),
    ("相关","有关"),("应用","运用"),("结合","联系"),("分析","考察"),("研究","分析"),
]
# 模板规则
RULES = [
    (r"综上所述[，,]", "总的来看"), (r"值得注意的是[，,]", "需要指出的是"),
    (r"不难发现[，,]", "可以看出"), (r"从理论层面看[，,]", "理论上讲"),
    (r"具有重要的现实意义", "有较强的实际应用价值"), (r"归根结底[，,]", "归根到底"),
    (r"为([^。，]{2,30})提供([^。，]{2,20})", lambda m: "在%s方面，%s更可靠" % (m.group(1),m.group(2))),
]

def rewrite(s):
    for a,b in SYN:
        if a in s and random.random()<0.8: s=s.replace(a,b)
    for pat,rep in RULES:
        try: s=re.sub(pat,rep,s)
        except: pass
    return s

def score(sents, stat, bm):
    if not sents: return []
    tok,model,dev=bm
    pt=stat_probs(stat,sents); pb=bert_score_per_sentence(tok,model,dev,sents,batch=64)
    fr=np.array([ds_fuse(float(a),float(b)) for a,b in zip(pt,pb)])
    gated,_=gated_doc_calibrate(fr,[0]*len(sents))
    return gated.tolist()

def main():
    import argparse
    ap=argparse.ArgumentParser(); ap.add_argument("--n", type=int, default=5000); a=ap.parse_args()
    stat=load_cls(); bm=load_bert(device="cuda")
    # 采样中文 AI 句（C-ReD + HC3, 混合）
    DATA=r"C:\Users\woshi\.dsh\aigc-detector\data"
    pool=[]
    for l in open(os.path.join(DATA,"ai_pub_samples.jsonl"),encoding="utf-8"):
        if not l.strip(): continue
        d=json.loads(l); src=d.get("source"); txt=(d.get("text") or "")
        if src in ("C-ReD-paper","HC3") and d.get("lang","zh")=="zh" and zhh(txt)>=25:
            pool.append(txt)
        if len(pool)>=a.n: break
    random.seed(42); random.shuffle(pool); pool=pool[:a.n]
    print("AI句池采样:", len(pool))
    # 生成改写候选
    cands=[rewrite(x) for x in pool]
    cands=[c for c in cands if c and zhh(c)>=15 and c not in pool]
    # 打分（原始 + 改写）
    o_sc=score(pool,stat,bm); c_sc=score(cands,stat,bm)
    pairs=[]
    for x,old,new in zip(pool,o_sc,c_sc):
        # 保留: 原始高AI 且 改写降分>0.03
        if old>=0.5 and new < old-0.03:
            pairs.append({"src_ai":x,"tgt_human":cands[list(pool).index(x)],"src_ai_prob":round(float(old),3),"tgt_ai_prob":round(float(new),3),"src":"synth"})
    # 去重保存
    seen=set(); out=[]
    for p in pairs:
        if p["src_ai"] in seen: continue
        seen.add(p["src_ai"]); out.append(p)
    outp=r"C:\Users\woshi\.dsh\aigc-detector\data\rewrite_pairs_synth.jsonl"
    with open(outp,"w",encoding="utf-8") as f:
        for p in out: f.write(json.dumps(p,ensure_ascii=False)+"\n")
    drops=[p['src_ai_prob']-p['tgt_ai_prob'] for p in out]
    print(f"=== 自助生成改写语料 ===")
    print(f"采样 {len(pool)} AI句 -> 生成已验证配对 {len(out)} 对 (原始均值 {np.mean([p['src_ai_prob'] for p in out]):.3f} -> 改写均值 {np.mean([p['tgt_ai_prob'] for p in out]):.3f})")
    print(f"保存 {outp}")
    for p in out[:2]: print(f"  AI: {p['src_ai'][:38]}...  人化: {p['tgt_human'][:38]}...")

if __name__=="__main__": main()
