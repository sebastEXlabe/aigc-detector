# -*- coding: utf-8 -*-
"""微信缓存全量挖掘：提取→按标题聚类→簇内找(高分原始,低分人化)文档对→句子对齐提(AI句->人化句)配对。
重活儿：建议无 -n 全量跑。用法：python mine_wechat_pairs.py [--limit 5000]
"""
import os, sys, re, glob, json, collections, random, hashlib
sys.path.insert(0, r"C:\Users\woshi\.dsh\aigc-detector")
import numpy as np
from docx import Document
from detector.dual_stream import load_bert, bert_score_per_sentence, fuse as ds_fuse
from detector.consistency import gated_doc_calibrate
from scripts.cross_validate import stat_probs, load_cls

WECHAT = r'D:\xwechat_files'
CACHE = r"C:\Users\woshi\.dsh\aigc-detector\data\_docx_txt_cache.jsonl"
def zhh(t): return len(re.findall(r"[\u4e00-\u9fff]", t))
def sss(t): return [s for s in re.split(r"(?<=[。！？；])", t) if s.strip()]

def _load_cache():
    c = {}
    if os.path.exists(CACHE):
        for l in open(CACHE, encoding="utf-8"):
            try:
                d = json.loads(l); c[d["h"]] = d["t"]
            except Exception: pass
    return c
def _save_cache(c):
    with open(CACHE, "w", encoding="utf-8") as f:
        for h, t in c.items(): f.write(json.dumps({"h": h, "t": t}, ensure_ascii=False) + "\n")

def docx_txt(fp, cache=None):
    h = hashlib.sha1((fp + str(os.path.getsize(fp))).encode("utf-8")).hexdigest()
    if cache is not None and h in cache: return cache[h]
    try:
        d = Document(fp); t = "\n".join(p.text.strip() for p in d.paragraphs if p.text.strip())
    except Exception:
        t = ""
    if cache is not None: cache[h] = t
    return t
def title(txt):
    for line in txt.split("\n"):
        line=line.strip()
        if len(line)>=8 and re.search(r'[\u4e00-\u9fff]{6,}',line): return line[:40]
    return None
def sim(a,b):
    sa=set(re.findall(r"[\u4e00-\u9fff]{2}",a)); sb=set(re.findall(r"[\u4e00-\u9fff]{2}",b))
    if not sa or not sb: return 0.0
    return len(sa&sb)/len(sa|sb)

def main():
    import argparse
    ap=argparse.ArgumentParser(); ap.add_argument("--limit",type=int,default=0)
    a=ap.parse_args()
    stat=load_cls(); bm=load_bert(device="cuda"); tok,model,dev=bm
    files=glob.glob(os.path.join(WECHAT,'**','*.docx'),recursive=True)
    print("缓存 docx:", len(files))
    cache=_load_cache()
    print("文本缓存命中:", len(cache), "(累计)") 
    # 提取 + 聚类（按标题）
    works=collections.defaultdict(list)
    for f in files:
        try:
            if os.path.getsize(f)<25000: continue
            t=docx_txt(f, cache)
            if zhh(t)<250: continue
            ti=title(t)
            if ti and not re.search(r'(毕业论文|毕业设计|学位论文|继续教育|自学考试|本科|大学|学院|学校)', ti):
                works[ti].append(f)
        except Exception: continue
    _save_cache(cache)
    # 只保留多版本簇
    clusters={t:f for t,f in works.items() if len(f)>=2}
    print("有效作品簇(标题非模板/学校名)含多版本:", len(clusters), " | 文件数:", sum(len(v) for v in clusters.values()))
    # 对每个簇打分, 找(高AI,低AI)文档对
    HEADER=re.compile(r'(UNIVERSITY|摘要|目录|参考文献|致谢|^\s*[A-Za-z ]{10,}\s*$)',re.I)
    outp=r"C:\Users\woshi\.dsh\aigc-detector\data\rewrite_pairs_wechat.jsonl"
    os.makedirs(os.path.dirname(outp), exist_ok=True)
    pairs=[]; seen=set()
    # 载入已有(续跑去重)
    if os.path.exists(outp):
        for l in open(outp,encoding="utf-8"):
            try:
                p=json.loads(l); pairs.append(p); seen.add(p["src_ai"])
            except Exception: pass
    out_fh=open(outp,"w",encoding="utf-8")  # 重写
    loaded_done=[p for p in pairs]
    def flushpairs(fp):
        fp.seek(0); fp.truncate()
        for p in pairs: fp.write(json.dumps(p,ensure_ascii=False)+"\n")
        fp.flush()
    for ci,(t,fs) in enumerate(list(clusters.items())):
        scored=[]
        for f in fs:
            txt=docx_txt(f, cache)
            sents=[s for s in sss(txt) if zhh(s)>=10]
            if not sents: continue
            try:
                pt=stat_probs(stat,sents); pb=bert_score_per_sentence(tok,model,dev,sents[:400],batch=64)
                fr=np.array([ds_fuse(float(a),float(b)) for a,b in zip(pt,pb)])
                g,_=gated_doc_calibrate(fr,[0]*len(sents))
                ov=float((g*np.array([len(s) for s in sents])).sum()/max(sum(len(s) for s in sents),1))
                scored.append((ov, f, txt))
            except Exception: continue
        if len(scored)<2: continue
        scored.sort(key=lambda x:-x[0])
        hi=scored[0]; lo=scored[-1]
        if hi[0]-lo[0]<0.08: continue  # 差异太小,不构成"降AIGC"
        got=0
        # 句子对齐：先对 hi/lo 两个文档各做一次批式打分(逐句fused)，再筛选降分对
        try:
            hs=[s for s in sss(hi[2]) if zhh(s)>=10][:250]
            ls=[s for s in sss(lo[2]) if zhh(s)>=10][:250]
            def score_sents(sents):
                if not sents: return []
                pt=stat_probs(stat,sents)
                pb=bert_score_per_sentence(tok,model,dev,sents,batch=32)
                return [ds_fuse(float(a),float(b)) for a,b in zip(pt,pb)]
            hsc=score_sents(hs); lsc=score_sents(ls)
        except Exception:
            hs=ls=hsc=lsc=None
        if hsc is not None:
            for oi,o in enumerate(hs):
                if hsc[oi]<0.5 or o in seen: continue  # 原始句必须较高AI且未采过
                # 找与其相似的低分句(限前若干候选)
                best=None; bl=None
                for li,l in enumerate(ls):
                    if lsc[li]>=hsc[oi]-0.03: continue  # 人化句必须明显更低
                    s=sim(o,l)
                    if s>=0.3 and abs(len(l)-len(o))/max(len(o),1)<0.6:
                        best=l; bl=lsc[li]; break
                if best is not None:
                    pairs.append({"src_ai":o,"tgt_human":best,
                                  "src_ai_prob":round(float(hsc[oi]),3),
                                  "tgt_ai_prob":round(float(bl),3),"src":"wechat_real"})
                    seen.add(o); got+=1
        # 每簇把新配对追加写入(可断点续采)
        flushpairs(out_fh)
        if (ci+1)%25==0:
            print(f"  已处理簇 {ci+1}/{len(clusters)} | 累计配对 {len(pairs)}", flush=True)
    out_fh.close()
    drops=[p['src_ai_prob']-p['tgt_ai_prob'] for p in pairs]
    print(f"=== 微信缓存挖掘 (AI句->人化句) 配对: {len(pairs)} ===")
    if drops: print(f"  平均降分 {np.mean(drops):.3f} ({np.mean([p['src_ai_prob'] for p in pairs]):.3f}->{np.mean([p['tgt_ai_prob'] for p in pairs]):.3f})")
    print("保存", outp)

if __name__=="__main__": main()
