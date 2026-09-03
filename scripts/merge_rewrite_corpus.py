# -*- coding: utf-8 -*-
"""合并真实人化 + 自助合成 改写语料 → 统一 rewrite_corpus.jsonl。
用法：python scripts/merge_rewrite_corpus.py
"""
import os, sys, json
sys.path.insert(0, r"C:\Users\woshi\.dsh\aigc-detector")
from scripts.pair_quality import clean_pairs
D=r"C:\Users\woshi\.dsh\aigc-detector\data"
def read(p, src_tag):
    out=[]
    if not os.path.exists(p): return out
    for l in open(p,encoding="utf-8"):
        l=l.strip()
        if not l: continue
        d=json.loads(l)
        d.setdefault("src", src_tag)
        out.append(d)
    return out
pairs=read(os.path.join(D,"rewrite_pairs.jsonl"),"real")+read(os.path.join(D,"rewrite_pairs_synth.jsonl"),"synth")+read(os.path.join(D,"rewrite_pairs_wechat.jsonl"),"wechat_real")
# 质量过滤(剔参考文献/封面/致谢等非正文污染)
pairs, dropped = clean_pairs(pairs)
# 去重(按src_ai) + 按强度排序(真实优先)
order={"real":0,"wechat_real":1,"synth":2}
seen=set(); uniq=[]
for p in sorted(pairs, key=lambda x: order.get(x.get("src"),3)):
    if p["src_ai"] in seen: continue
    seen.add(p["src_ai"]); uniq.append(p)
outp=os.path.join(D,"rewrite_corpus.jsonl")
with open(outp,"w",encoding="utf-8") as f:
    for p in uniq: f.write(json.dumps(p,ensure_ascii=False)+"\n")
import numpy as np
def grp(src):
    return [p for p in uniq if p.get("src")==src]
def sta(name):
    g=grp(name)
    if not g: return f"{name}:0"
    ds=[p['src_ai_prob']-p['tgt_ai_prob'] for p in g]
    return f"{name} {len(g)} (平均降分 {np.mean(ds):.3f})"
print("=== 合并改写语料 ===")
print(f"总 {len(uniq)} 对 (质量过滤剔除 {dropped}) | " + " | ".join(sta(n) for n in ("real","wechat_real","synth")))
print(f"保存 {outp}")
