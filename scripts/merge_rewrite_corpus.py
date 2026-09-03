# -*- coding: utf-8 -*-
"""合并真实人化 + 自助合成 改写语料 → 统一 rewrite_corpus.jsonl。
用法：python scripts/merge_rewrite_corpus.py
"""
import os, sys, json
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
pairs=read(os.path.join(D,"rewrite_pairs.jsonl"),"real")+read(os.path.join(D,"rewrite_pairs_synth.jsonl"),"synth")
# 去重(按src_ai) + 按强度排序(真实优先)
seen=set(); uniq=[]
for p in sorted(pairs, key=lambda x: 0 if x.get("src")=="real" else 1):
    if p["src_ai"] in seen: continue
    seen.add(p["src_ai"]); uniq.append(p)
outp=os.path.join(D,"rewrite_corpus.jsonl")
with open(outp,"w",encoding="utf-8") as f:
    for p in uniq: f.write(json.dumps(p,ensure_ascii=False)+"\n")
real=[p for p in uniq if p.get("src")=="real"]; syn=[p for p in uniq if p.get("src")=="synth"]
import numpy as np
print("=== 合并改写语料 ===")
print(f"总 {len(uniq)} 对 | 真实人化 {len(real)} (平均降分 {np.mean([p['src_ai_prob']-p['tgt_ai_prob'] for p in real]):.3f}) | 自助合成 {len(syn)} (平均降分 {np.mean([p['src_ai_prob']-p['tgt_ai_prob'] for p in syn]):.3f})")
print(f"保存 {outp}")
