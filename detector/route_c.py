# -*- coding: utf-8 -*-
"""路线C：本地困惑度/突发性检测（最终版）。
用 jieba 词级 n-gram。AI 信号 = "可预测句占比"（相对困惑度 < 1.0 的句子占比）。
"""
import os, re, math, json, sys, pickle
from collections import Counter
import jieba

class NgramLM:
    def __init__(self, n=3):
        self.n=n; self.ngrams=Counter(); self.contexts=Counter(); self.vocab=set()
        self.total=0
    def train(self, texts):
        for t in texts:
            words=[w for w in jieba.cut(re.sub(r"\s+","",t)) if w.strip()]
            if not words: continue
            for i in range(len(words)-self.n+1):
                g=tuple(words[i:i+self.n]); self.ngrams[g]+=1; self.contexts[g[:-1]]+=1
            for w in words: self.vocab.add(w)
        self.total=sum(self.ngrams.values())
    def prob(self, g):
        ctx=g[:-1]
        c=self.ngrams.get(g,0); cn=self.contexts.get(ctx,0)
        if cn==0: return 1.0/(self.total+len(self.vocab)+1)
        return (c+1)/(cn+len(self.vocab)+1)
    def perplexity(self, text):
        words=[w for w in jieba.cut(re.sub(r"\s+","",text)) if w.strip()]
        lps=[]
        for i in range(len(words)-self.n+1):
            g=tuple(words[i:i+self.n]); lps.append(math.log(max(self.prob(g),1e-12)))
        if not lps: return None
        return math.exp(-sum(lps)/len(lps))

def build_lm(dataset_path):
    texts=[]
    for l in open(dataset_path, encoding="utf-8"):
        if not l.strip(): continue
        d=json.loads(l)
        if isinstance(d,list):
            for x in d: texts.append(x["text"])
        elif "text" in d: texts.append(d["text"])
        elif "sentences" in d:
            for s in d["sentences"]: texts.append(s["text"])
    lm=NgramLM(n=3); lm.train(texts)
    human_perps=[]
    for l in open(dataset_path, encoding="utf-8"):
        if not l.strip(): continue
        d=json.loads(l)
        if isinstance(d,dict) and d.get("text") and d.get("prob",0.5)<0.4:
            p=lm.perplexity(d["text"])
            if p: human_perps.append(p)
    lm.global_basep=sum(human_perps)/len(human_perps) if human_perps else 20000
    return lm

def burstiness(sentences):
    lens=[len(s) for s in sentences if s]
    if len(lens)<2: return 0.0
    avg=sum(lens)/len(lens)
    if avg==0: return 0.0
    return math.sqrt(sum((l-avg)**2 for l in lens)/len(lens))/avg

def score_text(lm, sentences):
    """返回 (AI倾向0-1, 句子级相对困惑度列表, burstiness, basep)。"""
    perps=[lm.perplexity(s) for s in sentences]
    basep=getattr(lm,'global_basep',20000)
    rels=[p/basep if p else None for p in perps]
    valid=[r for r in rels if r]
    if not valid: return 0.0, rels, burstiness(sentences), basep
    # 可预测句占比：rel<1.0 → 比人类平均更可预测 → AI倾向
    predictable=sum(1 for r in valid if r<1.0)/len(valid)
    mean_rel=sum(valid)/len(valid)
    score=max(0.0,min(1.0, 0.6*predictable + 0.4*max(0.0,min(1.0, 1.0-mean_rel/2))))
    return score, rels, burstiness(sentences), basep

if __name__=="__main__":
    lm=build_lm(r"C:\Users\woshi\.dsh\aigc-detector\data\train_unified.jsonl")
    ai="综上所述，随着教育数字化转型的深入推进，具有重要的现实意义，值得注意的是，一方面，另一方面，从理论层面看，赋能。"
    human="本研究采用2018-2023年数据，回归系数0.412，t值3.21，通过Hausman检验确定固定效应模型，并用总资产收益率做了稳健性检验。"
    for name,t in [("AI",ai),("human",human),("AI*2",ai*2)]:
        s,rels,b,bp=score_text(lm,[t])
        print(f"{name}: score={s:.3f} rel={rels[0]:.3f}")
    import pickle
    os.makedirs(r"C:\Users\woshi\.dsh\aigc-detector\models", exist_ok=True)
    with open(r"C:\Users\woshi\.dsh\aigc-detector\models\n-gram-lm.pkl","wb") as f: pickle.dump(lm,f)
    print("LM saved")
