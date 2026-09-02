# -*- coding: utf-8 -*-
"""深流交叉验证标注正确性：用深流(RoBERTa)对各池打分，找"预期外"样本。
  human 池被判高AI -> 可能混入AI风格样本（污染）
  ai 池被判低AI   -> 可能混入真人/中性句（粒度问题）
在 WSL 运行。
"""
import os, sys, io, json, re, random
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
import torch, torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification

MODEL_PATH = "/mnt/c/Users/woshi/.dsh/aigc-detector/models/roberta_ft"
DATA = "/mnt/c/Users/woshi/.dsh/aigc-detector/data"

def zh_len(t): return len(re.findall(r"[\u4e00-\u9fff]", t))
def read_jsonl(path):
    out=[]
    for l in open(path, encoding="utf-8"):
        if l.strip():
            try: out.append(json.loads(l))
            except: pass
    return out

tok = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(dev); model.eval()

def score(sents):
    probs=[]
    with torch.no_grad():
        for i in range(0,len(sents),32):
            ch=sents[i:i+32]
            inp=tok(ch,truncation=True,max_length=200,padding="max_length",return_tensors="pt")
            inp={k:v.to(dev) for k,v in inp.items()}
            out=model(**inp).logits
            p=F.softmax(out,-1)[:,1].cpu().numpy(); probs.extend(p.tolist())
            del inp,out,p
            if torch.cuda.is_available(): torch.cuda.empty_cache()
    return probs

def audit(name, recs, expect, n=400):
    pool=[r["text"] for r in recs if zh_len(r.get("text","") or "")>=20]
    random.seed(7); random.shuffle(pool); pool=pool[:n]
    if not pool: print(f"[{name}] 无样本"); return
    p=score(pool)
    if expect=="human":
        # 判高AI的比例 = 潜在污染
        bad=float(sum(1 for x in p if x>=0.5)/len(p))
        print(f"[{name}] 期望human: 深流判AI均值={sum(p)/len(p):.3f} 判高AI占比={bad:.3f}")
        if bad>0.1:
            high=[(t,x) for t,x in zip(pool,p) if x>=0.5]
            print(f"    ⚠️ 潜在污染率 {bad:.2f}，例:")
            for t,x in high[:4]: print(f"      p={x:.2f} {t[:60]}")
    else:
        # 判低AI = 可能混真人/中性
        bad=float(sum(1 for x in p if x<0.5)/len(p))
        print(f"[{name}] 期望AI: 均值={sum(p)/len(p):.3f} 判低AI占比={bad:.3f}")
        if bad>0.1:
            low=[(t,x) for t,x in zip(pool,p) if x<0.5]
            print(f"    ⚠️ 可能混入真人/中性句 {bad:.2f}，例:")
            for t,x in low[:4]: print(f"      p={x:.2f} {t[:60]}")

print("=== 深流交叉验证标注正确性 ===")
print("\n[human 池]")
audit("human_corpus(cnki全文)", read_jsonl(os.path.join(DATA,"human_corpus.jsonl")), "human")
audit("human_self_train(自训练)", read_jsonl(os.path.join(DATA,"human_self_train.jsonl")), "human")
audit("human_pub(C-ReD/HC3/M4 human)", read_jsonl(os.path.join(DATA,"human_pub_samples.jsonl")), "human")
print("\n[ai 池]")
audit("ai_pub-C-ReD(中文学术AI)", [r for r in read_jsonl(os.path.join(DATA,"ai_pub_samples.jsonl")) if r.get("source")=="C-ReD-paper"], "ai")
