# -*- coding: utf-8 -*-
"""WSL 重新微调中文RoBERTa（用 cnki 文献库增强人类负样本，降低真实论文误判）。
数据：AI(train_unified prob>=0.4) + human(cnki文献库增强 + real_thesis + train_unified human)，平衡后训练。
模型：Hello-SimpleAI/chatgpt-detector-roberta-chinese 迁移
输出：/mnt/c/Users/woshi/.dsh/aigc-detector/models/roberta_ft
"""
import os, sys, io, json, torch, random
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
import numpy as np
from torch.utils.data import DataLoader, Dataset as TDataset
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_linear_schedule_with_warmup

BASE = "/mnt/c/Users/woshi/.dsh/aigc-detector"
DATA = os.path.join(BASE, "data", "train_unified.jsonl")
CNKI = os.path.join(BASE, "data", "human_cnki.jsonl")     # 文献库人类句
THESIS = os.path.join(BASE, "data", "human_positive.jsonl")  # 学生论文人类句
OUT = os.path.join(BASE, "models", "roberta_ft")
MODEL_ID = "Hello-SimpleAI/chatgpt-detector-roberta-chinese"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def read_list(path):
    out=[]
    for l in open(path, encoding="utf-8"):
        if not l.strip(): continue
        d=json.loads(l)
        if isinstance(d,list): out.extend(d)
        elif "text" in d: out.append(d)
        elif "sentences" in d: out.extend(d["sentences"])
    return out

class TextDS(TDataset):
    def __init__(self, texts, labels, tok, maxlen=128):
        self.enc=tok(texts,truncation=True,max_length=maxlen,padding="max_length",return_tensors="pt")
        self.labels=torch.tensor(labels)
    def __len__(self): return len(self.labels)
    def __getitem__(self,i): return {k:v[i] for k,v in self.enc.items()}, self.labels[i]

def main():
    # AI 样本
    ai=[]
    for r in read_list(DATA):
        if r.get("text") and r.get("prob",0.5)>=0.4: ai.append(r["text"])
    # 人类样本：cnki 语料为主(多样) + thesis + train_unified human
    human=[]
    for r in read_list(CNKI): human.append(r["text"])
    for r in read_list(THESIS): human.append(r["text"])
    for r in read_list(DATA):
        if r.get("text") and r.get("prob",0.5)<0.4: human.append(r["text"])
    # 平衡：人类取 AI 的 3.5 倍（深度模型要见足够多样人类，但不能压过AI信号）
    random.seed(42)
    target=int(len(ai)*2.4)
    random.shuffle(human)
    human_sel=human[:target]
    texts=ai+human_sel
    labels=[1]*len(ai)+[0]*len(human_sel)
    print("AI:",len(ai)," human:",len(human_sel)," 总:",len(texts))

    tok=AutoTokenizer.from_pretrained(MODEL_ID)
    model=AutoModelForSequenceClassification.from_pretrained(MODEL_ID)
    model.to(device); model.train()
    n=len(texts); ntr=int(n*0.85)
    order=np.random.RandomState(42).permutation(n)
    tr_idx,te_idx=order[:ntr],order[ntr:]
    tr_ds=TextDS([texts[i] for i in tr_idx],[labels[i] for i in tr_idx],tok)
    te_ds=TextDS([texts[i] for i in te_idx],[labels[i] for i in te_idx],tok)
    tr_ld=DataLoader(tr_ds,batch_size=16,shuffle=True)
    te_ld=DataLoader(te_ds,batch_size=16)
    epochs=3
    opt=torch.optim.AdamW(model.parameters(),lr=2e-5)
    total=len(tr_ld)*epochs
    sched=get_linear_schedule_with_warmup(opt,0,total)
    lossf=torch.nn.CrossEntropyLoss()
    print("开始训练(文献库增强)...")
    for ep in range(epochs):
        model.train(); tot=0; nb=0
        for batch,y in tr_ld:
            batch={k:v.to(device) for k,v in batch.items()}; y=y.to(device)
            opt.zero_grad(); out=model(**batch).logits; loss=lossf(out,y)
            loss.backward(); opt.step(); sched.step(); tot+=loss.item(); nb+=1
        model.eval(); preds=[]; probs=[]; gts=[]
        with torch.no_grad():
            for batch,y in te_ld:
                batch={k:v.to(device) for k,v in batch.items()}; out=model(**batch).logits
                proba=torch.softmax(out,-1)[:,1].cpu().numpy()
                preds.extend(proba>=0.5); probs.extend(proba); gts.extend(y.numpy())
        acc=accuracy_score(gts,preds); f1=f1_score(gts,preds); auc=roc_auc_score(gts,probs)
        print(f"epoch{ep+1}: loss={tot/nb:.4f} acc={acc:.4f} f1={f1:.4f} auc={auc:.4f}")
    os.makedirs(OUT,exist_ok=True)
    model.save_pretrained(OUT); tok.save_pretrained(OUT)
    print("已保存微调模型到",OUT)

if __name__=="__main__":
    main()

