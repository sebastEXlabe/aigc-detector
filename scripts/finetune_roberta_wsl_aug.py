# -*- coding: utf-8 -*-
"""WSL 深流自训练/增强版微调（中文RoBERTa）。
数据：
  AI 正样本 = train_unified AI + C-ReD中文学术 + HC3中文 + M4-zh-qa（大幅扩中文学术AI多样性）
  human 负样本 = cnki文献库(采样) + self_train高置信真人句 + thesis + train_unified human
平衡：human:AI ≈ 2.4:1
模型：Hello-SimpleAI/chatgpt-detector-roberta-chinese 迁移
输出：/mnt/c/Users/woshi/.dsh/aigc-detector/models/roberta_ft
用法：python finetune_roberta_wsl_aug.py [--max-ai 10000] [--epochs 3]
"""
import os, sys, io, json, torch, random, argparse
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
import numpy as np
from torch.utils.data import DataLoader, Dataset as TDataset
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_linear_schedule_with_warmup

BASE = "/mnt/c/Users/woshi/.dsh/aigc-detector"
DATA = os.path.join(BASE, "data")
OUT = os.path.join(BASE, "models", "roberta_ft")
MODEL_ID = "Hello-SimpleAI/chatgpt-detector-roberta-chinese"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def read_list(path):
    out = []
    if not os.path.exists(path):
        return out
    for l in open(path, encoding="utf-8"):
        if not l.strip(): continue
        d = json.loads(l)
        if isinstance(d, list): out.extend(d)
        elif "text" in d: out.append(d)
        elif "sentences" in d: out.extend(d["sentences"])
    return out

def zh_len(t):
    return len([c for c in t if '\u4e00' <= c <= '\u9fff'])

class TextDS(TDataset):
    def __init__(self, texts, labels, tok, maxlen=128):
        self.enc = tok(texts, truncation=True, max_length=maxlen, padding="max_length", return_tensors="pt")
        self.labels = torch.tensor(labels)
    def __len__(self): return len(self.labels)
    def __getitem__(self, i): return {k: v[i] for k, v in self.enc.items()}, self.labels[i]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-ai", type=int, default=10000)
    ap.add_argument("--max-human", type=int, default=30000)
    ap.add_argument("--epochs", type=int, default=3)
    a = ap.parse_args()
    random.seed(42); np.random.seed(42)

    # 1. AI 样本
    ai = []
    TU = os.path.join(DATA,"train_unified.jsonl")
    _guard = os.path.join(DATA,"train_unified_guarded.jsonl")
    if os.path.exists(_guard):
        TU = _guard  # 优先用深流守卫后的干净标签，避免"未检出→human"污染
    base_ai = [r["text"] for r in read_list(TU) if r.get("text") and r.get("prob",0.5)>=0.4]
    ai.extend(base_ai)
    # 公开中文学术 AI：C-ReD paper(论文文体) + HC3(问答) + M4-zh-qa
    pub_ai = read_list(os.path.join(DATA,"ai_pub_samples.jsonl"))
    cred = [r["text"] for r in pub_ai if r.get("source")=="C-ReD-paper" and r.get("lang","zh")=="zh"]
    hc3 = [r["text"] for r in pub_ai if r.get("source")=="HC3" and r.get("lang","zh")=="zh"]
    m4zh = [r["text"] for r in pub_ai if r.get("source")=="M4-zh-qa"]
    random.shuffle(cred); random.shuffle(hc3); random.shuffle(m4zh)
    ai.extend(cred[:4000])     # 中文学术论文（核心）
    ai.extend(hc3[:2500])      # 中文问答
    ai.extend(m4zh[:800])      # 中文百科问答
    if len(ai) > a.max_ai:
        random.shuffle(ai); ai = ai[:a.max_ai]
    print("AI:", len(ai), "(base=%d cred=%d hc3=%d m4zh=%d)" % (len(base_ai), min(len(cred),4000), min(len(hc3),2500), min(len(m4zh),800)))

    # 2. human 负样本
    human = []
    for r in read_list(os.path.join(DATA,"human_cnki.jsonl")): human.append(r.get("text",""))
    for r in read_list(os.path.join(DATA,"human_positive.jsonl")): human.append(r.get("text",""))
    for r in read_list(os.path.join(DATA,"human_self_train.jsonl")): human.append(r.get("text",""))  # 自训练高置信真人句
    for r in read_list(TU):
        if r.get("text") and r.get("prob",0.5)<0.4: human.append(r["text"])
    # 过滤空/短句
    human = [t for t in human if t and zh_len(t) >= 6]
    random.shuffle(human)
    target = int(len(ai) * 2.4)
    human_sel = human[:min(target, a.max_human)]
    print("human:", len(human_sel), "(池=%d)" % len(human))

    texts = ai + human_sel
    labels = [1]*len(ai) + [0]*len(human_sel)
    print("总:", len(texts), " ratio:", round(len(human_sel)/max(len(ai),1),2))

    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_ID)
    model.to(device); model.train()
    n = len(texts); ntr = int(n*0.85)
    order = np.random.RandomState(42).permutation(n)
    tr_idx, te_idx = order[:ntr], order[ntr:]
    tr_ds = TextDS([texts[i] for i in tr_idx],[labels[i] for i in tr_idx],tok)
    te_ds = TextDS([texts[i] for i in te_idx],[labels[i] for i in te_idx],tok)
    tr_ld = DataLoader(tr_ds, batch_size=16, shuffle=True)
    te_ld = DataLoader(te_ds, batch_size=16)
    epochs = a.epochs
    opt = torch.optim.AdamW(model.parameters(), lr=2e-5)
    total = len(tr_ld)*epochs
    sched = get_linear_schedule_with_warmup(opt, 0, total)
    lossf = torch.nn.CrossEntropyLoss()
    print("开始训练(增强版数据)... device=", device)
    for ep in range(epochs):
        model.train(); tot=0; nb=0
        for batch, y in tr_ld:
            batch={k:v.to(device) for k,v in batch.items()}; y=y.to(device)
            opt.zero_grad(); out=model(**batch).logits; loss=lossf(out,y)
            loss.backward(); opt.step(); sched.step(); tot+=loss.item(); nb+=1
        model.eval(); preds=[]; probs=[]; gts=[]
        with torch.no_grad():
            for batch, y in te_ld:
                batch={k:v.to(device) for k,v in batch.items()}; out=model(**batch).logits
                proba=torch.softmax(out,-1)[:,1].cpu().numpy()
                preds.extend(proba>=0.5); probs.extend(proba); gts.extend(y.numpy())
        acc=accuracy_score(gts,preds); f1=f1_score(gts,preds); auc=roc_auc_score(gts,probs)
        print(f"epoch{ep+1}: loss={tot/nb:.4f} acc={acc:.4f} f1={f1:.4f} auc={auc:.4f}")
    os.makedirs(OUT, exist_ok=True)
    model.save_pretrained(OUT); tok.save_pretrained(OUT)
    print("已保存微调模型到", OUT)

if __name__ == "__main__":
    main()
