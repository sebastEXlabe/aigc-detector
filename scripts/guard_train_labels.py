# -*- coding: utf-8 -*-
"""训练标签深流守卫：修正「报告未检出AIGC→human」的污染标签。

依据（实测）：train_unified 的 human 标签里 ~40% 被深流判 p>=0.5（来源平台漏检的AI）。
守卫原则：
  · AI 标签(high/medium/low) 原样保留
  · human 标签：深流 p<0.7 保留为 human；0.7<=p<0.9 剔除（疑似污染，弃用避免误导）；
    p>=0.9 改标为 AI（高度疑似漏检AI，作为正样本更好）
输出：data/train_unified_guarded.jsonl
用法：python scripts/guard_train_labels.py [--keep-thr 0.7] [--ai-thr 0.9]
"""
import os, sys, io, json, re, argparse
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.path.insert(0, r"C:\Users\woshi\.dsh\aigc-detector")
import numpy as np
from detector.dual_stream import load_bert, bert_score_per_sentence
BASE = r"C:\Users\woshi\.dsh\aigc-detector"
DATA = os.path.join(BASE, "data")

def read_recs(p):
    out = []
    for l in open(p, encoding="utf-8"):
        l = l.strip()
        if not l: continue
        try: d = json.loads(l)
        except: continue
        if isinstance(d, list): out.extend(d)
        else: out.append(d)
    return out

def zh(t): return len(re.findall(r"[\u4e00-\u9fff]", t))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--keep-thr", type=float, default=0.7)
    ap.add_argument("--ai-thr", type=float, default=0.9)
    a = ap.parse_args()
    tu = read_recs(os.path.join(DATA, "train_unified.jsonl"))
    print("train_unified:", len(tu))

    # 需要打分的 human 句（AI 标签直接保留，不重判）
    hu_idx = [i for i, r in enumerate(tu) if r.get("label") == "human" and zh((r.get("text") or "")) >= 10]
    hu_texts = [tu[i]["text"] for i in hu_idx]
    print("待守卫 human 句:", len(hu_texts))

    bm = load_bert(device="cuda")
    if bm is None:
        print("深流加载失败，中止"); return
    tok, model, dev = bm
    bp = bert_score_per_sentence(tok, model, dev, hu_texts, batch=32)
    bp = np.array(bp)

    from collections import Counter
    stats = Counter()
    out = []
    # 先拷贝所有记录，再按守卫改 human 部分
    for i, r in enumerate(tu):
        r = dict(r)
        if i in set(hu_idx):
            p = float(bp[hu_idx.index(i)])
            lab = r.get("label")
            if lab == "human":
                if p >= a.ai_thr:
                    r["label"] = "AI"; r["guard"] = "relabeled_ai"; r["deep_prob"] = round(p, 3)
                    stats["human->AI(p>=0.9)"] += 1
                elif p >= a.keep_thr:
                    stats["human dropped(0.7<=p<0.9)"] += 1
                    continue  # 剔除
                else:
                    r["guard"] = "kept_human"; r["deep_prob"] = round(p, 3)
                    stats["human kept(p<0.7)"] += 1
        out.append(r)

    outp = os.path.join(DATA, "train_unified_guarded.jsonl")
    with open(outp, "w", encoding="utf-8") as f:
        for r in out:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print("=== 守卫结果 ===")
    for k, v in stats.items(): print(f"  {k}: {v}")
    ai = sum(1 for r in out if r.get("label") in ("high","medium","low","AI"))
    hu = sum(1 for r in out if r.get("label") == "human")
    print(f"守卫后: 总{len(out)}  AI={ai}  human={hu}")
    print("已保存:", outp)

if __name__ == "__main__":
    main()
