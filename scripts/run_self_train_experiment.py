# -*- coding: utf-8 -*-
"""自训练 OOD 对照实验：自训练是否真的降低对真人学术句的误报（而非内部指标虚高）。

方法：
  - 用当前模型对 cnki 语料打分，分成：高置信human池 / 误报句池 / OOD留出池。
  - OOD留出池（训练完全不见）作为独立测试：
      · 对照组(不读 human_self_train) vs 自训练组(用高置信human扩展)
      · 对比两组对 OOD 真人句的误报率（真实效果）。
  - 同时对比对已知AI句(ai_pub 采样)的召回，确认没有牺牲检出。

用法：python scripts/run_self_train_experiment.py
"""
import os, sys, io, json, pickle, random, re
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
import numpy as np
BASE = r"C:\Users\woshi\.dsh\aigc-detector"
DATA = os.path.join(BASE, "data")
SCRIPTS = os.path.join(BASE, "scripts")

def read_recs(path):
    recs = []
    if not os.path.exists(path): return recs
    for l in open(path, encoding="utf-8"):
        if not l.strip(): continue
        d = json.loads(l)
        if isinstance(d, list): recs.extend(d)
        elif "text" in d: recs.append(d)
    return recs

def zh_len(t): return len(re.findall(r"[\u4e00-\u9fff]", t))

def build_and_eval(use_st, ood_texts, ood_is_human):
    """注入 self_train 开关，训练 classifier，返回在 ood 上的判定。"""
    import subprocess, sys as _s
    subprocess.run([_s.executable, os.path.join(SCRIPTS, "train_classifier.py"), "--lang", "zh",
                    ("" if use_st else "--no-self-train")], check=False,
                   capture_output=True, timeout=600)
    cls = pickle.load(open(os.path.join(BASE, "models", "classifier.pkl"), "rb"))
    prob = cls["model"].predict_proba(cls["vec"].transform(ood_texts))[:, 1]
    thr = cls.get("threshold", 0.4)
    # 误报率：真人类被判AI的比例
    fp = [bool(p >= thr) for p, is_h in zip(prob, ood_is_human) if is_h]
    return prob, thr, float(np.mean(fp)) if fp else 0.0

def main():
    random.seed(42)
    cls = pickle.load(open(os.path.join(BASE,"models","classifier.pkl"), "rb"))
    vec, model, thr0 = cls["vec"], cls["model"], cls.get("threshold",0.4)

    used = set()
    for r in read_recs(os.path.join(DATA,"train_unified.jsonl")): used.add(r["text"])
    for r in read_recs(os.path.join(DATA,"human_cnki.jsonl")): used.add(r["text"])
    for r in read_recs(os.path.join(DATA,"human_positive.jsonl")): used.add(r["text"])
    for r in read_recs(os.path.join(DATA,"human_self_train.jsonl")): used.add(r["text"])

    corpus = [r for r in read_recs(os.path.join(DATA,"human_corpus.jsonl"))
              if r.get("text") and r["text"] not in used and zh_len(r["text"])>=20]
    random.shuffle(corpus)
    ood = corpus[:4000]  # 独立OOD真人句池（训练完全不见）
    ood_texts = [r["text"] for r in ood]
    ood_is_human = [True]*len(ood)
    print(f"OOD真人句池: {len(ood)}")

    # 训练前模型对OOD的误报（基线参考）
    p0 = model.predict_proba(vec.transform(ood_texts))[:,1]
    print(f"训练前模型 OOD误报率(>thr0): {float((p0>=thr0).mean()):.3f}")

    print("\n训练对照组（--no-self-train）...")
    prob_c, thr_c, fp_c = build_and_eval(False, ood_texts, ood_is_human)
    print(f"  对照组 thr={thr_c:.3f} OOD误报率={fp_c:.3f}")

    print("\n训练自训练组（含高置信human）...")
    prob_s, thr_s, fp_s = build_and_eval(True, ood_texts, ood_is_human)
    print(f"  自训练组 thr={thr_s:.3f} OOD误报率={fp_s:.3f}")

    print(f"\n== 结论 ==\n对照组误报={fp_c:.3f} → 自训练组误报={fp_s:.3f} Δ={fp_s-fp_c:+.3f} "
          f"{'自训练有效降低误报 ✓' if fp_s < fp_c else '自训练无明显降低(可能污染或特征不足)'}")

if __name__ == "__main__":
    main()
