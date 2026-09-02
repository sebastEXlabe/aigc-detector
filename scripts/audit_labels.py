# -*- coding: utf-8 -*-
"""标注数据正确性审查。

目标（主人强调）：小心标注正确性。审查各数据源的标注是否有错位/污染。
方法：
  1. 对各来源抽样。
  2. 用【语料先验】检查：cnki 语料本应全是真人(human)，若发现明显AI痕迹(模板词/套话) -> 标注可疑。
  3. 用【模型交叉打分】检查：human 样本若被深流判为高AI -> 可疑(可能混入AI)；AI样本若被深流判低 -> 可疑。
  4. 人工特征：检测明显的模板套话(『综上所述』『从理论层面看』等)在 human 样本中出现的比例。

用法：python scripts/audit_labels.py
"""
import os, sys, io, json, re, random
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
BASE = r"C:\Users\woshi\.dsh\aigc-detector"
DATA = os.path.join(BASE, "data")

# AI 套话模板词（若是 human 数据里高比例出现，说明标注可能污染）
AI_PHRASES = [
    "综上所述", "从理论层面看", "值得注意的是", "不难发现", "无需赘言",
    "由此可见", "总而言之", "首先", "其次", "最后", "一方面", "另一方面",
    "随着...的", "赋能", "谱写", "勾勒出", "开启...新篇章", "具有重要意义",
    "本文旨在", "研究表明", "结果表明", "取得显著成效", "发展态势良好",
    "不置可否", "势在必行", "不容忽视", "日益凸显", "彰显了", "展望未来",
]

def zh_len(t): return len(re.findall(r"[\u4e00-\u9fff]", t))

def count_ai_phrases(t):
    hits = 0
    for p in AI_PHRASES:
        if p in t:
            hits += 1
    return hits

def read_jsonl(path):
    out = []
    if not os.path.exists(path): return out
    for l in open(path, encoding="utf-8"):
        if l.strip():
            try: out.append(json.loads(l))
            except: pass
    return out

def audit_pool(name, recs, expect, sample_n=800, min_zh=20, require_zh=True):
    """审查一个池：stat 期望文本属性。expect: 'human'|'ai'"""
    pool = [r for r in recs if zh_len(r.get("text","") or "") >= min_zh] if require_zh else recs
    random.seed(3); random.shuffle(pool)
    pool = pool[:sample_n]
    if not pool:
        print(f"  [{name}] 无样本"); return
    # 统计 AI 套话命中率
    high_phrase = sum(1 for r in pool if count_ai_phrases(r.get("text","")) >= 1)
    phrase_rate = high_phrase / len(pool)
    avg_len = sum(zh_len(r.get("text","")) for r in pool) / len(pool)
    print(f"  [{name}] 样={len(pool)} 均长={avg_len:.0f} 套话命中={phrase_rate:.3f}  (期望:{expect})")
    # 若期望 human 但套话命中过高 -> 警告
    if expect == "human" and phrase_rate > 0.15:
        print(f"     ⚠️ 套话命中率 {phrase_rate:.2f} 偏高，怀疑混入AI风格样本（标注可能不纯）")
    if expect == "human":
        # 抽样打印套话命中的样本（供人工判断）
        hits = [r["text"] for r in pool if count_ai_phrases(r.get("text","")) >= 1]
        if hits:
            print(f"     例: {hits[0][:80]}")
            print(f"     例: {hits[1][:80] if len(hits)>1 else ''}")

def main():
    print("=" * 60)
    print(" 标注数据正确性审查")
    print("=" * 60)
    print("\n[1] human 负样本池（期望全为真人，警惕混入AI）")
    audit_pool("human_corpus.jsonl(cnki全文语料)", read_jsonl(os.path.join(DATA,"human_corpus.jsonl")), "human")
    audit_pool("human_cnki.jsonl(采样9000)", read_jsonl(os.path.join(DATA,"human_cnki.jsonl")), "human")
    audit_pool("human_self_train.jsonl(自训练高置信句)", read_jsonl(os.path.join(DATA,"human_self_train.jsonl")), "human")
    audit_pool("human_pub_samples(公开human: C-ReD/HC3/M4)", read_jsonl(os.path.join(DATA,"human_pub_samples.jsonl")), "human")
    print("\n[2] AI 正样本池（期望全为AI，警惕混入真人/中性句）")
    audit_pool("ai_pub_samples.jsonl(公开AI)", read_jsonl(os.path.join(DATA,"ai_pub_samples.jsonl")), "ai")
    print("\n[3] train_unified(自有核心标注) 标签分布")
    tr = read_jsonl(os.path.join(DATA,"train_unified.jsonl"))
    from collections import Counter
    print("  label分布:", dict(Counter(r.get("label") for r in tr)))

if __name__ == "__main__":
    main()
