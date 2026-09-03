# -*- coding: utf-8 -*-
"""语料净化：用双门槛重筛出'真·忠实降重对'。
门槛：
  (1) 非节标题/目录/无信息句(启发式)。
  (2) 人化参考句 vs 原句 的语义相似度 >= 0.6(保意，用本机中文RoBERTa句向量cos)。
  (3) 降分(src_ai_prob - tgt_ai_prob) > 0(确实降了AIGC)。
输出：data/rewrite_corpus_clean.jsonl 记录每对(原句,参考句,两项打分) + 统计。
用法：python scripts/filter_clean_corpus.py
"""
import os, sys, re, json
sys.path.insert(0, r"C:\Users\woshi\.dsh\aigc-detector")
import numpy as np
import torch

D = r"C:\Users\woshi\.dsh\aigc-detector\data"

def is_heading(s):
    s = (s or "").strip()
    if not s or len(s) < 8: return True
    # 节/目录/编号标题
    if re.match(r'^\s*\d+(\.\d+)*[\s、.．]', s): return True
    if re.match(r'^\s*（[一二三四五六七八九十]+）', s): return True
    if re.match(r'^\s*[一二三四五六七八九十]+[\s、，,]', s): return True
    if re.search(r'^\s*(摘要|关键词|目录|参考文献|致谢|绪论|引言|结论|abstract|keywords|作者简介|基金项目)', s): return True
    return False

def main():
    from detector.dual_stream import load_bert, fuse as ds_fuse, bert_score_per_sentence
    from scripts.cross_validate import stat_probs, load_cls
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print("[clean] device:", dev, flush=True)
    stat = load_cls()
    bm = load_bert(device=dev); tok, model, bdevice = bm

    def embed(s):
        inp = tok(s, return_tensors="pt", truncation=True, max_length=160).to(bdevice)
        with torch.no_grad():
            out = model(**inp, output_hidden_states=True)
        h = out.hidden_states[-1]
        mask = inp["attention_mask"].unsqueeze(-1).float()
        vec = (h * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
        return vec[0].float().cpu()

    def aiscore(s):
        pt = stat_probs(stat, [s]); pb = bert_score_per_sentence(tok, model, bdevice, [s], batch=4)
        return float(ds_fuse(float(pt[0]), float(pb[0])))

    rows = [json.loads(l) for l in open(os.path.join(D, "rewrite_corpus.jsonl"), encoding="utf-8")]
    print("[clean] 总对:", len(rows), flush=True)
    clean = []
    stat_heads = 0; stat_lowsim = 0; stat_nodrop = 0
    for p in rows:
        src = (p.get("src_ai") or "").strip(); tgt = (p.get("tgt_human") or "").strip()
        if not src or not tgt: continue
        if is_heading(src) or is_heading(tgt):
            stat_heads += 1; continue
        # 保意
        try:
            s_src = aiscore(src); s_tgt = aiscore(tgt)
        except Exception:
            continue
        if not (s_src > s_tgt):  # 必须降分
            stat_nodrop += 1; continue
        e_s = embed(src); e_t = embed(tgt)
        sim = float(torch.nn.functional.cosine_similarity(e_s.unsqueeze(0), e_t.unsqueeze(0)).item())
        if sim < 0.6:
            stat_lowsim += 1; continue
        clean.append({**p, "sim_ref": round(sim, 3), "delta": round(s_src - s_tgt, 3)})
        if len(clean) % 200 == 0: print(f"  已保留 {len(clean)}", flush=True)

    outp = os.path.join(D, "rewrite_corpus_clean.jsonl")
    with open(outp, "w", encoding="utf-8") as f:
        for c in clean: f.write(json.dumps(c, ensure_ascii=False) + "\n")
    real = [c for c in clean if c.get("src") in ("real", "wechat_real")]
    syn = [c for c in clean if c.get("src") == "synth"]
    sims = [c["sim_ref"] for c in clean]; ds = [c["delta"] for c in clean]
    print("\n=== 净化统计 ===", flush=True)
    print(f"剔除 节标题/无信息 {stat_heads} | 无降分 {stat_nodrop} | 保意<0.6 {stat_lowsim}", flush=True)
    print(f"保留 {len(clean)} | 真实人化 {len(real)} | 合成 {len(syn)}", flush=True)
    print(f"平均保意 {np.mean(sims):.3f} | 平均降分 {np.mean(ds):.3f}", flush=True)
    print("保存", outp, flush=True)

if __name__ == "__main__":
    main()
