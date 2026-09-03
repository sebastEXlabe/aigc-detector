# -*- coding: utf-8 -*-
"""对训练好的 Qwen 降重改写模型做【诚实评估】：同时测两件事——
(1) 降AIGC：改写后检测器融合分是否下降(delta)。
(2) 保原意：改写与原文的语义相似度(用本机已加载的中文RoBERTa句向量做cos)。
关键：只看降分会误判(模型可能输出"灌水/截断/模板句"来刷低分)，必须同时看相似度，
      才能在"真降重"与"靠跑题刷低分"之间分清。对比基准：真实人化参考句的相似度。
用法：python scripts/evaluate_rewrite_qwen.py [--model Qwen/Qwen2.5-1.5B-Instruct]
"""
import os, sys, json, re, argparse
sys.path.insert(0, r"C:\Users\woshi\.dsh\aigc-detector")
import numpy as np
import torch

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=r"C:\Users\woshi\.dsh\aigc-detector\models\qwen_rewrite")
    ap.add_argument("--n", type=int, default=80)
    ap.add_argument("--max-len", type=int, default=128)
    args = ap.parse_args()

    import torch  # 先torch再transformers
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from detector.dual_stream import load_bert, fuse as ds_fuse, bert_score_per_sentence
    from scripts.cross_validate import stat_probs, load_cls

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print("[eval] device:", dev, flush=True)
    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token_id is None: tok.pad_token_id = tok.eos_token_id
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16).to(dev)
    model.eval()

    SYS = "你是论文降重助手，把下面这句话改写成更自然、更像人写的学术表达，保持原意不变，不要增删信息，只输出改写后的句子。"
    def build_prompt(s): return f"{SYS}\n原句：{s.strip()}\n改写："

    # 检测器(统计流 + 深度流) → 融合分
    stat = load_cls()
    bmtok = bmmod = bmdev = None
    try:
        bm = load_bert(device=dev); bmtok, bmmod, bmdev = bm
    except Exception as e:
        print("[eval] BERT load fail:", e, flush=True)
    def aiscore(s):
        pt = stat_probs(stat, [s])
        pb = bert_score_per_sentence(bmtok, bmmod, bmdev, [s], batch=4) if bmdev is not None else [0.5]
        return float(ds_fuse(float(pt[0]), float(pb[0])))

    # 语义相似度：用同一RoBERTa的最后隐藏层某处做句向量(取[CLS]/mean pooling)
    def embed(s):
        if bmdev is None: return None
        inp = bmtok(s, return_tensors="pt", truncation=True, max_length=160).to(bmdev)
        with torch.no_grad():
            out = bmmod(**inp, output_hidden_states=True)
        # 取最后隐藏层做 mean pooling(排除pad)
        h = out.hidden_states[-1]  # [1,T,H]
        mask = inp["attention_mask"].unsqueeze(-1).float()
        vec = (h * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
        return vec[0].float().cpu()
    def cos(a, b):
        if a is None or b is None: return None
        return float(torch.nn.functional.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).item())

    def rewrite(s):
        inp = build_prompt(s)
        enc = tok(inp, return_tensors="pt").to(dev)
        with torch.no_grad():
            g = model.generate(**enc, max_new_tokens=args.max_len, do_sample=False,
                               num_beams=4, no_repeat_ngram_size=2, early_stopping=True)
        return tok.decode(g[0][enc["input_ids"].shape[1]:], skip_special_tokens=True).strip()

    rows = [json.loads(l) for l in open(r"C:\Users\woshi\.dsh\aigc-detector\data\rewrite_corpus.jsonl", encoding="utf-8")]
    real = [p for p in rows if p.get("src") in ("real", "wechat_real")]
    import random; random.seed(7); random.shuffle(real)
    real = real[:args.n]

    # 基准：真实人化参考句相对原句的相似度(降重的"合理上限")
    base_sims = []
    res = []
    for p in real:
        src = p["src_ai"].strip(); tgt = p["tgt_human"].strip()
        s_src = aiscore(src)
        gen = rewrite(src)
        s_gen = aiscore(gen) if gen else s_src
        e_src = embed(src); e_gen = embed(gen) if gen else None
        sim_sg = cos(e_src, e_gen)          # 原句 vs 模型改写
        sim_st = cos(e_src, embed(tgt))     # 原句 vs 真实人化参考
        if sim_st is not None: base_sims.append(sim_st)
        res.append({"src_ai": src, "tgt_human": tgt, "gen": gen,
                    "src_ai_prob": round(s_src,3), "gen_ai_prob": round(s_gen,3),
                    "delta": round(s_src-s_gen,3),
                    "sim_model": round(sim_sg,3) if sim_sg is not None else None,
                    "sim_reference": round(sim_st,3) if sim_st is not None else None})
    import statistics
    ds = [r["delta"] for r in res]
    sg = [r["sim_model"] for r in res if r["sim_model"] is not None]
    print("\n=== 诚实评估 ===", flush=True)
    print(f"样本 {len(res)}", flush=True)
    print(f"降分: 成功 {sum(1 for d in ds if d>0)}/{len(ds)} ({100*sum(1 for d in ds if d>0)/len(ds):.0f}%) | 平均降分 {np.mean(ds):.3f}", flush=True)
    print(f"模型改写相似度: 平均 {np.mean(sg):.3f} (>=0.6算保意)", flush=True)
    if base_sims:
        print(f"真实人化参考相似度(基准): 平均 {np.mean(base_sims):.3f}", flush=True)
        print(f"  → 模型相似度 vs 参考基准: {np.mean(sg):.3f} vs {np.mean(base_sims):.3f}", flush=True)
    # 区分"真降重" vs "跑题刷低分"
    good = [r for r in res if r["sim_model"] is not None and r["sim_model"]>=0.6 and r["delta"]>0.15]
    game = [r for r in res if r["sim_model"] is not None and r["sim_model"]<0.4 and r["delta"]>0.15]
    print(f"真降重(相似度>=0.6 且 降分>0.15): {len(good)}", flush=True)
    print(f"跑题刷分(相似度<0.4 且 降分>0.15): {len(game)}", flush=True)
    print("\n--- 样例 ---", flush=True)
    for r in res[:8]:
        print(f"  原: {r['src_ai'][:44]} ({r['src_ai_prob']}) sim_vs_参考={r['sim_reference']}", flush=True)
        print(f"  改: {r['gen'][:44]} ({r['gen_ai_prob']}, 降{r['delta']}) sim_vs_原句={r['sim_model']}", flush=True)
    outp = r"C:\Users\woshi\.dsh\aigc-detector\data\rewrite_qwen_eval2.jsonl"
    with open(outp, "w", encoding="utf-8") as f:
        for r in res: f.write(json.dumps(r, ensure_ascii=False)+"\n")
    print(f"\n保存 {outp}", flush=True)

if __name__ == "__main__":
    main()
