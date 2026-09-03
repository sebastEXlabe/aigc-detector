# -*- coding: utf-8 -*-
"""段落级上下文改写训练语料构建：给真实配对补上语义锚点(所在段落上下文)。
思路：真正降重需要知道句子在讲什么(上下文)，孤句无锚点 → 模型只能编通用学术话。
这里从微信docx文本缓存里，为每个真实配对的 src_ai 句找到所在段落(及其前后句)，
构成 (段落上下文 → 段落改写) 训练对，让模型在有上下文的场景下学习"保意"+降重。
输出：data/rewrite_para_pairs.jsonl，每对含 src_para/tgt_para/embed_sentence。
用法：python scripts/build_para_pairs.py
"""
import os, sys, re, json
sys.path.insert(0, r"C:\Users\woshi\.dsh\aigc-detector")
import numpy as np

D = r"C:\Users\woshi\.dsh\aigc-detector\data"
def zhh(t): return len(re.findall(r"[\u4e00-\u9fff]", t))
def clean(s): return re.sub(r"\s+", "", (s or ""))

def main():
    # 载入缓存文档文本，一次性清洗
    raw = []
    for l in open(os.path.join(D, "_docx_txt_cache.jsonl"), encoding="utf-8"):
        try:
            d = json.loads(l); raw.append(d.get("t", ""))
        except Exception: pass
    docs = [(t, clean(t)) for t in raw if t and zhh(t) > 100]
    docs = [d for d in docs if d[1]]  # 非空
    print("缓存文档(含正文):", len(docs), flush=True)
    # 收集真实配对
    real = [json.loads(l) for l in open(os.path.join(D, "rewrite_corpus_clean.jsonl"), encoding="utf-8")]
    real = [p for p in real if p.get("src") in ("real", "wechat_real")]
    print("真实配对:", len(real), flush=True)

    out = []
    used = set()
    for p in real:
        src = (p.get("src_ai") or "").strip(); tgt = (p.get("tgt_human") or "").strip()
        if not src or not tgt or src in used: continue
        src_c = clean(src)
        if len(src_c) < 10: continue
        key = src_c[:20]
        found = None; found_clean = None
        for txt, tc in docs:  # tc 是清洗后的
            if key in tc:
                found = txt; found_clean = tc; break
        if found is None: continue
        # 切段，找含该句的段，取前后作上下文
        para_blocks = re.split(r"(?<=[。！？；\n])", found)
        block_idx = None
        for i, blk in enumerate(para_blocks):
            if src_c[:24] in clean(blk):
                block_idx = i; break
        if block_idx is None: continue
        blk = para_blocks[block_idx]
        blk_tgt = blk.replace(src, tgt) if src in blk else blk
        lo = max(0, block_idx - 1); hi = min(len(para_blocks), block_idx + 3)
        src_para = "".join(para_blocks[lo:hi])
        tgt_para = "".join(para_blocks[lo:block_idx]) + blk_tgt + "".join(para_blocks[block_idx+1: hi])
        if zhh(src_para) < 20: continue
        out.append({"src_para": src_para, "tgt_para": tgt_para,
                    "src_sentence": src, "tgt_sentence": tgt,
                    "src": p.get("src"), "sim_ref": p.get("sim_ref")})
        used.add(src)
        if len(out) % 50 == 0: print(f"  已构建 {len(out)}", flush=True)
    outp = os.path.join(D, "rewrite_para_pairs.jsonl")
    with open(outp, "w", encoding="utf-8") as f:
        for o in out: f.write(json.dumps(o, ensure_ascii=False) + "\n")
    print(f"=== 段落级配对: {len(out)} ===", flush=True)
    print("保存", outp, flush=True)

if __name__ == "__main__":
    main()
