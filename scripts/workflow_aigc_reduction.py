# -*- coding: utf-8 -*-
"""降 AIGC 引导工作流（方向A）：给定稿件，标出"该改写的句"（已对齐知网/维普口径），按 AI 概率排序，
你按序改写 → 复检检测器 AI 分数下降 → 降低知网/维普 AIGC 率。
用法：python workflow_aigc_reduction.py <稿件.docx|txt> [--topk 20]
"""
import os, sys, io, json, subprocess
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

def get_port():
    try: return int(open(r"C:\Users\woshi\.dsh\aigc-detector\last_port.txt").read().strip())
    except Exception: return 9000

def main():
    args = sys.argv[1:]
    if not args:
        print("用法: python workflow_aigc_reduction.py <稿件> [--topk 20]"); sys.exit(1)
    path = args[0]; topk = 20
    for i,a in enumerate(args):
        if a=="--topk" and i+1<len(args):
            try: topk=int(args[i+1])
            except: pass
    import requests
    port = get_port()
    try:
        r = requests.post(f"http://127.0.0.1:{port}/detect", json={"path": path, "top_k": topk}, timeout=180)
        res = r.json()
    except Exception as e:
        print(f"✗ 检测服务不可用: {e}"); sys.exit(1)
    overall = res.get("overall_ai_prob", 0); state = res.get("state","?")
    print("="*56)
    print(f"降 AIGC 引导: {os.path.basename(path)}")
    print("="*56)
    print(f"  当前 AI 概率: {overall*100:.1f}%   判定: {state}")
    print(f"  (检测器已对齐知网/维普: 标出的应改写句与知网/维普判的AI句高度一致)")
    hi = res.get("top_ai_sentences", [])
    print(f"\n  >>> 按 AI 概率排序的【应改写句清单】(改这些能降知网/维普 AIGC 率):")
    if not hi:
        print("  (无高 AI 句 — 该稿 AIGC 率已很低)")
    for i,h in enumerate(hi,1):
        tpls = ", ".join(h.get("templates",[]) or [])
        diag=[]
        if h.get("in_ai_island"): diag.append("AI密集段内")
        if h.get("isolated"): diag.append("孤立句可复核")
        print(f"  {i}. [{h.get('ai_prob',0)*100:.0f}%] {h.get('sentence','')[:45]}..." + (f"\n     模板: {tpls}" if tpls else "") + (f"  ({'; '.join(diag)})" if diag else ""))
    print(f"\n  ※ 改写后请重跑本脚本, 看 AI 概率是否下降; 降到目标阈值以下即视为通过知网/维普。")

if __name__ == "__main__":
    main()
