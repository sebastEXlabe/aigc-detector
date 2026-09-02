# -*- coding: utf-8 -*-
"""写作流程自动质检：稿件写完后自动调用 AIGC 检测服务做质检。
用法：python workflow_quality_check.py <稿件.docx|txt> [--threshold 0.35]
   - 调用本地 AIGC 服务 /detect 检测
   - 若 AI 概率超阈值，输出高风险段落 + 命中模板 + 改写建议
   - 在稿件生产流程末尾调用，作为交付前质检环节
"""
import os, sys, io, json, re, subprocess, time
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

def get_port():
    try:
        return int(open(r"C:\Users\woshi\.dsh\aigc-detector\last_port.txt").read().strip())
    except Exception:
        return 9000

def read_text(path):
    ext = os.path.splitext(path)[1].lower()
    if ext == ".docx":
        from docx import Document
        d = Document(path); return "\n".join(p.text for p in d.paragraphs)
    return open(path, encoding="utf-8", errors="ignore").read()

def quality_check(path, threshold=0.35):
    import requests
    port = get_port()
    BASE = f"http://127.0.0.1:{port}"
    # 调服务检测
    try:
        r = requests.post(f"{BASE}/detect", json={"path": path, "top_k": 10}, timeout=120)
        result = r.json()
    except Exception as e:
        print(f"✗ 检测服务不可用: {e}")
        result = None
    # 质检报告
    print("="*50)
    print(f"AIGC 质检报告: {os.path.basename(path)}")
    print("="*50)
    if result is None:
        print("  检测失败，跳过质检")
        return False
    overall = result.get("overall_ai_prob", 0)
    state = result.get("state", "?")
    verdict = result.get("verdict", "?")
    print(f"  稿件: {os.path.basename(path)}")
    print(f"  AI概率: {overall*100:.1f}%  |  判定: {state}  |  报告档位: {verdict}")
    print(f"  总字数: {result.get('total_chars')}  AI字符: {result.get('ai_chars')}  句子: {result.get('n_sentences')}")
    # 高风险段落
    hi = result.get("top_ai_sentences", [])
    if hi:
        print(f"\n  高风险段落 / 命中模板:")
        for h in hi:
            tpls = h.get('templates', [])
            print(f"    [{h.get('ai_prob',0)*100:.0f}%] {h.get('sentence','')[:50]}" + (f"\n       模板: {tpls}" if tpls else ""))
    # 判定
    if overall >= threshold:
        print(f"\n  ⚠️ 质检不通过：AI概率 {overall*100:.1f}% > 阈值 {threshold*100:.0f}%")
        print("  建议: 按上方高风险段落 + 模板，改写为更自然的原创表述")
        return False
    else:
        print(f"\n  ✅ 质检通过：AI概率 {overall*100:.1f}% ≤ 阈值 {threshold*100:.0f}%")
        return True

if __name__ == "__main__":
    args = sys.argv[1:]
    if not args:
        print("用法: python workflow_quality_check.py <稿件> [--threshold 0.35]"); sys.exit(1)
    path = args[0]
    threshold = 0.35
    for i,a in enumerate(args):
        if a=="--threshold" and i+1<len(args):
            try: threshold=float(args[i+1])
            except: pass
    ok = quality_check(path, threshold)
    sys.exit(0 if ok else 2)
