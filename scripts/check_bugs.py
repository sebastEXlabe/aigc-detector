# -*- coding: utf-8 -*-
"""AIGC 检测工具 BUG 定期检测脚本。
扫描 bug_tracker.jsonl，统计 open bug（按严重度/场景），输出待修复清单。
用法：python check_bugs.py [--all]   (--all 含已修复)
"""
import os, sys, io, json, collections
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
BASE = r"C:\Users\woshi\.dsh\aigc-detector"
BUG_FILE = os.path.join(BASE, "bug_tracker.jsonl")

def main():
    if not os.path.exists(BUG_FILE):
        print("bug_tracker.jsonl 不存在，无需检测"); return
    bugs = []
    for l in open(BUG_FILE, encoding="utf-8"):
        if l.strip():
            try: bugs.append(json.loads(l))
            except: pass
    open_bugs = [b for b in bugs if b.get("status") == "open"]
    print("="*50)
    print(f"AIGC 检测工具 BUG 检测报告")
    print("="*50)
    print(f"  总记录: {len(bugs)}  |  待修复(open): {len(open_bugs)}")
    # 按严重度
    sev = collections.Counter(b.get("severity") for b in open_bugs)
    print(f"  严重度分布: high={sev.get('high',0)} med={sev.get('med',0)} low={sev.get('low',0)}")
    # 待修复清单
    if open_bugs:
        print("\n  待修复 BUG 清单:")
        for b in sorted(open_bugs, key=lambda x: {"high":0,"med":1,"low":2}.get(x.get("severity"),3)):
            print(f"    [{b['id']}] {b.get('severity')} | {b.get('scene')} | {b.get('description','')[:50]} | 上报:{b.get('time')}")
    else:
        print("\n  ✅ 无待修复 open bug")
    print("="*50)
    # 退出码：有 high bug 返回 1 提示
    if sev.get('high',0) > 0:
        sys.exit(1)

if __name__ == "__main__":
    main()
