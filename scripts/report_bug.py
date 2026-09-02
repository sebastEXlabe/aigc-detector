# -*- coding: utf-8 -*-
"""AIGC 检测工具 BUG 上报脚本。
智能体在使用检测工具过程中发现 BUG，调用本脚本写入 bug_tracker.jsonl 统计文件。
用法：python report_bug.py "描述" [--severity high|med|low] [--scene 场景]
"""
import os, sys, io, json, datetime
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
BASE = r"C:\Users\woshi\.dsh\aigc-detector"
BUG_FILE = os.path.join(BASE, "bug_tracker.jsonl")

def report(desc, severity="med", scene="general"):
    """写入一条 bug 记录。返回记录 id。"""
    rec = {
        "id": f"BUG-{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}",
        "time": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "severity": severity,
        "scene": scene,
        "description": desc,
        "status": "open",
    }
    os.makedirs(BASE, exist_ok=True)
    with open(BUG_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"已上报 BUG [{rec['id']}] severity={severity} scene={scene}")
    print(f"  描述: {desc}")
    print(f"  已写入: {BUG_FILE}")
    return rec["id"]

if __name__ == "__main__":
    args = sys.argv[1:]
    if not args:
        print("用法: python report_bug.py \"BUG描述\" [--severity high|med|low] [--scene 场景]")
        sys.exit(1)
    desc = args[0]
    severity = "med"; scene = "general"
    for i, a in enumerate(args):
        if a == "--severity" and i+1 < len(args): severity = args[i+1]
        if a == "--scene" and i+1 < len(args): scene = args[i+1]
    report(desc, severity, scene)
