# -*- coding: utf-8 -*-
"""AIGC 检测工具 · 完整管线一键重建入口。
步骤：
  1. 训练统计流模型（classifier.pkl，TF-IDF+LR，用 train_balanced + cnki语料）
  2. 调用 WSL 微调深度流模型（roberta_ft，中文RoBERTa）
  3. 运行三场景验证
用法：
  python rebuild_all.py            # 完整重建（统计流+深流（WSL）+验证）
  python rebuild_all.py --stat-b   # 仅统计流
  python rebuild_all.py --bert     # 仅WSL深流微调
"""
import os, sys, io, subprocess, json, re
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
BASE = r"C:\Users\woshi\.dsh\aigc-detector"
SCRIPTS = os.path.join(BASE, "scripts")

def run(cmd, cwd=None):
    print(f"\n>>> {cmd}")
    r = subprocess.run(cmd, shell=True, cwd=cwd or SCRIPTS, capture_output=True, text=True, encoding="utf-8", errors="replace")
    if r.stdout: print(r.stdout)
    if r.stderr: print("[stderr]", r.stderr[-500:])
    return r.returncode

def main():
    args = sys.argv[1:]
    do_stat = "--bert" not in args
    do_bert = "--stat-b" not in args
    do_validate = True

    if do_stat:
        print("="*50); print("步骤1: 训练统计流 (TF-IDF+LR)"); print("="*50)
        rc = run("python train_classifier.py")
        if rc != 0:
            print("统计流训练失败，中止"); return

    if do_bert:
        print("="*50); print("步骤2: WSL 微调深度流 (中文RoBERTa)"); print("="*50)
        wsl_cmd = "wsl -- bash -c '. /home/sebast/aigcenv/bin/activate && python /mnt/c/Users/woshi/.dsh/aigc-detector/scripts/finetune_roberta_wsl.py'"
        rc = run(wsl_cmd)
        if rc != 0: print("深度流微调失败（检查WSL环境）")

    if do_validate:
        print("="*50); print("步骤3: 三场景验证"); print("="*50)
        run("python validate_three_scenarios.py")

    print("\n=== 管线完成 ===")

if __name__ == "__main__":
    main()
