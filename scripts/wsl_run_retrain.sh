#!/bin/bash
# WSL 深流重训脚本（方案2）。把输出写到日志（避免 utf-16 编码问题），日志在 Windows 侧可读。
set -u
LOG=/mnt/c/Users/woshi/.dsh/aigc-detector/logs/deep_retrain2.log
echo "==== 深流重训启动 $(date) ====" > $LOG
# 激活 venv
. /home/sebast/aigcenv/bin/activate 2>>$LOG || { echo "venv 激活失败" >>$LOG; exit 1; }
echo "venv: $(which python)" >> $LOG
echo "GPU:" >> $LOG
nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv,noheader 2>>$LOG | head -2
echo "torch:" >> $LOG
python -c "import torch; print('torch',torch.__version__,'cuda',torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none')" >> $LOG 2>&1
cd /mnt/c/Users/woshi/.dsh/aigc-detector/scripts
echo "==== 开始 finetune (guarded 数据) ====" >> $LOG
python finetune_roberta_wsl_aug.py --epochs 3 >> $LOG 2>&1
echo "==== 重训退出码 $? ====" >> $LOG
