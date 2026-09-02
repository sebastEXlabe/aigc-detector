#!/bin/bash
echo "=== GPU ==="
nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv,noheader 2>&1 | head -3
echo "=== torch/cuda ==="
. /home/sebast/aigcenv/bin/activate
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), 'device', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none')"
