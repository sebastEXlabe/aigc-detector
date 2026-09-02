# AIGC Detector — 本地 AIGC 文本检测工具

一个**本地运行**的中文 AIGC（AI生成内容）检测工具，面向学术论文/稿件生产质检。用双流融合（统计特征 + 深度RoBERTa）检测文本是否为 AI 生成，支持增量入库自动重训闭环。

## 特性

- **双流融合检测**：统计流 TF-IDF+LR 与深度流 微调RoBERTa 融合，三态判定，逐句定位 + 命中模板建议。
- **常驻 FastAPI 服务**：模型常驻内存，毫秒级检测，端口冲突自动切换。
- **自动闭环**：发现新 AIGC 报告 → 入库解析 → 自动重训 → 热更新模型（无需重启）。
- **报告参数对齐**：输出对齐官方 AIGC 报告口径（AI特征值/总字符/AI字符/句子数/档位判定）。
- **写作质检**：稿件生产/交付前自动质检（AI概率超阈值 → 定位高风险段落 + 改写建议）。
- **BUG 质量管理**：使用中上报 bug → bug_tracker → 定期检测修复。

## 架构

```
detector/        # 检测器模块（features/route_a/route_c/dual_stream/stylometric）
scripts/         # 服务、训练、入库、质检、bug上报/检测 脚本
data/            # 标注数据集（gitignore，不上传）
models/          # 模型（gitignore，不上传，可重训）
reports/         # 原始AIGC报告归档（gitignore，不上传）
```

## 快速开始

### 1. 常驻服务

```bash
python scripts/server.py --port 9000
# 端口冲突自动 +1，实际端口写 last_port.txt
```

### 2. 检测文本

```python
import requests, os
port = int(open("last_port.txt").read().strip())
r = requests.post(f"http://127.0.0.1:{port}/detect", json={"path": "论文.docx"})
res = r.json()  # overall_ai_prob/state/verdict/top_ai_sentences
```

### 3. 批量检测

```python
requests.post(f"http://127.0.0.1:{port}/detect_batch", json={"paths": ["a.docx","b.docx"]})
```

### 4. 入库新报告（自动重训）

```python
requests.post(f"http://127.0.0.1:{port}/ingest", json={"path": "AIGC_报告.pdf"})
```

### 5. 稿件质检

```bash
python scripts/workflow_quality_check.py 论文.docx --threshold 0.35
```

### 6. BUG 上报 / 检测

```bash
python scripts/report_bug.py "描述" --severity med --scene batch_detect
python scripts/check_bugs.py
```

## 接口

| 接口 | 说明 |
|---|---|
| `GET /health` | 健康检查 + 模型状态 |
| `POST /detect` | 单篇检测（含 verdict 档位判定） |
| `POST /detect_batch` | 批量检测多篇 |
| `POST /ingest` | 接收AIGC报告入库，自动触发重训 |
| `POST /train` | 手动触发重训（互斥锁保护） |
| `POST /report_bug` | BUG 上报到 bug_tracker |

## 判定口径

- **人类创作（0~40%）** / **疑似AI（40~60%）** / **AI生成（60~100%）**
- 检测输出：`ai_feature_rate`(AI特征值%)、`total_chars`、`ai_chars`、`n_sentences`、`verdict`

## 数据与模型

`data/`（标注数据集）、`models/`（统计流 classifier.pkl + 深度流 roberta_ft + n-gram LM）、`reports/`（原始AIGC报告归档）因体积较大或含隐私，通过 `.gitignore` 排除。模型可运行 `train_classifier.py` 重训，深度流可运行 `finetune_roberta_wsl.py`（WSL GPU）。

## License

MIT（示例）
