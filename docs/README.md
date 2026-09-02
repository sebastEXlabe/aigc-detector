# 本地 AIGC 检测工具（aigc-detector）

## 工具路径
`C:\Users\woshi\.dsh\aigc-detector`

> 说明：因系统将 `Documents`/`Desktop` 设为只读（OneDrive/安全策略托管），检测工具根目录放在 DSH 工作区 `~/.dsh/aigc-detector`。这是本机唯一可写入且可长期维护的位置。

## 目录结构
```
aigc-detector/
├── data/                       # 标注数据集
│   ├── train_unified.jsonl     # 统一训练集（4150句：知网+原文对照+Word标红版，合并去重）
│   ├── train_clean.jsonl       # 知网/HTML 句级（2510句，清洗后）
│   ├── train_dataset.jsonl     # 早期数据集（30篇→2775句）
│   └── word_annotations.jsonl  # Word标红版三档标签（2680条）
├── reports/                    # 归档的原始检测报告
│   ├── pdf/                    # 139份 PDF（231.5MB），含平台前缀
│   ├── word/                   # 22份 Word标红版（31.9MB）
│   └── html/                   # 14份 HTML（3.2MB）
├── detector/                   # 检测器代码
│   ├── features.py             # 路线A：AI特征库（模板句式+高频词）
│   └── route_a.py              # 路线A：特征计算器
├── scripts/                    # 数据解析/构建脚本
├── docs/                       # 说明文档
└── reports_manifest.json       # 归档清单（已复制数）
```

## 数据来源
从全盘（C盘 + D盘）扫描发现的 AIGC 检测报告，覆盖多个平台：
- **知网 AIGC**（cx.cnki.net 全文报告单）：片段级 + 原文全文
- **PaperPass / PaperYY**：逐句概率/三档颜色标注（HTML + Word标红版）
- **原文对照报告**（章节级）
- 维普/超星/万方等相关文件也已识别

## 标注数据规模
- 统一训练集：**4150 句**（human 2670 / high 815 / medium 611 / low 54）
- 标签类型：知网/HTML 为概率 0~1，Word标红版为三档（高/中/低/人类）

## 检测器路线
- **路线A**：特征工程/启发式（AI句式模板 + 高频词 + burstiness）
- **路线B**：标注数据训练分类器（sklearn）
- **路线C**：本地困惑度/burstiness

## 注
原始报告源文件部分位于微信缓存（`D:\xwechat_files`、`Documents\WXWork`），会随聊天清理而消失；已归档到 `reports/` 的是可访问版本。manifest 记录了复制结果。

## 路线B 模型现状（2026-09 优化后）

### 最优配置
- **二分类**（AI vs 人类）比三分类更可靠
- 特征：TF-IDF char n-gram（2,4），max_features=60000，sublinear_tf
- 分类器：LogisticRegression(C=1.0, class_weight="balanced") + sigmoid 概率校准
- 最优阈值：**0.404**（让 F1 最优）

### 实测性能（4150 句，20% 测试集）
- 准确率：**80.1%**
- F1（二分类）：**0.712**（最优阈值下 0.758）
- AUC：**0.879**

### 真实论文验证（《轻资产运营模式下企业财务风险管理研究》158句/13446字）
- 综合 AI 概率：**18.6% → ✅ 基本人类**
- 25.3% 句子被判 AI
- **关键洞察**：被判 AI 的多为学术框架性/概括性句子（"基于此，本文...识别...指标体系""研究结论可为...提供实操参考"）。这类句子与学生用 AI 写的摘要/结论/贡献高度相似，是真实混淆区。
- **研究方向**：需加入真实的"人类学术写作"负样本（而非仅用检测报告里被标绿的句子），或对框架性句子做专门校准。

### 三路线
- 路线B（主力）：校准二分类 TF-IDF，80.1% acc，AUC 0.879
- 路线A（定位）：AI 模板句式正则，输出修改建议
- 路线C（佐证）：jieba 词级 n-gram 相对困惑度 + 可预测句占比

## 最终成果（2026-09 优化后）

### 核心模型（路线B 主力）
- **二分类**（AI vs 人类）：char(2,4) TF-IDF + LogisticRegression(C=1.0, class_weight="balanced") + sigmoid校准
- **数据**：5541 句（含 1397 句真实人类学术论文句子增强负样本）
- **性能**：acc **86.6%**，F1 0.735，**AUC 0.92**，最优阈值 0.402

### 三场景实测（路线B核心分）
| 场景 | AI概率 | AI句 |
|---|---|---|
| AI套话（人工构造） | 45.5% | 5/7 |
| 知网判高度AI段落（真实报告原文） | 55.9% | 4/4 |
| 真实人类论文《企业财务风险管理》 | 2.3% | 0/158 |

### 三态判定（低误报）
- AI概率 ≥0.5：高度疑似AI生成
- 0.35~0.5：疑似AI（人工复核）
- 0.2~0.35：证据不足（倾向人类，少量AI痕迹）
- <0.2：基本人类撰写

### 调研方向落实（2024-2026 业界）
- **方向1（已落地）**：TF-IDF 主判 + 可解释文体特征佐证（MATTR/重复度/标点密度/功能词密度）+ 真实人类负样本 + 三态低误报输出
- **方向2（部分）**：路线C 用 jieba 词级 n-gram 相对困惑度 + 可预测句占比（弱佐证，权重0.2）
- **方向3（未做，可选）**：引入中文 RoBERTa 检测模型（Hello-SimpleAI/MGT-Mini）作深度信号，需 GPU/数据

### 关键经验
- **真实人类学术负样本**是降低"规范学术书面语误判"的关键：加入 1397 句真实论文句子后，真实论文误判从 25%降到0%，AUC 0.879→0.92。
- 业界共识：单一起点困惑度/统计值硬判断不可靠（GPTZero 已弃用），应多信号融合 + 重构负样本 + 低误报三态。

## 最终优化（利用已下载文献库构建真实人类负样本）

### 数据增强（关键突破）
- 从 cnki-hub 已下载文献库（2300篇学术PDF）**批量提取真实人类学术句子**
- 过滤去噪（去引用上标/图表/摘要/参考文献）后得 **60425 句**，去重 57809，抽样 9000 进训练
- 合并人类负样本：cnki学术语料 + real_thesis(1397) + train_unified人类句，**共 13121 句人类候选**
- **平衡采样 human:AI ≈ 2.3:1**（关键：避免过平衡导致对AI过度宽容）

### 最终模型性能（cnki增强 + 2.3:1平衡）
- 准确率 **~88-94%**，**F1 0.82-0.90**，**AUC 0.94-0.97**
- 三场景验证（固化模型）：
  - 知网判高度AI段落：AI概率 50.6%，2/4句判AI
  - 真实人类论文《企业财务风险管理》：AI概率 5.1%，仅1.3%句误判
  - AI套话：39.7%

### 关键经验
- **真实人类学术负样本是降误判的关键**：用已下载的学术文献库构建 13000+ 句人类语料，显著提升泛化。
- **平衡比例是精度-召回权衡的核心**：一味加人类样本会过平衡（全判人类）；2.3:1 达到最佳平衡。
- 数据文件：`data/human_cnki.jsonl`（cnki采样）、`data/human_corpus.jsonl`（全量77261句）、`data/human_positive.jsonl`（real_thesis）。

## 方向3：中文RoBERTa深度信号 + 双流融合（2026-09 完成）

### WSL 环境搭建
- 修复 WSL 网络（.wslconfig 从 mirrored 改 nat，重启后 eth0 + DNS + pypi 恢复）
- WSL Ubuntu 装 torch 2.13.0+cu130 + transformers 4.48.3 + datasets（清华源），GPU RTX 4060 Ti 可用
- **WSL 里 Trainer import 正常**（Windows 上 transformers5.16 Trainer 挂死，WSL 4.48 干净）

### 深度流模型（微调中文RoBERTa）
- 基座：Hello-SimpleAI/chatgpt-detector-roberta-chinese（102M）
- 用标注数据微调（AI样本 + cnki文献库人类负样本，平衡2.4:1）
- 微调后：**AUC 0.946**，acc 0.86
- 模型存至 models/roberta_ft/

### 双流智能加权融合
- 统计流(TF-IDF, AUC 0.94) + 深度流(RoBERTa, AUC 0.95) 融合
- 融合策略：两流一致→均值；**分歧时偏向统计流**（统计流用7w+文献库人类语料训练，对真实人类更准）

### 三场景实测（融合后）
| 场景 | 综合AI概率 | 判定 |
|---|---|---|
| 真实人类论文 | 16.9% | ✅ 基本人类（误判1.3%） |
| AI套话 | 43.6% | 🔶 疑似AI |
| 知网高度AI段落 | 54.0% | ⚠️ 高度疑似AI |

### 关键经验（方向3）
- 深度流对AI识别强（知网高度AI段落 74.3%），但对规范学术句偏敏感；统计流对真实人类更准——**双流互补是核心**。
- **分歧时偏向统计流**是平衡关键（统计流人类负样本训练更充分）。
- WSL 是干净理想的 GPU 训练环境（避开 Windows 依赖冲突），只需配置好网络。


## 服务化架构（常驻 + skill + 自动闭环 2026-09）

### 形态
- **常驻 FastAPI 服务**（`scripts/server.py`）+ **DSH skill**（`skills/aigc-detector`，已注册）双形态。
- 独立常驻进程：`start_service.bat`（VBS 开机自启），服务端口自动切换 + `last_port.txt` 记录实际端口（智能体调用前必读）。
- 单实例 PID 锁（`service.pid`）防多实例竞争；深度流后台异步加载（5秒就绪，统计流先用，深度流就绪热切换）。

### 接口
- `POST /detect`：稿件检测（含对齐报告参数 + `verdict` 档位判定）
- `POST /detect_batch`：批量检测多篇稿件（`paths`/`texts`）
- `POST /ingest`：接收AIGC报告→入库→自动触发重训
- `POST /train`：触发重训（互斥锁保护，防并发）
- `GET /health`：健康检查 + 模型状态

### 日志（实时终端窗口 + logs/service.log 双通道）
每次请求记录：时间/接口/路径/字符数/句数/耗时(统计流+深流各耗时)/两流均值/融合值/AI句数/三态/逐句高AI句/命中模板。
入库记录报告参数：平台/AI特征值/总字符/AI字符/报告编号/篇名/作者。

### 报告参数对齐（对应真实AIGC报告口径）
检测/入库输出对齐：`AI特征值`(ai_feature_rate)、`总字符数`(total_chars)、`AI特征字符数`(ai_chars)、`句子数`(n_sentences)、`报告编号`、`篇名`、`作者`、`平台`。
`verdict` 对照官方档位：<40%人类创作 / 40~60%疑似AI / ≥60%AI生成。

### 写作流程自动质检
`scripts/workflow_quality_check.py <稿件> [--threshold 0.35]`——稿件生产中/交付前自动调用检测服务，AI概率超阈值输出高风险段落+命中模板+改写建议。exit 0=通过，2=不通过。已接入写作流程。

### 自动闭环
`/ingest` 发现新AIGC报告 → 入库解析(句级标注) → 自动触发重训(统计流+WSL深流) → 热更新模型(无需重启)。幂等（去重）。

### 本次 bug 修复（详见 AGENTS.md 跳出固有思维规则）
单实例锁 / 深度流异步加载容错 / bat 绝对python路径 / 端口自动切换+last_port / 重训互斥锁 / GPU内存安全(动态batch+empty_cache) / 报告参数对齐 / docx 遍历段落+表格 / 字符级分句器(引号内句号·省略号·连续标点) / --target 参数脱节 / 边界请求(无效路径·非JSON)异常处理。

## 调用示例（智能体）
```python
import requests, os
port = int(open(r"C:\Users\woshi\.dsh\aigc-detector\last_port.txt").read().strip())
BASE = f"http://127.0.0.1:{port}"
r = requests.post(f"{BASE}/detect", json={"path": r"论文.docx"}, timeout=120)
# result: overall_ai_prob/state/verdict/top_ai_sentences/...
