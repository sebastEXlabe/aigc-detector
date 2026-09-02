# 贡献指南 (Contributing)

感谢你愿意为 AIGC Detector 贡献代码！

## 开发流程

1. **Fork** 仓库并创建你的特性分支：`git checkout -b my-feature`
2. 这里的核心是**尊重现有的架构**：双流融合（统计流 `detector/` + 深度流 RoBERTa），服务化（`scripts/server.py`）。
3. 提交你的改动，遵循清晰的提交信息。
4. 推送分支并提交 Pull Request。

## 开发要点

- **保持可测试**：新功能请附带或更新 `tests/test_smoke.py` 中的冒烟测试。
- **不动大文件**：`models/*.pkl` 和 `models/roberta_ft/*.safetensors` 走 Git LFS，勿用普通 git 提交超大文件。
- **隐私红线**：`reports/`（AIGC 检测报告，含论文篇名/作者等）**严禁上传**，已由 `.gitignore` 排除；贡献时不要添加任何含个人信息的真实数据。
- **敏感信息**：不要提交任何 API key、账号、密码（`.credentials.yaml`、`*.key` 等已在 `.gitignore`）。

## 代码规范

- Python 3.10+，遵循 PEP8。
- 中文注释为主（项目面向中文论文场景）。
- 运行本地测试：`python tests/test_smoke.py`

## 提交信息

建议格式：`类型(范围): 简短描述`
- `fix(server): 修复端口切换`
- `feat(detector): 新增文体特征`
- `docs: 更新 README`

## 报告 Bug

请使用 `.github/ISSUE_TEMPLATE/bug_report.md` 模板，或直接运行 `python scripts/report_bug.py "描述"` 记录到本地 bug_tracker（项目内部使用）。
