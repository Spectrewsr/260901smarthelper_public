# 常州产业招商知识平台（本地可运行 Demo）

这是一个只绑定本机 `127.0.0.1` 的网页版演示系统。它以 100 家常州企业、3 个园区、26 条落地配套记录和 30 个可追溯公开来源为底座，提供招商研判、精确筛选、距离估算、局部产业关系图谱、园区对标、落地配套、材料导出、企业台账和数据录入能力。

本版已经实现知识库 PDF 中以下 7 个建设模块：统一身份认证、产业链智能匹配、园区对标与洽谈、本地化落地配套、招商材料与企业台账、单智能体与三类知识库、基础数据规范化处理。**不包含**国产化/内网安全基础适配，以及项目实施、培训和一年运维服务。

## 一分钟启动

在 PowerShell 进入本目录后运行：

```powershell
.\run_demo.ps1
```

打开 [http://127.0.0.1:8000](http://127.0.0.1:8000)。按 `Ctrl+C` 停止服务。

脚本使用 Conda 环境 `changzhou-rag-demo`，不使用仓库内的 `.venv`。新电脑可直接执行：

```powershell
conda env create -f environment.yaml
conda activate changzhou-rag-demo
```

## 完整 Advanced RAG 模型准备

核心流程不需要任何云端 API 或密钥。需要完整检索链时，将两个已下载模型放到下列目录；模型与生成索引均被 Git 忽略。

```text
models/
  bge-m3/
  bge-reranker-v2-m3/
```

模型可用 Hugging Face 下载：

```powershell
conda activate changzhou-rag-demo
python -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='BAAI/bge-m3', local_dir='models/bge-m3', ignore_patterns=['onnx/*', 'README.md', 'imgs/*', 'long.jpg', '.gitattributes'])"
python -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='BAAI/bge-reranker-v2-m3', local_dir='models/bge-reranker-v2-m3', ignore_patterns=['onnx/*', 'README.md', 'imgs/*', 'long.jpg', '.gitattributes'])"
python scripts\build_index.py --data-dir data --embedding-backend bge-m3 --model-path models\bge-m3 --device cuda --batch-size 2 --max-length 768
```

RTX 3050（6 GB）上已验证上述配置。没有本地模型时，系统会清楚标明检索/重排后备状态，绝不会把后备路径伪装成 BGE 或 Cross-Encoder。

## 查询架构

```text
自然语言问题
  └─ 轻量单智能体（仅允许固定工具计划）
       ├─ 招商研判：Contextual Retrieval + SQLite FTS5/BM25 + BGE-M3
       │             └─ RRF 融合 ──> BAAI Cross-Encoder 重排 ──> 可追溯证据卡
       ├─ 产业关系：局部 SQLite Knowledge Graph / 最多两跳 GraphRAG
       ├─ 精确筛选：参数化 SQL（不调用 RAG）
       └─ 距离查询：SQLite RTree + Haversine + 本地道路系数估算（不调用 RAG）
```

单智能体联动三类核心知识库：`企业与产业链`、`园区招商洽谈`、`本地化落地配套`。页面会显示每次查询实际走过的工具和检索轨迹。

## 页面与功能

- `研判工作台`：自动/Hybrid/SQL/地理/图谱/园区/配套/项目综合七类路由；结果带来源链接、置信边界和执行轨迹。
- `企业目录`：按区县、产业赛道和关键词查 100 家企业；核心标签涵盖产业、细分行业、产品、能力与产业链角色。
- `园区对标`：展示产业适配、绿色低碳公开基础、载体供给、政策与服务依据四维证据矩阵；唯一数值为“公开资料完备度”，不充当招商优先级；可生成独资设立洽谈要点和 `0–30 / 31–90 / 91–180 天` 分阶段参考方案。
- `落地配套`：按“所选区县 + 常州市全域通用资料”生成政务办事、生活配套、交通区位配套包，显示适用范围、来源和核验日期。7 个企业数据覆盖的区县/功能区均有三类属地资料。
- `企业台账`：管理员和招商专员两级入口；管理员可见完整演示联系人字段，招商专员只见脱敏后的字段。页面可查看详情、更新阶段/负责人/下一步，并查看事件历史。
- `数据管理`：管理员可先预检 UTF-8 CSV 的来源 ID、置信度、必填字段和大小限制，再发布有效行；也支持人工录入完整的角色、产品/能力、输入/输出、目标客户行业和备注字段。所有可检索/导出文本在入库前识别并脱敏手机号、邮箱、身份证号，标准化后自动生成基础标签与可检索知识块。
- `导出`：标准化靶向招商清单 CSV；三套预置材料模板（产业链靶向招商、独资设立园区洽谈、落地配套协同）的 DOCX/PPTX 导出。

本地演示账号：

| 角色 | 账号 | 密码 |
| --- | --- | --- |
| 招商专员 | `officer` | `officer-demo-2026` |
| 管理员 | `admin` | `admin-demo-2026` |

## 数据与使用边界

- 原始资料位于 `data/raw/`。每条企业、园区和配套记录都保存 `source_id`，来源元数据保存在 `sources.csv`。
- 企业坐标若没有可核验公开坐标，使用的是明确标识为 `district_reference` 的区县演示参考点，不是企业地址；“车程”是本地道路系数估算，不是实时导航。
- 图谱中的 `potential` / `inferred` 边仅表示公开标签推导的待核验关联，不代表实际交易或合作。
- 导出材料和页面结论均保留“潜在线索、需核验”的边界说明。
- `data/derived/`（SQLite、向量索引）和 `data/exports/`（下载结果）是本机生成物，不提交 Git。
- 正常更新 `data/raw/` 后重新启动时，原始资料会重建为新证据缓存，同时保留人工录入企业、台账、导入审核与导出审计；数据库不能一致性读取时会拒绝自动覆盖，保护已有记录。

先检查原始数据，再手动重建索引：

```powershell
python scripts\validate_data.py --data-dir data --min-companies 100
python scripts\build_index.py --data-dir data --embedding-backend bge-m3 --model-path models\bge-m3 --device cuda --batch-size 2 --max-length 768
```

## 验收测试

以下命令会在临时副本中重建真实 BGE-M3 索引、加载本地 Cross-Encoder，并执行 10 个端到端样例；不会污染 `data/raw/`：

```powershell
python -m unittest -v tests.test_data_pipeline tests.test_platform_e2e
```

10 个样例覆盖：两级角色/脱敏和台账更新历史、真实 Hybrid RAG、三知识库单智能体联动、SQL 精确筛选、地理 SQL、局部 GraphRAG、园区证据矩阵、7 个区县/功能区的三类配套包、三模板 DOCX/PPTX/CSV 导出、CSV 预检发布与人工录入，并回归检查动态向量增量、hash 后备标识、敏感信息清洗和刷新并发写入留存。逐项的 7 模块 70 问验收见 [SEVEN_MODULE_70_QUESTION_ACCEPTANCE.md](SEVEN_MODULE_70_QUESTION_ACCEPTANCE.md)。

## 目录说明

```text
app/
  services/knowledge.py     SQLite、FTS、RTree、图谱、台账、导入
  services/advanced_rag.py Contextual + Hybrid + RRF + Cross-Encoder
  services/agent.py        受限工具计划的轻量单智能体
  services/exports.py      CSV / DOCX / PPTX 三模板导出
data/raw/                  可追溯的原始公开资料
tests/test_platform_e2e.py 10 项端到端验收
environment.yaml           可复现的 Conda 环境
```
