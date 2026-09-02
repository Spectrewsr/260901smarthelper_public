# 常州产业链招商助手（本地演示版）

这是一个可在本机运行的网页版 Demo：用常州公开企业资料检索潜在供应商、客户和合作方，并把每次研判展示为可回溯的企业卡片与来源链接。

页面包含两个工作区：

- `招商咨询`：输入行业、企业或招商需求，生成行业概况、潜在匹配、产业链、区域配套、招商建议和使用边界。
- `企业资料库`：按区县、产业赛道、产品和能力浏览资料，并打开每家企业的完整公开信息。

## 启动

本 Demo 使用 Conda 环境 `changzhou-rag-demo`。在 PowerShell 中进入此目录后运行：

```powershell
.\run_demo.ps1
```

脚本会定位该 Conda 环境；若网页核心依赖缺失，会安装 `requirements.txt`。启动成功后访问 [http://127.0.0.1:8000](http://127.0.0.1:8000)。按 `Ctrl+C` 停止服务。

首次使用可直接根据仓库中的环境文件创建环境：

```powershell
conda env create -f environment.yaml
conda activate changzhou-rag-demo
```

也可以手动启动：

```powershell
conda activate changzhou-rag-demo
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
```

## 数据与本地检索

企业资料应放在 `data/raw/`，其中至少包含：

```text
companies.csv
sources.csv
regional_assets.jsonl
```

先校验资料，再构建检索索引：

```powershell
conda activate changzhou-rag-demo
python scripts\validate_data.py --data-dir data
python scripts\build_index.py --data-dir data
```

即使还没有下载 Embedding 模型，Demo 也能运行：它会使用本地确定性的词汇检索作为后备，并在页面状态中说明当前检索方式。

若要启用已定下的 `BAAI/bge-m3`，先根据本机 CUDA/PyTorch 兼容性安装 GPU 版 PyTorch，再安装可选依赖、下载模型并构建索引：

```powershell
conda activate changzhou-rag-demo
python -m pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu126
python -m pip install -r requirements-embedding.txt
python -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='BAAI/bge-m3', local_dir='models/bge-m3', ignore_patterns=['onnx/*', 'README.md', 'imgs/*', 'long.jpg', '.gitattributes'])"
python scripts\build_index.py --data-dir data --embedding-backend bge-m3 --model-path models\bge-m3 --device cuda
```

为适配 6 GB 显存的 RTX 3050，索引构建默认使用批大小 4、最大长度 512。生成的模型和索引保存在 `models/`、`data/derived/`，不会进入 Git。

## 可选：DeepSeek V4 Flash 叙述生成

没有模型密钥时，系统始终会生成本地、规则化且可追溯的报告。配置一个 OpenAI Chat Completions 兼容服务后，模型只会润色本次检索到的证据叙述；企业清单、来源卡片和匹配标签仍由本地数据决定。

复制 `.env.example` 为 `.env`，再填写服务信息：

```dotenv
OPENCODEGO_API_KEY=你的密钥
OPENCODEGO_BASE_URL=服务方提供的兼容接口根地址
OPENCODEGO_MODEL=deepseek-v4-flash
```

如希望密钥保留在一个本地文件而不写入 `.env`，可改为：

```dotenv
OPENCODEGO_KEY_FILE=相对或绝对的本地密钥文件路径
OPENCODEGO_BASE_URL=服务方提供的兼容接口根地址
OPENCODEGO_MODEL=deepseek-v4-flash
```

密钥文件内容可以是单个密钥、`OPENCODEGO_API_KEY=...` 一行，或首行是独立密钥、后面附使用说明的本地笔记。该文件仅由后端在启动时读取，不会发送给浏览器、写入日志或加入 Git。请勿把密钥写进前端 JavaScript、CSV 或公开资料文件。

## 资料使用边界

- 企业信息应来自可公开访问或已获授权的资料；每条企业记录至少保留一个来源链接。
- 页面中的“潜在供应商 / 客户 / 合作方”只是基于产品、能力和产业链标签的初步线索，不代表企业间已有合作关系。
- 缺少证据时，Demo 不会补全注册资本、认证、产能、驾车时间或实时工商状态。
- 需要更新资料时，替换 `data/raw/` 中的文件并重新构建索引即可；网页服务会在下次请求时重新读取原始资料。
