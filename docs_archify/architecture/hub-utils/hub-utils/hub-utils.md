# Hub 交互工具（hub-utils）

> 本文是 `hub-utils` 域下的叶子子系统文档。域级总览见 `../hub-utils.md`。
>
> 源码基准：`transformers` v5.18.0.dev0，commit `5474a55e920f358d8382f3ecd3377edca979baa`。

## 1. 功能清单

| 能力 | 说明 | 源码路径 |
|------|------|----------|
| `cached_file()` | 从 Hub 下载/定位单个文件（带缓存+锁） | `src/transformers/utils/hub.py:239` |
| `cached_files()` | 批量下载多个文件 | `hub.py:299` |
| `push_to_hub` | 上传模型/分词器到 Hub（PushToHubMixin） | `utils/` + 各 Mixin |
| 缓存目录管理 | `HF_HOME`/`HUGGINGFACE_HUB_CACHE` 环境变量控制缓存路径 | `utils/constants.py` |
| 认证 | `HfFolder` token 管理、`hf auth login` 缓存 | `hub.py` |
| 离线模式 | `HF_HUB_OFFLINE`/`TRANSFORMERS_OFFLINE` 环境变量 | `hub.py` |
| 私有仓库 | token 鉴权访问 gated/private repo | `hub.py` |
| commit hash 记录 | `_commit_hash` 贯穿加载链 | `hub.py` |
| `ModelCard` | 模型卡片生成与 YAML front matter | `src/transformers/modelcard.py` |
| `get_task()` | 从 Hub 模型 README 推断任务名 | `hub.py` |

## 2. 核心类型

| 类型 | 位置 | 职责 |
|------|------|------|
| `cached_file()` | `hub.py:239` | 统一文件下载入口（缓存+锁+鉴权+离线降级） |
| `PushToHubMixin` | `utils/` | 上传 Mixin（Model/Tokenizer/Config/Pipeline 继承） |
| `ModelCard` | `modelcard.py` | 模型卡片 dataclass |
| `extract_commit_hash()` | `hub.py` | 从 cached 文件提取 commit hash |

## 3. 关键调用链

### 3.1 从 Hub 加载模型
1. `from_pretrained("model_id")` → `cached_file(model_id, CONFIG_NAME)` 下载 config.json；
2. 提取 `_commit_hash`；
3. 下载权重文件（`pytorch_model.bin`/`model.safetensors`）；
4. 离线模式（`HF_HUB_OFFLINE=1`）跳过网络，仅用本地缓存。

### 3.2 推送到 Hub
1. `model.push_to_hub(repo_id)` → 创建/克隆 repo；
2. 上传文件（config/权重/tokenizer）；
3. 提交 commit。

## 4-9. 概要

- **配置**：`HF_HOME`（缓存根目录）、`HF_HUB_OFFLINE`/`TRANSFORMERS_OFFLINE`（离线）、`HF_TOKEN`（认证）、`revision`（版本）。
- **错误**：网络失败时离线降级；gated repo 无 token 抛 `GatedRepoError`。
- **In-Scope**：`utils/hub.py`、`modelcard.py`、`utils/constants.py`。
- **Out-of-Scope**：HF Hub 远端服务、huggingface_hub 库（实际 HTTP 下载/上传）均"不在本仓库源码内"。
- **Python 适配口径**：按 Hub 交互 capability seam 分组；图以 architecture + sequence（下载流程）为主。

## 10. 图表清单与质量

| 图 | 文件 | 类型 | 质量档 |
|----|------|------|--------|
| Hub 交互架构图 | `hub-utils-architecture.html` | architecture | standard |
| 文件下载时序图 | `hub-utils-download-sequence.html` | sequence | standard |
