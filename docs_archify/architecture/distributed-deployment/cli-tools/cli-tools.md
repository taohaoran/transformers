# 命令行工具（cli-tools）

> 本文是 `distributed-deployment` 域下的叶子子系统文档。域级总览见 `../distributed-deployment.md`。
>
> 源码基准：`transformers` v5.18.0.dev0，commit `5474a55e920f358d8382f3ecd3377edca979baa`。

## 1. 功能清单

| 能力 | 说明 | 源码路径 |
|------|------|----------|
| `transformers-cli` 入口 | argparse 子命令注册 | `src/transformers/cli/transformers.py:35`（`main()`） |
| `add-new-model-like` | 从现有模型模板生成新模型骨架 | `cli/add_new_model_like.py` |
| `chat` | 命令行对话，支持多模型 | `cli/chat.py` |
| `download` | 从 Hub 下载模型/分词器 | `cli/download.py` |
| `serve` | 启动推理服务（FastAPI + OpenAI 兼容 API） | `cli/serve.py:41`（`Serve` 类） |
| `system` | 系统信息/环境检查 | `cli/system.py` |
| serving 子目录 | FastAPI 路由：`/v1/chat/completions`、`/v1/completions`、`/load_model` | `cli/serving/server.py:44` |
| serving model_manager | 模型加载与推理端点管理 | `cli/serving/model_manager.py` |

## 2. 核心类型

| 类型 | 位置 | 职责 |
|------|------|------|
| `Serve` | `cli/serve.py:41` | serve 子命令主类 |
| `build_server()` | `cli/serving/server.py:44` | FastAPI app 工厂 |
| `ReasoningMode` 枚举 | `cli/serve.py:35` | 推理模式 |
| `main()` | `cli/transformers.py:35` | CLI 入口，argparse 分发 |

## 3. 关键调用链

### 3.1 CLI 分发
1. `transformers-cli <subcommand>` → `main()` 解析 argparse；
2. 子命令 dispatch 到对应模块（chat/download/serve/system/add_new_model_like）。

### 3.2 serve 请求流
1. `build_server()` 创建 FastAPI app，注册 `/v1/chat/completions` 等端点；
2. 客户端 POST 请求 → lifespan 加载模型 → model_manager 推理；
3. 返回 OpenAI 兼容 JSON 响应。

## 4-9. 概要

- **配置**：`--port`/`--host`（serve）、`--model`（chat/serve）、`--revision`（download）。
- **错误**：模型未加载返回 HTTP 错误；参数校验由 argparse 处理。
- **In-Scope**：`cli/` 包。
- **Out-of-Scope**：FastAPI/Uvicorn 均"不在本仓库源码内"。
- **Python 适配口径**：按子命令分组；图以 architecture + dataflow（serve 请求流）为主。

## 10. 图表清单与质量

| 图 | 文件 | 类型 | 质量档 |
|----|------|------|--------|
| CLI 工具架构图 | `cli-tools-architecture.html` | architecture | standard |
| serve 请求数据流 | `cli-tools-serve-dataflow.html` | dataflow | standard |
