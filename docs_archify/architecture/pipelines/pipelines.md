# pipelines 域总览

> 本域包含以下叶子子系统；各叶子详情见对应文档。
> 源码基准：`transformers` v5.18.0.dev0（main），commit `5474a55e920f358d8382f3ecd3377edca979baa`。

## 1. 域职责

`pipelines` 域实现 HuggingFace transformers 的**推理管线层**：将原始输入（文本/图像/音频）封装为"预处理→模型推理→后处理"三步统一接口，用户一行 `pipeline("task")` 即可完成推理。本域包含框架基类（`Pipeline`）、工厂函数（`pipeline()`）、任务注册表（`SUPPORTED_TASKS`）以及 24 个具体任务管线实现。

## 2. 叶子索引

| 叶子 | 文档 | 架构图 | 时序图/数据流图 | 职责一句话 |
|------|------|--------|-----------------|-----------|
| pipelines-framework | [pipelines-framework.md](pipelines-framework/pipelines-framework.md) | [架构图](pipelines-framework/pipelines-framework-architecture.html) | [工厂时序](pipelines-framework/pipelines-framework-factory-sequence.html) · [调用时序](pipelines-framework/pipelines-framework-call-sequence.html) | Pipeline 基类 + pipeline() 工厂 + 批量迭代器 |
| pipelines-tasks | [pipelines-tasks.md](pipelines-tasks/pipelines-tasks.md) | [架构图](pipelines-tasks/pipelines-tasks-architecture.html) | — | 24 个具体任务管线的三步实现 |

## 3. 域级机制细节

### 任务注册表（SUPPORTED_TASKS）
`SUPPORTED_TASKS`（`__init__.py:141`）是任务名→`{impl, pt, default, type}` 的静态映射，`TASK_ALIASES`（`:136`）提供别名（`sentiment-analysis→text-classification` 等）。`PipelineRegistry.check_task()` 运行时解析别名并返回 `targeted_task`。

### 三步推理抽象
所有管线继承 `Pipeline` 基类，实现三个抽象方法：
- `preprocess(input)`：原始输入 → dict[tensors]
- `_forward(tensors)`：模型推理 → ModelOutput
- `postprocess(outputs)`：ModelOutput → 人类可读结果

`Pipeline.__call__` 自动路由单条（`run_single`）或批量（`get_iterator` + DataLoader）。

### 批量推理迭代器
`pt_utils.py` 提供 `PipelineDataset`（Map-style）、`PipelineIterator`（反批次）、`PipelineChunkIterator`（长输入分块）、`PipelinePackIterator`（按 `is_last` 打包），支持惰性流式推理。
