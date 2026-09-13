# 框架集成（integrations-frameworks）

> 本文是 `distributed-deployment` 域下的叶子子系统文档。域级总览见 `../distributed-deployment.md`。
>
> 源码基准：`transformers` v5.18.0.dev0，commit `5474a55e920f358d8382f3ecd3377edca979baa`。

## 1. 功能清单

| 族 | 代表文件 | 说明 | 源码路径 |
|----|----------|------|----------|
| 训练框架 | `deepspeed.py` | DeepSpeed 集成：`HfDeepSpeedConfig`、`deepspeed_init`、ZeRO-3 权重加载 | `src/transformers/integrations/deepspeed.py` |
| 训练框架 | `fsdp.py` | PyTorch FSDP 集成（与 distributed-core 配合） | `integrations/fsdp.py` |
| 训练框架 | `accelerate.py` | accelerate 集成（device_map 自动分片） | `integrations/accelerate.py` |
| 微调 | `peft.py` | PEFT/LoRA 集成：`PeftAdapterMixin`、`maybe_load_adapters` | `integrations/peft.py:57/662` |
| 硬件/后端 | `flash_attention.py` | Flash Attention 前向替换 | `integrations/flash_attention.py:26` |
| 硬件/后端 | `tpu.py`、`neuron.py`、`openvino.py`、`onnxruntime.py`、`torchao.py` | 各硬件后端适配 | `integrations/` |
| 实验追踪 | `wandb.py`、`comet_ml.py`、`mlflow.py`、`tensorboard.py`、`clearml.py`、`codecarbon.py`、`dagshub.py` | 训练日志回调 | `integrations/integration_utils.py:104-193` |
| 超参搜索 | `optuna.py`、`ray_tune.py`、`sigopt.py` | HPO 集成 | `integrations/` |
| 统一检测 | `integration_utils.py` | 50+ 个 `is_*_available()` 检测函数 | `integrations/integration_utils.py` |
| 注意力后端 | `sdpa_attention.py`、`flex_attention.py`、`flash_paged.py`、`eager_paged.py` | 注意力实现切换 | `integrations/` |

**深读覆盖**：深读 `deepspeed.py`（HfDeepSpeedConfig + deepspeed_init）、`peft.py`（PeftAdapterMixin）、`flash_attention.py`（flash_attention_forward）、`integration_utils.py`（is_*_available 检测体系）；其余 50+ 文件按族列表说明。

## 2. 核心类型与接口清单

| 类型 | 位置 | 职责 |
|------|------|------|
| `HfDeepSpeedConfig(DeepSpeedConfig)` | `deepspeed.py:57` | transformers 侧 DeepSpeed 配置封装 |
| `deepspeed_init(trainer, ...)` | `deepspeed.py:585` | Trainer 初始化 DeepSpeed 引擎 |
| `PeftAdapterMixin` | `peft.py:57` | PEFT adapter 混入基类 |
| `maybe_load_adapters()` | `peft.py:662` | from_pretrained 时自动加载 LoRA adapter |
| `flash_attention_forward()` | `flash_attention.py:26` | Flash Attention 前向替换函数 |
| `is_*_available()` 族 | `integration_utils.py:104-193` | 50+ 个可选依赖检测函数 |
| `TrainerCallback` 子类 | 各追踪文件 | Wandb/TensorBoard/MLflow 等日志回调 |

## 3. 关键调用链

### 3.1 DeepSpeed 训练初始化
1. Trainer 检测 `args.deepspeed` 配置；
2. `HfDeepSpeedConfig` 解析 JSON 配置文件；
3. `deepspeed_init(trainer, num_training_steps)` 构造 DeepSpeed engine（模型+优化器+LR scheduler）；
4. ZeRO-3 模式下 `_load_state_dict_into_zero3_model` 分片加载权重。

### 3.2 PEFT adapter 加载
1. `from_pretrained(model_id)` 检测目录下是否有 `adapter_config.json`；
2. `maybe_load_adapters` 调 PEFT 库加载 LoRA 权重；
3. `PeftAdapterMixin` 提供 `load_adapter/unload/set_adapter` 接口。

### 3.3 集成检测模式
- 每个集成文件顶部定义 `is_xxx_available()`（try import + 版本检查）；
- 主代码在使用前调用检测函数，不可用时优雅降级（warning 而非 error）。

## 4. 配置项

| 配置 | 行为 | 位置 |
|------|------|------|
| `--deepspeed`（TrainingArguments） | DeepSpeed JSON 配置路径 | `deepspeed.py` |
| `load_in_4bit`/`load_in_8bit` | bitsandbytes 量化加载 | `integrations/bitsandbytes.py` |
| `attn_implementation` | `"flash_attention_2"`/`"sdpa"`/`"eager"` | `flash_attention.py` |
| `WANDB_DISABLED` | 禁用 Wandb | `integration_utils.py` |

## 5-9. 概要

- **错误语义**：可选依赖未安装时 warning + 降级；必需依赖缺失时 ImportError。
- **并发**：各集成独立；DeepSpeed 多进程训练由外部 `deepspeed` 启动器管理。
- **In-Scope**：`src/transformers/integrations/` 全部集成文件。
- **Out-of-Scope**：DeepSpeed/PEFT/Wandb/TensorBoard/Flash-Attention 本体均"不在本仓库源码内"。
- **Python 适配口径**：按 capability seam（训练/微调/硬件/追踪/HPO）分组；图以 architecture（集成生态图）为主。

## 10. 图表清单与质量

| 图 | 文件 | 类型 | 质量档 |
|----|------|------|--------|
| 集成生态架构图 | `integrations-frameworks-architecture.html` | architecture | standard |

**覆盖范围披露**：深读 4 个代表文件（deepspeed/peft/flash_attention/integration_utils）；其余 50+ 文件按族列表说明。
