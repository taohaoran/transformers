# training-args（TrainingArguments 参数体系）

> 本文是 `trainer-training` 域下的叶子子系统文档。域级总览见 `../trainer-training.md`。
> 本文只展开 **TrainingArguments 参数体系与 HfArgumentParser 解析器**，不重复展开训练循环（见 `../trainer-core/`）。
>
> 源码基准：`transformers` main 分支，commit `5474a55e920f358d8382f3ecd3377edca979baa`。

## 1. 功能清单

| 能力 | 说明 | 源码路径 |
|------|------|----------|
| `TrainingArguments` | 训练参数数据类：100+ 参数控制训练行为 | `src/transformers/training_args.py:180` |
| `Seq2SeqTrainingArguments` | Seq2Seq 任务扩展参数 | `src/transformers/training_args_seq2seq.py:29` |
| `HfArgumentParser` | 从 dataclass 自动生成 argparse 解析器 | `src/transformers/hf_argparser.py:111` |
| 命令行解析 | `parse_args_into_dataclasses()` | `hf_argparser.py:272` |
| YAML/JSON 配置文件 | `parse_yaml_file()` / `parse_json_file()` | `hf_argparser.py:408/386` |
| 批量参数组 | 训练批次：per_device_train_batch_size × gradient_accumulation_steps | `training_args.py:195/260` |
| 学习率参数组 | learning_rate / weight_decay / warmup_ratio / lr_scheduler_type | `training_args.py:210/221` |
| 精度参数组 | fp16 / bf16 / tf32 | `training_args.py:282/285` |
| 分布式参数组 | deepspeed / fsdp / fsdp_config | `training_args.py:673` |
| 策略参数组 | logging_strategy / evaluation_strategy / save_strategy | `training_args.py:357/470` |

## 2. 核心类型与接口清单

| 类型 / 函数 | 位置 | 职责 |
|-------------|------|------|
| `TrainingArguments` | `training_args.py:180` | `@dataclass`；100+ 训练超参，带类型注解与默认值 |
| `Seq2SeqTrainingArguments` | `training_args_seq2seq.py:29` | 继承 TrainingArguments；增加 predict_with_generate/generation_max_length |
| `HfArgumentParser` | `hf_argparser.py:111` | 继承 argparse.ArgumentParser；从 dataclass 自动构建 CLI |
| `SchedulerType` | `training_args.py` | 学习率调度策略枚举 |
| `OptimizerNames` | `training_args.py` | 优化器名称枚举 |

## 3. 关键调用链

### 3.1 从命令行到 TrainingArguments

1. 用户脚本调用 `HfArgumentParser(TrainingArguments).parse_args_into_dataclasses()`。
2. `_add_dataclass_arguments()` 遍历 dataclass 字段，自动生成 `--per_device_train_batch_size` 等 CLI 参数。
3. 解析结果返回 `TrainingArguments` 实例。

### 3.2 从 YAML 配置文件加载

1. `HfArgumentParser.parse_yaml_file("config.yaml")` 读取 YAML 文件。
2. 映射到 dataclass 字段，类型校验。
3. 返回 TrainingArguments 实例。

## 4. 配置项

| 参数分组 | 关键参数 | 默认值 | 说明 |
|----------|----------|--------|------|
| 批次 | `per_device_train_batch_size` | `8` | 每设备训练 batch |
| | `gradient_accumulation_steps` | `1` | 梯度累积步数 |
| | `per_device_eval_batch_size` | `8` | 每设备评估 batch |
| 学习率 | `learning_rate` | `5e-5` | AdamW 初始学习率 |
| | `weight_decay` | `0.0` | 权重衰减 |
| | `warmup_ratio` / `warmup_steps` | `0.0` / `0` | warmup 比例/步数 |
| | `lr_scheduler_type` | `"linear"` | 调度策略 |
| | `max_grad_norm` | `1.0` | 梯度裁剪 |
| 精度 | `fp16` / `bf16` | `False` | 混合精度 |
| | `tf32` | `False` | TensorFloat-32 |
| 策略 | `logging_strategy` | `"steps"` | 日志策略 |
| | `evaluation_strategy` | `"no"` | 评估策略 |
| | `save_strategy` | `"steps"` | 保存策略 |
| | `save_total_limit` | `None` | 最大 checkpoint 数 |
| 分布式 | `deepspeed` | `None` | DeepSpeed 配置 |
| | `fsdp` | `False` | FSDP 启用 |
| | `local_rank` | `-1` | 本地 rank |
| 其他 | `output_dir` | `"output"` | 输出目录 |
| | `seed` | `42` | 随机种子 |
| | `max_steps` / `num_train_epochs` | `-1` / `3` | 训练步数/轮数 |

## 5. 错误与重试语义

- **参数冲突**：如 `save_strategy="best"` 需要 `load_best_model_at_end=True`。
- **类型校验**：HfArgumentParser 自动类型转换（str → int/float/bool/enum）。
- **不支持的组合**：如 `dataloader_drop_last` 与分布式采样器的兼容性检查。
- **无重试**：参数解析失败直接退出。

## 6. 并发细节

- **无并发**：参数解析是单线程同步操作。
- **数据类不可变**：`TrainingArguments` 是 `@dataclass(frozen=False)`，运行时可修改但不推荐。

## 7. 系统边界

**In-Scope（本仓库源码内）**
- `src/transformers/training_args.py`：TrainingArguments
- `src/transformers/training_args_seq2seq.py`：Seq2SeqTrainingArguments
- `src/transformers/hf_argparser.py`：HfArgumentParser

**Out-of-Scope（不在本仓库源码内）**
- argparse 标准库实现
- PyYAML 解析库

## 8. 与相邻子系统交互

- **上游 → 本叶子**：用户脚本创建 `TrainingArguments(...)` 并传入 `Trainer(...)`。
- **本叶子 → 下游**：
  - → Trainer 训练循环（见 trainer-core）：`args.per_device_train_batch_size`、`args.gradient_accumulation_steps`
  - → 优化器/调度器（见 trainer-optimizer）：`args.learning_rate`、`args.lr_scheduler_type`

## 9. 语言专项适配口径

- Python 库，按"参数定义 / 参数解析 / Seq2Seq 扩展"capability seam 划分。
- 图类型：architecture（参数体系与解析器关系）；sequence/lifecycle 不适用（参数是静态配置）。

## 10. 图表清单与质量

| 图 | 文件 | 类型 | archify 质量档 |
|----|------|------|----------------|
| TrainingArguments 体系架构图 | `training-args-architecture.html` | architecture | standard（降档披露） |

JSON IR 源文件位于 `json/` 目录。sequence/lifecycle 不产出：参数体系是静态数据类，无状态机或时序语义。
