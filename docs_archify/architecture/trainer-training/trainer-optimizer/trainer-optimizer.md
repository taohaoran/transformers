# trainer-optimizer（优化器与学习率调度）

> 本文是 `trainer-training` 域下的叶子子系统文档。域级总览见 `../trainer-training.md`。
> 本文只展开**优化器创建、学习率调度器、Seq2SeqTrainer 扩展与超参搜索**，不重复展开训练循环（见 `../trainer-core/`）。
>
> 源码基准：`transformers` main 分支，commit `5474a55e920f358d8382f3ecd3377edca979baa`。

## 1. 功能清单

| 能力 | 说明 | 源码路径 |
|------|------|----------|
| `create_optimizer()` | 优化器工厂：按 `optim` 参数分发到 15+ 种优化器实现 | `src/transformers/trainer.py:1229` |
| `create_scheduler()` | 学习率调度器工厂：按 `lr_scheduler_type` 选择调度策略 | `src/transformers/trainer.py:1305` |
| `_get_adamw_torch()` | PyTorch AdamW 优化器创建 | `src/transformers/trainer_optimizer.py:201` |
| `_get_adafactor()` | Adafactor 优化器（显存高效） | `src/transformers/trainer_optimizer.py:195` |
| `_get_sgd()` / `_get_adagrad()` / `_get_rmsprop()` | 经典优化器 | `trainer_optimizer.py:321/333/343` |
| `_get_galore_optimizer()` | GaLore（梯度低秩投影）优化器 | `trainer_optimizer.py:355` |
| `_get_apollo_optimizer()` | Apollo 优化器 | `trainer_optimizer.py:388` |
| `_get_lomo_optimizer()` | LOMO（MeZO 风格）优化器 | `trainer_optimizer.py:417` |
| `get_linear_schedule_with_warmup` | 线性 warmup + 线性衰减 | `src/transformers/optimization.py:107` |
| `get_cosine_schedule_with_warmup` | 余弦退火 + warmup | `optimization.py:143` |
| `get_cosine_with_hard_restarts_schedule_with_warmup` | 硬重启余弦退火 | `optimization.py:188` |
| `get_polynomial_decay_schedule_with_warmup` | 多项式衰减 | `optimization.py:242` |
| `get_constant_schedule_with_warmup` | 常数学习率 + warmup | `optimization.py:80` |
| `get_inverse_sqrt_schedule` | 逆平方根衰减（T5 风格） | `optimization.py:296` |
| `get_wsd_schedule` | Warmup-Stable-Decay 调度 | `optimization.py:508` |
| `get_scheduler()` | 统一调度器入口（枚举分发） | `optimization.py:960` |
| `Seq2SeqTrainer` | 序列到序列训练器：eval 时生成 + 指标计算 | `src/transformers/trainer_seq2seq.py:55` |
| 超参搜索集成 | Optuna / Ray Tune / SigOpt / Wandb 后端 | `src/transformers/hyperparameter_search.py` |

## 2. 核心类型与接口清单

| 类型 / 函数 | 位置 | 职责 |
|-------------|------|------|
| `OptimizerContext` | `trainer_optimizer.py:55` | 优化器创建上下文（模型、参数组、args） |
| `Trainer.create_optimizer()` | `trainer.py:1229` | 优化器工厂主入口；参数分组（weight_decay 排除 bias/LayerNorm） |
| `Trainer.create_scheduler()` | `trainer.py:1305` | 调度器工厂；计算 warmup_steps |
| `get_scheduler()` | `optimization.py:960` | 统一调度器枚举分发 |
| `Seq2SeqTrainer` | `trainer_seq2seq.py:55` | 继承 Trainer；`evaluate()` 调用 `model.generate()` 计算 ROUGE/BLEU |
| `BestRun` | `trainer_utils.py:410` | 超参搜索最佳结果 NamedTuple |

## 3. 关键调用链

### 3.1 优化器创建（`trainer.py:1229-1305`）

1. **参数分组**：`get_decay_parameter_names()` 收集不需要权重衰减的参数（bias、LayerNorm.weight）。
2. **优化器选择**：按 `args.optim` 枚举分发到 `trainer_optimizer.py` 中的 `_get_*` 函数。
3. **特殊优化器**：GaLore/Apollo/LOMO 有独立的参数组处理逻辑。

### 3.2 学习率调度（`trainer.py:1305-1408`）

1. **warmup 计算**：`num_warmup_steps = args.warmup_steps` 或 `int(args.warmup_ratio * num_training_steps)`。
2. **调度器选择**：按 `args.lr_scheduler_type` 枚举分发到 `optimization.py` 中的 `get_*_schedule` 函数。
3. **调度器步进**：训练循环中每步 `lr_scheduler.step()`。

### 3.3 Seq2SeqTrainer.evaluate()（`trainer_seq2seq.py:139`）

1. 调用 `super().evaluate()` 获取预测。
2. 若 `predict_with_generate=True`：调用 `model.generate()` 生成序列。
3. 计算 ROUGE/BLEU 指标（由 `compute_metrics` 函数提供）。

## 4. 配置项

| 配置项 | 默认 / 行为 | 位置 |
|--------|-------------|------|
| `optim` | `"adamw_torch"`；可选 adafactor/sgd/adagrad/rmsprop/galore/apollo/lomo 等 | `training_args.py` |
| `learning_rate` | `5e-5`；AdamW 初始学习率 | `training_args.py` |
| `weight_decay` | `0.0`；权重衰减系数 | `training_args.py` |
| `adam_beta1` / `adam_beta2` | `0.9` / `0.999`；Adam 动量系数 | `training_args.py` |
| `max_grad_norm` | `1.0`；梯度裁剪范数 | `training_args.py` |
| `lr_scheduler_type` | `"linear"`；可选 constant/cosine/cosine_with_restarts/polynomial 等 | `training_args.py` |
| `warmup_ratio` / `warmup_steps` | `0.0` / `0`；学习率 warmup 比例或步数 | `training_args.py` |
| `num_cycles` | 余弦重启次数 | `training_args.py` |
| `power` | 多项式衰减指数 | `training_args.py` |
| `predict_with_generate` | `False`；Seq2Seq eval 时是否生成 | `training_args_seq2seq.py` |

## 5. 错误与重试语义

- **不支持的优化器**：`optim` 枚举值不匹配时抛 `ValueError`。
- **不支持的调度器**：`lr_scheduler_type` 不匹配时抛 `ValueError`。
- **低秩优化器**：GaLore/Apollo 需要额外配置，参数不满足时抛错。
- **无重试**：优化器/调度器创建是一次性操作，失败直接终止。

## 6. 并发细节

- **参数分组**：`create_optimizer()` 按参数名模式分组（decay vs no_decay），不同组可有不同 weight_decay。
- **分布式优化器**：DeepSpeed/FSDP 由 Accelerate 封装优化器创建（`deepspeed_init` / `delay_optimizer_creation`）。
- **无多线程**：优化器和调度器均为单线程同步操作。

## 7. 系统边界

**In-Scope（本仓库源码内）**
- `src/transformers/trainer_optimizer.py`：优化器工厂
- `src/transformers/optimization.py`：调度器实现
- `src/transformers/trainer_seq2seq.py`：Seq2SeqTrainer
- `src/transformers/hyperparameter_search.py`：超参搜索

**Out-of-Scope（不在本仓库源码内）**
- PyTorch 优化器/调度器实现
- Optuna / Ray Tune / SigOpt / Wandb 超参搜索框架——不在本仓库源码内
- bitsandbytes / torchao 等外部优化器库

## 8. 与相邻子系统交互

- **上游 → 本叶子**：`Trainer._prepare_for_training()` 调用 `create_optimizer()` + `create_scheduler()`。
- **本叶子 → 下游**：
  - → `training_step()` 中 `optimizer.step()` / `scheduler.step()`
  - → Seq2SeqTrainer → `model.generate()`（见 generation-loop 叶子）

## 9. 语言专项适配口径

- Python 库，按"优化器工厂 / 调度器工厂 / Seq2Seq 扩展 / 超参搜索"capability seam 划分。
- 图类型：architecture（优化器-调度器组件关系）；sequence/lifecycle 不适用（创建是一次性工厂模式）。

## 10. 图表清单与质量

| 图 | 文件 | 类型 | archify 质量档 |
|----|------|------|----------------|
| 优化器与调度器架构图 | `trainer-optimizer-architecture.html` | architecture | standard（降档披露） |

JSON IR 源文件位于 `json/` 目录。sequence/lifecycle 不产出：优化器/调度器创建是一次性工厂调用，无状态机或时序语义。
