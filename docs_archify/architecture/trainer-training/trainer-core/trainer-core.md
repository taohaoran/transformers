# trainer-core（Trainer 核心训练循环）

> 本文是 `trainer-training` 域下的叶子子系统文档。域级总览见 `../trainer-training.md`。
> 本文只展开 **Trainer 核心训练循环与回调系统**，不重复展开优化器/调度器创建（见 `../trainer-optimizer/`）、
> TrainingArguments 参数体系（见 `../training-args/`）、数据整理器（见 `../data-collators-loss/`）。
>
> 源码基准：`transformers` main 分支，commit `5474a55e920f358d8382f3ecd3377edca979baa`。

## 1. 功能清单

| 能力 | 说明 | 源码路径 |
|------|------|----------|
| `Trainer.train()` | 训练主入口：模型准备、checkpoint 恢复、调用 `_inner_training_loop` | `src/transformers/trainer.py:1408` |
| `_inner_training_loop()` | 核心训练循环：epoch 迭代、step 迭代、前向/反向/优化器步进 | `src/transformers/trainer.py:1527` |
| `training_step()` | 单步训练：前向计算 loss → `accelerator.backward(loss)` | `src/transformers/trainer.py:1975` |
| `_maybe_log_save_evaluate()` | 按 IntervalStrategy 触发日志/保存/评估 | `src/transformers/trainer.py:2156` |
| `evaluate()` / `predict()` | 评估与预测流程 | `src/transformers/trainer.py:2688` / `:2994` |
| `CallbackHandler` | 回调分发器：按训练阶段依次调用所有已注册回调 | `src/transformers/trainer_callback.py:429` |
| `TrainerCallback` | 回调基类：`on_train_begin`/`on_step_end`/`on_evaluate` 等 15+ 钩子 | `src/transformers/trainer_callback.py:295` |
| `TrainerState` | 训练状态：global_step、epoch、best_metric、log_history | `src/transformers/trainer_callback.py:35` |
| `TrainerControl` | 训练控制：`should_training_stop`/`should_epoch_stop` 等布尔标志 | `src/transformers/trainer_callback.py:234` |
| `IntervalStrategy` | 日志/评估策略枚举：`NO`/`STEPS`/`EPOCH` | `src/transformers/trainer_utils.py:389` |
| 分布式集成 | Accelerate 封装 DDP/FSDP/DeepSpeed/TPU | `src/transformers/trainer.py`（多处分支） |

## 2. 核心类型与接口清单

| 类型 / 函数 | 位置 | 职责 |
|-------------|------|------|
| `Trainer` | `trainer.py:259` | 核心训练类；组合 model/args/callbacks/optimizer/scheduler/accelerator |
| `TrainerCallback` | `trainer_callback.py:295` | 回调接口；子类覆盖各 `on_*` 钩子 |
| `CallbackHandler` | `trainer_callback.py:429` | 继承 TrainerCallback；遍历所有回调分发钩子调用 |
| `TrainerState` | `trainer_callback.py:35` | 训练状态数据类；序列化到 `trainer_state.json` |
| `TrainerControl` | `trainer_callback.py:234` | 控制流标志；回调可设置 `should_training_stop=True` 提前终止 |
| `IntervalStrategy` | `trainer_utils.py:389` | 枚举：日志/评估/保存的触发策略 |
| `SaveStrategy` | `trainer_utils.py:395` | 枚举：`NO`/`STEPS`/`EPOCH`/`BEST` |
| `EvalPrediction` | `trainer_utils.py:210` | 评估输出 NamedTuple：predictions + label_ids + metrics |

## 3. 关键调用链

### 3.1 `train()` → `_inner_training_loop()` 完整流程

1. **train() 准备**（`trainer.py:1408-1525`）：
   - a. 种子设置、梯度检查点启用、NEFTune hooks、调试选项。
   - b. checkpoint 恢复（模型权重 + TrainerState + 回调状态）。
   - c. `find_executable_batch_size()` 包装 `_inner_training_loop`（支持 auto_find_batch_size）。
2. **_inner_training_loop() 初始化**（`trainer.py:1527-1620`）：
   - a. `get_train_dataloader()` 获取数据加载器。
   - b. `set_initial_training_values()` 计算 epochs/steps/batch_size。
   - c. `_init_training_state()` 初始化 TrainerState。
   - d. `_prepare_for_training()`：创建 optimizer/scheduler，`accelerator.prepare()` 封装模型。
3. **Epoch 循环**（`trainer.py:1608-1617`）：
   - a. `callback_handler.on_epoch_begin()`。
   - b. `_run_epoch()`：遍历 dataloader，每步执行：
     - `callback_handler.on_step_begin()`
     - `training_step(model, inputs)`：前向 + `accelerator.backward(loss)`
     - 梯度累积达到 `gradient_accumulation_steps` 后：梯度裁剪 → optimizer.step() → scheduler.step() → zero_grad()
     - `callback_handler.on_step_end()`
     - `_maybe_log_save_evaluate()`：按策略触发日志/保存/评估
   - c. `callback_handler.on_epoch_end()`
4. **收尾**：`_finalize_training()` → `on_train_end()` → 返回 TrainOutput。

### 3.2 回调钩子序列

`on_train_begin` → [epoch_begin → step_begin → training_step → step_end → (log/save/eval)] × N → epoch_end] × N → train_end

## 4. 配置项

| 配置项 | 默认 / 行为 | 位置 |
|--------|-------------|------|
| `per_device_train_batch_size` | 每设备训练 batch 大小 | `training_args.py` |
| `gradient_accumulation_steps` | 梯度累积步数 | `training_args.py` |
| `max_steps` / `num_train_epochs` | 训练总步数 / 总轮数 | `training_args.py` |
| `logging_steps` / `logging_strategy` | 日志频率（IntervalStrategy） | `training_args.py` |
| `evaluation_strategy` / `eval_steps` | 评估触发策略 | `training_args.py` |
| `save_strategy` / `save_steps` | checkpoint 保存策略 | `training_args.py` |
| `save_total_limit` | 保留的最大 checkpoint 数 | `training_args.py` |
| `learning_rate` / `weight_decay` / `warmup_steps` | 优化器超参（见 trainer-optimizer） | `training_args.py` |
| `fp16` / `bf16` | 混合精度训练 | `training_args.py` |
| `gradient_checkpointing` | 梯度检查点（省显存） | `training_args.py` |
| `dataloader_drop_last` | 丢弃最后不完整 batch | `training_args.py` |

## 5. 错误与重试语义

- **OOM 自动重试**：`find_executable_batch_size()` 包装训练循环，OOM 时自动减半 batch 重试。
- **梯度裁剪**：`max_grad_norm` 触发时 `torch.nn.utils.clip_grad_norm_()`。
- **NaN/Inf 检测**：`DebugUnderflowOverflow` 调试选项检测 loss 异常。
- **checkpoint 恢复**：支持从 checkpoint 恢复模型/优化器/scheduler/训练状态，`ignore_data_skip=False` 时快进 dataloader。
- **异常传播**：训练中异常直接传播；`handle_batch_error` 钩子可在评估中处理单 batch 错误。

## 6. 并发细节

- **分布式训练**：通过 Accelerate 抽象 DDP/FSDP/DeepSpeed/TPU；`accelerator.backward(loss)` 处理分布式梯度同步。
- **梯度累积**：`gradient_accumulation_steps` 步内不执行 optimizer.step，仅累积梯度。
- **混合精度**：GradScaler 由 Accelerate 管理；`fp16`/`bf16` 自动切换。
- **上下文并行**：`_prepare_context_parallel_inputs()` 支持 Context Parallelism。
- **无显式线程**：训练循环单线程同步执行；多 GPU 通过分布式后端并行。

## 7. 系统边界

**In-Scope（本仓库源码内）**
- `src/transformers/trainer.py`：Trainer 核心类
- `src/transformers/trainer_callback.py`：回调系统
- `src/transformers/trainer_utils.py`：EvalPrediction/IntervalStrategy 等
- `src/transformers/trainer_pt_utils.py`：分布式采样器/设备工具

**Out-of-Scope（不在本仓库源码内）**
- Accelerate 库（分布式封装、GradScaler、prepare）——不在本仓库源码内
- DeepSpeed / FSDP / TorchXLA 后端——不在本仓库源码内
- 优化器/调度器实现细节（见 trainer-optimizer 叶子）

## 8. 与相邻子系统交互

- **上游 → 本叶子**：用户代码实例化 `Trainer(model, args, train_dataset, eval_dataset, tokenizer, data_collator)` 并调用 `.train()`。
- **本叶子 → 下游**：
  - → `training_step()` → 模型 `forward()` + `accelerator.backward()`
  - → `CallbackHandler` → 各 TrainerCallback 实现
  - → 优化器/scheduler（见 trainer-optimizer 叶子）
  - → 数据整理器（见 data-collators-loss 叶子）
  - → Accelerate 分布式后端

## 9. 语言专项适配口径

- Python 库，按"训练循环 / 回调系统 / 状态管理 / 分布式集成"capability seam 划分。
- 图类型：architecture（组件关系）+ lifecycle（训练循环状态机）+ sequence（单步训练时序）。
- 外部边界：Accelerate/DeepSpeed/FSDP 标注为外部组件。

## 10. 图表清单与质量

| 图 | 文件 | 类型 | archify 质量档 |
|----|------|------|----------------|
| Trainer 组件架构图 | `trainer-core-architecture.html` | architecture | standard（降档披露） |
| 训练循环状态机 | `trainer-core-lifecycle.html` | lifecycle | standard（降档披露） |
| 单步训练时序图 | `trainer-core-sequence.html` | sequence | showcase |

JSON IR 源文件位于 `json/` 目录。
