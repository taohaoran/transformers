# trainer-training 域总览

> 本域包含以下叶子子系统；各叶子详情见对应文档。
> 源码基准：`transformers` main 分支，commit `5474a55e920f358d8382f3ecd3377edca979baa`。

## 1. 域职责

trainer-training 域实现了 HuggingFace transformers 的训练框架，提供从数据加载到模型保存的完整训练流水线。核心是 `Trainer` 类，它封装了训练循环、回调系统、分布式训练集成，并通过 `TrainingArguments` 提供 100+ 可配置参数。

## 2. 叶子索引

| 叶子 | 文档 | 架构图 | 时序图 / 状态机 | 职责一句话 |
|------|------|--------|----------------|-----------|
| trainer-core | [trainer-core.md](trainer-core/trainer-core.md) | [架构图](trainer-core/trainer-core-architecture.html) | [时序图](trainer-core/trainer-core-sequence.html) · [状态机](trainer-core/trainer-core-lifecycle.html) | Trainer 核心训练循环与回调系统 |
| trainer-optimizer | [trainer-optimizer.md](trainer-optimizer/trainer-optimizer.md) | [架构图](trainer-optimizer/trainer-optimizer-architecture.html) | — | 优化器/调度器工厂与 Seq2SeqTrainer |
| training-args | [training-args.md](training-args/training-args.md) | [架构图](training-args/training-args-architecture.html) | — | TrainingArguments 参数体系与解析器 |
| data-collators-loss | [data-collators-loss.md](data-collators-loss/data-collators-loss.md) | [架构图](data-collators-loss/data-collators-loss-architecture.html) | — | DataCollator 系列与专用损失函数 |

## 3. 域级机制细节

### 3.1 训练循环状态机

`Trainer._inner_training_loop()`（`trainer.py:1527`）实现了完整的训练状态机：
- **初始化**：数据加载器、TrainingState、优化器/调度器、模型封装
- **Epoch 循环**：on_epoch_begin → 遍历 dataloader → on_epoch_end
- **Step 循环**：on_step_begin → training_step（forward+backward）→ 梯度累积 → optimizer.step() → on_step_end → _maybe_log_save_evaluate
- **收尾**：on_train_end → 返回 TrainOutput

### 3.2 回调系统

`CallbackHandler`（`trainer_callback.py:429`）统一分发 15+ 个 `on_*` 钩子。回调可通过 `TrainerControl` 设置 `should_training_stop=True` 提前终止训练。默认回调包括 PrinterCallback、ProgressCallback、EarlyStoppingCallback 等。

### 3.3 分布式策略抽象

Trainer 通过 Accelerate 库抽象了多种分布式训练后端（DDP/FSDP/DeepSpeed/TPU）。用户只需设置 `TrainingArguments` 中的 `deepspeed`/`fsdp` 参数，Trainer 自动选择对应的封装路径，无需用户手动管理多 GPU 通信。
