# 分布式核心（distributed-core）

> 本文是 `distributed-deployment` 域下的叶子子系统文档。域级总览见 `../distributed-deployment.md`。本文展开 transformers 内置的分布式并行工具（区别于外部 accelerate/deepspeed）。
>
> 源码基准：`transformers` v5.18.0.dev0（main 分支），commit `5474a55e920f358d8382f3ecd3377edca979baa`。

## 1. 功能清单

| 能力 | 说明 | 源码路径 |
|------|------|----------|
| 张量并行（TP） | 模型层权重按列/行切分到多 GPU，使用 PyTorch DTensor | `src/transformers/distributed/tensor_parallel.py` |
| `TensorParallelLayer` 基类 | 定义 shard_param / transform_inputs / context_around_forward / transform_output 四阶段接口 | `tensor_parallel.py:123` |
| `ColwiseParallel` | 列并行：Linear 权重按 dim=0 切分，输入复制，输出 AllReduce | `tensor_parallel.py:161` |
| `RowwiseParallel` | 行并行：Linear 权重按 dim=1 切分，输入需 AllGather | `tensor_parallel.py:244` |
| `SequenceParallel` | 序列维并行 | `tensor_parallel.py:413` |
| MoE 专家并行 | `MoeExpertsParallel`/`EpRouterParallel`/`MoeIdentityParallel` 专家切分 | `tensor_parallel.py:588/658/647` |
| `tp_plan` 注册 | 模型配置中声明各层的并行策略映射 | `tensor_parallel.py:45`（`verify_tp_plan`） |
| 流水线并行（PP） | 模型按层切分到多 stage，stage 间 P2P 传 hidden states | `src/transformers/distributed/pipeline_parallel.py` |
| `PipelineStage` | stage 元数据：rank/邻接 rank/P2P 通信（isend/irecv） | `pipeline_parallel.py:52` |
| `apply_pipeline_parallelism()` | 按 pp_mesh 切分模型层到各 rank | `pipeline_parallel.py:183` |
| `pipeline_parallel_naive_forward()` | 朴素流水线前向（无 micro-batch 调度） | `pipeline_parallel.py:247` |
| FSDP 封装 | `apply_fully_sharded_data_parallelism()` 对 PyTorch FSDP 的配置封装 | `src/transformers/distributed/fsdp.py:187` |
| FSDP plan 展开 | `expand_fsdp_plan()`/`verify_fsdp_plan()` 分片策略校验 | `fsdp.py:145/164` |
| DTensor 分片工具 | `DtensorShardOperation` 枚举与 `_dtensor_from_local_like` | `src/transformers/distributed/sharding_utils.py:39/321` |
| 分布式初始化与 checkpoint | `initialize_distributed_mesh`、`gather_full_state_dict`、分布式保存/加载 | `src/transformers/distributed/utils.py:132/179/198` |
| 分布式配置 | `DistributedConfig` 数据类 | `src/transformers/distributed/configuration_utils.py` |
| `mixin.py` | 模型分布式混入 | `src/transformers/distributed/mixin.py` |

## 2. 核心类型与接口清单

| 类型 | 位置 | 职责 |
|------|------|------|
| `TensorParallelLayer` | `tensor_parallel.py:123` | TP 策略基类：`install_forward` 包装原 forward 为 tp_forward（pre→context→post） |
| `ColwiseParallel` | `tensor_parallel.py:161` | 列并行：`shard_param` 用 `distribute_tensor` Shard(n-2)；输出 AllReduce |
| `RowwiseParallel` | `tensor_parallel.py:244` | 行并行：输入 AllGather，输出 Shard(-1) |
| `MoeExpertsParallel` | `tensor_parallel.py:588` | MoE 专家切分到不同 rank |
| `EpRouterParallel` | `tensor_parallel.py:658` | 专家并行路由层 |
| `ParallelInterface(GeneralInterface)` | `tensor_parallel.py:762` | transformers.GeneralInterface 体系的 TP 实现入口 |
| `PipelineStage` | `pipeline_parallel.py:52` | stage 通信封装：`communicate("send_forward"/"recv_forward")` P2P |
| `PipelineIdentityLayer(nn.Identity)` | `pipeline_parallel.py:41` | stage 边界的恒等层占位 |
| `apply_pipeline_parallelism()` | `pipeline_parallel.py:183` | 模型 PP 主入口 |
| `apply_fully_sharded_data_parallelism()` | `fsdp.py:187` | FSDP 主入口 |
| `DtensorShardOperation` | `sharding_utils.py:39` | DTensor 分片操作枚举 |
| `initialize_distributed_mesh()` | `utils.py:132` | 创建 DeviceMesh（TP/PP/DP 三维） |

## 3. 关键调用链

### 3.1 张量并行应用链

1. 模型 config 中声明 `tp_plan: dict[layer_name_pattern, strategy_name]`；
2. `verify_tp_plan(expected_keys, tp_plan)` 校验 plan 覆盖所有参数（`tensor_parallel.py:45`）；
3. 逐层遍历模型参数，`_get_parameter_tp_plan` 匹配层名模式（`tensor_parallel.py:74`）；
4. 对每个匹配的层实例化对应 `TensorParallelLayer` 子类（Colwise/Rowwise/...）；
5. `shard_param()` 调 `distribute_tensor(local_param, mesh, [Shard(dim)])` 把本地参数包成 DTensor；
6. `install_forward()` 包装 `module.forward`：pre-forward 变换输入 → local DTensor context → 原 forward → post-forward 变换输出（AllReduce/AllGather）。

### 3.2 流水线并行前向链

1. `apply_pipeline_parallelism(model, pp_mesh)` 构造 `PipelineStage(pp_mesh)`；
2. `layer_range_for_rank(rank, num_layers)` 计算每层所属 rank（`pipeline_parallel.py:109`）；
3. 非首 stage 前向时 `communicate("recv_forward")` 从 prev_rank irecv hidden_states；
4. 本 stage 层计算 hidden_states；
5. 非末 stage `communicate("send_forward")` isend 到 next_rank；
6. `pipeline_parallel_naive_forward` 串联上述 P2P（当前为朴素版本，无 micro-batch 交错调度）。

### 3.3 FSDP 应用链

1. `_get_fsdp_policy_kwargs(distributed_config)` 取分片策略（`fsdp.py:64`）；
2. `expand_fsdp_plan` 展开层名通配符为具体模块名；
3. `verify_fsdp_plan(module_names, fsdp_plan)` 校验；
4. `apply_fully_sharded_data_parallelism` 调 `torch.distributed.fsdp.fully_shard(module, ...)` 包装。

## 4. 配置项

| 配置 | 行为 | 位置 |
|------|------|------|
| `tp_plan`（model config） | 层名→并行策略名映射 | `tensor_parallel.py:45` |
| `pp_mesh`（DeviceMesh） | 1D 流水线设备网格 | `pipeline_parallel.py:55` |
| `fsdp_plan`（model config） | FSDP 分片策略映射 | `fsdp.py:164` |
| `DistributedConfig` | 分布式配置数据类 | `configuration_utils.py` |
| `comm_on_cpu` | gloo 后端时 P2P 走 CPU | `pipeline_parallel.py:63` |
| `use_local_output`（Colwise） | 是否输出 local tensor（推理内核优化） | `tensor_parallel.py:164` |

## 5. 错误与重试语义

- **TP 不可整除校验**（`tensor_parallel.py:183-189`）：Colwise 输出维不能被 tp_size 整除时抛 `ValueError`。
- **PP 未注册层**（`pipeline_parallel.py:126`）：`find_rank_for_key` 对未知 key 返回 None（不报错，告警）。
- **FSDP plan 校验**（`fsdp.py:164`）：未识别的 plan 键抛错。
- 分布式通信失败由 PyTorch `dist` 层抛出（不在本仓库源码内）。
- 无自动重试；分布式初始化失败直接终止。

## 6. 并发细节

- **多 GPU 进程模型**：transformers 分布式工具假设 PyTorch `torchrun` 已启动多进程，每进程绑定一个 GPU。
- **P2P 通信**：PP stage 间用 `dist.P2POp(isend/irecv)` + `dist.batch_isend_irecv` 批量收发（`pipeline_parallel.py:100-103`）；CUDA 通信后 `torch.cuda.synchronize()`。
- **DTensor 惰性通信**：TP 中 AllReduce/AllGather 封装在 `transform_output_post_forward` 中，由 DTensor 操作触发。
- **DeviceMesh**：`initialize_distributed_mesh` 创建 TP×PP×DP 三维网格（`utils.py:132`）。
- **Checkpoint 一致性**：`gather_full_state_dict` 从各 rank 收集分片参数聚合成完整 state_dict（`utils.py:179`）。
- 当前 PP 为朴素前向（无 1F1B micro-batch 调度），注释标注 TODO 按层类型/参数量平衡 stage。

## 7. 系统边界

**In-Scope**
- `src/transformers/distributed/`：TP/PP/FSDP 工具、DTensor 分片、分布式 checkpoint、配置

**Out-of-Scope**
- PyTorch Distributed（`torch.distributed`、`DeviceMesh`、`DTensor`、`fully_shard`、`P2POp`）——外部库
- accelerate 库的 `device_map` 自动分片——外部库（见 integrations-frameworks）
- DeepSpeed 引擎——外部库
- 具体模型层的 TP plan 声明——见 models-registry 域
- `torchrun` 启动器——外部工具

## 8. 与相邻子系统交互

- **模型加载 → distributed 工具**：`PreTrainedModel.from_pretrained` 后调用 `apply_tensor_parallel` / `apply_pipeline_parallelism` / `apply_fully_sharded_data_parallelism`。
- **distributed → accelerate**：`update_fsdp_plugin_peft` 与 accelerate 集成（`fsdp.py:257`）。
- **distributed → Trainer**：训练时由 Trainer 调用 FSDP 封装（见 trainer-training 域）。
- **integrations → distributed**：DeepSpeed/FSDP 集成文件调用本包工具（见 `../integrations-frameworks/`）。

## 9. 语言专项适配口径

Python 项目。按并行策略 capability seam 分组（TP/PP/FSDP）。图以 architecture（策略组件）+ dataflow（TP 权重切分数据流）为主。外部依赖 PyTorch Distributed 标注"不在本仓库源码内"。不适用"单二进制"，按包结构+依赖图表达。

## 10. 图表清单与质量

| 图 | 文件 | 类型 | archify 质量档 |
|----|------|------|----------------|
| 分布式并行策略架构图 | `distributed-core-architecture.html` | architecture | showcase |
| 张量并行权重切分数据流 | `distributed-core-tp-dataflow.html` | dataflow | standard |

JSON IR 源文件位于 `json/` 目录。
