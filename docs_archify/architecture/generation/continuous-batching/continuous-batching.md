# continuous-batching（连续批处理引擎）

> 本文是 `generation` 域下的叶子子系统文档。域级总览见 `../generation.md`。
> 本文只展开 **vLLM 风格连续批处理引擎**，不重复展开单次 generate 循环（见 `../generation-loop/`）。
>
> 源码基准：`transformers` main 分支，commit `5474a55e920f358d8382f3ecd3377edca979baa`。

## 1. 功能清单

| 能力 | 说明 | 源码路径 |
|------|------|----------|
| `ContinuousBatchingManager` | 高层管理器：启动/停止后台推理线程，管理生命周期 | `src/transformers/generation/continuous_batching/continuous_api.py:574` |
| `ContinuousBatchProcessor` | 连续批处理主循环：prepare_next_batch → generation_step → update_batch | `src/transformers/generation/continuous_batching/continuous_api.py:214` |
| `FIFOScheduler` | FIFO 调度器：按到达顺序调度等待队列中的请求 | `src/transformers/generation/continuous_batching/scheduler.py:331` |
| `PrefillFirstScheduler` | Prefill 优先调度器：优先调度 prefill 请求 | `src/transformers/generation/continuous_batching/scheduler.py:380` |
| `ModelRunner` | 模型单步前向执行：padding、CUDA graph capture、采样 | `src/transformers/generation/continuous_batching/model_runner.py:29` |
| `BlockManager` | 物理 KV Cache 块管理：分配/释放/引用计数/块共享 | `src/transformers/generation/continuous_batching/cache_manager.py:58` |
| `FullAttentionCacheAllocator` | 全注意力 KV Cache 分配器（paged attention 块表） | `src/transformers/generation/continuous_batching/cache_manager.py:365` |
| `SlidingAttentionCacheAllocator` | 滑动窗口注意力 KV Cache 分配器 | `src/transformers/generation/continuous_batching/cache_manager.py:456` |
| `OffloadingManager` | CPU/GPU 显存卸载：显存不足时将请求 KV 换出到 CPU | `src/transformers/generation/continuous_batching/offloading_manager.py:55` |
| `RequestState` | 请求状态对象：跟踪 token 序列、块分配、完成判定 | `src/transformers/generation/continuous_batching/requests.py:124` |
| `OutputRouter` | 输出路由：将生成结果交付给等待的 future | `src/transformers/generation/continuous_batching/continuous_api.py:84` |
| `CBLogitsProcessor` | 连续批处理专用 logits 处理器 | `src/transformers/generation/continuous_batching/cb_logits_processors.py` |

## 2. 核心类型与接口清单

| 类型 / 函数 | 位置 | 职责 |
|-------------|------|------|
| `ContinuousBatchProcessor` | `continuous_api.py:214` | 主循环三阶段：`prepare_next_batch()` / `_generation_step()` / `update_batch()` |
| `Scheduler` (ABC) | `scheduler.py:22` | 调度基类；`schedule_batch()` 返回待处理请求列表 |
| `RequestState` | `requests.py:124` | 单个请求的完整状态：prompt、生成 tokens、块分配、状态枚举 |
| `RequestStatus` | `requests.py:83` | 请求生命周期枚举：`PENDING(0)` / `PREFILLING(1)` / `DECODING(2)` / `FINISHED(3)` / `FAILED(4)` |
| `BlockManager` | `cache_manager.py:58` | 物理块分配：`get_free_blocks()` / `free_blocks()` / 引用计数 |
| `CacheAllocator` (ABC) | `cache_manager.py:297` | 逻辑块→物理块映射：`fill_block_table()` / `get_read/write_indices()` |
| `ModelRunner` | `model_runner.py:29` | 执行单次前向 + 采样：`compute_batch()` / `_sample()` |
| `OffloadingManager` | `offloading_manager.py:55` | CPU 卸载：`offload_requests()` / `restore_scheduled_requests()` |
| `GenerationOutput` | `requests.py:94` | 单次输出：prompt_ids + generated_tokens + logprobs |

## 3. 关键调用链

### 3.1 连续批处理主循环（`continuous_api.py:381-560`）

1. **`prepare_next_batch()`**（`continuous_api.py:381`）：
   - a. `_update_tp_group_state()`：从 TP driver 获取新请求/取消/停止信号。
   - b. `scheduler.clear_cancelled_requests()`：清理取消请求。
   - c. `scheduler.schedule_batch(max_batch_tokens, num_pages)`：调度器从等待队列选择请求组成 batch。
   - d. 若返回 `None`（Cache 满）：循环 `offloading_manager.offload_requests()` 卸载非活跃请求，再重新调度。
   - e. `offloading_manager.restore_scheduled_requests()`：将刚调度的请求从 CPU 恢复到 GPU。
   - f. `model_runner.maybe_pad_inputs()`：padding 到静态尺寸（torch.compile 友好）。
   - g. `inputs_and_outputs.prepare_batch_tensors()`：构建 batch 张量。
2. **`_generation_step(model)`**（`continuous_api.py:551`）：调用 `model_runner.compute_batch()` 执行单次前向 + 采样。
3. **`update_batch()`**（`continuous_api.py:445`）：
   - a. `inputs_and_outputs.prepare_batch_update()`：从 GPU 取回新 token。
   - b. 遍历请求：`state.update_and_check_completion(token, logprob)` 更新 token 序列。
   - c. 完成的请求：`scheduler.finish_request()` + `output_router.deliver_batch()`。
   - d. Fork 请求：`cache.fork_request()` + `cache.copy_cache()`。

### 3.2 请求生命周期

- **PENDING → PREFILLING**：调度器从等待队列选取请求，分配块，执行 prefill。
- **PREFILLING → DECODING**：prefill 完成后，`has_new_token=True` 且 `generated_len()==0`，切换到 DECODING。
- **DECODING → FINISHED**：生成 EOS 或达到 max_new_tokens。
- **DECODING → PENDING（卸载）**：显存不足时被 `OffloadingManager` 换出到 CPU。

## 4. 配置项

| 配置项 | 默认 / 行为 | 位置 |
|--------|-------------|------|
| `max_batch_tokens` | 单 batch 最大 token 数（pre-fill + decode 总和） | `continuous_api.py:218` |
| `max_requests_per_batch` | 单 batch 最大请求数 | `scheduler.py:29` |
| `block_size` | 每个 KV Cache 块的 token 数（paged attention 页大小） | `cache_manager.py:78` |
| `gpu_blocks` | GPU 上物理 KV Cache 块总数 | `cache_manager.py` |
| `cpu_offload_space_gib` | CPU 卸载空间上限（GiB） | `offloading_manager.py:62` |
| `safety_margin` | Cache 分配安全余量比例 | `scheduler.py:29` |
| `use_async_batching` | 异步批处理模式（请求可在 batch 执行中加入/退出） | `continuous_api.py:214` |
| `streaming` | 流式输出模式 | `requests.py:124` |

## 5. 错误与重试语义

- **Cache 满且无法卸载**：`RuntimeError("No requests can be scheduled and no requests can be offloaded.")`（`continuous_api.py:424`）。
- **请求取消**：`set_request_cancellation()` 标记，`clear_cancelled_requests()` 清理并释放 CPU 缓存。
- **TP 组错误传播**：`record_fatal_error()` 记录全局错误，所有后续请求失败。
- **请求错误隔离**：`_handle_request_error()` 捕获单请求异常，标记 FAILED，不影响其他请求。
- **无重试**：请求失败直接终止；模型前向错误传播到全局停止。

## 6. 并发细节

- **后台线程模型**：`ContinuousBatchingManager.start()` 启动后台线程执行推理循环；主线程通过 `generate_batch()` 提交请求。
- **异步批处理**：`use_async_batching=True` 时，请求可在 batch 执行中途加入/退出，无需等待当前 batch 完成。
- **CUDA Graph**：`ModelRunner._capture_graph()` 捕获 decode 阶段的 CUDA graph 减少 launch 开销。
- **TP 通信**：`_update_tp_group_state()` 与 Tensor Parallel driver 通信协调多 GPU 状态。
- **OutputRouter 线程安全**：`deliver()` 将结果放入 future，等待方在另一个线程接收。

## 7. 系统边界

**In-Scope（本仓库源码内）**
- `src/transformers/generation/continuous_batching/` 全部文件

**Out-of-Scope（不在本仓库源码内）**
- 模型 forward 计算本身
- PyTorch CUDA 实现、CUDA Graph capture 底层
- vLLM 参考实现（仅架构借鉴）
- Tensor Parallel 通信库（NCCL 等）

## 8. 与相邻子系统交互

- **上游 → 本叶子**：`generate()` 检测到 `cache_implementation="paged"` 时分发到 `generate_batch()`（`utils.py:2526`）。
- **本叶子 → 下游**：
  - → 模型 `forward()`：通过 `ModelRunner.compute_batch()` 执行前向
  - → KV Cache：`BlockManager` 分配/释放/共享物理块
  - → OffloadingManager：显存不足时 CPU 换入换出

## 9. 语言专项适配口径

- Python 库，按"调度 / 执行 / 缓存管理 / 卸载"四个 capability seam 划分。
- 图类型：architecture（引擎组件关系）+ lifecycle（请求状态机）。
- 外部边界：CUDA/NCCL 标注为外部组件。
- 部署维度：后台线程 + 主线程双线程模型，无多二进制。

## 10. 图表清单与质量

| 图 | 文件 | 类型 | archify 质量档 |
|----|------|------|----------------|
| 连续批处理引擎架构图 | `continuous-batching-architecture.html` | architecture | standard（降档披露） |
| 请求生命周期状态机 | `continuous-batching-lifecycle.html` | lifecycle | standard（降档披露） |

JSON IR 源文件位于 `json/` 目录。
