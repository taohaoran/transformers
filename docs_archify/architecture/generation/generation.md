# generation 域总览

> 本域包含以下叶子子系统；各叶子详情见对应文档。
> 源码基准：`transformers` main 分支，commit `5474a55e920f358d8382f3ecd3377edca979baa`。

## 1. 域职责

generation 域实现了 HuggingFace transformers 的文本生成能力，覆盖从单次生成到连续批处理的完整生成栈。核心入口是 `GenerationMixin.generate()`，它根据 `GenerationConfig` 参数选择生成策略（greedy/sample/beam/assisted），并通过 KV Cache 管理高效地逐 token 生成。

## 2. 叶子索引

| 叶子 | 文档 | 架构图 | 时序图 / 状态机 | 职责一句话 |
|------|------|--------|----------------|-----------|
| generation-loop | [generation-loop.md](generation-loop/generation-loop.md) | [架构图](generation-loop/generation-loop-architecture.html) | [时序图](generation-loop/generation-loop-sequence.html) · [状态机](generation-loop/generation-loop-lifecycle.html) | generate() 核心调度与生成循环 |
| generation-assisted | [generation-assisted.md](generation-assisted/generation-assisted.md) | [架构图](generation-assisted/generation-assisted-architecture.html) | [时序图](generation-assisted/generation-assisted-sequence.html) | 推测解码（草稿模型+目标模型验证） |
| generation-watermarking | [generation-watermarking.md](generation-watermarking/generation-watermarking.md) | [架构图](generation-watermarking/generation-watermarking-architecture.html) | [时序图](generation-watermarking/generation-watermarking-sequence.html) | 文本水印注入与检测 |
| continuous-batching | [continuous-batching.md](continuous-batching/continuous-batching.md) | [架构图](continuous-batching/continuous-batching-architecture.html) | [状态机](continuous-batching/continuous-batching-lifecycle.html) | vLLM 风格连续批处理引擎 |

## 3. 域级机制细节

### 3.1 生成策略统一入口

`GenerationMixin.generate()`（`utils.py:2395`）是所有生成策略的统一入口。它根据 `GenerationConfig` 中的 `generation_mode`（由 `_get_generation_mode()` 推断）分发到：
- `_greedy_search()`：贪心解码
- `_sample()`：采样解码（temperature/top-p/top-k）
- `_beam_search()` / `_beam_sample()`：束搜索
- `_assisted_decoding()`：推测解码

### 3.2 KV Cache 跨叶子复用

所有生成策略共享同一套 KV Cache 管理机制：`past_key_values` 张量在生成步间传递，避免重复计算 prompt 的注意力。推测解码（generation-assisted）和连续批处理（continuous-batching）在此基础上扩展了 Cache 复用（草稿模型与目标模型共享 KV、paged attention 块管理）。

### 3.3 Logits 处理器链与停止准则链

`_get_logits_processor()` 和 `_get_stopping_criteria()` 构建两条责任链：
- **LogitsProcessor 链**：temperature → top-k → top-p → repetition_penalty → watermarking → 采样
- **StoppingCriteria 链**：max_length → eos_token → 自定义准则

水印（generation-watermarking）作为 LogitsProcessor 链中的一环注入。
