# generation-loop（生成核心循环）

> 本文是 `generation` 域下的叶子子系统文档。域级总览见 `../generation.md`。
> 本文只展开**自回归生成的统一调度入口与核心解码循环**，不重复展开推测解码（见 `../generation-assisted/`）、
> 水印注入（见 `../generation-watermarking/`）、连续批处理（见 `../continuous-batching/`）。
>
> 源码基准：`transformers` main 分支，commit `5474a55e920f358d8382f3ecd3377edca979baa`。

## 1. 功能清单

| 能力 | 说明 | 源码路径 |
|------|------|----------|
| 统一生成入口 `generate()` | 接收 prompt 与生成参数，完成配置准备、输入准备、Cache 准备、处理器/准则链构建，最终分发到具体解码方法 | `src/transformers/generation/utils.py:2395`（`GenerationMixin.generate`） |
| 生成模式分发 | 根据 `GenerationConfig` 推导 `GenerationMode`，通过 `GENERATION_MODES_MAPPING` 映射到 `_sample`/`_beam_search`/`_assisted_decoding` | `src/transformers/generation/configuration_utils.py:82`（`GenerationMode` 枚举）；`src/transformers/generation/utils.py:139`（`GENERATION_MODES_MAPPING`） |
| 贪心/采样解码循环 `_sample` | 支持 greedy（`do_sample=False`，argmax）与 multinomial sampling（`do_sample=True`，softmax+multinomial）的自回归循环 | `src/transformers/generation/utils.py:2917`（`GenerationMixin._sample`） |
| Beam Search 循环 `_beam_search` | 维护 `num_beams` 条候选序列，每步 top-K 筛选、early stopping heuristic、length penalty、finished beam 管理 | `src/transformers/generation/utils.py:3369`（`GenerationMixin._beam_search`） |
| GenerationConfig 参数体系 | 集中管理 100+ 生成参数（长度/策略/Cache/logits 操作/输出控制），支持 `validate()`、`from_pretrained()`、`save_pretrained()` | `src/transformers/generation/configuration_utils.py:100`（`GenerationConfig`） |
| LogitsProcessor 链构建 | 根据 GenerationConfig 自动组装 CFG→repetition_penalty→no_repeat_ngram→bad_words→min_length→采样 warper（温度/top-k/top-p/min-p）→水印→归一化 | `src/transformers/generation/utils.py:1252`（`_get_logits_processor`） |
| StoppingCriteria 链构建 | 自动组装 MaxLength/MaxTime/StopString/EosToken/Confidence 准则 | `src/transformers/generation/utils.py:1487`（`_get_stopping_criteria`） |
| 输入预处理链 | `_prepare_model_inputs`→`_prepare_special_tokens`→`_prepare_attention_mask/position_ids`→`_expand_inputs_for_generation`→`_prepare_cache_for_generation` | `src/transformers/generation/utils.py:775/2168/904/880/1039/2058` |
| 文本流式输出 | `TextStreamer`（逐 token 打印）、`TextIteratorStreamer`（迭代器）、`AsyncTextIteratorStreamer`（异步迭代器） | `src/transformers/generation/streamers.py:42/157/226` |
| 延迟停止检查优化 | `DeferredStopCheck` 将停止准则检查从每个前向步后推迟到 Cache 记录阶段，减少 GPU 同步点 | `src/transformers/generation/utils.py:390`（`DeferredStopCheck`） |
| Prefill 阶段封装 | `_prefill` 封装首次前向（处理 `logits_to_keep`、encoder-decoder 编码器前向、多模态输入清理） | `src/transformers/generation/utils.py:4065` |

## 2. 核心类型与接口清单

| 类型 / 函数 | 位置 | 职责 |
|-------------|------|------|
| `GenerationMixin` | `utils.py:482` | 所有生成式模型 mixin，继承 `ContinuousMixin`，实现 `generate()` 及全部解码方法 |
| `GenerationConfig` | `configuration_utils.py:100` | 生成配置数据类，持有全部生成参数；`get_generation_mode()` 推导解码模式 |
| `GenerationMode` | `configuration_utils.py:82` | 枚举：`greedy_search`/`sample`/`beam_search`/`beam_sample`/`assisted_generation` 等 |
| `LogitsProcessorList` | `logits_process.py:63` | 继承 `list`，按顺序对 logits 施加所有处理器；`__call__(input_ids, scores)` 链式调用 |
| `LogitsProcessor` | `logits_process.py:49` | 抽象基类，子类实现 `__call__(input_ids, scores) -> scores` |
| `StoppingCriteriaList` | `stopping_criteria.py:618` | 继承 `list`，`__call__` 返回 bool 张量标记哪些序列应停止 |
| `StoppingCriteria` | `stopping_criteria.py:48` | 抽象基类，子类实现 `__call__(input_ids, scores) -> bool` |
| `BaseStreamer` | `streamers.py:28` | 流式输出抽象接口：`put(value)` / `end()` |
| `StopCheck` / `DeferredStopCheck` | `utils.py:372/390` | 停止检查封装：同步版每步检查；延迟版利用 Cache 异步记录减少同步点 |
| `GenerateDecoderOnlyOutput` / `GenerateEncoderDecoderOutput` | `utils.py:171/207` | 非 beam 生成输出 dataclass（含 sequences/scores/attentions/hidden_states/past_key_values） |
| `GenerateBeamDecoderOnlyOutput` / `GenerateBeamEncoderDecoderOutput` | `utils.py:255/299` | beam 生成输出 dataclass（额外含 beam_indices/beam_scores） |

## 3. 关键调用链

### 3.1 `generate()` 完整调度流程（`utils.py:2395-2812`）

1. **步骤 0a**（`utils.py:2503`）：若 `custom_generate` 为字符串，从 Hub 加载自定义生成函数并直接返回；若为 Callable，先完成准备步骤后调用。
2. **步骤 0b**（`utils.py:2526`）：若 `cache_implementation="paged"`，切换到连续批处理模式（`generate_batch`），本叶子不展开（见 continuous-batching 叶子）。
3. **步骤 1**（`utils.py:2590-2639`）：`_prepare_generation_config()` 合并用户 kwargs 与默认配置；`get_generation_mode(assistant_model)` 推导模式；`GENERATION_MODES_MAPPING[mode]` 反射到解码方法（`_sample`/`_beam_search`/`_assisted_decoding`）。
4. **步骤 2**（`utils.py:2642-2643`）：初始化空的 `LogitsProcessorList()` 与 `StoppingCriteriaList()`（用户可传入自定义实例）。
5. **步骤 3**（`utils.py:2649-2681`）：`_prepare_model_inputs()` 确定 `input_ids`/`inputs_embeds`/多模态输入；`_prepare_special_tokens()` 准备 eos/pad/decoder_start 张量；decoder-only 模型检测 right-padding 并警告。
6. **步骤 4**（`utils.py:2683-2707`）：准备 `attention_mask`、`position_ids`；encoder-decoder 模型执行编码器前向并缓存 `encoder_outputs`。
7. **步骤 5**（`utils.py:2709-2733`）：encoder-decoder 生成 `decoder_input_ids`；decoder-only 直接使用 prompt；`_expand_inputs_for_generation()` 按 `num_beams`/`num_return_sequences` 扩展 batch 维。
8. **步骤 6**（`utils.py:2735-2752`）：`_prepare_generated_length()` 计算最终 `max_length`/`min_length`（优先 `max_new_tokens`）；`logits_to_keep=1` 优化首次前向。
9. **步骤 7**（`utils.py:2754-2778`）：`_prepare_cache_for_generation()` 根据 `cache_implementation` 实例化 `DynamicCache`/`StaticCache`/`QuantizedCache` 等。
10. **步骤 8**（`utils.py:2780-2799`）：`_get_logits_processor()` 组装全部 logits 处理器；`_get_stopping_criteria()` 组装全部停止准则。
11. **步骤 9**（`utils.py:2801-2810`）：调用 `decoding_method(input_ids, logits_processor, stopping_criteria, generation_config, **model_kwargs)`。

### 3.2 `_sample()` 自回归解码循环（`utils.py:2917-3134`）

1. **初始化**（`utils.py:2959-3021`）：读取配置；`unfinished_sequences = ones(batch_size)` 跟踪未完成序列；`_prefill()` 执行首次前向；根据 Cache 类型选择 `StopCheck` 或 `DeferredStopCheck`。
2. **主循环**（`utils.py:3024-3087`，`while _has_unfinished_sequences()`）：
   - a. 非首次迭代时 `prepare_inputs_for_generation()` 准备单 token 输入（利用 KV Cache 只传最后一个 token）。
   - b. `model_forward(**model_inputs)` 执行前向，取 `outputs.logits[:, -1]`。
   - c. `_update_model_kwargs_for_generation()` 更新 Cache。
   - d. `logits_processor(input_ids, next_token_logits)` 链式处理 logits。
   - e. token 选择：`do_sample=True` → softmax + `torch.multinomial`；`do_sample=False` → `torch.argmax`。
   - f. 已完成序列用 `pad_token_id` 填充。
   - g. `input_ids = cat([input_ids, next_tokens])` 追加 token。
   - h. `stopping_criteria(input_ids, scores)` 标记完成序列；`stop_check()` 检查是否全部完成。
   - i. `del outputs` 释放大张量引用。
3. **收尾**（`utils.py:3089-3134`）：`stop_check.finish()` 如有延迟回退步数则 `_undo_generation_steps()`；`streamer.end()`；按 `return_dict_in_generate` 返回 dataclass 或 tensor。

### 3.3 `_beam_search()` Beam Search 循环（`utils.py:3369+`）

1. 初始化 beam 状态：`beams_to_keep = (1 + n_eos_tokens) * num_beams`；维护 `beam_scores`、`beam_was_beam_finished`、`finished_beams`。
2. 每步前向后，对 `(batch * num_beams)` 条候选的 logits 施加 `logits_processor`，按 `do_sample` 决定 top-K 选择还是无放回采样。
3. `_get_top_k_continuations()` 筛选 top-K 续接；`_update_finished_beams()` 管理已完成 beam；`_get_running_beams_for_next_iteration()` 决定存活 beam。
4. early stopping：`_check_early_stop_heuristic()` 判断是否已有足够高质量候选。

## 4. 配置项

| 配置项 | 默认 / 行为 | 位置 |
|--------|-------------|------|
| `max_new_tokens` / `max_length` | 控制最大生成长度；`max_new_tokens` 优先于 `max_length` | `configuration_utils.py:133` |
| `min_new_tokens` / `min_length` | 最小生成长度，配合 eos 强制前不允许停止 | `configuration_utils.py:138` |
| `do_sample` | `False`（贪心）；`True` 时启用 multinomial sampling | `configuration_utils.py:154` |
| `num_beams` | `1`（无 beam search）；>1 启用 beam search | `configuration_utils.py:156` |
| `temperature` | `1.0`；<1 降低熵，>1 增加随机性 | `configuration_utils.py:188` |
| `top_k` | 默认 `0`（不启用）；保留概率最高的 K 个 token | `configuration_utils.py:190` |
| `top_p` | `1.0`；核采样，保留累计概率 ≥ top_p 的最小 token 集 | `configuration_utils.py:192` |
| `min_p` | `None`；按最高概率 token 比例的最小概率阈值 | `configuration_utils.py:195` |
| `repetition_penalty` | `1.0`；对已出现 token 的 logits 除以 penalty | `logits_process.py:306` |
| `no_repeat_ngram_size` | `0`；禁止生成已出现的 n-gram | `logits_process.py:1073` |
| `bad_words_ids` | `None`；禁止生成指定 token 序列 | `logits_process.py:1395` |
| `length_penalty` | `1.0`；>1 偏好长序列，<1 偏好短序列（beam 用） | `configuration_utils.py`（beam 参数组） |
| `early_stopping` | beam search 停止策略：`bool`/`"never"` | `configuration_utils.py:140` |
| `use_cache` | `True`；启用 KV Cache 加速解码 | `configuration_utils.py:163` |
| `cache_implementation` | `"dynamic"`；可选 `static`/`offloaded`/`quantized`/`paged` | `configuration_utils.py:166` |
| `output_scores` / `output_attentions` / `output_hidden_states` | 控制是否输出中间量 | `configuration_utils.py`（输出控制组） |
| `return_dict_in_generate` | `False`；`True` 返回 dataclass 而非纯 tensor | `configuration_utils.py`（输出控制组） |
| `pad_token_id` / `eos_token_id` / `bos_token_id` | 特殊 token 张量，在 `_prepare_special_tokens()` 中实例化 | `configuration_utils.py` |

## 5. 错误与重试语义

- **生成模式校验**（`utils.py:1686` `_validate_generation_mode`）：在分发前校验模式与参数组合一致性（如 `num_beams>1` 时 `do_sample` 的行为），不合法直接抛 `ValueError`。
- **生成长度校验**（`utils.py:1798` `_validate_generated_length`）：检查 `max_length` ≥ prompt 长度，否则抛错。
- **模型 kwargs 校验**（`utils.py:1743` `_validate_model_kwargs`）：过滤模型不接受的 kwargs，给出警告。
- **停止准则校验**（`stopping_criteria.py:636` `validate_stopping_criteria`）：若用户未提供任何停止准则且 `max_length=None`，自动注入 `MaxLengthCriteria` 防止无限循环。
- **GPU 同步容错**（`utils.py:2814` `_has_unfinished_sequences`）：FSDP/DeepSpeed ZeRO-3 多 GPU 场景下，`synced_gpus=True` 时即使本地 peer 已完成也继续空转，避免其他 GPU 死锁。
- **无重试机制**：生成循环是确定性前向计算，不涉及网络重试；失败直接传播异常。

## 6. 并发细节

- **单线程同步执行**：`_sample`/`_beam_search` 均为单线程自回归循环，每次前向同步等待 GPU 计算完成。
- **GPU 同步点优化**：`DeferredStopCheck`（`utils.py:390`）利用 Cache 的异步记录机制，将停止准则检查从"每个前向步后立即同步"推迟到"Cache 内部批量记录"，减少 GPU 同步次数。仅在支持 `activate_past_recording` 的 Cache 类型上启用。
- **KV Cache 共享**：`past_key_values` 作为 `model_kwargs` 的一部分在循环中原地更新，无需拷贝；`_update_model_kwargs_for_generation()`（`utils.py:1069`）负责更新。
- **无多线程/GIL 竞争**：整个生成循环在主线程执行，无显式线程或 asyncio 并发（`AsyncTextIteratorStreamer` 是消费端异步，不影响生成端）。
- **`del outputs` 内存管理**（`utils.py:3087`）：每步显式删除 `outputs` 引用，防止首次迭代的大 logits 张量在循环中滞留显存。

## 7. 系统边界

**In-Scope（本仓库源码内）**
- `src/transformers/generation/utils.py`：`GenerationMixin` 全部生成调度与解码循环
- `src/transformers/generation/configuration_utils.py`：`GenerationConfig` 参数体系
- `src/transformers/generation/logits_process.py`：全部 `LogitsProcessor`/`LogitsWarper` 实现
- `src/transformers/generation/stopping_criteria.py`：全部 `StoppingCriteria` 实现
- `src/transformers/generation/streamers.py`：`TextStreamer` 等流式输出类

**Out-of-Scope（不在本仓库源码内）**
- 模型 `forward()` 前向计算本身（由各模型 `modeling_*.py` 实现，不在本叶子展开）
- PyTorch 框架（`torch.multinomial`/`torch.argmax`/CUDA 计算）——不在本仓库源码内
- 推测解码的候选生成器（见 `generation-assisted` 叶子）
- 水印 logits 处理器的详细算法（见 `generation-watermarking` 叶子）
- vLLM 风格连续批处理引擎（见 `continuous-batching` 叶子）
- 已弃用模式（contrastive_search/dola/group_beam/constrained_beam）——迁移至 `transformers-community` Hub 仓库

## 8. 与相邻子系统交互

- **上游 → 本叶子**：用户代码调用 `model.generate(input_ids, ...)`；各模型类（`*ForCausalLM`）通过继承 `GenerationMixin` 获得生成能力。
- **本叶子 → 下游**：
  - → 模型 `forward(**model_inputs)`：每步前向计算，返回 logits 与 Cache。
  - → `LogitsProcessorList`：链式修改 logits 分布。
  - → `StoppingCriteriaList`：判断是否终止。
  - → `BaseStreamer.put()`：逐 token 流式输出。
  - → `generation-assisted`：当 `assistant_model` 或 `prompt_lookup_num_tokens` 存在时，分发到 `_assisted_decoding()`。
  - → `continuous-batching`：当 `cache_implementation="paged"` 时，分发到 `generate_batch()`。

## 9. 语言专项适配口径

本项目为 Python 库，按 capability seam 分组：
- **分组**：按"生成调度入口 / 解码循环 / 参数配置 / logits 变换 / 停止控制 / 流式输出"六个 capability seam 划分，而非按 class 层级。
- **图类型**：以 architecture（组件结构）+ lifecycle（生成循环状态机）为主；sequence（generate 调用链）作为辅助。
- **外部边界**：标注 PyTorch CUDA 计算、Hub 模型仓库、第三方推理引擎为外部组件。
- **部署维度**：Python 包，单进程调用，无多二进制部署概念；`custom_generate` 支持从 Hub 加载外部生成逻辑。

## 10. 图表清单与质量

| 图 | 文件 | 类型 | archify 质量档 |
|----|------|------|----------------|
| 生成核心组件架构图 | `generation-loop-architecture.html` | architecture | showcase |
| 生成循环状态机图 | `generation-loop-lifecycle.html` | lifecycle | showcase |
| generate() 调用时序图 | `generation-loop-sequence.html` | sequence | showcase |

JSON IR 源文件位于 `json/` 目录。
