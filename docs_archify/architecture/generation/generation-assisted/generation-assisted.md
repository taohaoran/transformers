# generation-assisted（推测解码 / 辅助生成）

> 本文是 `generation` 域下的叶子子系统文档。域级总览见 `../generation.md`。
> 本文只展开**推测解码（speculative decoding）的候选生成-验证-接受循环**，不重复展开核心生成循环（见 `../generation-loop/`）。
>
> 源码基准：`transformers` main 分支，commit `5474a55e920f358d8382f3ecd3377edca979baa`。

## 1. 功能清单

| 能力 | 说明 | 源码路径 |
|------|------|----------|
| 推测解码主循环 `_assisted_decoding` | 草稿模型生成 N 个候选 token → 目标模型一次前向验证 → 接受匹配前缀 + 1 个 bonus token | `src/transformers/generation/utils.py:3723` |
| `CandidateGenerator` 基类 | 抽象接口：`get_candidates()` 返回候选序列，`update_candidate_strategy()` 根据接受率调整策略 | `src/transformers/generation/candidate_generator.py:39` |
| `AssistedCandidateGenerator` | 基于草稿小模型的候选生成器；草稿模型自回归生成 `num_assistant_tokens` 个候选 | `src/transformers/generation/candidate_generator.py:80` |
| `AssistedCandidateGeneratorDifferentTokenizers` | 草稿模型与目标模型词表不同时的候选生成器；含 token 映射与对角序列对齐 | `src/transformers/generation/candidate_generator.py:341` |
| `PromptLookupCandidateGenerator` | 无需草稿模型，从输入 prompt 中查找 n-gram 重复作为候选（prompt lookup decoding） | `src/transformers/generation/candidate_generator.py:1018` |
| `UniversalSpeculativeDecodingGenerator` | 自推测解码（USD）：用目标模型自身的早期退出层生成候选 | `src/transformers/generation/candidate_generator.py:899` |
| `EarlyExitCandidateGenerator` | 基于目标模型层提前退出的候选生成 | `src/transformers/generation/candidate_generator.py:1174` |
| `MTPCandidateGenerator` | Multi-Token Prediction 候选生成器（MTP 头） | `src/transformers/generation/candidate_generator.py:1423` |
| `_speculative_sampling` | 采样模式下的推测接受/拒绝算法（speculative decoding paper Algorithm 1） | `src/transformers/generation/utils.py:4153` |
| `AssistantToTargetTranslator` | 不同词表间的 token ID 映射（草稿→目标） | `src/transformers/generation/candidate_generator.py:682` |
| 候选生成器选择 `_get_candidate_generator` | 根据 generation_config 参数自动选择 CandidateGenerator 子类 | `src/transformers/generation/utils.py:1125` |

## 2. 核心类型与接口清单

| 类型 / 函数 | 位置 | 职责 |
|-------------|------|------|
| `CandidateGenerator` | `candidate_generator.py:39` | 抽象基类；`get_candidates(input_ids)` → `(candidate_ids, candidate_logits)`；`update_candidate_strategy(input_ids, scores, num_matches)` |
| `AssistedCandidateGenerator` | `candidate_generator.py:80` | 草稿模型驱动的候选生成；管理草稿模型的 generation_config、kwargs、device 同步 |
| `PromptLookupCandidateGenerator` | `candidate_generator.py:1018` | 无草稿模型；在 prompt 中滑动窗口查找 n-gram 匹配，取后续 token 作为候选 |
| `UniversalSpeculativeDecodingGenerator` | `candidate_generator.py:899` | 自推测解码；用目标模型早期层输出生成候选 |
| `_speculative_sampling` | `utils.py:4153` | 采样接受算法：`p(x)/q(x)` 阈值接受，拒绝时重采样 |
| `AssistantToTargetTranslator` | `candidate_generator.py:682` | 词表映射：草稿模型 token ID → 目标模型 token ID |
| `num_assistant_tokens` | `configuration_utils.py` | 草稿模型每轮生成的候选 token 数；heuristic 策略动态调整 |

## 3. 关键调用链

### 3.1 推测解码主循环（`utils.py:3723-4063`）

1. **初始化**（`utils.py:3764-3800`）：要求 `use_cache=True` 且不能用 StaticCache；`cache.activate_past_recording()` 启用 Cache 记录；`_get_candidate_generator()` 选择候选生成器。
2. **主循环**（`while _has_unfinished_sequences()`）：
   - a. **获取候选**（`utils.py:3850`）：`candidate_generator.get_candidates(input_ids)` → `(candidate_input_ids, candidate_logits)`。
   - b. **目标模型批量前向**（`utils.py:3875-3905`）：将候选序列整体输入目标模型，一次前向获得 `candidate_length + 1` 个位置的 logits（`logits_to_keep = candidate_length + 1`）。
   - c. **logits 处理**（`utils.py:3913-3917`）：逐位置施加 logits_processor。
   - d. **接受/拒绝**（`utils.py:3919-3955`）：
     - 采样模式 + 有草稿 logits：调用 `_speculative_sampling()` 执行 speculative decoding 论文接受算法。
     - 贪心模式：`selected_tokens = new_logits.argmax(-1)`；`n_matches = (候选 == selected).cumprod().sum()`；接受前 `n_matches + 1` 个 token（最后一个 bonus token 来自目标模型）。
   - e. **KV Cache 裁剪**（`utils.py:3967`）：`past_key_values.crop(-(candidate_length - n_matches))` 丢弃未接受候选的 KV。
   - f. **策略更新**（`utils.py:3970`）：`update_candidate_strategy()` — 全匹配则 `num_assistant_tokens += 2`，否则 `max(1, -1)`。
3. **收尾**：返回生成结果。

### 3.2 PromptLookupDecoding 候选生成（`candidate_generator.py:1062`）

1. 从 `max_matching_ngram_size` 向下遍历 ngram_size（2→1）。
2. 在 input_ids 上创建滑动窗口，找与末尾 ngram 匹配的位置。
3. 取匹配位置后续 `num_output_tokens` 个 token 作为候选。
4. 用 logits_processor 检查候选 token 是否被禁止（设为 -inf 则截断）。
5. 截断 EOS token 后续；未找到匹配则返回原输入（退化为普通自回归）。

## 4. 配置项

| 配置项 | 默认 / 行为 | 位置 |
|--------|-------------|------|
| `assistant_model` | 草稿小模型实例；传入后启用推测解码 | `utils.py:2403`（generate 参数） |
| `prompt_lookup_num_tokens` | >0 时启用 PromptLookupDecoding，无需草稿模型 | `candidate_generator.py:1018` |
| `num_assistant_tokens` | 草稿模型每轮生成的候选数；默认启发式调整 | `configuration_utils.py` |
| `num_assistant_tokens_schedule` | `"heuristic"`/`"heuristic_transient"`/`"constant"` | `candidate_generator.py:232` |
| `assistant_confidence_threshold` | 草稿模型置信度阈值；低于此停止草稿生成 | `candidate_generator.py:130` |
| `assistant_ensemble_weight` | 贪心模式下草稿/目标 logits 融合权重 | `utils.py:3940` |
| `use_mtp` | 启用 Multi-Token Prediction 候选生成 | `configuration_utils.py:158` |

## 5. 错误与重试语义

- **Cache 类型限制**：推测解码要求 DynamicCache（不支持 StaticCache），否则抛 `ValueError`（`utils.py:3764`）。
- **batch_size 限制**：仅支持 `batch_size=1`，>1 抛 `ValueError`（`utils.py:3843`）。
- **无匹配退化**：若候选全部被拒绝（`n_matches=0`），仍接受 1 个 bonus token，退化为普通贪心/采样，不报错。
- **长度预算保护**：接受 token 数超过 `max_length - cur_len` 时截断（`utils.py:3958`）。
- **Cache crop 容错**：`crop(0)` 仍调用以收缩 sliding window/linear attention cache（`utils.py:3967`）。

## 6. 并发细节

- **单 batch 串行**：`batch_size=1` 约束下完全串行执行。
- **草稿模型与目标模型设备分离**：候选生成在草稿模型设备上执行，候选 tokens 复制到目标模型设备（`candidate_input_ids.to(self.device)`）。
- **Cache 记录模式**：`activate_past_recording()` 启用 Cache 异步记录，配合 `DeferredStopCheck` 减少同步点。
- **KV Cache 裁剪**：每轮根据 `n_matches` 裁剪未接受 token 的 KV 条目，避免显存浪费。

## 7. 系统边界

**In-Scope（本仓库源码内）**
- `src/transformers/generation/candidate_generator.py`：全部 CandidateGenerator 子类
- `src/transformers/generation/utils.py`：`_assisted_decoding`、`_speculative_sampling`、`_get_candidate_generator`

**Out-of-Scope（不在本仓库源码内）**
- 草稿模型本身的 forward 计算（由草稿模型的 `modeling_*.py` 实现）
- 目标模型 forward 计算（见 generation-loop 叶子）
- speculative decoding 数学理论（论文 arxiv 2211.17192）

## 8. 与相邻子系统交互

- **上游 → 本叶子**：`generate()` 检测到 `assistant_model` 或 `prompt_lookup_num_tokens` 参数，通过 `GENERATION_MODES_MAPPING` 分发到 `_assisted_decoding()`。
- **本叶子 → 下游**：
  - → 草稿模型 `assistant_model.generate()`：生成候选 token 序列
  - → 目标模型 `forward()`：批量验证候选
  - → KV Cache：`activate_past_recording()` + `crop()`
  - → LogitsProcessorList：对验证 logits 逐位置施加

## 9. 语言专项适配口径

- Python 库，按"候选生成策略"capability seam 划分（草稿模型/PromptLookup/USD/MTP）。
- 图类型：architecture（组件关系）+ sequence（候选-验证-接受时序）；lifecycle 不适用（核心循环已在 generation-loop 叶子覆盖）。
- 外部边界：草稿模型为独立模型实例，标注为外部依赖。

## 10. 图表清单与质量

| 图 | 文件 | 类型 | archify 质量档 |
|----|------|------|----------------|
| 推测解码组件架构图 | `generation-assisted-architecture.html` | architecture | standard（布局标签密度高，降档披露） |
| 候选-验证-接受时序图 | `generation-assisted-sequence.html` | sequence | showcase |

JSON IR 源文件位于 `json/` 目录。lifecycle 图不单独产出：推测解码循环的状态机与 generation-loop 共享，避免重复。
