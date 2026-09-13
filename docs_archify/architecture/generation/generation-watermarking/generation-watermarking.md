# generation-watermarking（文本水印）

> 本文是 `generation` 域下的叶子子系统文档。域级总览见 `../generation.md`。
> 本文只展开**文本水印的注入与检测机制**，不重复展开 logits 处理器链的组装（见 `../generation-loop/`）。
>
> 源码基准：`transformers` main 分支，commit `5474a55e920f358d8382f3ecd3377edca979baa`。

## 1. 功能清单

| 能力 | 说明 | 源码路径 |
|------|------|----------|
| 水印注入 `WatermarkLogitsProcessor` | 在生成过程中对"绿名单"token 的 logits 加偏置 bias，使模型倾向于选择绿名单 token | `src/transformers/generation/logits_process.py:2389` |
| 绿名单生成 `_get_greenlist_ids` | 根据前 context_width 个 token 哈希种子 → RNG 打乱词表 → 前 `greenlist_ratio * vocab_size` 个 token 为绿名单 | `src/transformers/generation/logits_process.py:2486` |
| `lefthash` 种子方案 | 绿名单仅依赖前一个 token 的哈希（论文 Algorithm 2） | `src/transformers/generation/logits_process.py:2478` |
| `selfhash` 种子方案 | 绿名单依赖当前候选 token 自身（论文 Algorithm 3），通过拒绝采样确定 | `src/transformers/generation/logits_process.py:2494` |
| 水印检测 `WatermarkDetector` | 对文本逐 n-gram 统计绿名单命中频率，计算 z-score 统计检验 | `src/transformers/generation/watermarking.py:71` |
| z-score 统计检验 | `z = (green_count - γ*N) / sqrt(N*γ*(1-γ))`；z > threshold 判定为有水印 | `src/transformers/generation/watermarking.py:180` |
| `WatermarkingConfig` | 水印配置：greenlist_ratio、bias、seeding_scheme、context_width、hashing_key | `src/transformers/generation/configuration_utils.py:1453` |
| SynthID 文本水印 | Google SynthID 水印 logits 处理器与检测器 | `src/transformers/generation/logits_process.py:2562`；`src/transformers/generation/watermarking.py:483` |
| 贝叶斯水印检测器 | 基于神经网络的贝叶斯水印检测模型 | `src/transformers/generation/watermarking.py:350` |

## 2. 核心类型与接口清单

| 类型 / 函数 | 位置 | 职责 |
|-------------|------|------|
| `WatermarkLogitsProcessor` | `logits_process.py:2389` | LogitsProcessor 子类；`__call__(input_ids, scores)` 对绿名单 token 加 bias |
| `WatermarkDetector` | `watermarking.py:71` | 检测器；`__call__(input_ids, z_threshold=3.0)` 返回布尔预测 |
| `WatermarkingConfig` | `configuration_utils.py:1453` | 水印参数配置；`construct_processor(vocab_size, device)` 实例化处理器 |
| `SynthIDTextWatermarkLogitsProcessor` | `logits_process.py:2562` | SynthID 水印处理器 |
| `SynthIDTextWatermarkDetector` | `watermarking.py:483` | SynthID 检测器 |
| `_get_greenlist_ids` | `logits_process.py:2486` | 核心：种子→打乱词表→绿名单 |
| `_compute_z_score` | `watermarking.py:180` | 统计检验：绿名单命中率的 z-score |

## 3. 关键调用链

### 3.1 水印注入（每步生成）

1. `generate()` 的 `_get_logits_processor()` 检测到 `watermarking_config`（`utils.py:1475`），调用 `construct_processor()` 实例化 `WatermarkLogitsProcessor`，追加到处理器链末尾（在所有采样 warper 之后、renormalize 之前）。
2. 每个生成步，`WatermarkLogitsProcessor.__call__(input_ids, scores)`：
   - a. `set_seed(input_ids[-context_width:])`：lefthash 模式用 `hash_key * last_token` 作种子；selfhash 模式用前 context_width 个 token 的查表值乘积作种子。
   - b. `torch.randperm(vocab_size, generator=rng)`：用种子 RNG 打乱整个词表。
   - c. 取前 `greenlist_size = int(vocab_size * greenlist_ratio)` 个 token ID 为绿名单。
   - d. `scores[greenlist_ids] += bias`：对绿名单 token 的 logits 加偏置。

### 3.2 水印检测

1. `WatermarkDetector.__call__(input_ids, z_threshold=3.0)`：
   - a. 去除 BOS token。
   - b. `_score_ngrams_in_passage()`：对每个 n-gram（长度 = context_width + 1），用 LRU 缓存的 `_get_ngram_score_cached(prefix, target)` 判断 target token 是否在 prefix 对应的绿名单中。
   - c. 统计 `green_token_count`（绿名单命中数）和 `num_tokens_scored`（总 n-gram 数）。
   - d. `_compute_z_score()`：z = (green_count - γ*N) / sqrt(N*γ*(1-γ))。
   - e. `prediction = z > z_threshold`。
   - f. 可选计算 p-value 和 confidence。

## 4. 配置项

| 配置项 | 默认 / 行为 | 位置 |
|--------|-------------|------|
| `greenlist_ratio` | `0.25`；绿名单占词表比例 γ | `logits_process.py:2430` |
| `bias` | `2.0`；绿名单 token logits 偏置强度；推荐 [0.5, 2.0] | `logits_process.py:2431` |
| `hashing_key` | `15485863`（第 100 万素数）；生产环境应替换为私钥 | `logits_process.py:2432` |
| `seeding_scheme` | `"lefthash"`；可选 `"selfhash"` | `logits_process.py:2433` |
| `context_width` | `1`；种子使用的前 N 个 token 数 | `logits_process.py:2434` |
| `z_threshold` | `3.0`；检测器 z-score 阈值 | `watermarking.py:191` |

## 5. 错误与重试语义

- **种子方案校验**：`seeding_scheme` 非 `lefthash`/`selfhash` 抛 `ValueError`（`logits_process.py:2468`）。
- **绿名单比例校验**：`greenlist_ratio` 必须在 (0, 1) 开区间内（`logits_process.py:2470`）。
- **短序列警告**：`input_ids` 长度 < `context_width` 时跳过本步水印并警告（`logits_process.py:2519`）。
- **检测序列长度校验**：文本短于 `context_width + 1` 抛 `ValueError`（`watermarking.py:227`）。
- **无重试**：水印注入是确定性 logits 变换，检测是统计计算，无重试机制。

## 6. 并发细节

- **LRU 缓存**：检测器用 `functools.lru_cache(maxsize=128)` 缓存 `_get_ngram_score()`，避免对重复 n-gram 重复哈希/打乱（`watermarking.py:143`）。
- **RNG 确定性**：水印注入用 `torch.Generator` 而非全局 RNG，保证种子可复现（`logits_process.py:2476`）。
- **无多线程**：注入和检测均为单线程同步计算。

## 7. 系统边界

**In-Scope（本仓库源码内）**
- `src/transformers/generation/logits_process.py`：`WatermarkLogitsProcessor`、`SynthIDTextWatermarkLogitsProcessor`
- `src/transformers/generation/watermarking.py`：`WatermarkDetector`、贝叶斯检测器、SynthID 检测器
- `src/transformers/generation/configuration_utils.py`：`WatermarkingConfig`

**Out-of-Scope（不在本仓库源码内）**
- 水印论文理论（arxiv 2306.04634）
- SynthID 外部模型（Google 训练的贝叶斯检测器权重）
- PyTorch 哈希/RNG 实现

## 8. 与相邻子系统交互

- **上游 → 本叶子**：`_get_logits_processor()` 在处理器链末尾追加 `WatermarkLogitsProcessor`（`utils.py:1475`）。
- **本叶子 → 下游**：
  - → LogitsProcessorList：水印处理器作为链中最后一个处理器（renormalize 之前）。
  - → 用户代码：`WatermarkDetector` 独立调用，对已有文本进行水印检测。

## 9. 语言专项适配口径

- Python 库，按"水印注入 / 水印检测"两个 capability seam 划分。
- 图类型：architecture（注入-检测组件关系）+ sequence（水印注入单步流程）；lifecycle 不适用（水印是无状态每步变换）。
- 外部边界：SynthID 外部模型权重标注为外部组件。

## 10. 图表清单与质量

| 图 | 文件 | 类型 | archify 质量档 |
|----|------|------|----------------|
| 水印注入与检测架构图 | `generation-watermarking-architecture.html` | architecture | standard（降档披露） |
| 水印注入单步流程图 | `generation-watermarking-sequence.html` | sequence | showcase |

JSON IR 源文件位于 `json/` 目录。lifecycle 图不产出：水印注入是每步确定性 logits 变换，无状态机语义。
