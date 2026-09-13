# Sequence-to-Sequence 模型（seq2seq-models）

> 本文是 `models-registry` 域下的叶子子系统文档。域级总览见 `../models-registry.md`。
> 本叶子覆盖**编码器-解码器（encoder-decoder）家族**：encoder 读源序列，decoder 经 cross-attention 生成目标序列。
>
> 源码基准：HuggingFace transformers v5.18.0.dev0（main），commit `5474a55e920f358d8382f3ecd3377edca979baa`。

## 1. 功能清单（覆盖范围披露）

**深读家族：**

| 家族 | 能力 | 源码路径 |
|------|------|----------|
| T5 | 文本到文本统一框架；相对位置偏置；encoder-decoder 条件生成 | `models/t5/modeling_t5.py`（`T5Attention:176`、`_relative_position_bucket:217`、`T5LayerNorm:50`） |
| BART | 去噪预训练；学习式位置嵌入；encoder/decoder 各自堆叠 | `models/bart/modeling_bart.py`（`BartEncoderLayer:260`、`BartDecoderLayer:311`、`BartEncoder:463`、`BartDecoder:552`） |
| Pegasus | BART 变体，句间 gap 句子生成预训练 | `models/pegasus/` |

**仅列表说明（不逐个深读）：** Marian、MBart、NLLB、M2M100、ProphetNet、Blenderbot、mBART 等。
它们共享"双向 encoder + 因果 decoder + cross-attention"骨架，差异在分词、位置编码与预训练目标。

## 2. 核心类型与接口清单（以 T5/BART 为代表）

| 类型 | 位置 | 职责 |
|------|------|------|
| `T5Attention` | `t5/modeling_t5.py:176` | 自/cross 注意力；`is_decoder`（`:186`）区分 encoder/decoder 层；用相对位置偏置而非绝对位置嵌入 |
| `_relative_position_bucket` | `:217` | 把相对距离分桶（小距离精确、大距离对数桶），作为注意力偏置 |
| `T5LayerNorm` | `:50` | T5 式均方根 LayerNorm（无均值减） |
| `T5LayerFF` | `:126` | 前馈（DenseAct 或 GatedAct） |
| `BartEncoderLayer` / `BartDecoderLayer` | `bart/modeling_bart.py:260`/`:311` | encoder 双向层 / decoder 因果层；decoder 层含对 encoder 输出的 cross-attention |
| `BartLearnedPositionalEmbedding` | `:74` | BART 学习式位置嵌入（替代 T5 的相对偏置） |
| `BartEncoder` / `BartDecoder` | `:463`/`:552` | 编码器堆叠 / 解码器堆叠 |

## 3. 关键调用链

**T5ForConditionalGeneration 前向（`t5/modeling_t5.py`）**
1. encoder：输入嵌入 → N 层 `T5Attention(is_decoder=False)` 双向自注意力（带相对位置偏置）→ FFN，得 `encoder_outputs`。
2. decoder：目标序列嵌入 → 因果自注意力（`is_decoder=True`）→ **cross-attention**：Q 来自 decoder，K/V 来自 `encoder_outputs`（`encoder_outputs` 作为 kwarg 传入）→ FFN。
3. 输出 logits 过 LM head，逐位置预测目标 token；生成时 `GenerationMixin` 以 KV cache 自回归展开。

**BART**：`BartEncoder`（`:463`）双向读源；`BartDecoder`（`:552`）每层先因果自注意力，再 cross-attention 关注 encoder 隐状态。位置用 `BartLearnedPositionalEmbedding`（`:74`）。

**关键数据流**：`encoder_outputs`（含 last_hidden_state/attention mask）从 encoder 跨接给 decoder 的 cross-attention——这是 seq2seq 区别于纯 decoder 的核心。

## 4. 配置项

| 配置项 | 默认 / 行为 | 位置 |
|--------|-------------|------|
| `model_type` | `"t5"` / `"bart"` | `configuration_t5.py`、`configuration_bart.py` |
| `is_decoder` | 标记该层是否为 decoder（决定是否加因果掩码与 cross-attention） | `modeling_t5.py:186` |
| `d_ff` / `d_model` / `num_layers` | 前馈维 / 模型维 / 层数 | 各配置 |
| `relative_attention_num_buckets` / `max_distance` | T5 相对位置桶数与上限 | `modeling_t5.py:217` |
| `is_encoder_decoder` | 标记条件生成模型（生成配置） | `configuration_utils` |

## 5. 错误与重试语义

- 前向为张量计算；cross-attention 中 encoder/decoder 序列长度不匹配时由 PyTorch 报 shape 错。
- 生成时 `encoder_outputs` 未传入或维度不符由生成循环校验。无业务重试/退避。

## 6. 并发细节

- 同步张量计算；encoder 全序列并行，decoder 自回归逐步。
- KV cache：decoder 侧缓存自注意力历史；cross-attention 的 K/V 来自一次性编码的 `encoder_outputs`，不重复计算。
- 无 Python 线程模型。

## 7. 系统边界

**In-Scope（本仓库源码内）**
- `models/t5/`、`models/bart/`、`models/pegasus/` 的 modeling/configuration。

**Out-of-Scope（不在本仓库源码内）**
- `GenerationMixin` 生成循环（`src/transformers/generation/`）。
- 不在本叶子：纯 encoder（encoder-models）、纯 decoder（decoder-models）、多模态 seq2seq（multimodal/speech）。

## 8. 与相邻子系统交互

- 上游：用户经 `AutoModelForSeq2SeqLM`（auto-registry）按 `model_type` 分发。
- 下游：混入 `GenerationMixin` 做条件生成；`encoder_outputs` 跨接是核心数据流。
- 生成关系：文件由 modular/`# Copied from` 机制产生（modular-system）。

## 9. 语言专项适配口径（Python）

- **分组**：按 capability seam——"encoder 编码"与"decoder 解码 + cross-attention"。
- **图类型**：architecture（encoder-decoder 组件与 cross-attention 数据走向）为主；数据流在 MD 文字说明。不产出 lifecycle。
- **外部边界**：PyTorch 注意力核外部。

## 10. 图表清单与质量

| 图 | 文件 | 类型 | archify 质量档 |
|----|------|------|----------------|
| encoder-decoder 架构（T5 代表） | `seq2seq-models-architecture.html` | architecture | **showcase**（0 错误通过） |

JSON IR 源：`json/seq2seq-models-architecture.json`。
不另出 sequence：cross-attention 数据流已在 architecture 中以连线表达。

**覆盖范围披露**：深读 T5/BART/Pegasus 三个家族；其余（Marian、MBart、NLLB、M2M100、ProphetNet、Blenderbot 等）仅按共性列表说明，未逐类读源码。
