# Decoder-only 模型（decoder-models）

> 本文是 `models-registry` 域下的叶子子系统文档。域级总览见 `../models-registry.md`。
> 本叶子覆盖**因果语言模型（decoder-only）家族**：自回归生成，下三角注意力掩码。
>
> 源码基准：HuggingFace transformers v5.18.0.dev0（main），commit `5474a55e920f358d8382f3ecd3377edca979baa`。

## 1. 功能清单（覆盖范围披露）

**深读家族（逐类读源码）：**

| 家族 | 能力 | 源码路径 |
|------|------|----------|
| LLaMA | decoder-only 基座：RMSNorm + RoPE + SwiGLU + GQA，KV cache | `models/llama/modeling_llama.py`（`LlamaRMSNorm:53`、`LlamaAttention:217`、`LlamaDecoderLayer:284`、`LlamaForCausalLM:421`） |
| Qwen2 | 类 LLaMA，支持 sliding window attention 与 GQA | `models/qwen2/configuration_qwen2.py:42`（`num_key_value_heads:66`、`use_sliding_window:74`） |
| GPT2 | 早期 decoder-only 代表（学习式位置嵌入、LayerNorm） | `models/gpt2/` |
| Gemma | Google decoder-only，类 LLaMA 结构 | `models/gemma/` |
| Mistral | LLaMA 变体 + sliding window attention | `models/mistral/` |

**仅列表说明（不逐个深读，架构与 LLaMA/GPT2 共性）：** Falcon、Phi、Mixtral（MoE 路由）、Qwen2.5、
StableLM、Mamba、Gemma2/3、OLMo、DeepSeek（ dense/MoE）、CoHere 等。它们共享因果自注意力 + 前馈堆叠，
差异集中在归一化、激活函数、位置编码、注意力窗口与专家路由。

## 2. 核心类型与接口清单（以 LLaMA 为代表）

| 类型 | 位置 | 职责 |
|------|------|------|
| `LlamaRMSNorm` | `llama/modeling_llama.py:53` | 均方根归一化（等价 T5LayerNorm），无均值减项；`@use_kernel_forward_from_hub` 可换 Hub 优化核 |
| `LlamaRotaryEmbedding` | `:73` | RoPE 旋转位置编码，对 Q/K 施加角度旋转 |
| `LlamaMLP` | `:163` | SwiGLU 门控升维/降维 |
| `repeat_kv`（内联） | `:182` | GQA：把 `num_key_value_heads` 个 K/V 复制扩到 `num_attention_heads` |
| `LlamaAttention` | `:217` | 因果自注意力；`num_key_value_groups = num_attention_heads // num_key_value_heads`（`:225`）实现 GQA；KV 投影到 KV 头数（`:234`） |
| `LlamaDecoderLayer` | `:284` | Pre-Norm：`input_layernorm` → Attention → `post_attention_layernorm` → MLP；继承 `GradientCheckpointingLayer` |
| `LlamaModel` | `:347` | token 嵌入 + N 层 decoder + 最终 `norm` + `rotary_emb` |
| `LlamaForCausalLM` | `:421` | `LlamaModel` + `lm_head`，混入 `GenerationMixin` 提供 `generate` |

## 3. 关键调用链

**LlamaForCausalLM 前向（`modeling_llama.py`）**
1. token 嵌入 → `LlamaModel`（`:347`）。
2. 逐层 `LlamaDecoderLayer.forward`（`:295`）：`hidden = input_layernorm(hidden)` → `LlamaAttention`（因果掩码 + RoPE 旋转 Q/K + GQA 复制 K/V）→ 残差；再 `post_attention_layernorm` → `LlamaMLP`（SwiGLU）→ 残差。
3. 全部层后过最终 `LlamaRMSNorm`（`:357`）。
4. `LlamaForCausalLM.forward`（`:438`）经 `lm_head` 出 logits；训练时算交叉熵，生成时由 `GenerationMixin` 用 KV cache 自回归逐 token 预测。

**Qwen2 与 LLaMA 差异**（`qwen2/configuration_qwen2.py`）：默认 `num_key_value_heads=32`（GQA）；`use_sliding_window`（`:74`）开启后 `sliding_window=4096`（`:75`），并对靠后的层限制窗口（`:91`）。其余堆叠与 LLaMA 同构。

**Mistral**：在 LLaMA 基础上引入 sliding window attention；**Mixtral** 把 MLP 换成 MoE 路由（按 token 选专家）。

## 4. 配置项

| 配置项 | 默认 / 行为 | 位置 |
|--------|-------------|------|
| `model_type` | `"llama"` / `"qwen2"` | `configuration_llama.py`、`configuration_qwen2.py:42` |
| `num_hidden_layers` / `num_attention_heads` | 层数 / Q 头数 | 各配置 |
| `num_key_value_heads` | GQA 的 KV 头数；Qwen2 默认 32 | `configuration_qwen2.py:66` |
| `use_sliding_window` / `sliding_window` | 滑窗注意力开关与窗口大小 | `configuration_qwen2.py:74` |
| `rms_norm_eps` | RMSNorm 数值 eps | `modeling_llama.py:292` |
| `attention_bias` | Q/K/V/O 线性层是否带 bias | `modeling_llama.py:234` |
| `attn_implementation` | eager/sdpa/flash_attention_* | auto 层透传 |

## 5. 错误与重试语义

- 前向/生成为张量计算，shape 错配由 PyTorch 报错。
- KV cache 不匹配（如传入旧 `past_key_values` 长度）由 `PreTrainedModel`/`generate` 校验。
- 权重加载缺失/多余 key 统一报告。无业务重试/退避。

## 6. 并发细节

- 前向同步张量计算；`GradientCheckpointingLayer` 以显存换速度。
- **KV cache**：自回归生成时缓存历史 K/V，避免每步重算——这是 decoder 区别于 encoder 的核心运行时结构（由 `GenerationMixin` 管理，`src/transformers/generation/`，不在本叶子展开）。
- flash-attention 内部 CUDA 并行不在本仓库源码内。

## 7. 系统边界

**In-Scope（本仓库源码内）**
- `models/llama/`、`models/qwen2/`、`models/gpt2/`、`models/gemma/`、`models/mistral/` 的 modeling/configuration。

**Out-of-Scope（不在本仓库源码内）**
- `GenerationMixin`/`generate` 采样与 beam search 循环：在 `src/transformers/generation/`，属相邻子系统。
- flash-attention 内核、CUDA 并行。
- 不在本叶子：encoder（encoder-models）、seq2seq（seq2seq-models）。

## 8. 与相邻子系统交互

- 上游：用户经 `AutoModelForCausalLM`（auto-registry 叶子）按 `model_type` 分发。
- 下游：`LlamaForCausalLM` 混入 `GenerationMixin`（generation 子系统）做自回归生成；隐状态可被多模态模型（multimodal-models）作语言模型部分复用。
- 生成关系：文件由 modular 源/`# Copied from` 同步产生（modular-system 叶子）。

## 9. 语言专项适配口径（Python）

- **分组**：按 capability seam——"解码器堆叠（RMSNorm+Attention+MLP）"与"生成头（LM head+GenerationMixin）"。
- **图类型**：architecture（decoder block 组件）为主；前向与 KV cache 机制在 MD 文字说明。不产出 lifecycle：无训练/运行状态机（生成循环在 generation 子系统，不在本叶子）。
- **外部边界**：flash-attn、PyTorch 标注外部。

## 10. 图表清单与质量

| 图 | 文件 | 类型 | archify 质量档 |
|----|------|------|----------------|
| decoder-only 架构（LLaMA 代表） | `decoder-models-architecture.html` | architecture | **showcase**（0 错误通过） |

JSON IR 源：`json/decoder-models-architecture.json`。
不另出 sequence：decoder 前向为确定性堆叠调用，文字调用链已表达。

**覆盖范围披露**：深读 LLaMA/Qwen2/GPT2/Gemma/Mistral 五个家族；其余 decoder 家族（Falcon、Phi、Mixtral、Qwen2.5、StableLM、Mamba、DeepSeek 等）仅按共性列表说明，未逐类读源码。
