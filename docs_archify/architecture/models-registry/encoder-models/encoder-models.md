# Encoder 类模型（encoder-models）

> 本文是 `models-registry` 域下的叶子子系统文档。域级总览见 `../models-registry.md`。
> 本叶子覆盖**双向（非因果）Transformer encoder 家族**：以 BERT 为范式，叠 N 层双向自注意力。
>
> 源码基准：HuggingFace transformers v5.18.0.dev0（main），commit `5474a55e920f358d8382f3ecd3377edca979baa`。

## 1. 功能清单（覆盖范围披露）

**深读家族（逐类读源码）：**

| 家族 | 能力 | 源码路径 |
|------|------|----------|
| BERT | 双向 encoder 基座 + 多种任务头（MLM/NSP/序列分类/token 分类/问答/多选） | `models/bert/modeling_bert.py`（`BertModel:594`、`BertForMaskedLM:909`、`BertForSequenceClassification:1072`） |
| RoBERTa | BERT 改良：无 `token_type_ids`、动态 padding、GottBERT 等 | `models/roberta/modeling_roberta.py`（`RobertaModel:551`） |
| DeBERTa | 解耦注意力（内容与位置解耦）+ 增强掩码 | `models/deberta/modeling_deberta.py`（`DebertaModel:627`） |
| ELECTRA | 生成器-判别器架构，判别式预训练 | `models/electra/modeling_electra.py`（`ElectraDiscriminatorPredictions:465`、`ElectraModel:545`） |

**仅列表说明（不逐个深读，架构与 BERT 共性）：** AlBERT、XLNet、XLM、Flaubert、CamemBERT、
XLMRoberta、ConvBert、MobileBERT、SqueezeBert、FNet、LXMERT、ERNIE、BigBird、Longformer 等。
它们共享"嵌入 + N 层双向自注意力 + 残差 LayerNorm + 任务头"骨架，差异集中在嵌入表、位置编码与预训练任务。

## 2. 核心类型与接口清单（以 BERT 为代表）

| 类型 | 位置 | 职责 |
|------|------|------|
| `BertEmbeddings` | `modeling_bert.py:53` | token + position + `token_type_embeddings`（`:60`）三路嵌入相加 + LayerNorm + dropout |
| `BertSelfAttention` | `:139` | 双向（非因果）缩放点积自注意力，支持 `attn_implementation`（eager/sdpa/flash） |
| `BertAttention` | `:296` | SelfAttention + SelfOutput（残差+LN） |
| `BertLayer` | `:354` | Attention → Intermediate → Output，继承 `GradientCheckpointingLayer` 支持梯度检查点 |
| `BertEncoder` | `:419` | 堆叠 `config.num_hidden_layers` 个 `BertLayer` |
| `BertPooler` | `:451` | 取 `[CLS]` 隐状态过 `nn.Linear`+tanh（用于分类句向量） |
| `BertForMaskedLM` / `BertForSequenceClassification` / `BertForTokenClassification` | `:909`/`:1072`/`:1251` | 同一 `BertModel` backbone 接不同任务头 |
| `ElectraDiscriminatorPredictions` | `electra/modeling_electra.py:465` | ELECTRA 判别器头，预测每个 token 是否被替换 |

## 3. 关键调用链

**BertForMaskedLM 前向（`modeling_bert.py`）**
1. `BertEmbeddings.forward`（`:68`）：`inputs_embeds + token_type_embeddings + position_embeddings` → LN → dropout（`:100`-`101`）。
2. `BertEncoder.forward`（`:425`）逐层调用 `BertLayer.forward`（`:374`）。
3. 每层：`BertAttention`（SelfAttention → SelfOutput 残差 LN）→ `BertIntermediate`（GELU 升维）→ `BertOutput`（降维残差 LN）。
4. `BertModel`（`:594`）输出 sequence hidden states；`BertForMaskedLM`（`:909`）接 `BertLMPredictionHead`（`:483`）预测被 mask token。

**RoBERTa 与 BERT 差异**（`roberta/modeling_roberta.py`）：`RobertaEmbeddings`（`:56`）不使用 `token_type_ids`，改用 `token_type_ids` 默认全 0 与 GPT-2 式字节级 BPE；其余堆叠结构与 BERT 同构。当前 main 分支中 RoBERTa 已为**独立实现**（非 `# Copied from` BERT）。

**ELECTRA 判别**：`ElectraModel`（`electra/modeling_electra.py:545`）为共享 encoder；`ElectraForPreTraining` 接 `ElectraDiscriminatorPredictions`（`:465`）逐 token 二分类"是否被生成器替换"。

## 4. 配置项

| 配置项 | 默认 / 行为 | 位置 |
|--------|-------------|------|
| `model_type` | `"bert"` | `configuration_bert.py:42` |
| `num_hidden_layers` | 堆叠层数（BERT-base=12） | 各 `configuration_*.py` |
| `hidden_size` / `num_attention_heads` | 隐藏维 / 注意力头数 | 同上 |
| `type_vocab_size` | token_type 嵌入表大小 | `configuration_bert.py` |
| `attn_implementation` | eager/sdpa/flash_attention_* | 经 `_BaseAutoModelClass` 透传 |
| `gradient_checkpointing` | `BertLayer` 继承 `GradientCheckpointingLayer` 可开 | `modeling_bert.py:354` |

## 5. 错误与重试语义

- 模型前向为纯张量计算，错误多为 shape 不匹配（如 `token_type_ids` 维度）时由 PyTorch 抛 `RuntimeError`。
- 权重加载阶段的缺失/多余 key 由 `PreTrainedModel.from_pretrained` 统一报告（missing/unexpected keys），不在本叶子。
- 无网络重试/退避；无业务错误恢复。

## 6. 并发细节

- 前向为同步张量计算，无 Python 线程/协程模型；`GradientCheckpointingLayer` 以显存换算力（反向时重算前向），不引入并发。
- 注意力计算在 PyTorch 内部并行（SDPA/flash 内部 CUDA 核并行，不在本仓库源码内）。
- KV cache 在纯 encoder 双向模型中不适用（非自回归生成）。

## 7. 系统边界

**In-Scope（本仓库源码内）**
- `models/bert/`、`models/roberta/`、`models/deberta/`、`models/electra/` 的 modeling/configuration/tokenization。

**Out-of-Scope（不在本仓库源码内）**
- PyTorch 注意力核、CUDA 并行；tokenizers/sentencepiece 分词后端。
- 不在本叶子：decoder 因果模型（decoder-models）、seq2seq（seq2seq-models）。

## 8. 与相邻子系统交互

- 上游：用户经 `AutoModel`/`AutoModelForMaskedLM`（auto-registry 叶子）按 `model_type="bert"` 分发到本家族。
- 下游：输出隐状态 → 任务头；或作为 encoder 接入多模态/seq2seq（见对应叶子）。
- 生成关系：各家族文件由 modular 源或 `# Copied from` 同步机制产生（modular-system 叶子）。

## 9. 语言专项适配口径（Python）

- **分组**：按 capability seam——"共享 encoder 堆叠（BertEncoder/Layer/Attention）"与"任务头族"两条 seam。
- **图类型**：architecture（组件堆叠）为主；前向调用链在 MD 第 3 节文字化，不另出 sequence（单层结构清晰，文字已足够）。不产出 dataflow/lifecycle：无管道数据流或训练状态机。
- **外部边界**：PyTorch 注意力核、tokenizers 标注外部。

## 10. 图表清单与质量

| 图 | 文件 | 类型 | archify 质量档 |
|----|------|------|----------------|
| encoder 家族架构（BERT 代表） | `encoder-models-architecture.html` | architecture | **standard**（子组件边界连线约束回退 standard；已如实披露） |

JSON IR 源：`json/encoder-models-architecture.json`。
不另出 sequence：BERT 前向为确定性堆叠调用，文字调用链已表达；无状态机故不产 lifecycle。

**覆盖范围披露**：深读 BERT/RoBERTa/DeBERTa/ELECTRA 四个家族；其余约数十个 encoder 家族（AlBERT、XLNet、XLM、CamemBERT、XLMRoberta 等）仅按共性列表说明，未逐类读源码。
