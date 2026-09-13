# 语音模型（speech-models）

> 本文是 `models-registry` 域下的叶子子系统文档。域级总览见 `../models-registry.md`。
> 本叶子覆盖**语音/音频模型**：语音识别、语音翻译、神经音频编解码。
>
> 源码基准：HuggingFace transformers v5.18.0.dev0（main），commit `5474a55e920f358d8382f3ecd3377edca979baa`。

## 1. 功能清单（覆盖范围披露）

**深读家族：**

| 家族 | 能力 | 源码路径 |
|------|------|----------|
| Whisper | 多语言语音识别：log-mel 输入 + encoder-decoder | `models/whisper/modeling_whisper.py`（`shift_tokens_right:69`、`WhisperPositionalEmbedding:205`、`WhisperAttention:242`） |
| SeamlessM4T | 统一语音翻译：Conformer encoder + 语音/文本多模态 | `models/seamless_m4t/modeling_seamless_m4t.py`（`SeamlessM4TConformerEncoderLayer:622`、`SeamlessM4TGenerationOutput:121`） |
| EnCodec | 神经音频编解码：encoder + 残差向量量化 + decoder | `models/encodec/modeling_encodec.py`（`EncodecConv1d:82`、`EncodecLSTM:236`、`EncodecResnetBlock:252`） |

**仅列表说明（不逐个深读）：** Wav2Vec2、Hubert、Speech2Text、Clap、MusicGen、Bark、
SpeechT5、MMS、Canary 等。它们共享"音频特征 → encoder → 文本/音频 token"骨架。

## 2. 核心类型与接口清单（以 Whisper/EnCodec 为代表）

| 类型 | 位置 | 职责 |
|------|------|------|
| `WhisperAttention` | `whisper/modeling_whisper.py:242` | 自/cross 注意力；`is_decoder`（`:250`）区分 encoder/decoder 层 |
| `shift_tokens_right` | `:69` | seq2seq 训练时把 decoder 输入右移一位并填 `decoder_start_token_id` |
| `WhisperPositionalEmbedding` | `:205` | decoder 位置嵌入 |
| `SeamlessM4TConformerEncoderLayer` | `seamless_m4t/modeling_seamless_m4t.py:622` | Conformer 结构（自注意力 + 卷积模块 + FFN） |
| `EncodecConv1d` / `EncodecConvTranspose1d` | `encodec/modeling_encodec.py:82`/`:179` | 一维卷积下采样/上采样 |
| EnCodec quantizer | `encodec/` | 残差向量量化，把连续音频特征离散化为 token |

## 3. 关键调用链

**WhisperForConditionalGeneration 前向**
1. 输入 log-mel 谱（由 `WhisperFeatureExtractor` 从波形计算，不在本叶子）。
2. Whisper encoder 双向编码声学特征。
3. Whisper decoder 因果自注意力 + 对 encoder 输出的 cross-attention；`shift_tokens_right`（`whisper/modeling_whisper.py:69`）构造 decoder 输入。
4. 出文本 token 概率，`generate` 自回归转录；`is_decoder`（`:250`）标志区分层类型。

**EnCodec 编解码**
1. `EncodecConv1d` 序列下采样波形 → `EncodecResnetBlock`/LSTM 提特征。
2. quantizer 残差向量量化为离散音频 token。
3. `EncodecConvTranspose1d` 上采样重建波形——用于语音/音乐生成模型的离散 token 化。

**SeamlessM4T**：Conformer encoder 编码语音，统一支持语音→文本/语音→语音多任务翻译。

## 4. 配置项

| 配置项 | 默认 / 行为 | 位置 |
|--------|-------------|------|
| `model_type` | `"whisper"`/`"seamless_m4t"`/`"encodec"` | 各配置 |
| `num_mel_bins` / `sampling_rate` | Whisper log-mel 特征维与采样率 | `configuration_whisper.py` |
| `max_source_positions` / `max_target_positions` | encoder/decoder 最大长度 | 同上 |
| `target_lang` / `language` | Whisper/SeamlessM4T 目标语言 | 生成配置 |
| `compress` / `quantization` | EnCodec 压缩率与量化组数 | `configuration_encodec.py` |

## 5. 错误与重试语义

- 前向为张量计算；log-mel 帧长与 `max_source_positions` 不符时报错。
- 音频预处理（log-mel）在 feature extractor，不在本叶子。无业务重试。

## 6. 并发细节

- 同步张量计算；encoder 全序列并行，decoder 自回归。
- Whisper 复用 seq2seq 的 KV cache 模式（cross-attention K/V 来自一次性 encoder）。
- EnCodec 的量化为同步查表。无 Python 线程模型。

## 7. 系统边界

**In-Scope（本仓库源码内）**
- `models/whisper/`、`models/seamless_m4t/`、`models/encodec/`。

**Out-of-Scope（不在本仓库源码内）**
- log-mel 特征提取（`WhisperFeatureExtractor`，在 audio_utils/feature extraction）；torchaudio/soundfile 音频 IO：外部。
- 不在本叶子：多模态语音-视觉联合（multimodal）。

## 8. 与相邻子系统交互

- 上游：用户经 `AutoModelForSpeechSeq2Seq`/`AutoModelForCTC` 等（auto-registry）分发；音频经 `AutoFeatureExtractor` 预处理。
- 本叶子 → 下游：EnCodec 编码的音频 token 可被音乐/语音生成模型复用；`GenerationMixin` 做解码。
- Whisper 结构与 seq2seq-models（T5/BART）同构。

## 9. 语言专项适配口径（Python）

- **分组**：按 capability seam——"声学编码"与"文本/音频 token 解码/重建"。
- **图类型**：architecture（多家族并列）为主。不产出 lifecycle。
- **外部边界**：torchaudio/soundfile/PyTorch 标注外部。

## 10. 图表清单与质量

| 图 | 文件 | 类型 | archify 质量档 |
|----|------|------|----------------|
| 语音家族架构 | `speech-models-architecture.html` | architecture | **standard**（连线标签布局约束回退 standard；已如实披露） |

JSON IR 源：`json/speech-models-architecture.json`。
不另出 sequence：前向数据流已在 architecture 连线表达。

**覆盖范围披露**：深读 Whisper/SeamlessM4T/EnCodec 三个家族；其余（Wav2Vec2、Hubert、Speech2Text、Clap、MusicGen、Bark、SpeechT5 等）仅按共性列表说明，未逐类读源码。
