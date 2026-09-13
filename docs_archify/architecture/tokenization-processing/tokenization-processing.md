# tokenization-processing（分词与多模态处理）域总览

> 本域包含以下叶子子系统；各叶子详情见对应文档。
> 源码基准：HuggingFace transformers `main`，commit `5474a55e920f358d8382f3ecd3377edca979baa`。

## 1. 域职责

`tokenization-processing` 域负责模型输入的预处理：

- 文本分词（Fast 基于 Rust `tokenizers`，Slow 纯 Python / SentencePiece / Mistral）。
- 多模态输入统一处理（图像/音频/视频预处理 + 文本分词）。
- 对话模板渲染（`apply_chat_template`）。
- 编码结果容器（`BatchEncoding`/`BatchFeature`）。

本域的输出是模型 `forward` 的 `input_ids`/`attention_mask`/`pixel_values`/`audio_values`/`video_values` 等张量。

## 2. 叶子索引

| 叶子 | 文档 | 架构图 | 时序图 | 职责一句话 |
|------|------|--------|--------|-----------|
| tokenization-utils-base | [tokenization-utils-base.md](tokenization-utils-base/tokenization-utils-base.md) | [架构图](tokenization-utils-base/tokenization-utils-base-architecture.html) | [时序图](tokenization-utils-base/tokenization-utils-base-sequence.html) | `PreTrainedTokenizerBase` 基类与 Fast tokenizer 绑定 |
| tokenization-slow | [tokenization-slow.md](tokenization-slow/tokenization-slow.md) | [架构图](tokenization-slow/tokenization-slow-architecture.html) | — | 纯 Python 慢分词器、SentencePiece、Mistral 适配 |
| multimodal-processing | [multimodal-processing.md](multimodal-processing/multimodal-processing.md) | [架构图](multimodal-processing/multimodal-processing-architecture.html) | — | ProcessorMixin 多模态统一处理（图像/音频/视频） |

## 3. 域级机制细节

### 3.1 Fast vs Slow 双轨

- **Fast 路径**：`PreTrainedTokenizerFast` 通过 PyO3 绑定 Rust `tokenizers` 库，分词在 Rust 侧完成，Python 侧只做参数编排。性能高（10-100x），但调试不透明。
- **Slow 路径**：`PythonBackend` 纯 Python 实现，`Trie` 加速不拆分 token 查找。慢但易扩展调试。
- **自动转换**：只有慢分词器文件时，`convert_slow_tokenizer.Converter` 自动构建 Fast tokenizer，用户无感知。

### 3.2 ProcessorMixin 统一多模态接口

`ProcessorMixin` 把 tokenizer + image processor + audio processor + video processor 组合成单一 `__call__` 接口。多模态模型（CLIP、LLaVA、Whisper 等）的 Processor 都继承此 mixin，对外暴露统一的 `processor(text=..., images=..., audio=..., videos=...)` 调用方式。

### 3.3 chat_template

`apply_chat_template` 用 Jinja2 渲染对话历史为模型特定格式的文本。模板字符串存在 `tokenizer_config.json` 中，可按模型自定义。
