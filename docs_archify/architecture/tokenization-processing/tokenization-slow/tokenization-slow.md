# tokenization-slow（纯 Python 分词器、SentencePiece 与 Mistral）

> 本文是 `tokenization-processing` 域下的叶子子系统文档。域级总览见 `../tokenization-processing.md`。
> 本文只展开慢分词器实现；Fast 分词器基类见
> [`../tokenization-utils-base/`](../tokenization-utils-base/tokenization-utils-base.md)。
>
> 源码基准：HuggingFace transformers `main`，commit `5474a55e920f358d8382f3ecd3377edca979baa`。

## 1. 功能清单

| 能力 | 说明 | 源码路径 |
|------|------|----------|
| `PreTrainedTokenizer`（`PythonBackend`） | 纯 Python 慢分词器基类 | `src/transformers/tokenization_python.py:400` |
| `Trie` / `ExtensionsTrie` | 前缀树，加速不拆分 token 查找 | `tokenization_python.py:45`、`:276` |
| `tokenize`/`_tokenize` | 文本→token 列表（纯 Python） | `:625`、`:680` |
| `_convert_token_to_id`/`_convert_id_to_token` | token↔id 字典查表 | 基类抽象，子类实现 |
| `SentencePieceBackend` | SentencePiece C++ 库的 Python 适配 | `src/transformers/tokenization_utils_sentencepiece.py:45` |
| `SentencePieceExtractor` | 从 spiece.model 抽 vocab/merges | `:289` |
| `MistralCommonBackend` | Mistral/mistral-common 库适配 | `src/transformers/tokenization_mistral_common.py:186` |
| `MistralTokenizerType` 枚举 | `tekken`/`spm`/`tiktoken` 类型 | `:158` |
| `encode`/`decode`/`batch_decode` | Mistral 专用编解码 | `:377`、`:433`、`:500` |

## 2. 核心类型与接口清单

| 类型 / 函数 | 位置 | 职责 |
|-------------|------|------|
| `PythonBackend`（`PreTrainedTokenizer`） | `tokenization_python.py:400` | 慢分词器基类，继承 `PreTrainedTokenizerBase` |
| `Trie` | `:45` | 前缀树，`split(text)` 快速切分不拆分 token |
| `SentencePieceBackend` | `tokenization_utils_sentencepiece.py:45` | 持有 `sentencepiece.SentencePieceProcessor` 句柄 |
| `MistralCommonBackend` | `tokenization_mistral_common.py:186` | 持有 `mistral_common` 库的 tokenizer 句柄 |
| `MistralTokenizerType` | `:158` | 枚举 tokenizer 后端类型 |

## 3. 关键调用链

### 3.1 纯 Python 分词 `tokenizer.tokenize("文本")`

1. `PythonBackend.tokenize`（`tokenization_python.py:625`）：先查 `Trie` 命中不拆分 token（如特殊 token），剩余文本交 `_tokenize`。
2. `_tokenize`（`:680`）由子类实现（BERT 按 wordpiece、GPT2 按 BPE、T5 按 SentencePiece 等）。
3. `_convert_token_to_id` 把 token 字符串查 vocab 字典得 id。
4. 与 Fast 路径相比，无 Rust 加速，纯 Python 循环，性能低但易调试。

### 3.2 SentencePiece 分词

1. `SentencePieceBackend.__init__`（`tokenization_utils_sentencepiece.py:60`）加载 `.model` 文件到 `sentencepiece.SentencePieceProcessor`（外部 C++ 库）。
2. `_tokenize`（`:204`）调 `sp.EncodeAsPieces(text)`。
3. `_convert_token_to_id`（`:223`）调 `sp.PieceToId`。

### 3.3 Mistral 分词

1. `MistralCommonBackend.__init__`（`tokenization_mistral_common.py:222`）按 `MistralTokenizerType` 选择 tekken/spm/tiktoken 后端。
2. `encode`（`:377`）调 mistral-common 库（外部）。

## 4. 配置项

| flag / option | 默认 / 行为 | 位置 |
|---------------|-------------|------|
| `model_max_length` | 模型最大序列长度 | 基类 |
| `do_lower_case` | 大小写处理 | 子类 |
| `bos_token`/`eos_token`/`pad_token`/`unk_token` | 特殊 token | 基类 |
| `split_special_tokens` | 是否拆分特殊 token | 基类 |

## 5. 错误与重试语义

- **SentencePiece .model 文件缺失**：`sentencepiece` 库抛 `OSError`，上抛。
- **未知 token**：`_convert_token_to_id` 返回 `unk_token_id`，不抛错。
- **Mistral 后端类型未知**：`MistralTokenizerType` 枚举校验失败抛 `ValueError`。
- 无网络重试。

## 6. 并发细节

- 纯 Python 分词器非线程安全（Trie、vocab dict 可变）。
- SentencePiece C++ processor 本身线程安全（只读模型）。
- 无 goroutine/线程。

## 7. 系统边界

**In-Scope（本仓库源码内）**
- `PythonBackend` 基类与 Trie。
- `SentencePieceBackend` 适配层。
- `MistralCommonBackend` 适配层。

**Out-of-Scope（不在本仓库源码内）**
- `sentencepiece` C++ 库（外部）。
- `mistral-common` Python 库（外部）。
- 各具体慢分词器（`models/<x>/tokenization_<x>.py`）——本叶子只定义基类与适配。

## 8. 与相邻子系统交互

- **上游**：tokenization-utils-base 叶子的 `PreTrainedTokenizerBase` 是本叶子基类。
- **本叶子 → 外部**：sentencepiece、mistral-common 库。
- **下游**：具体模型分词器继承本叶子。

## 9. 语言专项适配口径

Python 项目，采用 TS 口径的 Python 适配：
1. **分组**：按 capability seam 分组——本叶子 = "慢分词器后端" seam。
2. **图类型**：architecture（组件边界）。
3. **外部边界**：sentencepiece/mistral-common 标注外部。
4. **部署维度**：不适用单二进制。

## 10. 图表清单与质量

| 图 | 文件 | 类型 | archify 质量档 |
|----|------|------|----------------|
| 慢分词器组件架构图 | `tokenization-slow-architecture.html` | architecture | showcase |

JSON IR 源文件位于 `json/` 目录。
