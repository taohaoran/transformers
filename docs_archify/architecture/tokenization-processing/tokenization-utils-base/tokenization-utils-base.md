# tokenization-utils-base（PreTrainedTokenizerBase 与 Fast Tokenizer 基类）

> 本文是 `tokenization-processing` 域下的叶子子系统文档。域级总览见 `../tokenization-processing.md`。
> 本文只展开分词器基类与 Fast tokenizer；纯 Python 慢分词器见
> [`../tokenization-slow/`](../tokenization-slow/tokenization-slow.md)。
>
> 源码基准：HuggingFace transformers `main`，commit `5474a55e920f358d8382f3ecd3377edca979baa`。

## 1. 功能清单

| 能力 | 说明 | 源码路径 |
|------|------|----------|
| `PreTrainedTokenizerBase` | 所有分词器的抽象基类 | `src/transformers/tokenization_utils_base.py:962` |
| `BatchEncoding` | `UserDict` 子类，封装 forward 输出（input_ids/attention_mask/token_type_ids） | `:195` |
| `TruncationStrategy` 枚举 | `LONGEST_FIRST`/`ONLY_FIRST`/`ONLY_SECOND`/`DO_NOT_TRUNCATE` | `:154` |
| `CharSpan`/`TokenSpan` NamedTuple | token↔char 映射 | `:166`、`:179` |
| `from_pretrained` | 分词器下载/缓存/加载 | `:1489` |
| `encode`/`decode` | 文本↔id 序列转换 | `:2241`、`:2853` |
| `__call__` / `batch_encode_plus` | 批量编码入口（padding/truncation/return_tensors） | `:2418` |
| `pad` | 批量 padding | `:2568` |
| `apply_chat_template` | 对话模板渲染入口 | `:2989` |
| `encode_message_with_chat_template` | 消息级模板应用 | `:3165` |
| `add_special_tokens`/`add_tokens` | 动态添加特殊 token | `:1102`、`:1209` |
| `save_pretrained`/`_save_pretrained` | 词表保存 | `:1976`、`:2137` |
| `convert_tokens_to_ids`/`convert_ids_to_tokens` | token↔id 互转 | `:1456`、`:1472` |
| `PreTrainedTokenizerFast`（`TokenizersBackend`） | 基于 Rust `tokenizers` 库的快分词器 | `src/transformers/tokenization_utils_tokenizers.py:88` |
| `is_fast` 标识 | `True` 表示 Rust 后端 | `:539` |
| `save_vocabulary` | 保存 fast tokenizer 的 `tokenizer.json` | `:556` |
| `update_post_processor` | 注册后处理器（BERT 式 `<s></s>`） | `:637` |
| `convert_slow_tokenizer.py` | 慢→快转换器族 | `src/transformers/convert_slow_tokenizer.py:220` 起 |
| `Converter` 基类 | 各模型慢分词器→fast 转换器 | `:220` |
| `SpmConverter` | SentencePiece 模型→fast 转换 | `:634` |
| `SentencePieceExtractor` | 从 .model 抽 vocab/merges | `:146` |

## 2. 核心类型与接口清单

| 类型 / 函数 | 位置 | 职责 |
|-------------|------|------|
| `PreTrainedTokenizerBase` | `tokenization_utils_base.py:962` | 分词器基类，继承 `PushToHubMixin` |
| `BatchEncoding` | `:195` | 编码结果容器，支持 `.to(device)`/`.convert_to_tensors` |
| `TruncationStrategy` | `:154` | 截断策略枚举 |
| `PreTrainedTokenizerFast`（`TokenizersBackend`） | `tokenization_utils_tokenizers.py:88` | Fast 分词器基类，持有 Rust `TokenizerFast` 句柄 |
| `Converter` | `convert_slow_tokenizer.py:220` | 慢→快转换器抽象 |
| `SpmConverter` | `:634` | SentencePiece 转换器 |

## 3. 关键调用链

### 3.1 `tokenizer("文本")` 编码链

1. `__call__`（`tokenization_utils_base.py:2418`）调 `batch_encode_plus`。
2. Fast 路径：`PreTrainedTokenizerFast._encode_plus` 调 Rust `tokenizers` 库（外部）的 `Tokenizer.encode_batch`，返回 `EncodingFast` 句柄，包装成 `BatchEncoding`。
3. Slow 路径：`tokenize`（`:2211`）→ `convert_tokens_to_ids`（`:1456`）→ `pad`/`truncate`。
4. `apply_chat_template`（`:2989`）先按 Jinja2 模板渲染对话历史为单串，再走上述编码。

### 3.2 `PreTrainedTokenizer.from_pretrained` 选择 Fast/Slow

1. 调 `_from_pretrained`（`:1751`）下载 `tokenizer_config.json`/`vocab.json`/`merges.txt`/`tokenizer.json`。
2. 若仓库有 `tokenizer.json`（Fast 格式），实例化 `PreTrainedTokenizerFast`。
3. 否则实例化慢分词器（`PreTrainedTokenizer`，见 tokenization-slow 叶子）；可通过 `use_fast=True` 调 `convert_slow_tokenizer.Converter` 自动转 Fast。

### 3.3 慢→快自动转换

1. `PreTrainedTokenizer.from_pretrained` 检测到只有慢分词器文件（如 `spiece.model`）。
2. 调对应 `Converter`（如 `SpmConverter`，`convert_slow_tokenizer.py:634`）。
3. Converter 读慢分词器，构建 Rust `tokenizers.Tokenizer` 对象，返回 `PreTrainedTokenizerFast`。

## 4. 配置项

| flag / option | 默认 / 行为 | 位置 |
|---------------|-------------|------|
| `padding` | `False`；可选 `max_length`/`longest`/`do_not_pad` | `__call__:2418` |
| `truncation` | `False`；按 `max_length` 截断 | 同上 |
| `max_length` | 模型 max_position_embeddings | 同上 |
| `return_tensors` | `None`；`pt`/`tf`/`np` | 同上 |
| `return_attention_mask` | `True` | 同上 |
| `add_special_tokens` | `True` | `encode:2241` |
| `use_fast` | `True`（from_pretrained），优先 Fast | `_from_pretrained:1751` |
| `chat_template` | `None`；Jinja2 模板字符串 | `apply_chat_template:2989` |
| `return_token_type_ids` | 自动判断 | 同上 |

## 5. 错误与重试语义

- **词表文件缺失**：`cached_file` 抛 `OSError`，包装为"Can't load tokenizer"。
- **截断策略与 max_length 冲突**：`TruncationStrategy` 不匹配时抛 `ValueError`。
- **Fast tokenizer 不可用**：`tokenizers` Rust 库未装时，`PreTrainedTokenizerFast` 实例化抛 `ImportError`，自动回退慢分词器。
- **chat_template 渲染错误**：Jinja2 异常上抛，提示模板语法错误。
- 网络重试由 HF Hub 库负责。

## 6. 并发细节

- Fast tokenizer 的 Rust `Tokenizer` 内部持有可变状态（added tokens），非线程安全；多线程并发 `__call__` 需外部加锁。
- 无 Python 层 goroutine/线程。
- Rust `tokenizers` 库（外部）内部用 rayon 做并行分词（Cargo feature，非本仓库源码）。

## 7. 系统边界

**In-Scope（本仓库源码内）**
- `PreTrainedTokenizerBase` 基类与 `BatchEncoding`。
- `PreTrainedTokenizerFast` 的 Python 绑定层。
- 慢→快转换器族。

**Out-of-Scope（不在本仓库源码内）**
- Rust `tokenizers` 库（`huggingface/tokenizers` crate）：实际分词算法。
- SentencePiece C++ 库（外部）。
- Jinja2 模板引擎（外部）。
- HF Hub 下载/缓存。
- 各具体模型分词器（`models/<x>/tokenization_<x>.py`）——本叶子只定义基类。

## 8. 与相邻子系统交互

- **上游**：用户代码 / `pipeline` / 模型 `forward` 前调用本叶子。
- **本叶子 → tokenization-slow 叶子**：慢分词器基类与 SentencePiece 适配。
- **本叶子 → 外部**：Rust tokenizers、sentencepiece、Jinja2、HF Hub。
- **下游**：模型 `forward(input_ids=..., attention_mask=...)` 消费 `BatchEncoding`。

## 9. 语言专项适配口径

Python 项目，采用 TS 口径的 Python 适配：
1. **分组**：按 capability seam 分组——本叶子 = "分词器基类与 Fast 绑定" seam。
2. **图类型**：architecture + sequence（encode 调用链）。
3. **外部边界**：Rust tokenizers/sentencepiece/Jinja2 标注外部。
4. **部署维度**：不适用单二进制。

## 10. 图表清单与质量

| 图 | 文件 | 类型 | archify 质量档 |
|----|------|------|----------------|
| 分词器组件架构图 | `tokenization-utils-base-architecture.html` | architecture | standard（showcase 回退，如实披露） |
| encode 调用时序图 | `tokenization-utils-base-sequence.html` | sequence | showcase |

JSON IR 源文件位于 `json/` 目录。
