# core-framework（核心框架）域总览

> 本域包含以下叶子子系统；各叶子详情见对应文档。
> 源码基准：HuggingFace transformers `main`，commit `5474a55e920f358d8382f3ecd3377edca979baa`。

## 1. 域职责

`core-framework` 域是 HuggingFace transformers 的 PyTorch 模型层核心框架，负责：

- 模型配置的定义、加载、保存（`PreTrainedConfig` + `AutoConfig` 工厂）。
- 模型基类 `PreTrainedModel` 的完整加载链（`from_pretrained`：配置→实例化→权重→tie→eval）。
- 模型层构建块（注意力掩码、FlashAttention 适配、RoPE、激活函数、初始化）。
- 权重磁盘 IO（safetensors/bin 分片加载、KV cache、远端代码、GGUF）。

本域是所有具体模型（`models/<x>/modeling_<x>.py`）的公共底座。

## 2. 叶子索引

| 叶子 | 文档 | 架构图 | 时序图 / 数据流图 | 职责一句话 |
|------|------|--------|-------------------|-----------|
| configuration-utils | [configuration-utils.md](configuration-utils/configuration-utils.md) | [架构图](configuration-utils/configuration-utils-architecture.html) | [时序图](configuration-utils/configuration-utils-sequence.html) | `PreTrainedConfig` 基类与 `AutoConfig` 注册工厂 |
| modeling-utils | [modeling-utils.md](modeling-utils/modeling-utils.md) | [架构图](modeling-utils/modeling-utils-architecture.html) | [时序图](modeling-utils/modeling-utils-sequence.html) | `PreTrainedModel` 基类与 `from_pretrained` 加载链 |
| modeling-layers | [modeling-layers.md](modeling-layers/modeling-layers.md) | [架构图](modeling-layers/modeling-layers-architecture.html) | — | 层构建块、注意力掩码、RoPE、激活函数 |
| weight-io | [weight-io.md](weight-io/weight-io.md) | [架构图](weight-io/weight-io-architecture.html) | [数据流图](weight-io/weight-io-dataflow.html) | 权重加载、KV cache、远端代码、GGUF |

## 3. 域级机制细节

### 3.1 加载链总览

用户调 `AutoModel.from_pretrained(name_or_path)` 后，跨四个叶子的完整路径：

1. **configuration-utils**：`AutoConfig.from_pretrained` 下载/读取 `config.json`，按 `model_type` 惰性注册映射实例化具体 `Config`。
2. **modeling-utils**：`PreTrainedModel.from_pretrained` 用 config 在 meta device 上实例化模型结构。
3. **weight-io**：`_get_resolved_checkpoint_files` 定位权重文件，`WeightConverter` 做键名重命名，逐参数写回。
4. **modeling-layers**：模型 `__init__` 内按 `config.attn_implementation`/`config.rope_scaling` 选择层实现。

### 3.2 配置驱动的静态分派

整个框架的核心设计是**配置驱动**：所有运行时行为（dtype、注意力后端、RoPE 类型、激活函数、量化）都由 `config` 对象决定，而非用户显式传参。这使得同一模型代码可通过不同 config 切换后端。

### 3.3 惰性注册机制

`CONFIG_MAPPING`（`_LazyConfigMapping`）和 `MODEL_MAPPING` 等注册表都采用惰性导入：启动期只存 `model_type → 类名字符串`，首次访问时才 `importlib.import_module` 对应模型模块。这避免了导入全部 300+ 模型的启动开销。
