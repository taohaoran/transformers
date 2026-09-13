# Auto 注册表（auto-registry）

> 本文是 `models-registry` 域下的叶子子系统文档。域级总览见 `../models-registry.md`，本文只展开
> `src/transformers/models/auto/` 这一"按名称动态分发到具体模型类"的注册机制；具体模型家族的内部架构
> 分别见 encoder-models / decoder-models 等叶子。
>
> 源码基准：HuggingFace transformers v5.18.0.dev0（main），commit `5474a55e920f358d8382f3ecd3377edca979baa`。

## 1. 功能清单

| 能力 | 说明 | 源码路径 |
|------|------|----------|
| 通用配置工厂 | `AutoConfig.from_pretrained` 按 `config.json` 中的 `model_type` 选中具体 `XxxConfig` | `models/auto/configuration_auto.py:301`（`AutoConfig.from_pretrained`） |
| 惰性配置映射 | `_LazyConfigMapping`：`model_type` 字符串 → 配置类，首次访问才 import 模型包 | `models/auto/configuration_auto.py:93` |
| 模型工厂分发 | `_BaseAutoModelClass.from_pretrained/from_config`：按 config 类型选模型类，处理本地/远程代码分支 | `models/auto/auto_factory.py:199` |
| 惰性模型映射 | `_LazyAutoMapping`：Config 类 → 模型/分词器类，`OrderedDict` 子类，`__getitem__` 时才 `importlib.import_module` | `models/auto/auto_factory.py:580` |
| 字符串注册表 | `CONFIG_MAPPING_NAMES`、`MODEL_*_MAPPING_NAMES` 等纯字符串 `OrderedDict`（由 `utils/check_auto.py` 自动生成） | `models/auto/auto_mappings.py:24` 起 |
| 任务头自动类 | `AutoModel`、`AutoModelForCausalLM`、`AutoModelForSeq2SeqLM` 等约 40 个自动类，各自绑定一张 `_MAPPING` | `models/auto/modeling_auto.py:2220` 起 |
| 分词器/处理器自动类 | `AutoTokenizer`、`AutoProcessor`、`AutoImageProcessor`、`AutoFeatureExtractor`、`AutoVideoProcessor` | `models/auto/tokenization_auto.py:647`、`processing_auto.py`、`image_processing_auto.py`、`feature_extraction_auto.py`、`video_processing_auto.py` |
| 远程代码注册 | `trust_remote_code` 时经 `config.auto_map` 用 `get_class_from_dynamic_module` 动态加载并注册进 `_extra_content` | `models/auto/auto_factory.py:382`；`dynamic_module_utils.py`（不在本叶子） |
| 文档字符串自动填充 | `replace_list_option_in_docstrings` 把"List options"占位替换为全部可用 model_type 列表 | `models/auto/configuration_auto.py:249` |
| backbone 自动加载 | `_BaseAutoBackboneClass`：Hub 上不存在的 id 回退到 timm backbone | `models/auto/auto_factory.py:435` |

> 说明：任务给定的文件清单中提到的 `modeling_flax_auto.py`、`modeling_tf_auto.py`、`conversion_mapping.py`、
> `fusion_mapping.py`、`auto_backbone.py` 在当前 main 分支（v5.18.0.dev0）的 `models/auto/` 下**不存在**——
> Flax/TF 自动模型已随框架后端调整移出；backbone 能力合并进 `_BaseAutoBackboneClass`。本叶子按实际文件分析。

## 2. 核心类型与接口清单

| 类型 / 函数 | 位置 | 职责 |
|-------------|------|------|
| `_BaseAutoModelClass` | `auto_factory.py:199` | 所有 AutoModel 类的基类；`__init__` 直接抛错，只能 `from_pretrained/from_config`；实现统一的本地/远程分发 |
| `_LazyAutoMapping` | `auto_factory.py:580` | 继承 `OrderedDict[type[PreTrainedConfig], tuple]`；`__getitem__` 用 `_reverse_config_mapping` 反查 model_type 再惰性 import；`_extra_content` 存远程注册的覆盖项 |
| `_LazyConfigMapping` | `configuration_auto.py:93` | 字符串 key（model_type）→ 配置类的惰性映射；`CONFIG_MAPPING` 即其实例 |
| `_LazyLoadAllMappings` | `configuration_auto.py:156` | 首次访问时**一次性** import 所有模型包并合并映射（用于需要全集的场景，代价是慢） |
| `AutoConfig` | `configuration_auto.py:278` | 非惰性分发：读 `config.json` 的 `model_type`/`auto_map` 后委托具体配置类 |
| `_get_model_class(config, model_mapping)` | `auto_factory.py:183` | 映射值为列表时，按 `config.architectures` 名称匹配具体头类，缺省取第一个 |
| `model_type_to_module_name(key)` | `configuration_auto.py:64` | model_type 字符串 → 包名（`-` 转 `_`，查 `SPECIAL_MODEL_TYPE_TO_MODULE_NAME`） |
| `auto_class_update(cls, ...)` | `auto_factory.py:485` | 用 `copy_func` 复制 `from_pretrained/from_config`，替换 docstring 并注入可选模型列表 |
| `add_generation_mixin_to_remote_model` | `auto_factory.py:548` | 兼容旧 Hub 远程模型：动态给类挂上 `GenerationMixin` |

## 3. 关键调用链

**链路 A：`AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3-8b")`（本地模型主路径）**
1. `_BaseAutoModelClass.from_pretrained`（`auto_factory.py:266`）先 `cached_file(config.json)` 尽早拿到 `commit_hash`（`:292`）。
2. 若非已构造 config，调用 `AutoConfig.from_pretrained`（`auto_factory.py:341`）读出 `model_type="llama"`。
3. `has_local_code = type(config) in cls._model_mapping`（`:361`）；命中后 `_get_model_class`（`:183`）在 `MODEL_FOR_CAUSAL_LM_MAPPING[type(config)]` 里按 `config.architectures`（如 `LlamaForCausalLM`）选出具体类。
4. `_LazyAutoMapping.__getitem__`（`auto_factory.py:601`）经 `_reverse_config_mapping["LlamaConfig"]="llama"` → `_load_attr_from_module`（`:617`）`importlib.import_module(".llama", "transformers.models")`，再 `getattribute_from_module` 取出 `LlamaForCausalLM`。
5. 调用 `LlamaForCausalLM.from_pretrained(..., config=config)`（`:407`）下载权重并实例化。

**链路 B：远程代码模型（`trust_remote_code=True`）**
1. config 含 `auto_map`，且 `AutoModelForCausalLM` 在 `auto_map` 中（`:360`）。
2. `resolve_trust_remote_code` 判定信任后，`get_class_from_dynamic_module` 从 Hub 拉取并执行远程建模文件（`:383`）。
3. `cls.register(config.__class__, model_class, exist_ok=True)` 与 `model_class.register_for_auto_class(auto_class=cls)` 把类注入 `_LazyAutoMapping._extra_content`（`:387`、`auto_factory.py:689`），后续同会话直接命中。

**链路 C：`AutoConfig.from_pretrained`（configuration_auto.py:301）**
1. `PreTrainedConfig.get_config_dict` 拉 config.json；读 `model_type` 字段。
2. 若有 `auto_map["AutoConfig"]` 且允许远程 → 动态加载配置类；否则 `CONFIG_MAPPING["llama"]` 惰性取到 `LlamaConfig`（`configuration_auto.py:419`）。
3. 对 `model_type=="mistral"` 且含 `layer_types` 有启发式重定向到 `ministral`（`:412`）。

## 4. 配置项

| 配置项 | 默认 / 行为 | 位置 |
|--------|-------------|------|
| `trust_remote_code` | 默认 `False`；显式 `True` 或交互确认后才执行 Hub 上远程代码 | `auto_factory.py:144`、`resolve_trust_remote_code` |
| `revision` / `code_revision` | 权重/代码分别的 git 版本定位，默认 `main` | `auto_factory.py:280` |
| `torch_dtype="auto"` / `quantization_config` | 加载时从 kwargs 剥离并回填，避免污染 config | `auto_factory.py:333` |
| `_from_auto` | 内部标记，告知下游"由 Auto 工厂调用" | `auto_factory.py:269` |
| `adapter_kwargs` / peft | 检测到 adapter 配置文件时改从 `base_model_name_or_path` 加载基座 | `auto_factory.py:304` |
| `SPECIAL_MODEL_TYPE_TO_MODULE_NAME` | 个别 model_type 到包名非平凡映射（如 `parakeet_tdt→parakeet`） | `auto_mappings.py:742` |

## 5. 错误与重试语义

- 无法识别 config 类型时抛 `ValueError`，列出 `_model_mapping` 中全部可用配置类名（`auto_factory.py:410`）。
- `CONFIG_MAPPING` 找不到 `model_type` 时抛 `ValueError`，提示升级 transformers 或从源码安装（`configuration_auto.py:421`）。
- 远程代码信任不通过（`trust_remote_code=False` 且未交互确认）时走本地分支；本地也无该 config 类则报"Unrecognized configuration class"。
- 注册冲突：`_LazyAutoMapping.register` 对已被官方占用的 model_type 抛 `ValueError`（`auto_factory.py:676`）；但对复用官方 config 的远程注册直接跳过，以保证 `trust_remote_code=False` 时仍回退官方实现（`:685`）。
- 无自动重试/退避；网络下载重试由 `cached_file` / `huggingface_hub`（不在本仓库源码内）负责。

## 6. 并发细节

- 本叶子是纯 Python 导入/分发层，**无 goroutine/线程模型**（Go 口径不适用）。`_LazyAutoMapping._modules` 字典在多线程首次并发 import 同一包时依赖 Python 导入锁（importlib 全局锁），`_modules` 缓存非显式加锁，但重复 import 是幂等的。
- 无 workqueue/channel；`OrderedDict` 保序仅影响 docstring 列表顺序，不影响分发正确性。
- context 不适用：分发是同步函数调用，超时/取消由后续权重下载与模型 forward 阶段处理。

## 7. 系统边界

**In-Scope（本仓库源码内）**
- `models/auto/` 全部文件：工厂基类、字符串映射表、各 Auto 自动类的定义与 docstring 填充。
- `model_type ↔ 包名 ↔ 配置类 ↔ 任务头模型类` 的注册表结构。

**Out-of-Scope（不在本仓库源码内）**
- 具体模型家族（BERT/LLaMA/T5…）的前向实现：见 encoder/decoder/seq2seq/vision/multimodal/speech 各叶子。
- `huggingface_hub` 下载与缓存、`tokenizers`/`sentencepiece` 后端、PyTorch 权重加载：均为外部依赖。
- `dynamic_module_utils.get_class_from_dynamic_module` 的远程沙箱执行细节：在 `src/transformers/dynamic_module_utils.py`，不在本叶子。
- 不在本叶子：模型家族本身的架构（各模型叶子）。

## 8. 与相邻子系统交互

- 上游 → 本叶子：用户脚本 `from transformers import AutoModel, AutoConfig`；`PreTrainedModel.from_pretrained` 内部也复用 config 解析。
- 本叶子 → 下游：`_LazyAutoMapping` → `importlib.import_module("transformers.models.<family>")` → 各模型家族包（如 `models/llama`、`models/bert`）。
- 本叶子 → 外部：经 `cached_file` 访问 HuggingFace Hub 拉 `config.json` 与权重；远程代码经 `get_class_from_dynamic_module` 拉 Hub 上的建模文件。
- 与 modular-system 叶子的关系：新模型通过 `modular_*.py` + `make fix-repo` 生成家族包后，其 `cls.model_type` 会被 `utils/check_auto.py` 写入 `auto_mappings.py`，即本叶子注册表的自动生成来源。

## 9. 语言专项适配口径（Python）

- **分组**：按 capability seam 分组——"配置解析（AutoConfig/_LazyConfigMapping）"与"模型/分词器分发（_BaseAutoModelClass/_LazyAutoMapping）"两条独立 seam，分别由 `configuration_auto.py` 与 `auto_factory.py` 承担，互不依赖对方运行时状态。
- **图类型**：以 architecture（注册表组件与边界）+ sequence（`from_pretrained` 分发时序）为主；不产出 lifecycle——Auto 工厂无训练/运行状态机，仅一次性分发，故不适用。
- **外部边界**：HuggingFace Hub、tokenizers、sentencepiece、timm、peft 均标注为外部组件，不在本仓库源码内。
- **依赖图**：`auto_mappings.py` 是纯字符串数据（无 import 模型类），保证 `import transformers` 时不触发 516 个模型包的全量加载——这是整个注册表的性能关键。

## 10. 图表清单与质量

| 图 | 文件 | 类型 | archify 质量档 |
|----|------|------|----------------|
| Auto 注册表域结构 | `auto-registry-architecture.html` | architecture | **showcase**（0 错误通过） |
| from_pretrained 分发时序 | `auto-registry-sequence.html` | sequence | **standard**（首版 participant 标签过宽/自消息跨度不足，回退 standard；已如实披露） |

JSON IR 源文件位于 `json/` 目录：`auto-registry-architecture.json`、`auto-registry-sequence.json`。
不产出 dataflow/lifecycle：本叶子为同步分发逻辑，无管道数据流或状态机；architecture 与 sequence 已完整表达结构与时序。
