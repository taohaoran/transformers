# modeling-utils（PreTrainedModel 基类与权重加载链）

> 本文是 `core-framework` 域下的叶子子系统文档。域级总览见 `../core-framework.md`。
> 本文只展开 `PreTrainedModel` 基类的加载/保存/钩子体系；具体注意力/层构建见
> [`../modeling-layers/`](../modeling-layers/modeling-layers.md)，磁盘缓存与远端代码见
> [`../weight-io/`](../weight-io/weight-io.md)。
>
> 源码基准：HuggingFace transformers `main`，commit `5474a55e920f358d8382f3ecd3377edca979baa`。

## 1. 功能清单

| 能力 | 说明 | 源码路径 |
|------|------|----------|
| `PreTrainedModel` 基类 | 所有 PyTorch 模型的基类，继承 `nn.Module`、`ModuleUtilsMixin`、`EmbeddingAccessMixin` 等 | `src/transformers/modeling_utils.py:1074` |
| `from_pretrained` 完整加载链 | 配置加载 → dtype 解析 → 实例化 → 权重加载 → tie_weights → eval | `modeling_utils.py:3819` |
| `_load_pretrained_model` | 实际加载 state_dict 到模型，处理分片/缺失键/意外键 | `modeling_utils.py:4357` |
| `_finalize_model_loading` | 加载收尾：缺失键/意外键校验、tie_weights | `modeling_utils.py:4471` |
| `save_pretrained` | 序列化模型权重（safetensors/bin）+ config + generation_config | `modeling_utils.py:3238` |
| 注意力实现调度 | `_flash_attn_can_dispatch`/`_sdpa_can_dispatch`/`set_attn_implementation` 后端选择与回退 | `:1583`、`:1658`、`:1959` |
| 权重绑定向量化 | `tie_weights`、`_get_tied_weight_keys`、`remove_tied_weights_from_state_dict` | `:2526`、`:378`、`:438` |
| 嵌入层缩放 | `resize_token_embeddings`/`_resize_token_embeddings`/`_get_resized_embeddings` | `:2629`、`:2688`、`:2727` |
| 梯度检查点 | `gradient_checkpointing_enable`/`_set_gradient_checkpointing` | `:3106`、`:3174` |
| 输入梯度要求 | `enable_input_require_grads`/`disable_input_require_grads`（PEA/LoRA 场景） | `:2134`、`:2178` |
| `ModuleUtilsMixin` | `device`/`dtype`/`num_parameters` 便捷属性 | `:900` |
| `EmbeddingAccessMixin` | `get_input_embeddings`/`set_input_embeddings`/`get_output_embeddings` 抽象 | `:965` |
| `LoadStateDictConfig` dataclass | 权重加载配置聚合（dtype/device_map/quantizer/weight_mapping 等） | `:171` |
| checkpoint 文件解析 | `_get_resolved_checkpoint_files`（分片 index.json 定位） | `:535` |
| `ModelOutput` 体系 | 所有 forward 输出的 dataclass 基类族（`BaseModelOutput`/`CausalLMOutputWithPast` 等 30+） | `src/transformers/modeling_outputs.py:24`–`1190` |
| `pytorch_utils` 工具 | `Conv1D`、`apply_chunking_to_forward`、`prune_linear_layer`、`meshgrid` | `src/transformers/pytorch_utils.py:95`、`:124`、`:61`、`:202` |
| 初始化 | `_init_weights`/`initialize_weights`/`post_init` | `:2292`、`:2384`、`:1294` |

## 2. 核心类型与接口清单

| 类型 / 函数 | 位置 | 职责 |
|-------------|------|------|
| `PreTrainedModel` | `modeling_utils.py:1074` | 模型基类；子类化时 `__init_subclass__`（`:1223`）注册 `config_class` 与 auto 映射 |
| `LoadStateDictConfig` | `:171` | dataclass，聚合 `from_pretrained` 加载期所有参数，传给 `_load_pretrained_model` |
| `ModuleUtilsMixin` | `:900` | 提供 `model.device`/`model.dtype`/`num_parameters` 便捷属性 |
| `EmbeddingAccessMixin` | `:965` | 抽象 `get_input_embeddings` 等，供 `tie_weights` 与 `resize_token_embeddings` 调用 |
| `ModelOutput`（来自 `utils.generic`） | `modeling_outputs.py` 顶部导入 | dataclass 基类，支持 `.to_tuple()`、键访问 |
| `BaseModelOutput` / `CausalLMOutputWithPast` 等 | `modeling_outputs.py:24`、`:610` | 各 forward 输出签名，字段含 `last_hidden_state`/`past_key_values`/`logits`/`hidden_states`/`attentions` |
| `Conv1D` | `pytorch_utils.py:95` | GPT-2 风格 1D 卷积（实际是转置 Linear） |
| `apply_chunking_to_forward` | `pytorch_utils.py:124` | 把长序列 FFN 分块前向以省显存 |
| `get_hf_quantizer` | 导入自 `integrations` | 返回量化器（GPTQ/AWQ/BitsAndBytes 等，外部）或 None |
| `_get_device_map` | `:4294` 调用 | 经 accelerate 计算 device_map（外部库） |

## 3. 关键调用链

### 3.1 `PreTrainedModel.from_pretrained("bert-base-uncased")` 加载链

1. `from_pretrained`（`modeling_utils.py:3819`）解析参数：`quantization_config`、`device_map`、`dtype`、`use_safetensors`、`low_cpu_mem_usage`、`gguf_file`、`adapter_kwargs` 等。
2. 调 `maybe_load_adapters`（`:4159`）处理 LoRA 适配器；`check_and_set_device_map`（`:4164`）校验 device_map。
3. 若 `config` 非 `PreTrainedConfig` 实例，调 `config_class.from_pretrained(..., return_unused_kwargs=True)`（`:4178`）——见 configuration-utils 叶子——拿到 config 与 `model_kwargs`。
4. `get_hf_quantizer`（`:4208`）根据 `quantization_config` 返回量化器（外部，不在本仓库源码内）。
5. `_get_resolved_checkpoint_files`（`:4218`）定位权重文件：safetensors 单文件 / 分片 + `model.safetensors.index.json` / pytorch bin / GGUF。
6. `_get_dtype`（`:4237`）根据 checkpoint 文件、config.dtype、用户传入 dtype 解析最终 dtype。
7. 注册 fusion patches（`:4250`）与 kernel patches（`:4256`，可选）。
8. `get_init_context`（`:4263`）返回 `init_empty_weights`/`init_on_meta_device` 上下文管理器（低内存加载关键：模型先在 meta device 上实例化，不占显存）。
9. `with ContextManagers(model_init_context): model = cls(config, ...)`（`:4266`）实例化模型结构；`hf_quantizer.preprocess_model`（`:4271`）替换量化模块。
10. `dtype_plan = model._get_dtype_plan(dtype)`（`:4284`）；`weight_conversions = get_model_conversion_mapping(...)`（`:4287`）处理权重键名映射。
11. 组装 `LoadStateDictConfig`（`:4297`），调 `cls._load_pretrained_model(model, state_dict, checkpoint_files, load_config)`（`:4314`）实际加载权重。
12. `_finalize_model_loading`（`:4315`）做缺失/意外键校验、`tie_weights`、`model.eval()`（`:4316`）。
13. 若 `can_generate()`，调 `adjust_generation_fn`（`:4322`）加载 generation_config；多 device/disk offload 时 `accelerate_dispatch`（`:4334`）注入钩子。

### 3.2 `save_pretrained("./out")` 保存链

1. `save_pretrained`（`:3238`）：先 `_save_pretrained` 收集 state_dict，`tie_weights` 后只保存一份绑定权重。
2. 写 `model.safetensors`（优先）或 `pytorch_model.bin`；大模型写分片 + `index.json`。
3. 调 `config.save_pretrained`（见 configuration-utils 叶子）写 `config.json`；写 `generation_config.json`（若存在）。

## 4. 配置项

| flag / option | 默认 / 行为 | 位置 |
|---------------|-------------|------|
| `dtype` | `None`（`"auto"`），从 checkpoint 推断 | `from_pretrained:3819` kwargs |
| `device_map` | `None`；`"auto"` 经 accelerate 分片到多 GPU/磁盘 | `:4164` |
| `low_cpu_mem_usage` | `None`（自动推断），为 `True` 时 meta device 实例化 | `:4263` init context |
| `use_safetensors` | `None`，优先 safetensors | `:3830` |
| `weights_only` | `True`，`torch.load` 仅反序列化张量（安全） | `:3831` |
| `ignore_mismatched_sizes` | `False`，尺寸不匹配时不报错 | `:3825` |
| `variant` | `None`，权重文件变体（如 `fp16`） | `_get_resolved_checkpoint_files:535` |
| `gguf_file` | `None`，GGUF 格式加载入口 | `:4232` |
| `quantization_config` | `None`，GPTQ/AWQ/BnB 等量化 | `:4208` |
| `attn_implementation` | `None`，`eager`/`sdpa`/`flash_attention_2/3/4` | `:4202` |
| `offload_folder` | `None`，磁盘 offload 目录 | `:4302` |
| `max_memory` | `None`，device_map 分片内存上限 | `_get_device_map` |
| `torch_dtype`（历史） | 兼容参数，映射到 `dtype` | configuration 叶子 |
| `tie_weights` 行为 | 加载后自动调用 | `:4315` |

## 5. 错误与重试语义

- **缺失键**：`_load_pretrained_model` 对比 checkpoint state_dict 与模型 `state_dict()`，缺失键进入 `loading_info["missing_keys"]`；若关键键缺失（非 `tie_weights` 或 `ignore_mismatched_sizes`），`_finalize_model_loading` 抛 `RuntimeError`。
- **意外键**：checkpoint 有而模型无的键进入 `unexpected_keys`，仅 warning（旧版 checkpoint 兼容）。
- **尺寸不匹配**：`ignore_mismatched_sizes=False` 时抛 `RuntimeError`，提示标签数不匹配。
- **权重文件缺失**：`_get_resolved_checkpoint_files` 找不到任何权重文件时抛 `OSError`，提示使用 `from_pretrained` 或检查目录。
- **GGUF 无 accelerate**：`gguf_file` 传入但 `is_accelerate_available()` 为 False 时抛 `ValueError`（`:4153`）。
- **未知 attention 实现**：`_check_and_adjust_attn_implementation`（`:1716`）在请求 `flash_attention_2` 但环境不满足时降级到 `sdpa`/`eager` 并 warning。
- 本叶子不做网络重试；下载重试由 HF Hub 库负责。

## 6. 并发细节

- 本叶子为单进程 PyTorch 模型，无 goroutine/线程池；`from_pretrained` 全程同步。
- **低内存加载**是关键并发/内存优化：`get_init_context`（`:3729`）在 `low_cpu_mem_usage=True` 时返回 `init_empty_weights()` 或 `torch.device("meta")` 上下文，模型先在 meta 上构建结构（不分配显存），再逐个张量加载权重——这不是并发，而是流式内存分配。
- **device_map 分片**：accelerate（外部）把不同子模块放到不同 GPU/磁盘；本叶子只组装 `LoadStateDictConfig`，分片执行由 accelerate 完成。
- **梯度检查点**：`gradient_checkpointing_enable`（`:3106`）用 `torch.utils.checkpoint` 在前向时丢弃中间激活、反向时重算——以算力换显存，无线程。
- 无 `context.Context`；PyTorch 用 `torch.cuda.Stream`/`torch.autograd` 管理设备与梯度流（外部）。

## 7. 系统边界

**In-Scope（本仓库源码内）**
- `PreTrainedModel` 基类全部方法（加载/保存/tie/resize/梯度检查点/注意力调度）。
- `LoadStateDictConfig`、`_get_resolved_checkpoint_files`、`_get_dtype`。
- `ModelOutput` dataclass 体系（`modeling_outputs.py`）。
- `pytorch_utils.py` 工具函数（`Conv1D`/`apply_chunking_to_forward`/`prune_linear_layer`）。

**Out-of-Scope（不在本仓库源码内）**
- PyTorch 本身（`torch.nn.Module`、`torch.safetensors`、`torch.cuda`）。
- `accelerate` 库：device_map 分片、disk offload、钩子注入。
- `safetensors` 库：权重序列化格式。
- 量化后端：GPTQ/AWQ/BitsAndBytes/optimum-quanto（外部）。
- FlashAttention / SDPA 内核实现（外部库）；本叶子只做调度选择。
- 具体模型 `__init__`（`models/<x>/modeling_<x>.py`）——本叶子只定义基类与 `_init_weights` 钩子。
- 配置加载本身：见 [`../configuration-utils/`](../configuration-utils/configuration-utils.md)。
- 磁盘缓存/HF Hub 下载：见 [`../weight-io/`](../weight-io/weight-io.md)。

## 8. 与相邻子系统交互

- **上游**：用户代码 / `AutoModel.from_pretrained` / `pipeline` → 本叶子。
- **本叶子 → configuration-utils 叶子**：`config_class.from_pretrained`（`:4178`）拿到 `PreTrainedConfig`。
- **本叶子 → weight-io 叶子**：`_get_resolved_checkpoint_files` 调 `cached_file`/分片索引解析；`_load_pretrained_model` 调 `core_model_loading.py` 的实际加载函数。
- **本叶子 → modeling-layers 叶子**：模型 `__init__` 内构建层时读取 `config.attn_implementation` 选择 `eager`/`sdpa`/`flash` 实现。
- **本叶子 → 外部**：`get_hf_quantizer`（量化）、`accelerate`（分片）、`safetensors`（格式）、FlashAttention（内核）。

## 9. 语言专项适配口径

Python 项目，采用 TS 口径的 Python 适配：

1. **分组**：按 capability seam 分组——本叶子 = "模型基类与加载链" capability seam；与"配置定义"（configuration-utils）、"层构建"（modeling-layers）、"磁盘 IO"（weight-io）分离。
2. **图类型**：architecture（组件边界）+ sequence（`from_pretrained` 加载时序）为主；ModelOutput 体系是静态 dataclass 继承，不产 lifecycle。
3. **外部边界**：PyTorch/accelerate/safetensors/FlashAttention/量化后端均标注外部。
4. **部署维度**：不适用"单二进制"；用包结构表达——`modeling_utils.py`（基类+加载）+ `modeling_outputs.py`（输出体系）+ `pytorch_utils.py`（张量工具）。

## 10. 图表清单与质量

| 图 | 文件 | 类型 | archify 质量档 |
|----|------|------|----------------|
| PreTrainedModel 组件架构图 | `modeling-utils-architecture.html` | architecture | standard（showcase 因标签间距约束回退，如实披露） |
| from_pretrained 加载时序图 | `modeling-utils-sequence.html` | sequence | showcase |

JSON IR 源文件位于 `json/` 目录。
