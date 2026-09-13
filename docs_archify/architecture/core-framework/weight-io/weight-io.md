# weight-io（权重加载、缓存、远端代码与 GGUF）

> 本文是 `core-framework` 域下的叶子子系统文档。域级总览见 `../core-framework.md`。
> 本文只展开磁盘/HF Hub 权重 IO；模型基类加载链见 [`../modeling-utils/`](../modeling-utils/modeling-utils.md)。
>
> 源码基准：HuggingFace transformers `main`，commit `5474a55e920f358d8382f3ecd3377edca979baa`。

## 1. 功能清单

| 能力 | 说明 | 源码路径 |
|------|------|----------|
| `WeightConverter`/`WeightTransform` 体系 | 权重键名重命名与张量转换算子 | `src/transformers/core_model_loading.py:769`、`:1152` |
| 转换算子族 | `Chunk`/`Concatenate`/`Interleave`/`MergeModulelist`/`SplitModulelist`/`Transpose`/`Conv3dToLinear` 等 | `core_model_loading.py:112`–`:657` |
| `WeightRenaming`/`GroupWeightRename`/`PrefixChange` | 键名重命名规则 | `:992`、`:1032`、`:1098` |
| `set_param_for_module`/`offload_and_maybe_resave_param` | 把张量写回模块参数、磁盘 offload 重存 | `:1343`、`:1386` |
| `spawn_materialize` | 子进程物化张量（低内存） | `:1244` |
| `Cache` 基类族 | 推理 KV cache：`DynamicCache`/`StaticCache`/`QuantizedCache`/`EncoderDecoderCache`/`MtpCache` | `src/transformers/cache_utils.py:1283`、`:1756`、`:1848`、`:1903`、`:1966`、`:2125` |
| `CacheLayerMixin` 及各层类型 | `DynamicLayer`/`StaticLayer`/`DynamicSlidingWindowLayer`/`QuantizedLayer`/`LinearAttentionLayer` 等 | `cache_utils.py:27`–`:1212` |
| `get_layer_types_and_kwargs` | 从 config 推导层类型与 kwargs | `:1720` |
| `safetensors_conversion.py` | 旧 pytorch bin → safetensors 转换 | `src/transformers/safetensors_conversion.py` |
| `file_utils.py` | 文件工具（薄封装，实际在 utils/） | `src/transformers/file_utils.py` |
| 远端代码加载 | `create_dynamic_module`/`get_class_from_dynamic_module`/`custom_object_save` | `src/transformers/dynamic_module_utils.py:101`、`:516`、`:626` |
| `resolve_trust_remote_code` | 交互决定是否信任远端代码 | `:712` |
| `get_cached_module_file` | 远端模块文件缓存（带 hash） | `:346` |
| GGUF 加载 | `modeling_gguf_pytorch_utils.py` | `src/transformers/modeling_gguf_pytorch_utils.py` |

## 2. 核心类型与接口清单

| 类型 / 函数 | 位置 | 职责 |
|-------------|------|------|
| `ConversionOps`（ABC） | `core_model_loading.py:81` | 张量转换算子抽象基类 |
| `WeightTransform` | `:769` | 权重变换规则基类 |
| `WeightRenaming` / `GroupWeightRename` / `PrefixChange` | `:992`、`:1032`、`:1098` | 键名重命名子类 |
| `WeightConverter` | `:1152` | 组合变换规则，应用于 state_dict |
| `Cache` | `cache_utils.py:1283` | KV cache 抽象基类 |
| `DynamicCache` / `StaticCache` | `:1756` / `:1848` | 动态增长 / 静态预分配 cache |
| `get_class_from_dynamic_module` | `dynamic_module_utils.py:516` | 从 Hub 远端 .py 文件加载类 |
| `resolve_trust_remote_code` | `:712` | 交互式安全确认 |

## 3. 关键调用链

### 3.1 权重文件定位与加载（modeling-utils 调用本叶子）

1. modeling-utils 叶子的 `_get_resolved_checkpoint_files` 调 `cached_file`（HF Hub 库）定位 `model.safetensors` 或 `pytorch_model.bin`，分片时读 `model.safetensors.index.json` 得到分片文件名映射。
2. 调 `WeightConverter`（`core_model_loading.py:1152`）应用键名重命名（旧 checkpoint 兼容）。
3. `safetensors.safe_open`（外部库）mmap 打开权重文件；`set_param_for_module`（`:1343`）把张量逐参数写回模型。
4. 低内存模式下 `spawn_materialize`（`:1244`）在子进程物化张量。
5. 磁盘 offload 时 `offload_and_maybe_resave_param`（`:1386`）把参数重存到磁盘。

### 3.2 trust_remote_code 远端类加载

1. `AutoConfig.from_pretrained`（configuration-utils 叶子）检测 `config_dict["auto_map"]["AutoConfig"]`。
2. `resolve_trust_remote_code`（`dynamic_module_utils.py:712`）交互式确认。
3. `get_cached_module_file`（`:346`）下载远端 .py 到本地缓存（带 source hash）。
4. `get_class_from_dynamic_module`（`:516`）importlib 加载模块并取出类。

### 3.3 KV cache 生命周期

1. 模型 `forward` 首次调用时创建 `DynamicCache`（`cache_utils.py:1756`）或 `StaticCache`（`:1848`，预分配固定大小）。
2. 每步生成 `update()` 追加 key/value；`get_seq_length()` 查询当前长度。
3. 生成结束后丢弃。

## 4. 配置项

| flag / option | 默认 / 行为 | 位置 |
|---------------|-------------|------|
| `use_safetensors` | `None`，优先 safetensors | modeling-utils 叶子 |
| `variant` | `None`，权重文件变体 | 同上 |
| `offload_folder` | `None`，磁盘 offload 目录 | 同上 |
| `trust_remote_code` | `None`，`resolve_trust_remote_code` 交互确认 | `dynamic_module_utils.py:712` |
| `cache_dir` | `None`，HF 缓存目录 | configuration-utils 叶子 |
| `local_files_only` | `False`，只读本地缓存 | 同上 |
| `gguf_file` | `None`，GGUF 文件名 | `modeling_gguf_pytorch_utils.py` |

## 5. 错误与重试语义

- **权重文件缺失**：`cached_file` 抛 `OSError`，modeling-utils 包装为友好错误。
- **shard 缺失**：`index.json` 列出的分片文件缺失时抛 `OSError`。
- **远端代码未信任**：`resolve_trust_remote_code` 在非交互环境下抛 `EnvironmentError`，引导用户显式传 `trust_remote_code=True`。
- **远端模块 hash 不匹配**：`get_cached_module_file` 检测到本地缓存与远端 hash 不一致时重新下载。
- **GGUF 架构不支持**：`load_gguf_checkpoint` 抛 `ValueError`，提示架构未实现。
- 网络重试由 HF Hub 库负责。

## 6. 并发细节

- `spawn_materialize`（`core_model_loading.py:1244`）用 `multiprocessing` 子进程物化张量，主进程通过队列接收——这是本叶子唯一的并发点。
- `DynamicCache` 非线程安全；生成循环单线程顺序调用。
- 无 `context.Context` 传播。
- mmap 权重文件由 OS 页缓存管理（外部）。

## 7. 系统边界

**In-Scope（本仓库源码内）**
- 权重键名转换算子族（`core_model_loading.py`）。
- KV cache 体系（`cache_utils.py`）。
- 远端代码加载（`dynamic_module_utils.py`）。
- GGUF 加载（`modeling_gguf_pytorch_utils.py`）。
- safetensors 转换（`safetensors_conversion.py`）。

**Out-of-Scope（不在本仓库源码内）**
- `huggingface_hub` 库：实际下载/缓存/etag 校验。
- `safetensors` 库：张量序列化格式。
- `torch.load`/`torch.serialization`：pytorch bin 反序列化。
- `sentencepiece`/`gguf` 解析后端（外部）。
- PyTorch `torch.cuda`/`torch.meshgrid`（外部）。

## 8. 与相邻子系统交互

- **上游**：modeling-utils 叶子的 `_load_pretrained_model` 调用本叶子。
- **本叶子 → configuration-utils**：读 `config._commit_hash`、`config.model_type`。
- **本叶子 → 外部**：HF Hub 下载、safetensors 库、torch.load、GGUF 解析。
- **下游**：具体模型 forward 调 `Cache` 的 `update`/`get_seq_length`。

## 9. 语言专项适配口径

Python 项目，采用 TS 口径的 Python 适配：
1. **分组**：按 capability seam 分组——本叶子 = "权重 IO 与缓存" seam。
2. **图类型**：dataflow（权重加载管道）+ architecture（组件边界）。
3. **外部边界**：HF Hub/safetensors/torch 标注外部。
4. **部署维度**：不适用单二进制。

## 10. 图表清单与质量

| 图 | 文件 | 类型 | archify 质量档 |
|----|------|------|----------------|
| 权重加载数据流图 | `weight-io-dataflow.html` | dataflow | standard（showcase 标签间距失败回退，如实披露） |
| 组件架构图 | `weight-io-architecture.html` | architecture | standard（showcase 回退，如实披露） |

JSON IR 源文件位于 `json/` 目录。
