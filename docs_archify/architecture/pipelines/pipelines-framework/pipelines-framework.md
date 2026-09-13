# 管线框架（pipelines-framework）

> 本文是 `pipelines` 域下的叶子子系统文档。域级总览见 `../pipelines.md`，本文只展开 **Pipeline 基类框架与 `pipeline()` 工厂函数** 的职责边界，不展开各具体任务管线（分别见 `../pipelines-tasks/`）。
>
> 源码基准：`transformers` v5.18.0.dev0（main 分支），commit `5474a55e920f358d8382f3ecd3377edca979baa`。

## 1. 功能清单

| 能力 | 说明 | 源码路径 |
|------|------|----------|
| `Pipeline` 基类 | 所有推理管线的抽象基类，定义"输入→预处理→模型推理→后处理→输出"五步框架 | `src/transformers/pipelines/base.py:757`（`class Pipeline`） |
| 三步推理抽象 | `preprocess` / `_forward` / `postprocess` 三个抽象方法，由各任务管线实现 | `base.py:1151` / `1159` / `1172` |
| 参数分段清洗 | `_sanitize_parameters` 将 `__init__` 与 `__call__` 的 kwargs 拆为 preprocess/forward/postprocess 三组参数 | `base.py:1138` |
| `__call__` 推理入口 | 统一推理入口，支持单条/列表/数据集/生成器输入，自动选择 run_single 或 DataLoader 批量迭代 | `base.py:1217` |
| `forward()` 设备放置与推理上下文 | 包装 `_forward`：`torch.no_grad()` 推理上下文、输入/输出张量设备搬运 | `base.py:1183` |
| `load_model()` 模型加载 | 字符串模型 ID 时按 `model_classes` 元组依次尝试 `from_pretrained`，首个成功即停；dtype 失败回退 float32 | `base.py:187` |
| `load_assistant_model()` | 投机解码辅助模型加载，校验主/辅模型词表一致性 | `base.py:313` |
| `pipeline()` 工厂函数 | 任务名→管线类的统一工厂：配置加载、默认模型选择、组件（tokenizer/processor）自动解析、模型加载、实例化 | `src/transformers/pipelines/__init__.py:671` |
| `SUPPORTED_TASKS` 注册表 | 任务名→`{impl, pt, default, type}` 映射，含默认模型与 revision | `__init__.py:141` |
| `TASK_ALIASES` 别名表 | `sentiment-analysis→text-classification`、`ner→token-classification`、`text-to-speech→text-to-audio` | `__init__.py:136` |
| `PipelineRegistry` | 任务注册表封装：`check_task` 别名解析、`register_pipeline` 动态注册 | `base.py:1347` |
| 批量推理迭代器 | `PipelineDataset`（Map-style）、`PipelineIterator`（Iterable-style 反批次）、`PipelineChunkIterator`（变长分块）、`PipelinePackIterator`（按 `is_last` 打包） | `src/transformers/pipelines/pt_utils.py:8/23/156/201` |
| 批次 collate | `pad_collate_fn` 按 tokenizer/feature_extractor 配置自动 padding；`no_collate_fn` 单条直通 | `base.py:124` / `73` |
| `ChunkPipeline` | 长输入分块基类：`preprocess` 产出多个分块，逐块推理后合并后处理 | `base.py:1315` |
| 数据格式 IO | `PipelineDataFormat` 抽象 + JSON/CSV/管道(stdin/stdout)三种实现 | `base.py:392/505/549/591` |
| 设备自动选择 | 按 CUDA→MLU→MUSA→NPU→HPU→XPU→MPS→CPU 优先级自动选择推理设备 | `base.py:846-863` |
| Chat 输入检测 | 检测消息列表自动包装为 `Chat` 对象（对话式管线） | `base.py:1221-1247` |

## 2. 核心类型与接口清单

| 类型 / 函数 | 位置 | 职责 |
|-------------|------|------|
| `Pipeline(_ScikitCompat, PushToHubMixin)` | `base.py:757` | 管线基类，持有 model/tokenizer/processor 等组件，定义推理生命周期 |
| `ChunkPipeline(Pipeline)` | `base.py:1315` | 长输入分块管线基类，覆盖 `run_single`/`get_iterator` |
| `ArgumentHandler(ABC)` | `base.py:382` | 各管线参数处理抽象接口 |
| `PipelineException(Exception)` | `base.py:365` | 管线推理异常，携带 task/model/reason |
| `PipelineDataFormat` / `Json/Csv/PipedPipelineDataFormat` | `base.py:392/549/505/591` | 文件/stdin 数据读写格式 |
| `_ScikitCompat(ABC)` | `base.py:639` | 兼容 sklearn `fit/predict` 接口的混入基类 |
| `load_model()` | `base.py:187` | 多模型类尝试加载 + dtype 回退 |
| `pipeline()` 工厂函数 | `__init__.py:671` | 对外主入口，任务分发与组件装配 |
| `PipelineRegistry` | `base.py:1347` | 任务注册表（`SUPPORTED_TASKS` + `TASK_ALIASES` 的运行时封装） |
| `PipelineDataset(Dataset)` | `pt_utils.py:8` | Map-style 数据集包装，`__getitem__` 调用 `preprocess` |
| `PipelineIterator(IterableDataset)` | `pt_utils.py:23` | 迭代器包装，将 DataLoader 批次拆回单条（`loader_batch_size` 反批次） |
| `PipelineChunkIterator(PipelineIterator)` | `pt_utils.py:156` | 展平嵌套生成器（preprocess 产出多块） |
| `PipelinePackIterator(PipelineIterator)` | `pt_utils.py:201` | 按 `is_last` 标记将分块重新打包为完整样本 |
| `KeyDataset` / `KeyPairDataset` | `pt_utils.py:301/313` | HuggingFace `datasets` 列名→管线参数的键值映射 |
| `pad_collate_fn` | `base.py:124` | 批次 padding 整理函数工厂 |

## 3. 关键调用链

### 3.1 `pipeline()` 工厂构造链（`__init__.py:671-1127`）

1. **参数校验**（`:844-879`）：task 与 model 不能同时为空；model 与 tokenizer/feature_extractor 不能单独提供；`Path` 转字符串。
2. **commit hash 解析**（`:881-901`）：先调用 `cached_file()` 拉取 `config.json` 以尽早获取 `_commit_hash`（私有仓库/离线模式安全降级）。
3. **配置实例化**（`:906-937`）：`config` 为字符串时 `AutoConfig.from_pretrained`；model 为字符串且 config 缺失时，先检测 PEFT adapter 文件（`find_adapter_config_file`），再 `AutoConfig.from_pretrained(model)`。
4. **任务分发**（`:939-979`）：
   - 优先检查 `config.custom_pipelines`（远程代码自定义管线，需 `trust_remote_code=True`，通过 `get_class_from_dynamic_module` 动态加载）。
   - 否则走内置 `SUPPORTED_TASKS`：`check_task(task)` 解析别名，取 `targeted_task["impl"]` 作为 `pipeline_class`。
5. **默认模型兜底**（`:982-992`）：用户未传 model 时，从 `SUPPORTED_TASKS[task]["default"]` 取默认模型名与 revision。
6. **模型加载**（`:1034-1043`）：字符串 model 调 `load_model()`，按 `targeted_task["pt"]` 元组依次尝试模型类。
7. **组件自动解析**（`:1050-1104`）：根据 pipeline 类的 `_load_tokenizer/_load_image_processor/_load_feature_extractor/_load_processor/_load_video_processor` 标志位，分别调用 `_resolve_tokenizer` 等函数加载对应组件。
8. **实例化返回**（`:1127`）：`pipeline_class(model=model, task=task, **kwargs)`。

### 3.2 `Pipeline.__call__` 推理链（`base.py:1217-1299`）

1. **输入类型检测**（`:1221-1247`）：generator 先 peek 首条判断是否为 Chat 消息；list/tuple/KeyDataset 检测 `is_valid_message` 自动包装为 `Chat`。
2. **参数融合**（`:1260-1265`）：`_sanitize_parameters(**kwargs)` 拆分运行时参数，与 `__init__` 时的 `_preprocess_params` 等做非覆盖式合并。
3. **输入分发**（`:1274-1299`）：
   - `list` → `get_iterator()` 后 `list()` 急切消费返回列表；
   - `Dataset`/`Generator` → 返回惰性迭代器（不物化全部输入）；
   - `ChunkPipeline` → 取迭代器首个结果（分块合并后单条返回）；
   - 单条输入 → `run_single()`：`preprocess → forward → postprocess` 同步三步。
4. **批量迭代链**（`get_iterator()` `:1192-1215`）：
   - `PipelineDataset(inputs, preprocess, preprocess_params)` 包装输入；
   - `DataLoader(dataset, num_workers, batch_size, collate_fn)` 多进程预处理；
   - `PipelineIterator(dataloader, forward, forward_params, loader_batch_size=batch_size)` 反批次；
   - `PipelineIterator(model_iterator, postprocess, postprocess_params)` 后处理。

### 3.3 `forward()` 设备与推理上下文链（`base.py:1183-1190`）

1. `device_placement()` 上下文（若模型在多设备上）；
2. `get_inference_context()` 默认返回 `torch.no_grad`；
3. `_ensure_tensor_on_device` 把输入张量搬到 `self.device`；
4. 调用子类 `_forward(model_inputs, **forward_params)`；
5. 输出张量搬回 CPU（`torch.device("cpu")`）以释放 GPU 显存。

## 4. 配置项

| 参数 / flag | 默认 / 行为 | 位置 |
|-------------|-------------|------|
| `device` | `None`→自动选择首个可用加速器（CUDA/MLU/MUSA/NPU/HPU/XPU/MPS），无加速器→CPU；`-1` 强制 CPU | `base.py:799, 823-863` |
| `device_map` | 传入 `model_kwargs`，`"auto"` 时由 accelerate 自动分片；与 `device` 互斥 | `__init__.py:801, 994-1005` |
| `dtype` | `"auto"`（按模型保存精度加载）；可传 `"float16"`/`torch.bfloat16` 等 | `__init__.py:813, 1026-1029` |
| `batch_size` | `None`→1（单条推理）；`>1` 启用 DataLoader 批量 | `base.py:937, 1254-1258` |
| `num_workers` | `None`→0（主进程预处理）；`>0` 多进程 DataLoader | `base.py:938, 1249-1253` |
| `binary_output` | `False`；`True` 时输出 pickle 二进制（大张量场景） | `base.py:800, 869` |
| `use_fast` | `True`；优先使用 `PreTrainedTokenizerFast` | `__init__.py:681` |
| `revision` | `None`；模型仓库分支/tag/commit | `__init__.py:680, 787` |
| `token` | `None`；Hub 认证 token 或 `True`（读 `hf auth login` 缓存） | `__init__.py:682` |
| `trust_remote_code` | `None`；是否允许执行 Hub 仓库中的自定义代码 | `__init__.py:686` |
| `assistant_model` | `None`；投机解码辅助模型（仅 generate 管线） | `base.py:884` |
| `TOKENIZERS_PARALLELISM` | 批量推理时自动设为 `"false"`（与 DataLoader 多进程冲突） | `base.py:1206-1208` |
| `_load_processor/_load_tokenizer/...` | 类级标志位（`True/None/False`），控制工厂函数是否加载对应组件 | `base.py:779-783` |

## 5. 错误与重试语义

- **模型加载失败回退**（`base.py:235-278`）：`load_model()` 按 `model_classes` 元组逐个尝试 `from_pretrained`，捕获 `OSError/ValueError/TypeError/RuntimeError` 后记录 traceback 继续下一个类；全部失败时拼接所有 traceback 抛出 `ValueError`。
- **dtype 设备回退**（`base.py:247-265`）：若因 dtype 在目标设备不支持（如消费级 GPU 不支持 bf16），自动以 `torch.float32` 重试一次并 warning。
- **未知任务**（`base.py:1357-1364`）：`PipelineRegistry.check_task()` 对未注册任务抛 `KeyError`，列出全部可用任务名。
- **设备不可用**（`base.py:832-843`）：XPU/HPU 设备字符串不可用时抛 `ValueError`，建议改用 `device="cpu"`。
- **adapter 冲突**（`__init__.py:994-1005`）：`device_map` 与 `model_kwargs["device_map"]` 同时存在抛 `ValueError`；`device` 与 `device_map` 同时存在 warning 并以 `device` 为准。
- **空输入**（`base.py:1227-1228`）：generator 立即 `StopIteration` 时 inputs 置空列表。
- 管线框架本身不做自动重试；模型加载的"多类尝试"是唯一的容错路径。

## 6. 并发细节

- **DataLoader 多进程预处理**（`base.py:1212`）：`num_workers>0` 时由 PyTorch DataLoader 多进程并行执行 `preprocess`（tokenizer/图像处理在 worker 进程），主线程负责 `forward`（GPU 推理）与 `postprocess`。
- **tokenizer 并行禁用**（`base.py:1206-1208`）：DataLoader 多进程已开启并行，自动设 `TOKENIZERS_PARALLELISM=false` 避免 fork 重复。
- **惰性迭代**（`base.py:1284-1289`）：Dataset/Generator 输入返回 `PipelineIterator` 惰性链，输入不物化到内存，适合流式推理。
- **`PipelineIterator` 反批次**（`pt_utils.py:119-153`）：GPU 批量推理结果在迭代器中逐条拆出，使 `postprocess` 看到的是单条 batch_size=1 的结果（保持与单条推理一致的接口）。
- **`PipelinePackIterator` 状态机**（`pt_utils.py:251-298`）：内部维护 `accumulator` 与 `is_last` 标记，将分块结果重新聚合为完整样本，是有状态迭代器。
- **GPU 顺序调用警告**（`base.py:1268-1272`）：同一管线在 CUDA 上被顺序调用超过 10 次时 warning 建议改用 Dataset 批量（避免逐条同步开销）。
- 本叶子不使用 Python 线程锁/异步 IO；并发模型完全依赖 PyTorch DataLoader 的多进程模型。

## 7. 系统边界

**In-Scope（本仓库源码内）**
- `src/transformers/pipelines/base.py`：Pipeline 基类、load_model、数据格式 IO、PipelineRegistry
- `src/transformers/pipelines/pt_utils.py`：批量推理迭代器族
- `src/transformers/pipelines/__init__.py`：pipeline() 工厂、SUPPORTED_TASKS/TASK_ALIASES 注册表

**Out-of-Scope（不在本仓库源码内）**
- PyTorch（`torch`、`torch.utils.data.DataLoader`、`torch.no_grad`）——外部依赖
- tokenizers（Rust 实现的快速分词器后端）——外部依赖
- HuggingFace Hub 远端服务（模型下载、gated repo 鉴权）——外部服务
- accelerate（`device_map="auto"` 分片、`hf_device_map`）——外部库，不在本仓库
- 各具体任务管线的 preprocess/_forward/postprocess 实现——见 `../pipelines-tasks/`
- 模型本身（`PreTrainedModel` 子类、`generate()` 方法）——见 models-registry 域

## 8. 与相邻子系统交互

- **用户代码 → `pipeline()` 工厂**：传入任务名/模型名 → 返回实例化的 `Pipeline` 子类。
- **`pipeline()` → Auto 系列**：`AutoConfig.from_pretrained`、`AutoTokenizer.from_pretrained`（见 models-registry/auto-registry）。
- **`pipeline()` → Hub 下载层**：`cached_file()`、`extract_commit_hash()`（见 `../../hub-utils/hub-utils/`）。
- **`Pipeline.__call__` → 任务管线子类**：调用子类实现的 `preprocess/_forward/postprocess`（见 `../pipelines-tasks/`）。
- **`Pipeline.forward()` → 模型**：`self._forward()` 内部调用 `model(**inputs)` 或 `model.generate(**inputs)`。
- **`pipeline()` → PEFT 集成**：`find_adapter_config_file()` 检测 LoRA adapter（见 `../../distributed-deployment/integrations-frameworks/`）。

## 9. 语言专项适配口径

本仓库为 **Python** 项目，按 Python 适配口径执行：

1. **分组方式**：按业务领域 + capability seam 分组。本叶子属于"管线框架"capability seam，与具体任务管线（pipelines-tasks）分离——框架定义抽象基类与工厂，任务族实现三步方法。
2. **图类型选择**：以 architecture（组件边界）+ sequence（`__call__` 推理调用链）为主；不产出 lifecycle（管线无显式状态机，call_count 仅为计数器）。
3. **外部边界标注**：PyTorch、tokenizers、Hub 远端、accelerate 均标注"不在本仓库源码内"。
4. **部署维度**：不适用"单二进制"分析；改用包结构 + 依赖图表达（`pipelines/` 包内模块依赖关系）。
5. **并发模型**：Python 无 goroutine；并发通过 PyTorch DataLoader 多进程实现，已在第 6 节展开。

## 10. 图表清单与质量

| 图 | 文件 | 类型 | archify 质量档 |
|----|------|------|----------------|
| 管线框架组件架构图 | `pipelines-framework-architecture.html` | architecture | showcase |
| `pipeline()` 工厂构造时序图 | `pipelines-framework-factory-sequence.html` | sequence | showcase |
| `__call__` 推理调用时序图 | `pipelines-framework-call-sequence.html` | sequence | showcase |

JSON IR 源文件位于 `json/` 目录（3 个文件，与 HTML 一一对应）。
