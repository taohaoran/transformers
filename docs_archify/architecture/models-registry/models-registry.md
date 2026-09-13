# models-registry 域总览

> 本域是 HuggingFace transformers 仓库 `src/transformers/models/` 的组织与分发层：
> 它把约 516 个模型家族以"注册表 + 惰性加载 + 代码生成"的方式统一对外暴露为 `Auto*` 接口。
> 源码基准：v5.18.0.dev0（main），commit `5474a55e920f358d8382f3ecd3377edca979baa`。

## 1. 域职责

`models-registry` 域回答三个问题：
1. **用户怎么用**：`AutoModel`/`AutoConfig`/`AutoTokenizer` 等通用工厂如何按 checkpoint 的 `model_type`
   动态分发到约 516 个模型家族中的具体类（见 auto-registry 叶子）。
2. **几百个模型文件怎么维护**：`modular_*.py` 模板 + LibCST 自动生成 + `# Copied from` 遗留同步机制（见 modular-system 叶子）。
3. **模型家族怎么分门别类**：按架构范式分为 encoder / decoder / seq2seq / vision / multimodal / speech 六类，
   每类有代表家族与共性骨架。

域的代码根：`src/transformers/models/`（518 目录、约 2722 个 py 文件），其中自动分发层在 `models/auto/`。

## 2. 叶子索引

| 叶子 | 文档 | 架构图 | 其他图 | 职责一句话 |
|------|------|--------|--------|-----------|
| auto-registry | [auto-registry.md](auto-registry/auto-registry.md) | [架构图](auto-registry/auto-registry-architecture.html) | [时序图](auto-registry/auto-registry-sequence.html) | `Auto*` 工厂按 `model_type` 惰性分发到具体模型类 |
| modular-system | [modular-system.md](modular-system/modular-system.md) | [架构图](modular-system/modular-system-architecture.html) | [数据流图](modular-system/modular-system-dataflow.html) | modular 模板生成 + `# Copied from` 同步维护模型文件 |
| encoder-models | [encoder-models.md](encoder-models/encoder-models.md) | [架构图](encoder-models/encoder-models-architecture.html) | — | 双向 encoder（BERT/RoBERTa/DeBERTa/ELECTRA） |
| decoder-models | [decoder-models.md](decoder-models/decoder-models.md) | [架构图](decoder-models/decoder-models-architecture.html) | — | 因果 decoder-only（LLaMA/Qwen2/GPT2/Gemma/Mistral） |
| seq2seq-models | [seq2seq-models.md](seq2seq-models/seq2seq-models.md) | [架构图](seq2seq-models/seq2seq-models-architecture.html) | — | encoder-decoder（T5/BART/Pegasus） |
| vision-models | [vision-models.md](vision-models/vision-models.md) | [架构图](vision-models/vision-models-architecture.html) | — | 纯视觉（ViT/CLIP/Swin/DETR） |
| multimodal-models | [multimodal-models.md](multimodal-models/multimodal-models.md) | [架构图](multimodal-models/multimodal-models-architecture.html) | — | 视觉-语言多模态大模型（LLaVA/Qwen2-VL/BLIP/Florence-2） |
| speech-models | [speech-models.md](speech-models/speech-models.md) | [架构图](speech-models/speech-models-architecture.html) | — | 语音/音频（Whisper/SeamlessM4T/EnCodec） |

## 3. 域级机制细节

**自动注册体系（auto-registry 核心）**
- `auto_mappings.py` 是纯字符串 `OrderedDict`（`CONFIG_MAPPING_NAMES`、`MODEL_*_MAPPING_NAMES`），
  由 `utils/check_auto.py --fix_and_overwrite` 自动生成，**绝不手改**。
- `_LazyAutoMapping` / `_LazyConfigMapping` 在首次访问某 `model_type` 时才 `importlib.import_module`
  对应 `models/<family>/` 包——这是 `import transformers` 不必加载全部 516 家族的性能关键。
- 三方映射链：`model_type`（config.json）→ `Config` 类 → 任务头 `Model` 类；
  分词器/处理器/图像处理器各有独立 `*_MAPPING_NAMES`。
- `trust_remote_code` 经 `config.auto_map` 把 Hub 上的远程类动态注册进 `_extra_content`。

**modular 生成机制（modular-system 核心）**
- 现行机制：`modular_<name>.py` 可直接继承其他模型，`make fix-repo` 用 LibCST
  （`utils/modular_model_converter.py`）生成独立 `modeling_*.py` 等文件，文件头带"自动生成、勿手改"横幅。
- 遗留机制：`# Copied from <source>` 注释标记复制块，由 `utils/check_copies.py` 同步；**禁止新增**。
- 陷阱：`attr = AttributeError()` 是删除继承属性的指令，非占位符；其他模型可能继承同一 modular 源。

**516 模型家族的组织方式**
- 每个家族一个目录 `models/<family>/`，含 `configuration_<family>.py`、`modeling_<family>.py`、
  `tokenization_*.py`、`image_processing_*.py`（按需）；新家族由 modular 源 + `cls.model_type` 注册进自动表。
- 家族按架构范式归入本域的六个模型类叶子；本分析**只深读各类代表家族，其余列表说明**（见各叶子第 1 节覆盖范围披露）。

## 4. 域级图

![models-registry 域无独立域级图，各叶子图已覆盖组件与时序]

> 本域未单出域级大图：auto-registry 的架构图与时序图已表达注册分发机制，六个模型类叶子的架构图已表达各范式组件。
