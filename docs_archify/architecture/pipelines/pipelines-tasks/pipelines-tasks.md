# 任务管线族（pipelines-tasks）

> 本文是 `pipelines` 域下的叶子子系统文档。域级总览见 `../pipelines.md`。本文展开 **24 个具体任务管线** 的三步实现差异，框架基类见 `../pipelines-framework/`。
>
> 源码基准：`transformers` v5.18.0.dev0（main 分支），commit `5474a55e920f358d8382f3ecd3377edca979baa`。

## 1. 功能清单

本叶子覆盖 `src/transformers/pipelines/` 下 24 个任务管线文件（不含 base/pt_utils/__init__/audio_utils），按模态族分组：

| 族 | 管线文件 | 管线类 | 源码路径 |
|----|----------|--------|----------|
| **文本-生成** | text_generation.py | `TextGenerationPipeline` | `pipelines/text_generation.py:23` |
| **文本-分类** | text_classification.py | `TextClassificationPipeline` | `pipelines/text_classification.py` |
| **文本-MLM** | fill_mask.py | `FillMaskPipeline` | `pipelines/fill_mask.py` |
| **文本-NER** | token_classification.py | `TokenClassificationPipeline`（ChunkPipeline） | `pipelines/token_classification.py:92` |
| **文本-零样本** | zero_shot_classification.py | `ZeroShotClassificationPipeline` | `pipelines/zero_shot_classification.py` |
| **文本-特征** | feature_extraction.py | `FeatureExtractionPipeline` | `pipelines/feature_extraction.py` |
| **文本-翻译/摘要/QA** | text2text_generation.py（Translation/Summarization） | `TranslationPipeline`/`SummarizationPipeline` | `pipelines/`（本版本合并入 text2text 族） |
| **文本-表格QA** | table_question_answering.py | `TableQuestionAnsweringPipeline` | `pipelines/table_question_answering.py` |
| **文本-任意到任意** | any_to_any.py | `AnyToAnyPipeline` | `pipelines/any_to_any.py` |
| **文本-转音频** | text_to_audio.py | `TextToAudioPipeline` | `pipelines/text_to_audio.py` |
| **图像-分类** | image_classification.py | `ImageClassificationPipeline` | `pipelines/image_classification.py:73` |
| **图像-特征** | image_feature_extraction.py | `ImageFeatureExtractionPipeline` | `pipelines/image_feature_extraction.py` |
| **图像-分割** | image_segmentation.py | `ImageSegmentationPipeline` | `pipelines/image_segmentation.py` |
| **图像-检测** | object_detection.py | `ObjectDetectionPipeline` | `pipelines/object_detection.py` |
| **图像-零样本分类** | zero_shot_image_classification.py | `ZeroShotImageClassificationPipeline` | `pipelines/zero_shot_image_classification.py` |
| **图像-零样本检测** | zero_shot_object_detection.py | `ZeroShotObjectDetectionPipeline` | `pipelines/zero_shot_object_detection.py` |
| **图像-深度** | depth_estimation.py | `DepthEstimationPipeline` | `pipelines/depth_estimation.py` |
| **图像-掩码生成** | mask_generation.py | `MaskGenerationPipeline` | `pipelines/mask_generation.py` |
| **图像-关键点** | keypoint_matching.py | `KeypointMatchingPipeline` | `pipelines/keypoint_matching.py` |
| **多模态-图生文** | image_text_to_text.py | `ImageTextToTextPipeline` | `pipelines/image_text_to_text.py` |
| **多模态-文档QA** | document_question_answering.py | `DocumentQuestionAnsweringPipeline` | `pipelines/document_question_answering.py` |
| **音频-ASR** | automatic_speech_recognition.py | `AutomaticSpeechRecognitionPipeline`（ChunkPipeline） | `pipelines/automatic_speech_recognition.py:146` |
| **音频-分类** | audio_classification.py | `AudioClassificationPipeline` | `pipelines/audio_classification.py` |
| **音频-零样本** | zero_shot_audio_classification.py | `ZeroShotAudioClassificationPipeline` | `pipelines/zero_shot_audio_classification.py` |
| **音频-视频分类** | video_classification.py | `VideoClassificationPipeline` | `pipelines/video_classification.py` |
| **音频-图分类** | image_classification.py（同上） | — | — |

**深读覆盖**：本表"深读"标记为：`TextGenerationPipeline`、`TokenClassificationPipeline`、`ImageClassificationPipeline`、`AutomaticSpeechRecognitionPipeline` 四个代表管线；其余 20 个管线按族列说明（结构同三步抽象，差异在 preprocess 输入模态与 postprocess 格式化）。

## 2. 核心类型与接口清单

| 类型 | 位置 | 职责 |
|------|------|------|
| `TextGenerationPipeline(Pipeline)` | `text_generation.py:23` | 文本生成：`_pipeline_calls_generate=True`，preprocess 支持 Chat 输入（`apply_chat_template`），_forward 调 `model.generate()` |
| `TokenClassificationPipeline(ChunkPipeline)` | `token_classification.py:92` | NER：preprocess 输出 offset_mapping，postprocess 做 softmax + BIO 合并 + 跨分块去重 |
| `ImageClassificationPipeline(Pipeline)` | `image_classification.py:73` | 图像分类：preprocess 调 image_processor，_forward 裸 forward，postprocess softmax + top_k |
| `AutomaticSpeechRecognitionPipeline(ChunkPipeline)` | `automatic_speech_recognition.py:146` | 语音识别：preprocess 按 chunk_length_s 切音频，_forward 调 generate，postprocess 解码 + 时间戳 |
| `AggregationStrategy` 枚举 | `token_classification.py` | NER 合并策略：NONE/SIMPLE/FIRST/AVERAGE/MAX |
| `FillMaskPipeline` | `fill_mask.py` | MLM：postprocess 取 mask 位置 top_k 候选 |
| `ZeroShotClassificationPipeline` | `zero_shot_classification.py` | 零样本：preprocess 构造 NLI 假设对，postprocess 归一化候选标签概率 |
| `ObjectDetectionPipeline` | `object_detection.py` | 目标检测：postprocess 输出 bounding box + label + score |

## 3. 关键调用链

### 3.1 TextGenerationPipeline（`text_generation.py`）

1. **preprocess**（`:301-367`）：输入可为字符串或 `Chat` 对象；Chat 走 `tokenizer.apply_chat_template(messages, add_generation_prompt=..., return_tensors="pt")`；纯文本走 `tokenizer(prefix + prompt_text, return_tensors="pt")`；`handle_long_generation="hole"` 时截断输入尾部为生成留空间。
2. **_forward**（`:369-430`）：弹出 `prompt_text`，调整 `max_length/min_length`（prefix 长度补偿），调用 `model.generate(**model_inputs, **generate_kwargs)`。
3. **postprocess**（`:432-507`）：`tokenizer.decode` 输出 token IDs，剥离 prompt 部分，返回 `[{"generated_text": "..."}]`（含 Chat 模式下的 assistant 消息）。

### 3.2 TokenClassificationPipeline（`token_classification.py`，ChunkPipeline）

1. **preprocess**（`:246-301`）：tokenizer 编码句子，保留 `offset_mapping`（字符级偏移）、`special_tokens_mask`、`word_ids`；长句子切分多个分块（每块带 `is_last` 标记）。
2. **_forward**（`:302-324`）：标准 `model(**model_inputs)` 返回 logits。
3. **postprocess**（`:325-373`）：
   - logits float16/bf16 转 float32 后 softmax 得 scores；
   - `gather_pre_entities`：逐 token 过滤特殊 token，映射 word→char 偏移；
   - `aggregate(pre_entities, strategy)`：按 `AggregationStrategy` 将 B-X/I-X 序列合并为实体（如 `B-PER`+`I-PER` → `PER` 实体）；
   - `aggregate_overlapping_entities`：多分块重叠区域去重（取较长或较高分实体）。

### 3.3 ImageClassificationPipeline（`image_classification.py`）

1. **preprocess**（`:183-189`）：调 `self.image_processor(image, return_tensors="pt")` 得 `pixel_values`。
2. **_forward**（`:190-193`）：裸 `model(pixel_values=...)`（不调 generate）。
3. **postprocess**（`:194-230`）：logits 经 `function_to_apply`（sigmoid/softmax/none）后取 top_k，输出 `[{"score": float, "label": str}]`。

### 3.4 AutomaticSpeechRecognitionPipeline（`automatic_speech_recognition.py`，ChunkPipeline）

1. **preprocess**（`:379-517`）：加载音频数组，按 `chunk_length_s` 切分为 mel 频谱分块（`feature_extractor`），每块带 stride 重叠；长音频产出多个分块。
2. **_forward**（`:518-634`）：调 `model.generate(...)`（CTC 或 Seq2Seq 模型），支持时间戳与语言识别。
3. **postprocess**（`:635-745`）：tokenizer 解码为文本，合并多分块转录，可选输出 word-level 时间戳。

## 4. 配置项

| 参数 | 默认 / 行为 | 位置 |
|------|-------------|------|
| `top_k`（分类类管线） | 5；返回前 k 个类别 | `image_classification.py:108` |
| `function_to_apply` | `None`（自动推断）；可选 sigmoid/softmax/none | `image_classification.py:108` |
| `aggregation_strategy`（NER） | `AggregationStrategy.NONE`；SIMPLE/FIRST/AVERAGE/MAX | `token_classification.py:144` |
| `ignore_labels`（NER） | `["O"]`；过滤非实体标签 | `token_classification.py:327` |
| `chunk_length_s`（ASR） | 0（不切分）；>0 时按秒切音频 | `automatic_speech_recognition.py:283` |
| `stride_length_s`（ASR） | None；分块重叠秒数 | 同上 |
| `return_timestamps`（ASR） | False；`"char"` 或 `"word"` 时间戳 | `automatic_speech_recognition.py:518` |
| `handle_long_generation`（生成） | None；`"hole"` 截断输入为生成留空间 | `text_generation.py:305` |
| `continue_final_message`（生成/Chat） | None；自动检测末条是否 assistant | `text_generation.py:330` |
| `num_beams` / `max_new_tokens` | 透传 `model.generate()` | `text_generation.py:369` |

## 5. 错误与重试语义

- **长输入超限**：`TextGenerationPipeline` 在 `handle_long_generation="hole"` 时截断输入尾部（`text_generation.py:355-365`）；超限则 `ValueError`。
- **NER 合并冲突**：`aggregate_overlapping_entities` 对重叠实体取较长者或较高分者（`token_classification.py:382-390`），不报错。
- **图像分类超时**：`ImageClassificationPipeline` 支持 `timeout` 参数（URL 图片下载超时）。
- **ASR 分块边界**：stride 重叠区在 postprocess 合并时去重；CTC 模型与 Seq2Seq 模型走不同解码路径。
- 任务管线本身不做重试；异常直接向上抛出到 `Pipeline.__call__`。

## 6. 并发细节

- **ChunkPipeline 分块**：NER、ASR 等长输入管线继承 `ChunkPipeline`，`preprocess` 返回生成器逐块产出，`PipelineChunkIterator` 展平后由 DataLoader 批量推理，`PipelinePackIterator` 按 `is_last` 重新聚合（见 pipelines-framework 叶子）。
- **多进程预处理**：`num_workers>0` 时 tokenizer/图像处理器在 DataLoader worker 进程执行；GPU 推理在主进程。
- 任务管线内部无线程/锁；并发完全继承自基类。

## 7. 系统边界

**In-Scope**
- `src/transformers/pipelines/*.py`：24 个任务管线文件的三步实现
- `AggregationStrategy` 等枚举与工具函数

**Out-of-Scope**
- PyTorch 模型前向/`generate()` 实现——见 models-registry 域
- tokenizer/image_processor/feature_extractor 的具体实现——见 tokenization-processing 域
- 管线基类与工厂——见 `../pipelines-framework/`
- Chat 模板渲染（Jinja2）——见 `../../hub-utils/utils-generic/chat_template_utils.py`
- HF Hub 默认模型下载——见 `../../hub-utils/hub-utils/`

## 8. 与相邻子系统交互

- **`Pipeline.__call__` → 任务管线子类**：基类调用子类 `preprocess/_forward/postprocess`（见 `../pipelines-framework/`）。
- **任务管线 → tokenizer/processor**：preprocess 调 `self.tokenizer(...)` / `self.image_processor(...)`。
- **任务管线 → 模型**：_forward 调 `self.model(**inputs)` 或 `self.model.generate(**inputs)`。
- **任务管线 → Hub**：通过 `SUPPORTED_TASKS` 默认模型 ID 从 Hub 拉取（见 `../../hub-utils/hub-utils/`）。

## 9. 语言专项适配口径

Python 项目：按"模态族 + 任务"capability seam 分组。图以 architecture（族层级）为主；不产出 sequence（三步调用链已在 pipelines-framework 的 call-sequence 图中统一表达，任务管线仅差异化 postprocess）。外部依赖（PyTorch、tokenizers、Jinja2）均标注"不在本仓库源码内"。

## 10. 图表清单与质量

| 图 | 文件 | 类型 | archify 质量档 |
|----|------|------|----------------|
| 任务管线族架构图 | `pipelines-tasks-architecture.html` | architecture | showcase |

JSON IR 源文件位于 `json/` 目录。

**覆盖范围披露**：深读 `TextGenerationPipeline`（`text_generation.py`）、`TokenClassificationPipeline`（`token_classification.py`）、`ImageClassificationPipeline`（`image_classification.py`）、`AutomaticSpeechRecognitionPipeline`（`automatic_speech_recognition.py`）4 个代表管线；其余 20 个管线按族列表说明（三步结构一致，差异在 preprocess 输入模态与 postprocess 格式化逻辑）。
