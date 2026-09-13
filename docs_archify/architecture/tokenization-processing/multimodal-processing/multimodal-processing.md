# multimodal-processing（Processor 多模态统一处理）

> 本文是 `tokenization-processing` 域下的叶子子系统文档。域级总览见 `../tokenization-processing.md`。
> 本文只展开多模态处理器基类与图像/音频/视频预处理；文本分词见
> [`../tokenization-utils-base/`](../tokenization-utils-base/tokenization-utils-base.md)。
>
> 源码基准：HuggingFace transformers `main`，commit `5474a55e920f358d8382f3ecd3377edca979baa`。

## 1. 功能清单

| 能力 | 说明 | 源码路径 |
|------|------|----------|
| `ProcessorMixin` | 多模态处理器基类（文本+图像+音频+视频统一） | `src/transformers/processing_utils.py:602` |
| `_LazyAutoProcessorMapping` | AutoProcessor 惰性注册映射 | `:98` |
| `TextKwargs`/`ImagesKwargs`/`VideosKwargs`/`AudioKwargs`/`ProcessingKwargs` TypedDict | 各模态参数契约 | `:166`、`:235`、`:309`、`:386`、`:438` |
| `MultiModalData` | 多模态数据容器 | `:568` |
| `prepare_prompt_input` | 多模态 prompt 组装 | `:2368` |
| `BaseImageProcessor` | 图像处理器基类 | `src/transformers/image_processing_utils.py:60` |
| `is_valid_size_dict`/`get_size_dict`/`select_best_resolution` | 尺寸字典工具 | `:542`、`:584`、`:634` |
| `BatchFeature` | 特征批处理容器（UserDict） | `src/transformers/feature_extraction_utils.py:58` |
| `FeatureExtractionMixin` | 特征提取基类 | `:266` |
| `load_audio`/`load_audio_torchcodec`/`load_audio_librosa` | 音频加载（多后端） | `src/transformers/audio_utils.py:198`、`:282`、`:292` |
| `get_audio_filetype`/`_resolve_audio_source` | 音频格式探测 | `:113`、`:176` |
| `conv1d_output_length` | 音频卷积输出长度计算 | `:378` |
| `VideoMetadata`/`is_valid_video`/`make_batched_videos` | 视频元数据与批处理 | `src/transformers/video_utils.py:80`、`:130`、`:185` |
| `ChannelDimension` 枚举 | 通道维（channels_first/last） | `src/transformers/image_utils.py:85` |
| `ImageType`/`AnnotationFormat` 枚举 | 图像类型/标注格式 | `:90`、`:102` |
| `is_pil_image`/`get_image_type`/`valid_images`/`make_list_of_images` | 图像校验与批处理 | `:98`、`:108`、`:135`、`:164` |
| `to_numpy_array` | 图像→numpy | `:279` |

> 注：任务说明中提及的 `src/transformers/transforms/` 与 `src/transformers/backends/` 目录在当前 commit 下不存在；图像变换逻辑已并入 `image_utils.py` 与 `image_processing_utils.py`，后端检测逻辑并入 `processing_utils.py` 的库可用性检查（`is_torch_available`/`is_vision_available` 等）。

## 2. 核心类型与接口清单

| 类型 / 函数 | 位置 | 职责 |
|-------------|------|------|
| `ProcessorMixin` | `processing_utils.py:602` | 多模态处理器基类，组合 tokenizer + image/audio/video processor |
| `MultiModalData` | `:568` | 持有 text/images/videos/audio 的数据容器 |
| `BatchFeature` | `feature_extraction_utils.py:58` | 特征批处理结果（UserDict 子类） |
| `BaseImageProcessor` | `image_processing_utils.py:60` | 图像处理器基类，`__call__` 做 resize/normalize/to_tensor |
| `FeatureExtractionMixin` | `feature_extraction_utils.py:266` | 特征提取基类 |
| `load_audio` | `audio_utils.py:198` | 按后端选择加载音频为 numpy 数组 |
| `VideoMetadata` | `video_utils.py:80` | 视频帧元数据 Mapping |

## 3. 关键调用链

### 3.1 `processor(text=..., images=...)` 统一处理链

1. `ProcessorMixin.__call__`（`processing_utils.py:602` 子类）接收 text/images/videos/audio。
2. 文本侧：调 `PreTrainedTokenizerBase.__call__`（tokenization-utils-base 叶子）。
3. 图像侧：调 `BaseImageProcessor.__call__` 做 resize/normalize/to_tensor。
4. 音频侧：调 `load_audio` 加载，再做重采样/对数梅尔谱。
5. 视频侧：`make_batched_videos` 组装，逐帧走图像处理。
6. 输出合并为 `BatchFeature`/`BatchEncoding` 字典。

### 3.2 音频加载后端选择

1. `load_audio`（`audio_utils.py:198`）`backend="auto"` 时按可用性选 torchcodec→librosa→soundfile。
2. `_resolve_audio_source`（`:176`）把 URL/路径转 bytes 或数组。
3. `get_audio_filetype`（`:113`）探测格式。

## 4. 配置项

| flag / option | 默认 / 行为 | 位置 |
|---------------|-------------|------|
| `do_resize`/`size`/`resample` | 图像 resize 参数 | `image_processing_utils.py` |
| `do_normalize`/`image_mean`/`image_std` | 归一化参数 | 同上 |
| `return_tensors` | `pt`/`np`/`tf` | `processing_utils.py` |
| `sampling_rate` | 16000，音频重采样率 | `audio_utils.py:198` |
| `backend`（audio） | `auto`，可选 `torchcodec`/`librosa`/`soundfile` | 同上 |
| `do_pad`/`do_convert_rgb` | 图像填充/转 RGB | image processor 子类 |

## 5. 错误与重试语义

- **图像不是 PIL/numpy/tensor**：`valid_images` 抛 `ValueError`。
- **音频加载失败**：`load_audio` 后端都不可用时抛 `ImportError`，提示安装 librosa。
- **视频帧无效**：`is_valid_video_frame` 抛 `ValueError`。
- 网络音频下载失败：`_fetch_audio_bytes` 抛 `requests` 异常（外部库）。
- 无本叶子级重试。

## 6. 并发细节

- 本叶子为同步 numpy/torch 张量计算，无 goroutine/线程。
- `load_audio` 的 `requests` 下载（外部）有 `timeout` 参数。
- 图像变换用 PIL（外部）/torchvision（外部）后端，本叶子只做调度。

## 7. 系统边界

**In-Scope（本仓库源码内）**
- `ProcessorMixin` 与多模态参数契约 TypedDict。
- `BaseImageProcessor` 基类与尺寸工具。
- `BatchFeature`/`FeatureExtractionMixin`。
- 音频加载/格式探测。
- 视频元数据与批处理。
- 图像校验/批处理/类型枚举。

**Out-of-Scope（不在本仓库源码内）**
- PIL/Pillow、torchvision、librosa、soundfile、torchcodec 库。
- NumPy/PyTorch 张量后端。
- 具体模型 Processor（`models/<x>/processing_<x>.py`）——本叶子只定义基类。
- HF Hub 下载（图像/音频 URL）。

## 8. 与相邻子系统交互

- **上游**：用户代码 / `pipeline` / 多模态模型 forward 前调用本叶子。
- **本叶子 → tokenization-utils-base 叶子**：文本侧调 tokenizer。
- **本叶子 → modeling-utils 叶子**：输出 `input_ids`/`pixel_values`/`audio_values`/`video_values` 喂给模型 forward。
- **本叶子 → 外部**：PIL/torchvision/librosa/soundfile/torchcodec。

## 9. 语言专项适配口径

Python 项目，采用 TS 口径的 Python 适配：
1. **分组**：按 capability seam 分组——本叶子 = "多模态预处理" seam。
2. **图类型**：architecture（组件边界）。
3. **外部边界**：PIL/torchvision/librosa 标注外部。
4. **部署维度**：不适用单二进制。

## 10. 图表清单与质量

| 图 | 文件 | 类型 | archify 质量档 |
|----|------|------|----------------|
| 多模态处理器组件架构图 | `multimodal-processing-architecture.html` | architecture | standard（showcase 回退，如实披露） |

JSON IR 源文件位于 `json/` 目录。
