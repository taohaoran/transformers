# 多模态模型（multimodal-models）

> 本文是 `models-registry` 域下的叶子子系统文档。域级总览见 `../models-registry.md`。
> 本叶子覆盖**视觉-语言多模态大模型**：图像 + 文本联合输入，用 LLM 解码。
>
> 源码基准：HuggingFace transformers v5.18.0.dev0（main），commit `5474a55e920f358d8382f3ecd3377edca979baa`。

## 1. 功能清单（覆盖范围披露）

**深读家族：**

| 家族 | 能力 | 源码路径 |
|------|------|----------|
| LLaVA | 视觉编码器 + MLP 投影 + LLM，图像 token 拼进文本序列 | `models/llava/modeling_llava.py`（`LlavaMultiModalProjector:87`、`LlavaModel:130`，`vision_tower:133`、`multi_modal_projector:135`、`language_model:136`） |
| Qwen2-VL | 视觉编码器 + 动态分辨率（视觉旋转位置编码）+ Qwen2 LLM | `models/qwen2_vl/modeling_qwen2_vl.py`（`Qwen2VLVisionRotaryEmbedding:228`、`Qwen2VLVisionBlock:468`） |
| Florence-2 | 统一多任务视觉-语言框架 | `models/florence2/` |
| BLIP | 图文对齐 + 图文匹配（ITM）+ 条件生成 | `models/blip/modeling_blip.py`（`BlipVisionEmbeddings:172`、`BlipTextEmbeddings:247`） |

**仅列表说明（不逐个深读）：** InstructBlip、LlavaNext、Idefics、Pix2Struct、Blip-2、Flamingo、
Chameleon、Aria、Qwen2-VL 系、DeepSeek-VL 等。它们共享"视觉塔 → 投影 → LLM"三幕结构。

## 2. 核心类型与接口清单（以 LLaVA 为代表）

| 类型 | 位置 | 职责 |
|------|------|------|
| `LlavaMultiModalProjector` | `llava/modeling_llava.py:87` | 两层 MLP，把视觉特征维度对齐到 LLM 隐藏维 |
| `LlavaModel` | `:130` | 组合 `vision_tower`（`AutoModel.from_config`，`:133`）+ `multi_modal_projector`（`:135`）+ `language_model`（`:136`） |
| 前向融合 | `:154`/`:174`/`:250` | 图像过 vision_tower → projector；把投影后图像特征"替换"输入序列中的图像占位 token（`:180` 按 patch_size 计算位置）→ 拼好的混合序列喂 `language_model`（`:250`） |
| `Qwen2VLVisionRotaryEmbedding` | `qwen2_vl/modeling_qwen2_vl:228` | 视觉侧旋转位置编码，支持动态分辨率（图像随尺寸变化 token 数） |
| `Blip` 双头 | `blip/modeling_blip.py` | 除条件生成外，另有图文匹配（ITM）与图文相似度分支 |

## 3. 关键调用链

**LlavaForConditionalGeneration 前向（`llava/modeling_llava.py`）**
1. 图像 `pixel_values` 进 `vision_tower`（`:154`）取视觉特征。
2. `multi_modal_projector`（`:174`）把视觉特征投影到 LLM 维。
3. 按 `image_sizes/patch_size`（`:180`）把投影特征写入输入 embedding 中对应图像占位 token 位置。
4. 混合（图像+文本）序列进 `language_model`（`:250`）因果解码；`LlavaForCausalLM` 出 logits，`generate` 自回归生成文本回答。

**Qwen2-VL**：视觉塔自带 `Qwen2VLVisionRotaryEmbedding`（`:228`）按图像实际分辨率动态算位置，图像 token 数随分辨率变化；文本侧复用 Qwen2 decoder。

**BLIP**：在图文编码基础上额外训练图文匹配头（ITM），做检索/对齐；与 LLaVA 的"纯生成"取向不同。

## 4. 配置项

| 配置项 | 默认 / 行为 | 位置 |
|--------|-------------|------|
| `model_type` | `"llava"`/`"qwen2_vl"`/`"florence2"`/`"blip"` | 各配置 |
| `vision_config` / `text_config` | 子配置：视觉塔与 LLM 各自配置（composite config） | `configuration_llava.py` |
| `image_token_index` / 图像占位 | 输入序列中图像占位 token id | tokenizer/processor |
| `vision_feature_select_layer` | 取视觉塔哪一层输出 | llava 配置 |
| `min/max_pixels` | Qwen2-VL 动态分辨率上下限 | `configuration_qwen2_vl.py` |

## 5. 错误与重试语义

- 多模态前向为张量拼接；图像 token 数与占位不匹配时报 shape 错。
- composite config（`vision_config`+`text_config`）在 auto 层有 `get_text_config` 处理（见 auto-registry）。无业务重试。

## 6. 并发细节

- 同步张量计算；视觉塔常冻结或低学习率微调。
- 生成时 LLM 侧 KV cache 复用 decoder-models 机制；视觉特征每图只编码一次。
- 无 Python 线程模型。

## 7. 系统边界

**In-Scope（本仓库源码内）**
- `models/llava/`、`models/qwen2_vl/`、`models/florence2/`、`models/blip/`。

**Out-of-Scope（不在本仓库源码内）**
- 视觉塔底层（ViT/SigLIP，见 vision-models）、LLM 底层（见 decoder-models）均为组合复用，本叶子只做融合编排。
- 图像预处理在 `image_processing_auto`/processor，不在本叶子。

## 8. 与相邻子系统交互

- 上游：用户经 `AutoModelForImageTextToText`/`AutoModelForMultimodalLM`（auto-registry）分发；图像经 image processor 预处理。
- 本叶子 → 下游：复用 vision-models（视觉塔）与 decoder-models（LLM）作为子模块；`GenerationMixin` 做生成。
- composite config 经 auto-registry 的 `get_text_config` 拆分。

## 9. 语言专项适配口径（Python）

- **分组**：按 capability seam——"视觉编码与投影"与"LLM 融合解码"。
- **图类型**：architecture（三幕融合编排）为主。不产出 lifecycle。
- **外部边界**：复用的视觉塔/LLM 标注为相邻子系统；无新外部依赖。

## 10. 图表清单与质量

| 图 | 文件 | 类型 | archify 质量档 |
|----|------|------|----------------|
| 多模态架构（LLaVA 代表） | `multimodal-models-architecture.html` | architecture | **standard**（连线标签布局约束回退 standard；已如实披露） |

JSON IR 源：`json/multimodal-models-architecture.json`。
不另出 sequence：融合数据流已在 architecture 连线表达。

**覆盖范围披露**：深读 LLaVA/Qwen2-VL/Florence-2/BLIP 四个家族；其余（InstructBlip、LlavaNext、Idefics、Pix2Struct、Blip-2、Flamingo、Chameleon 等）仅按共性列表说明，未逐类读源码。
