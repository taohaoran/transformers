# 视觉模型（vision-models）

> 本文是 `models-registry` 域下的叶子子系统文档。域级总览见 `../models-registry.md`。
> 本叶子覆盖**纯视觉输入模型**：图像分类、目标检测、图文对比等。
>
> 源码基准：HuggingFace transformers v5.18.0.dev0（main），commit `5474a55e920f358d8382f3ecd3377edca979baa`。

## 1. 功能清单（覆盖范围披露）

**深读家族：**

| 家族 | 能力 | 源码路径 |
|------|------|----------|
| ViT | patch embedding + 标准 Transformer encoder 做图像分类 | `models/vit/modeling_vit.py`（`ViTPatchEmbeddings:42`、`ViTEmbeddings:72`、`ViTAttention:192`） |
| CLIP | 图像编码器 + 文本编码器双塔 + 图文对比损失 | `models/clip/modeling_clip.py`（`contrastive_loss:47`、`CLIPVisionEmbeddings:138`、`CLIPTextEmbeddings:221`） |
| Swin | 分层 Transformer + 移位窗口（shifted window）注意力 | `models/swin/` |
| DETR | CNN backbone + transformer encoder-decoder + 匈牙利匹配目标检测 | `models/detr/modeling_detr.py`（`DetrConvEncoder:242`、`load_backbone:255`） |

**仅列表说明（不逐个深读）：** ConvNeXt、ResNet、MobileNet、SegFormer、DPT、Yolos、BeIT、 DeiT、
PoolFormer、MobileViT、DepthAnything 等。它们共享"图像 → patch/特征图 → transformer/CNN → 任务头"骨架。

## 2. 核心类型与接口清单（以 ViT/CLIP/DETR 为代表）

| 类型 | 位置 | 职责 |
|------|------|------|
| `ViTPatchEmbeddings` | `vit/modeling_vit.py:42` | 把 `pixel_values` 切成固定 patch 并线性投影为 patch token（`:62` forward） |
| `ViTEmbeddings` | `:72` | patch 嵌入 + 位置嵌入（含 `[CLS]` 式可学习 token） |
| `contrastive_loss` / `image_text_contrastive_loss` | `clip/modeling_clip.py:47`/`:51` | 图文相似度矩阵的对称交叉熵损失 |
| `CLIPVisionEmbeddings` / `CLIPTextEmbeddings` | `:138`/`:221` | 双塔各自的输入嵌入 |
| `DetrConvEncoder` | `detr/modeling_detr.py:242` | 经 `load_backbone`（`:255`）加载 timm/AutoBackbone CNN，输出多尺度特征图；`replace_batch_norm`（`:260`）冻结 BN |
| DETR decoder | `detr/` | object queries 经 encoder-decoder 输出检测框 + 类，训练用匈牙利匹配（检测头/损失在 `modeling_detr.py`） |

## 3. 关键调用链

**ViTForImageClassification 前向**
1. `ViTPatchEmbeddings.forward`（`vit/modeling_vit.py:62`）：`pixel_values` → patch token 序列。
2. `ViTEmbeddings`（`:136`）加位置嵌入 → `ViTEncoder` 双向自注意力堆叠。
3. 取 `[CLS]` 特征过分类头出 logits。

**CLIP 双塔**
1. `CLIPVisionModel` 出图像嵌入，`CLIPTextModel` 出文本嵌入。
2. 各自 L2 归一化后算相似度矩阵；`image_text_contrastive_loss`（`clip/modeling_clip.py:51`）对称地对比学习，`logit_scale` 可学习温度。

**DETR 检测**
1. `DetrConvEncoder`（`detr/modeling_detr.py:242`）CNN backbone 提特征图。
2. 特征图展平为 token 进 transformer encoder-decoder；decoder 用一组可学习 object queries 并行输出 N 个检测预测。
3. 训练时预测与真值框用匈牙利匹配做二分图指派（损失与匹配逻辑在本家族文件内）。

## 4. 配置项

| 配置项 | 默认 / 行为 | 位置 |
|--------|-------------|------|
| `model_type` | `"vit"`/`"clip"`/`"swin"`/`"detr"` | 各 `configuration_*.py` |
| `image_size` / `patch_size` | ViT 输入图与 patch 尺寸 | `configuration_vit.py` |
| `num_channels` | 输入通道（RGB=3） | 同上 |
| `backbone` / `use_timm_backbone` | DETR 的 CNN backbone 选择 | `configuration_detr.py` |
| `attn_implementation` | eager/sdpa/flash | auto 层透传 |

## 5. 错误与重试语义

- 前向为张量计算；图像分辨率与 `image_size`/patch 数不符时报 shape 错。
- DETR backbone 加载失败（timm 不可用）由 `requires_backends`/`load_backbone` 报错。无业务重试。

## 6. 并发细节

- 同步张量计算；视觉模型非自回归，无 KV cache。
- CNN backbone 的卷积在 PyTorch 内并行（外部）；梯度检查点可选。

## 7. 系统边界

**In-Scope（本仓库源码内）**
- `models/vit/`、`models/clip/`、`models/swin/`、`models/detr/`。

**Out-of-Scope（不在本仓库源码内）**
- timm、torchvision、PIL、CUDA 卷积核：外部依赖。
- 不在本叶子：多模态视觉-语言融合（multimodal-models）、语音视觉（speech）。

## 8. 与相邻子系统交互

- 上游：`AutoModelForImageClassification` 等（auto-registry）分发；图像经 `AutoImageProcessor`（image_processing_auto）预处理。
- 下游：视觉编码器常被多模态模型（multimodal-models，如 LLaVA/Qwen2-VL）复用为视觉塔。
- DETR backbone 经 `backbone_utils.load_backbone` 复用 timm。

## 9. 语言专项适配口径（Python）

- **分组**：按 capability seam——"图像编码（patch/CNN）"与"任务头（分类/检测/对比）"。
- **图类型**：architecture（多家族组件并列）为主。不产出 lifecycle。
- **外部边界**：timm/torchvision/PIL 标注外部。

## 10. 图表清单与质量

| 图 | 文件 | 类型 | archify 质量档 |
|----|------|------|----------------|
| 视觉家族架构 | `vision-models-architecture.html` | architecture | **showcase**（0 错误通过） |

JSON IR 源：`json/vision-models-architecture.json`。
不另出 sequence：前向为确定性编码堆叠，文字调用链已表达。

**覆盖范围披露**：深读 ViT/CLIP/Swin/DETR 四个家族；其余（ConvNeXt、ResNet、MobileNet、SegFormer、DPT、Yolos、DeiT 等）仅按共性列表说明，未逐类读源码。
