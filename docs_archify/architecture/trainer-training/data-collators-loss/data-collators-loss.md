# data-collators-loss（数据整理器与损失函数）

> 本文是 `trainer-training` 域下的叶子子系统文档。域级总览见 `../trainer-training.md`。
> 本文只展开 **DataCollator 系列与专用损失函数**，不重复展开训练循环（见 `../trainer-core/`）。
>
> 源码基准：`transformers` main 分支，commit `5474a55e920f358d8382f3ecd3377edca979baa`。

## 1. 功能清单

| 能力 | 说明 | 源码路径 |
|------|------|----------|
| `DataCollatorMixin` | 数据整理器基类；定义 `__call__` 接口 | `src/transformers/data/data_collator.py:37` |
| `DefaultDataCollator` | 默认整理器：堆叠 batch 张量 | `data_collator.py:95` |
| `DataCollatorWithPadding` | 动态 padding：按 batch 内最长序列 pad | `data_collator.py:191` |
| `DataCollatorForLanguageModeling` | MLM 数据整理：随机掩码 + 生成 labels | `data_collator.py:621` |
| `DataCollatorForWholeWordMask` | WWM：整词掩码（BERT-Chinese 风格） | `data_collator.py:1021` |
| `DataCollatorForSeq2Seq` | Seq2Seq 整理：decoder_input_ids 生成 + label padding=-100 | `data_collator.py:489` |
| `DataCollatorForTokenClassification` | Token 分类：label 对齐与 padding | `data_collator.py:243` |
| `DataCollatorForMultipleChoice` | 多选任务整理 | `data_collator.py:422` |
| `DataCollatorForPermutationLanguageModeling` | PLM（XLNet 风格）排列语言模型整理 | `data_collator.py:1141` |
| `ImageLoss` | 目标检测基类损失 | `src/transformers/loss/loss_for_object_detection.py:85` |
| `HungarianMatcher` | 匈牙利匹配器（DETR 系列标签分配） | `loss_for_object_detection.py:285` |
| `RTDetrLoss` / `DeformableDetrImageLoss` / `GroundingDinoImageLoss` 等 | 各检测模型专用损失 | `loss_rt_detr.py:123` / `loss_deformable_detr.py:59` / `loss_grounding_dino.py:128` |
| `RNNTLoss` / `TDTLoss` | 语音识别损失（Transducer/TDT） | `loss_rnnt.py` / `loss_tdt.py` |

## 2. 核心类型与接口清单

| 类型 / 函数 | 位置 | 职责 |
|-------------|------|------|
| `DataCollatorMixin` | `data_collator.py:37` | 基类；`__call__(features)` → batch dict |
| `DataCollatorForLanguageModeling` | `data_collator.py:621` | MLM 核心：`mlm_probability` 掩码概率；`torch.rand` 选择掩码 token |
| `DataCollatorForSeq2Seq` | `data_collator.py:489` | 生成 `decoder_input_ids`（shift labels right） |
| `ImageLoss` | `loss_for_object_detection.py:85` | 检测损失基类：分类 + 框回归 + GIoU |
| `HungarianMatcher` | `loss_for_object_detection.py:285` | 二分图匹配：预测框与 GT 框最优分配 |

## 3. 关键调用链

### 3.1 MLM 数据整理（`DataCollatorForLanguageModeling.__call__`）

1. **padding**：将 batch 内所有序列 pad 到最长长度。
2. **掩码**：`torch.rand(input_ids.shape) < mlm_probability` 选择掩码候选。
3. **排除特殊 token**：不掩码 `special_tokens_mask` 中的 token。
4. **80/10/10 策略**：80% → `[MASK]`，10% → 随机 token，10% → 保持原 token。
5. **labels**：非掩码位置 label = -100（忽略），掩码位置 label = 原 token。

### 3.2 Seq2Seq 数据整理（`DataCollatorForSeq2Seq.__call__`）

1. **encoder padding**：对 `input_ids` padding + `attention_mask`。
2. **decoder_input_ids**：`labels` 右移一位（shift right），首 token 为 decoder_start_token_id。
3. **label padding**：`labels` 中 padding 位置设为 -100（CrossEntropyLoss ignore_index）。

### 3.3 检测损失计算

1. **匈牙利匹配**：`HungarianMatcher` 将预测框与 GT 框做最优二分匹配。
2. **损失计算**：分类损失（交叉熵）+ 框回归损失（L1 + GIoU）。
3. **归一化**：按正样本数归一化。

## 4. 配置项

| 配置项 | 默认 / 行为 | 位置 |
|--------|-------------|------|
| `mlm` | `True`；是否使用 MLM（False → CLM） | `data_collator.py:621` |
| `mlm_probability` | `0.15`；掩码概率 | `data_collator.py:621` |
| `pad_to_multiple_of` | `None`；pad 到倍数（tensor core 友好） | `data_collator.py:191` |
| `label_pad_token_id` | `-100`；Seq2Seq label padding 值 | `data_collator.py:489` |
| `tokenizer` | 必填；用于 pad/mask 特殊 token | 各 DataCollator |

## 5. 错误与重试语义

- **tokenizer 缺失**：MLM/Seq2Seq 整理器需要 tokenizer，未提供时抛错误。
- **input_ids 缺失**：`DefaultDataCollator` 要求 features 为 dict。
- **无重试**：数据整理是确定性转换，失败直接终止。

## 6. 并发细节

- **无并发**：数据整理是单 batch 同步操作。
- **确定性**：MLM 掩码用 `torch.rand`，受全局 seed 控制。

## 7. 系统边界

**In-Scope（本仓库源码内）**
- `src/transformers/data/data_collator.py`：DataCollator 系列
- `src/transformers/loss/`：检测/语音专用损失

**Out-of-Scope（不在本仓库源码内）**
- 模型前向计算本身
- PyTorch CrossEntropyLoss / L1Loss 等基础损失
- torchvision 等外部检测库

## 8. 与相邻子系统交互

- **上游 → 本叶子**：`Trainer.get_train_dataloader()` 使用 `data_collator` 参数。
- **本叶子 → 下游**：
  - → `training_step()`：整理后的 batch dict 传入模型 forward
  - → 模型损失计算：`labels` 字段用于 CrossEntropyLoss

## 9. 语言专项适配口径

- Python 库，按"数据整理 / 专用损失"capability seam 划分。
- 图类型：architecture（整理器族关系）；sequence/lifecycle 不适用（数据整理是无状态函数调用）。

## 10. 图表清单与质量

| 图 | 文件 | 类型 | archify 质量档 |
|----|------|------|----------------|
| DataCollator 与损失函数架构图 | `data-collators-loss-architecture.html` | architecture | standard（降档披露） |

JSON IR 源文件位于 `json/` 目录。sequence/lifecycle 不产出：数据整理是无状态 `__call__` 函数，损失计算是单次前向结果，无状态机或时序语义。
