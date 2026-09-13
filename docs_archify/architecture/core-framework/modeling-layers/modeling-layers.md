# modeling-layers（层构建块、注意力掩码、RoPE 与激活函数）

> 本文是 `core-framework` 域下的叶子子系统文档。域级总览见 `../core-framework.md`。
> 本文只展开模型层构建的通用组件；模型基类加载链见 [`../modeling-utils/`](../modeling-utils/modeling-utils.md)。
>
> 源码基准：HuggingFace transformers `main`，commit `5474a55e920f358d8382f3ecd3377edca979baa`。

## 1. 功能清单

| 能力 | 说明 | 源码路径 |
|------|------|----------|
| `GradientCheckpointingLayer` | 梯度检查点层基类 | `src/transformers/modeling_layers.py:52` |
| `GenericForSequenceClassification/TokenClassification/QuestionAnswering` | 通用分类头 | `modeling_layers.py:113`、`:249`、`:188` |
| `MtpModel`/`MtpLayer` | Multi-Token Prediction 层 | `modeling_layers.py:359`、`:311` |
| `AttentionMaskConverter` | 2D/4D 注意力掩码转换 | `src/transformers/modeling_attn_mask_utils.py:36` |
| `_prepare_4d_causal_attention_mask` | 生成 4D 因果掩码（eager） | `:324` |
| `_prepare_4d_causal_attention_mask_for_sdpa` | SDPA 用 4D 掩码 | `:377` |
| `_prepare_4d_attention_mask` / `_for_sdpa` | 非因果 4D 掩码 | `:435`、`:451` |
| FlashAttention 可用性检测 | `is_flash_attn_available`/`flash_attn_supports_top_left_mask` | `modeling_flash_attention_utils.py:53`、`:43` |
| FlashAttention 惰性导入 | `lazy_import_flash_attention`/`lazy_import_paged_flash_attention` | `:248`、`:275` |
| FA 输入 pad/unpad | `_unpad_input`/`_pad_input`/`_get_unpad_data`/`_upad_input` | `:299`、`:332`、`:351`、`:379` |
| `_flash_attention_forward` | FlashAttention 前向入口 | `:694` |
| RoPE 缩放计算 | `_compute_linear_scaling`/`_compute_dynamic_ntk`/`_compute_yarn`/`_compute_longrope`/`_compute_llama3` | `modeling_rope_utils.py:133`、`:269`、`:345`、`:486`、`:580` |
| `RotaryEmbeddingConfigMixin` | 配置 mixin，统一 RoPE 参数 | `:734` |
| `dynamic_rope_update` 装饰器 | 动态 RoPE 序列长度更新 | `:34` |
| 激活函数族 | `GELUTanh`/`NewGELU`/`GELUActivation`/`SiLU`/`Mish`/`FastGELU`/`QuickGELU`/`ReLU^2`/`XIELU` 等 | `src/transformers/activations.py:31`–`:353` |
| `get_activation` 工厂 | 按字符串名取激活函数类 | `activations.py:353` |
| 初始化方法 | `uniform_`/`normal_`/`xavier_uniform_`/`xavier_normal_`/`kaiming_*`/`trunc_normal_`/`orthogonal_` | `src/transformers/initialization.py:42`–`:158` |
| 函数式掩码组合 | `and_masks`/`or_masks`/`causal_mask_function`/`sliding_window_*`/`chunked_*`/`padding_mask_function` | `src/transformers/masking_utils.py:49`–`:206` |

## 2. 核心类型与接口清单

| 类型 / 函数 | 位置 | 职责 |
|-------------|------|------|
| `AttentionMaskConverter` | `modeling_attn_mask_utils.py:36` | 把 2D padding mask 转 4D 掩码；处理因果掩码、sliding window |
| `FlashAttentionKwargs` | `modeling_flash_attention_utils.py:570` | TypedDict，FlashAttention 所需 kwargs 契约 |
| `RopeParameters` | `modeling_rope_utils.py:678` | TypedDict，RoPE 缩放参数字典 |
| `RotaryEmbeddingConfigMixin` | `:734` | 配置 mixin，提供 `rope_scaling` 属性与参数归一化 |
| `ClassInstantier` | `activations.py:224` | OrderedDict 子类，支持 `ACT2CLS[act_string]()` 实例化 |
| `ACT2CLS`（OrderedDict） | `activations.py` 底部 | 激活名 → 类 注册表，`get_activation` 查表 |
| `GradientCheckpointingLayer` | `modeling_layers.py:52` | 带梯度检查点开关的层基类 |

## 3. 关键调用链

### 3.1 注意力掩码生成（eager vs SDPA）

1. 模型 `forward(attention_mask=...)` 收到 2D padding mask（shape `[batch, seq]`）。
2. eager 路径：`_prepare_4d_causal_attention_mask`（`modeling_attn_mask_utils.py:324`）→ `AttentionMaskConverter._prepare_4d_causal_attention_mask` 把 2D mask 扩展为 4D `[batch, 1, tgt, src]` 并叠加因果掩码。
3. SDPA 路径：`_prepare_4d_causal_attention_mask_for_sdpa`（`:377`）——SDPA 内核内部已支持因果，只需 2D mask 或全 False，避免 4D 掩码编译开销。
4. FlashAttention 路径：内核直接支持 `causal=True` 参数，不需显式掩码；`_unpad_input`（`modeling_flash_attention_utils.py:299`）把 padding token 压缩成变长序列。

### 3.2 RoPE 缩放选择

1. 模型 `__init__` 读 `config.rope_scaling`（dict，含 `type` 与 `factor`）。
2. `RotaryEmbeddingConfigMixin`（`modeling_rope_utils.py:734`）按 `type` 分派：`linear`→`_compute_linear_scaling_rope_parameters`（`:133`）；`dynamic`→`_compute_dynamic_ntk_parameters`（`:269`）；`yarn`→`_compute_yarn_parameters`（`:345`）；`longrope`→`:486`；`llama3`→`:580`。
3. `dynamic_rope_update`（`:34`）装饰器包装 forward，在序列长度超过训练长度时动态更新 inv_freq。

### 3.3 FlashAttention 后端选择

1. `PreTrainedModel._flash_attn_can_dispatch`（modeling-utils 叶子）检查版本与 head 数。
2. `lazy_import_flash_attention`（`modeling_flash_attention_utils.py:248`）惰性 import `flash_attn` 库（外部）。
3. `_upad_input`（`:379`）处理 padding，`_flash_attention_forward`（`:694`）调内核。

## 4. 配置项

| flag / option | 默认 / 行为 | 位置 |
|---------------|-------------|------|
| `attn_implementation` | `sdpa`（可用时），否则 `eager`；可选 `flash_attention_2/3/4` | modeling-utils 叶子 |
| `rope_scaling.type` | `None`；`linear`/`dynamic`/`yarn`/`longrope`/`llama3`/`ntk-by-parts` | `modeling_rope_utils.py` |
| `rope_scaling.factor` | 缩放因子 | 同上 |
| `activation`（config） | `"gelu"` 等，`get_activation` 查表 | `activations.py:353` |
| `gradient_checkpointing` | `False`；`model.gradient_checkpointing_enable()` 开启 | modeling-utils 叶子 |
| `torch_dtype` | 见 configuration-utils 叶子 | 同 |

## 5. 错误与重试语义

- FlashAttention 不可用时：`_flash_attn_can_dispatch` 返回 False，`_check_and_adjust_attn_implementation`（modeling-utils 叶子）降级到 `sdpa`/`eager` 并 warning，不抛错。
- RoPE 缩放类型未知：`rope_config_validation`（`modeling_rope_utils.py:1089`）抛 `ValueError`，列出支持的 type。
- 激活字符串未知：`get_activation`（`activations.py:353`）抛 `KeyError`，列出 `ACT2CLS` 可用键。
- 掩码形状不匹配：`AttentionMaskConverter._raise_on_mask_dimension_mismatch` 抛 `ValueError`。
- 本叶子不做网络重试。

## 6. 并发细节

- 本叶子为同步张量计算，无 goroutine/线程。
- RoPE inv_freq 计算在 `__init__` 一次完成；`dynamic_rope_update` 在每次 forward 检查序列长度（同步）。
- FlashAttention 内核自身在 CUDA 上并行（外部库），本叶子只做数据布局转换（pad/unpad）。
- 无 `context.Context`；PyTorch CUDA stream 管理设备同步（外部）。

## 7. 系统边界

**In-Scope（本仓库源码内）**
- 层构建块（`modeling_layers.py`）：分类头、MTP、梯度检查点层。
- 注意力掩码转换（`modeling_attn_mask_utils.py`）。
- FlashAttention 适配层（`modeling_flash_attention_utils.py`）：pad/unpad、kwargs 契约、惰性导入。
- RoPE 缩放计算（`modeling_rope_utils.py`）。
- 激活函数注册表（`activations.py`）与初始化方法（`initialization.py`）。
- 函数式掩码组合器（`masking_utils.py`）。

**Out-of-Scope（不在本仓库源码内）**
- FlashAttention / FlashAttention-3 / FlashAttention-4 CUDA 内核（Dao-AILab 外部库）。
- PyTorch SDPA 内核（`F.scaled_dot_product_attention`）。
- PyTorch 初始化器后端（`torch.nn.init`）。
- 各模型自身的 attention 模块（`models/<x>/modeling_<x>.py`）——本叶子只提供通用工具。
- 模型加载：见 [`../modeling-utils/`](../modeling-utils/modeling-utils.md)。

## 8. 与相邻子系统交互

- **上游**：modeling-utils 叶子的 `PreTrainedModel.__init__` / `forward` 调用本叶子。
- **本叶子 → modeling-utils**：读 `config.attn_implementation`、`config.rope_scaling`、`config.hidden_act`。
- **本叶子 → 外部**：`flash_attn` 库、`torch.nn.functional.scaled_dot_product_attention`、`torch.nn.init`。
- **下游**：具体模型 `models/<x>/modeling_<x>.py` 组合本叶子组件构建 Transformer block。

## 9. 语言专项适配口径

Python 项目，采用 TS 口径的 Python 适配：
1. **分组**：按 capability seam 分组——本叶子 = "层构建与注意力/RoPE 工具" seam。
2. **图类型**：architecture（组件边界）为主；RoPE 缩放分派是配置驱动的静态分派，不产 lifecycle。
3. **外部边界**：FlashAttention/SDPA 内核标注外部。
4. **部署维度**：不适用单二进制；用模块文件表达。

## 10. 图表清单与质量

| 图 | 文件 | 类型 | archify 质量档 |
|----|------|------|----------------|
| 层构建块组件架构图 | `modeling-layers-architecture.html` | architecture | showcase |

JSON IR 源文件位于 `json/` 目录。本叶子的调用链为静态分派（按 config 字段选择实现），不补充 sequence/dataflow 图。
