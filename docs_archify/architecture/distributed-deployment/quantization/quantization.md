# 量化框架（quantization）

> 本文是 `distributed-deployment` 域下的叶子子系统文档。域级总览见 `../distributed-deployment.md`。
>
> 源码基准：`transformers` v5.18.0.dev0，commit `5474a55e920f358d8382f3ecd3377edca979baa`。

## 1. 功能清单

| 能力 | 说明 | 源码路径 |
|------|------|----------|
| 量化抽象基类 | `HfQuantizer(ABC)`：`create_model`/`_process_model_before_weight_loading`/`_process_model_after_weight_loading`/`serialize` | `src/transformers/quantizers/base.py:73` |
| 量化配置基类 | `QuantizationConfig`（utils/quantization_config.py） | `quantizers/` + `utils/quantization_config.py` |
| Auto 选择器 | `AutoHfQuantizer`/`AutoQuantizationConfig` 按 config.quantization_config 选后端 | `quantizers/auto.py:154/195` |
| bitsandbytes 4bit | `quantizer_bnb_4bit.py` | `quantizers/` |
| bitsandbytes 8bit | `quantizer_bnb_8bit.py` | `quantizers/` |
| GPTQ | `quantizer_gptq.py` | `quantizers/` |
| AWQ | `quantizer_awq.py` | `quantizers/` |
| torchao | `quantizer_torchao.py` | `quantizers/` |
| 其他后端 | EETQ/HQQ/Quanto/FP8/SpQR/AQLM/GGUF/MXFP4/NVFP4 等 25+ 个 | `quantizers/` |
| QLoRA | `qlora_pipeline` 集成 | `quantizers/` |
| 注册表 | `register_quantization_config(method)`/`register_quantizer(name)` | `auto.py:306/322` |

**深读覆盖**：深读 `base.py`（HfQuantizer 抽象）、`auto.py`（Auto 选择器）、`quantizer_bnb_4bit.py` 代表后端；其余 25+ 后端文件按族列表说明。

## 2. 核心类型

| 类型 | 位置 | 职责 |
|------|------|------|
| `HfQuantizer(ABC)` | `base.py:73` | 量化器抽象：量化模型创建/权重加载前后钩子/序列化 |
| `AutoHfQuantizer` | `auto.py:195` | 按 quantization_config 自动选后端 |
| `AutoQuantizationConfig` | `auto.py:154` | 自动选量化配置类 |
| `get_hf_quantizer()` | `auto.py:338` | 工厂函数 |
| `get_keys_to_not_convert()` | `base.py:38` | 量化时跳过的层（lm_head/layernorm） |

## 3. 关键调用链

1. `from_pretrained(model_id, load_in_4bit=True)` 传入量化参数；
2. `AutoQuantizationConfig.from_dict(kwargs)` 构造量化配置；
3. `get_hf_quantizer(config, quantization_config, ...)` 选后端；
4. `quantizer.create_model(model)` 量化模型包装；
5. `_process_model_before_weight_loading`（准备量化层）→ 加载权重 → `_process_model_after_weight_loading`（后处理）；
6. `serialize()` 保存量化权重。

## 4-9. 概要

- **配置**：`load_in_4bit`/`load_in_8bit`、`bnb_4bit_quant_type`/`bnb_4bit_compute_dtype`、`gptq_bits`/`awq_bits` 等。
- **错误**：不支持的设备/量化组合抛 `ValueError`；后端未安装时 ImportError。
- **In-Scope**：`quantizers/` 包。
- **Out-of-Scope**：bitsandbytes/GPTQ/AWQ/torchao 量化库本体均"不在本仓库源码内"。
- **Python 适配口径**：按量化后端分组；图以 architecture + dataflow（量化数据流）为主。

## 10. 图表清单与质量

| 图 | 文件 | 类型 | 质量档 |
|----|------|------|--------|
| 量化框架架构图 | `quantization-architecture.html` | architecture | standard |
| 量化加载数据流 | `quantization-dataflow.html` | dataflow | standard |

**覆盖范围披露**：深读 base.py + auto.py + bnb_4bit 代表后端；其余 25+ 后端文件列表说明。
