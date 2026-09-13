# distributed-deployment 域总览

> 本域包含以下叶子子系统；各叶子详情见对应文档。
> 源码基准：`transformers` v5.18.0.dev0（main），commit `5474a55e920f358d8382f3ecd3377edca979baa`。

## 1. 域职责

`distributed-deployment` 域覆盖**模型训练后的部署与规模化运行**维度：多 GPU 分布式并行（TP/PP/FSDP）、外部框架集成（DeepSpeed/PEFT/Wandb 等）、模型导出（ONNX/ExecuTorch）、量化（25+ 后端）、命令行工具（CLI + 推理服务）。本域是 transformers 从"单机推理"走向"生产部署"的关键层。

## 2. 叶子索引

| 叶子 | 文档 | 架构图 | 时序图/数据流图 | 职责一句话 |
|------|------|--------|-----------------|-----------|
| distributed-core | [distributed-core.md](distributed-core/distributed-core.md) | [架构图](distributed-core/distributed-core-architecture.html) | [TP 数据流](distributed-core/distributed-core-tp-dataflow.html) | 内置 TP/PP/FSDP 分布式工具 |
| integrations-frameworks | [integrations-frameworks.md](integrations-frameworks/integrations-frameworks.md) | [架构图](integrations-frameworks/integrations-frameworks-architecture.html) | — | 50+ 外部框架集成（DeepSpeed/PEFT/追踪等） |
| exporters | [exporters.md](exporters/exporters.md) | [架构图](exporters/exporters-architecture.html) | — | ONNX/TorchDynamo/ExecuTorch 导出 |
| quantization | [quantization.md](quantization/quantization.md) | [架构图](quantization/quantization-architecture.html) | [量化数据流](quantization/quantization-dataflow.html) | 25+ 量化后端统一抽象（bnb/GPTQ/AWQ 等） |
| cli-tools | [cli-tools.md](cli-tools/cli-tools.md) | [架构图](cli-tools/cli-tools-architecture.html) | [serve 数据流](cli-tools/cli-tools-serve-dataflow.html) | transformers-cli 五子命令 + FastAPI 推理服务 |

## 3. 域级机制细节

### 多后端集成统一抽象
- **量化**：`HfQuantizer(ABC)` 基类定义 `create_model`/权重加载前后钩子/序列化接口；`AutoHfQuantizer` 按 `quantization_config` 自动选后端。
- **集成检测**：`integration_utils.py` 统一 `is_*_available()` 延迟导入检测，可选依赖缺失时优雅降级。
- **分布式**：`distributed/` 包提供原生 TP/PP/FSDP 工具，区别于外部 accelerate/deepspeed。

### 部署维度
本域覆盖模型部署的全维度：**精度**（量化）、**格式**（导出）、**规模**（分布式）、**服务**（CLI serve）、**生态**（集成）。
