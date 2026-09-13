# 模型导出（exporters）

> 本文是 `distributed-deployment` 域下的叶子子系统文档。域级总览见 `../distributed-deployment.md`。
>
> 源码基准：`transformers` v5.18.0.dev0，commit `5474a55e920f358d8382f3ecd3377edca979baa`。

## 1. 功能清单

| 能力 | 说明 | 源码路径 |
|------|------|----------|
| 导出统一接口 | `AutoHfExporter`/`AutoExportConfig` 自动选择导出后端 | `src/transformers/exporters/auto.py` |
| ONNX 导出 | `OnnxExporter(DynamoExporter)` 基于 torch.export/Dynamo | `exporters/exporter_onnx.py:87` |
| TorchDynamo 导出 | `DynamoExporter` 基类，`torch.export.export` 封装 | `exporters/exporter_dynamo.py` |
| ExecuTorch 导出 | `ExecutorchExporter` 移动端/边缘端导出 | `exporters/exporter_executorch.py` |
| 导出配置 | `ExportConfig` 数据类 | `exporters/configs.py` |
| 算子补丁 | ONNX 导出时 patch 不兼容算子（rms_norm/where/cummax 等） | `exporter_onnx.py:143-463` |
| IO 名消歧 | `disambiguate_io_names` 处理重名输入输出 | `exporter_onnx.py:167` |

## 2. 核心类型与接口

| 类型 | 位置 | 职责 |
|------|------|------|
| `DynamoExporter`（基类） | `exporter_dynamo.py` | torch.export 封装基类 |
| `OnnxExporter(DynamoExporter)` | `exporter_onnx.py:87` | ONNX 导出后端 |
| `ExecutorchExporter` | `exporter_executorch.py` | ExecuTorch 导出后端 |
| `AutoHfExporter` / `AutoExportConfig` | `auto.py` | 按任务/配置自动选择导出器 |

## 3. 关键调用链

1. 用户调 `export(model, target="onnx"|"executorch", task=...)`；
2. `AutoHfExporter` 按 target 选择 `OnnxExporter`/`ExecutorchExporter`；
3. `DynamoExporter` 调 `torch.export.export(model, args, dynamic_shapes=...)` 导出；
4. `OnnxExporter` 额外应用算子补丁（patch rms_norm/where/cummax 等 ONNX 不兼容算子）；
5. 导出为 ONNX 文件 + 验证输出。

## 4-9. 概要

- **配置**：`task`（导出任务名）、`opset`（ONNX opset 版本）、`dynamic_shapes`（动态轴）。
- **错误**：不支持的模型架构抛 `ValueError`；算子 patch 失败 warning。
- **In-Scope**：`exporters/` 包。
- **Out-of-Scope**：PyTorch Dynamo/torch.export、ONNX Runtime、ExecuTorch SDK 均"不在本仓库源码内"。
- **Python 适配口径**：按导出后端分组；图以 architecture 为主。

## 10. 图表清单与质量

| 图 | 文件 | 类型 | 质量档 |
|----|------|------|--------|
| 导出后端架构图 | `exporters-architecture.html` | architecture | standard |
