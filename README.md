# 裂纹快速语义分割

更新：2026-11-17。基于 TACK Tunnel B，验证“固定网格候选选择 + 原分辨率局部分割”的两级路线。数据、训练、整图重建和主机级联对照已完成；独立泛化、误报控制和目标设备部署尚未验收。

当前保留 `MobileNetV3SmallGridRanker` 一级对照和 `MobileNetV3LargeRefineUNet` 二级候选，质量基线为 **eager FP32（允许 TF32）**。evidence 新结构完成 30 轮训练和级联后因覆盖/召回下降停止。FP32 TorchScript 的 CUDA 多输入一致性仍有差异，FP16 导出也未验收，均不作为默认部署产物。

## 文档入口

| 文档 | 内容 |
| --- | --- |
| [项目状态](docs/project_status.md) | 当前结果、证据、阶段完成情况与剩余缺口 |
| [技术路线](docs/segmentation_roadmap.md) | 数据契约、两级模型、训练和评价细节、后续技术门槛 |
| [实验记录](docs/experiment_log.md) | 历史尝试、失败结果、审计修正和停止原因 |
| [迭代计划](docs/iteration_plan.md) | 已完成任务、下一阶段顺序及完成标准 |
| [飞书研究讨论](https://zcnl6gnomk0m.feishu.cn/wiki/QvQ0wcebmiVlqakop4AcA1vEniR) | 原始构思及阶段进展 |
| [飞书技术方案](https://zcnl6gnomk0m.feishu.cn/docx/HGKSdYgM0obyLaxh4zWcnkK6nQc) | 技术方案、实验与计划 |

## 数据与推理协议

1536 张 512×512 图片组成 128 个连续画面，每组 12 块：

```text
A B C D
E F G H
I L M N
```

二级使用 **512×512 训练**；系统接收 **2048×1536** 画面，按原始像素切为 12 块，一级评分选 top-6，二级分割选中块，按固定坐标重建完整掩膜。全覆盖对照使用同一二级模型处理全部 12 块。整图无候选跳过继续关闭。

按源图组固定划分为训练/验证/测试开发 90/25/13 组；完整画面阳性/阴性分别为 82/8、20/5、13/0。标签值 40 为裂纹，其余 0/160/200 为背景。测试集已经反复参与开发，不能作为独立盲测。原始数据与原始标注保持不变。

## 当前结果

同一 refine 二级 eager FP32、batch=6、预热3帧、重复3遍，RTX 5070 Ti Laptop GPU 上的历史统一协议结果：

| 测试开发集路径 | Dice | Recall | 平均/P95 |
| --- | ---: | ---: | ---: |
| 全覆盖12块 | 41.77% | 49.36% | 133.00/163.16 ms |
| grid top-6 | 44.86% | 48.27% | 102.20/109.65 ms |

块数减半，历史平均延迟降低约23.2%，同时 Recall 下降1.09个百分点；等召回优势、端侧收益与节能尚未证明。验证阴性整图5/5出现至少一个预测像素，误报仍是主要缺口。详细结果及证据路径见项目状态。

## 环境与运行

已有 `.venv`：Python 3.12、PyTorch 2.11.0+cu128。重建环境：

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -r requirements.lock.txt --extra-index-url https://download.pytorch.org/whl/cu128
```

数据准备和拼接（已有产物不同时拒绝覆盖）：

```bash
.venv/bin/python scripts/prepare_tack.py
.venv/bin/python scripts/stitch_tack.py
```

下面是复现实验的命令，不表示当前有训练正在运行。输出目录/文件必须不存在，使用新的运行名。

```bash
# 二级 refine
.venv/bin/python scripts/train.py --model mobilenetv3_large_refine \
  --epochs 100 --batch-size 4 --output outputs/runs/refine_reproduction

# 一级 grid；保留的历史权重是 dynamic_roi_grid_v1/best.pt
.venv/bin/python scripts/train_dynamic_roi.py --arch grid \
  --epochs 30 --output outputs/runs/grid_reproduction

# 全覆盖整图基线
.venv/bin/python scripts/benchmark_roi_cascade.py --mode full \
  --segmenter outputs/runs/tack_mobilenetv3_large_refine_v1/best.pt \
  --segmenter-arch mobilenetv3_large_refine --segmenter-format eager \
  --segmenter-precision fp32 --split val --batch-size 6 --warmup 3 --repeats 3 \
  --output outputs/reports/refine_full_val_reproduction.json

# 一级 top-6 + 二级 refine
.venv/bin/python scripts/benchmark_roi_cascade.py --mode ranked \
  --ranker outputs/runs/dynamic_roi_grid_v1/best.pt --ranker-arch grid \
  --segmenter outputs/runs/tack_mobilenetv3_large_refine_v1/best.pt \
  --segmenter-arch mobilenetv3_large_refine --segmenter-format eager \
  --segmenter-precision fp32 --split val --topk 6 --batch-size 6 --warmup 3 --repeats 3 \
  --output outputs/reports/refine_ranked_val_reproduction.json

# 现有针对性检查
.venv/bin/python -m unittest discover -s tests -v
```

`run_dynamic_roi_cascade.py` 保留为历史入口，其计时包含标注/指标且未预热；效率比较统一使用 `benchmark_roi_cascade.py`。不要误用早期 `dynamic_roi_mobilenetv3_small_v1` 权重加载 grid 架构。

## 文件与辅助工具

`src/` 为模型、数据和几何处理；`scripts/` 为执行入口；`data/TACK_Tunnel_Data/` 为原始数据；`data/processed/TACK_Tunnel_B/` 为划分与整图清单；`outputs/runs/` 保存训练产物；`outputs/reports/` 保存评价与分析；`docs/` 保存长期文档；`tmp/` 保存临时审计和备份。旧 Tunnel200 未用于当前训练，作者 CSV 未用于本次划分。

- 数据校验：`scripts/validate_tack.py`，正式结果见 [数据审计](outputs/reports/tack_dataset_validation/findings.md)。
- 困难集：`outputs/reports/difficulty_set_v3/`，含60个阳性块、27组及6张 contact sheet；自动类别只是人工复核索引。
- 曲线：`scripts/plot_training.py --run <运行目录>`，导出原始 CSV、PDF/SVG/PNG，不平滑或补造数据。
- 看板：`scripts/dashboard.py --run <运行目录> --port 8765`，仅本机只读；使用 `.venv/bin/python` 启动，不表示服务当前已运行。
- 历史 U-Net/ResNet34 结构图由 `scripts/draw_architectures.py` 生成，产物在 `outputs/reports/model_architecture/`。

### 模型架构图

PlotNeuralNet 是架构图后端。先根据模型代码和推理入口整理结构说明，包括输入输出尺寸、各层通道、跳连和推理分支，再写独立的 Python 绘图脚本。脚本生成 TikZ，由 Tectonic 编译成 PDF，再渲染 PNG 供版式检查。发现交叠、裁切或连线错误时，只改绘图脚本并重新预览。检查通过后再导出正式产物。

级联审计 v2 对应 `MobileNetV3SmallGridRanker` 和基础 `MobileNetV3LargeUNet`。帧输入为 2048×1536，切成 3×4 个 512×512 图块。Ranked 路径选 Top‑6，Full 路径使用全部 12 块。两路共用二级分割器，阈值化后按原坐标回填。结构以 `src/dynamic_roi.py`、`src/mobilenet_unet.py` 和 `scripts/benchmark_roi_cascade.py` 为准。

```bash
# 首次使用时准备 PlotNeuralNet 和 Tectonic
.venv/bin/python scripts/setup_architecture_tools.py

# 预览并检查 tmp/pdfs/model_architecture/ 下的 PNG 或 PDF
.venv/bin/python scripts/draw_cascade_architecture.py --preview

# 布局检查通过后，导出 PDF、SVG、PNG 和生成的 TeX
.venv/bin/python scripts/draw_cascade_architecture.py
```

预览和导出需要 Poppler 的 `pdftoppm`，SVG 导出使用 `pdftocairo`。正式产物位于 `outputs/reports/model_architecture/`，绘图脚本是版式的唯一编辑入口。
