# U-Net 裂纹分割

当前使用 TACK Tunnel B，从随机权重训练基础通道数 32 的 U-Net；不加载预训练模型。环境为 `.venv`，Python 3.12、PyTorch 2.11.0+cu128。

## 工作空间

```text
src/unet.py                         模型结构
src/dataset.py                      数据加载与同步增强
src/reproducibility.py              确定性设置、数据指纹及断点记录
src/dashboard/                     本地看板页面
scripts/prepare_tack.py              检查原始数据并生成分组划分
scripts/train.py                     训练、验证、测试入口
scripts/dashboard.py                 本地只读监控服务
requirements.lock.txt               精确依赖版本
data/TACK_Tunnel_Data/               原始图片、原始多类别标注与作者代码
data/processed/TACK_Tunnel_B/        新划分与样本清单，不复制原始图片
outputs/runs/                       每次运行的配置、权重、指标和预测
outputs/reports/                    数据审计与验证记录
```

旧 Tunnel200 数据保留，但当前训练入口不再使用。TACK 原始图片、标注和作者 CSV 不改动；作者 CSV 不用于本次训练。

## 数据划分

```bash
.venv/bin/python scripts/prepare_tack.py
```

随机种子 42，按最后一个切块尾缀之前的完整前缀分组，例如 `TB_Camera4_001532`。同一源图的 12 个切块必须处于同一集合；采用接近 70/20/10 的源图分组比例。

| 集合 | 源图组数 | 图片 | 含裂纹 | 无裂纹 |
| --- | ---: | ---: | ---: | ---: |
| 训练 | 90 | 1080 | 196 | 884 |
| 验证 | 25 | 300 | 43 | 257 |
| 测试 | 13 | 156 | 24 | 132 |

全部 1536 张图片均纳入划分，不删除无裂纹样本。裂纹标签严格取 `mask == 40`；背景 0、水渍 160、析出物 200 均映射为 0。这排除了同一源图切块跨集合；尚未保证不同相机或相邻采集帧没有场景重叠。生成结果含 `manifest.json`，记录每张图是否含裂纹。重复执行准备命令必须得到相同内容；若已有结果不同则拒绝覆盖。

## 训练策略

- 输入保持 512×512，RGB 转浮点后除以 255；全部模型参数从随机初始化开始更新。
- 训练采用带放回加权采样，正/负图片的期望比例为 1:1；每轮抽取 1080 次，样本可能重复，不保证每轮覆盖全部图片。全部负样本始终保留在候选池中。
- 仅训练增强：50% 水平翻转，加等概率 0/90/180/270 度旋转；图像与标注同步，无缩放、裁剪和插值。
- 损失为全体图片的 BCEWithLogitsLoss，加仅在有裂纹图片上计算的 Dice Loss。全负 batch 只计算 BCE，不除零、不虚构裂纹；不额外叠加类别权重。
- Adam，初始学习率 0.001；ReduceLROnPlateau 监控验证 Dice，耐心值 5 轮、衰减系数 0.5、最低学习率 0.00001。默认最多 100 轮、batch size 4；至少训练 30 轮，连续 15 轮验证 Dice 无超过 0.002 的有效提升后停止。微小改善不重置早停计数，但实际最高 Dice 的权重始终保存。
- 验证和测试使用原始自然正负比例，无增强、无均衡采样；以验证集全局裂纹 Dice 保存最佳权重，训练结束后重载该权重评估测试集。
- 阈值固定 0.5，报告全局裂纹 Dice、IoU、Precision、Recall；负样本产生的 FP 同样计入这些指标。此外记录无裂纹图片误报率（只要预测出一个裂纹像素就计为该图片误报）及无裂纹图片上的像素误报比例。

## 运行

```bash
# 冒烟：训练 3 个 batch，验证/测试各 2 个 batch，显式交替选取正负样本
.venv/bin/python scripts/train.py --smoke --output outputs/runs/tack_smoke

# 正式训练
.venv/bin/python scripts/train.py --epochs 100 --batch-size 4 --output outputs/runs/tack_baseline
```

冒烟的平衡子集只验证正负样本流程，不能代表自然比例测试集上的效果。正式训练启动状态见对应运行目录。输出目录必须尚不存在；重复运行时换用新目录。结果包括自动保存的 `train.log`、`config.json`、每轮更新的 `history.json`、`best.pt`、最终 `metrics.json`，以及正、负各一张原图/标注/预测对比图。

## 环境复现

已有 `.venv` 可以直接使用；重新创建时执行：

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -r requirements.lock.txt --extra-index-url https://download.pytorch.org/whl/cu128
```

## 可复现与断点记录

新启动的训练固定 Python、NumPy、PyTorch CPU/CUDA 和采样器随机状态，启用确定性算法，关闭 cuDNN benchmark 和 TF32。启动时保存源码、依赖版本、GPU/驱动/CUDA 信息、数据文件 SHA256 和划分快照。每轮保存 `last.pt`（模型、优化器、调度器、早停及随机状态），`best.pt` 仍按验证 Dice 选择。当前 CLI 没有断点恢复参数，完整状态供后续恢复实现和核对使用。

已在本机做两次独立 GPU 冒烟，模型权重逐位一致，指标及预测图片一致；不同软硬件版本不保证逐位一致。验证记录见 `outputs/reports/reproducibility_validation/results.json`。

现有 `tack_baseline` 在加入上述功能前已启动，当前暂停在第 8 轮第 67 个 batch，前 7 轮完成。该运行缺少启动时优化器/RNG 状态，不能事后保证精确复现；其 `source_at_pause/` 只是暂停时源码快照。新设置不会热更新已暂停的 Python 进程。暂停保留内存和显存；本次没有恢复或重启正式训练。

## 本地数据看板

地址：<http://127.0.0.1:8765>。服务仅监听本机，提供只读监控，没有训练控制按钮。

```bash
.venv/bin/python scripts/dashboard.py --run outputs/runs/tack_baseline --port 8765
```

每 5 秒读取训练进度、每轮 Loss/Dice/IoU/Precision/Recall、学习率、早停和正负样本数据；同时采集整卡显存与利用率、系统内存、训练进程 RSS、GPU 温度/功耗和 CPU 温度/利用率。硬件记录写入当前运行的 `resources.jsonl`，图表显示最近 720 次采样。缺失传感器显示为空，监测启动前的硬件数据不回填。

看板代码位于 `src/dashboard/`，监测逻辑位于 `src/monitoring.py`；服务启动信息与日志位于 `outputs/dashboard/local/`。浏览器视觉检查受当前环境无可用浏览器限制；静态资源、API、传感器和暂停状态已验证。

看板曲线分为训练/验证 Loss、验证 Dice/IoU、验证 Precision/Recall、学习率、负样本图片误报率、负样本像素误报率。所有训练图横轴为 Epoch；学习率使用阶梯线与对数纵轴，误报率以百分比显示，图片和像素口径分别作图。曲线采用固定配色、实线/虚线及不同点标记，图例与曲线一致，保留原始值，不做平滑。
