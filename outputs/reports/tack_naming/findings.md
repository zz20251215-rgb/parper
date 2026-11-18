# TACK Tunnel B 来源与文件命名

## 官方来源

- 论文：[TACK Tunnel Data (TTD): A Benchmark Dataset for Deep Learning-Based Defect Detection in Tunnels](https://arxiv.org/abs/2512.14477)，Andreas Sjölander、Valeria Belloni、Robel Fekadu、Andrea Nascetti，2025-12-16 提交的 arXiv 预印本。此次核查未确认另有正式期刊版本。
- [论文全文](https://arxiv.org/html/2512.14477v1)；[官方数据仓库](https://huggingface.co/datasets/TACK-project/TACK_Tunnel_Data)。本地 Git remote 与该官方仓库相符。
- 论文 Methods/Data Collection 说明采集传感器分辨率为 2448×2048，使用七台红外相机；Tunnel B 为预制混凝土衬砌，表面较平整。
- Dataset description 说明训练图片为 512×512 切块；论文文字将切块索引概述为 A–D，但与本地 B 的 12 个实际尾缀不一致，因此具体布局采用本地统计、边缘对比与目视拼接验证，不能仅凭论文的 A–D 描述推断。

## 本地验证结论

`TB_Camera4_001532_G.png` 可拆为 Tunnel B、Camera4、采集图像编号 001532、切块编号 G。同一 `TB_Camera4_001532` 前缀对应同一组空间切块，不是 12 次独立拍摄或 12 种缺陷类别。数字编号的物理时间/距离单位未找到官方解释，不能视作时间戳或里程。

全部 1536 张 B 图片恰好组成 128 组，每组均有 12 块，单块 512×512，布局为：

```text
A B C D
E F G H
I L M N
```

J、K 未出现，但每组模式一致，不是个别组缺文件；跳过这两个字母的命名原因未找到明确官方说明。对全部 128 组平均边缘灰度差进行分析，预期水平邻接约 3.01～5.26 灰度级，垂直邻接约 3.74～4.83；每条预期内部邻接均为对应方向的最低差候选。另目视核对了两组 4 列拼接的场景连续性。

可拼出的区域为 2048×1536，并非论文所说的完整传感器画幅 2448×2048。缺少裁切脚本和原始帧，不能恢复外围未提供像素，也不能确认裁切起点。

图片 G 对应标注 `TB_Camera4_001532_fuse_G_1band.png`。官方脚本 `get_expected_mask_basename` 证实配对规则；`1band` 标注为单通道。本地没有找到 `fuse` 一词的明确命名释义，不据此推断额外处理流程。

标注值为 0=背景、40=裂纹、160=水渍、200=析出物（leaching）。本地统计：263 张含裂纹、33 张含水渍、116 张含析出物，与论文 Table 2 相符；不同缺陷可以同图出现。只做裂纹分割时应以 `mask == 40` 转二值，不能将所有非零值都当裂纹。

本地 CSV 在源图前缀层面也有跨集合共享，具体数量见 `local_groups.json`；本次仅调查，没有改变文件、标注和划分。

## 证据

- `local_groups.json`：完整尾缀分布、源图组数和 CSV 源图交集。
- `seam_analysis.json`：全部 128 组的边缘匹配统计。
- `TB_Camera4_001532_4cols.jpg`：带切块标签的拼接预览。
- `stitched_example.png`、`stitched_mask.png`：该示例的无标签拼接图与标注，均为 2048×1536。
