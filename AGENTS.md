# 架构图

- 使用 PlotNeuralNet 作为模型架构图后端。先依据模型和推理入口整理网络结构，再在 `scripts/` 中编写调用后端的独立 Python 绘图脚本。
- 绘图脚本生成 TikZ，经 Tectonic 编译成预览 PDF，再渲染并检查 PNG 中的标签、连线、间距和裁切。版式修改只写回独立绘图脚本，再重新生成预览。
- `src/` 中的模型实现和 `.tools/PlotNeuralNet/` 是绘图输入和后端。生成的 TeX、PDF、SVG、PNG 都是构建产物。版式迭代期间仅更新预览，检查通过后才导出到 `outputs/reports/model_architecture/`。
