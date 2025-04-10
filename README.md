# Image to SVG Converter

这是一个将常规位图（如PNG、JPG、BMP等）转换为简化SVG图标的工具。它通过概率统计采样和颜色聚类，将图像简化为8种颜色，并生成矢量化的SVG文件。

## 功能
- 支持多种位图格式（PNG、JPG、BMP、TIFF等）。
- 可自定义图片边长和取样点数。
- 在保存SVG前提供量化图像预览。
- 生成的SVG文件按颜色分组，便于编辑。

## 安装
1. 确保安装Python 3.8或更高版本。
2. 安装依赖：
   ```bash
   pip install numpy opencv-python svgwrite