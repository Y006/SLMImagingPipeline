# SLM 成像系统控制与重建

## 简介

本项目是一个基于空间光调制器（SLM）的计算成像系统的完整解决方案。系统集成了硬件控制、数据采集、图像重建等核心功能，支持 PSF（点扩展函数）测量、被测物体成像采集，以及多种重建算法。项目采用模块化设计，通过 YAML 配置驱动工作流，支持 ADMM 和 Wiener 两种重建算法，为计算摄影研究提供一个开放且易于扩展的平台。

## 快速开始

### 1. 环境配置

使用 `uv` 配置项目环境：

```bash
# 安装依赖
uv sync

# 激活虚拟环境
source .venv/bin/activate  # macOS/Linux
.venv\Scripts\activate  # Windows
```

### 2. 运行 CLI 命令

所有操作通过命令行界面完成，以下是部分常用命令：

#### 数据采集
```bash
python cli.py --help

# 采集 PSF（点扩展函数）
python cli.py capture-psf --config configs/single_shot_config.yaml --repeat 3

# 采集测量数据
python cli.py capture-measurement --config configs/single_shot_config.yaml --repeat 5

# 系统标定/预览
python cli.py calibrate --config configs/single_shot_config.yaml

# 查看 PNG 图片中嵌入的实验元数据
python cli.py inspect path/to/image.png

# ADMM 算法重建
python cli.py admm --config configs/single_shot_config.yaml --note "测试运行"

# Wiener 滤波重建（预览模式）
python cli.py wiener --config configs/single_shot_config.yaml
```

### 3. 配置说明

所有参数通过 `configs/single_shot_config.yaml` 配置。

### 4. 输出结果

所有结果保存至 `output/{project_id}/` 目录：
- 原始采集数据保存为 PNG，并嵌入实验元数据
- 重建结果同样包含完整的配置和参数记录
- 可通过 `inspect` 命令查看嵌入的元数据

### 注意：

当没有硬件时，可以通过调整 `src/cli/capture.py` 里面的内容进行测试：

```python
# # 仿真硬件接口（用于测试）
# from src.hardware.mock_hardware import (
#     MockSLM as SLM,
#     MockHikCamera as HikCamera,
#     mock_display_image as display_image,
#     MV_Image_Bmp
# )

# 真实硬件接口
from src.hardware.camera import HikCamera, MV_Image_Bmp
from src.hardware.slm import SLM
from src.hardware.screen import display_image
```

## 显示器设计方案

> [!NOTE]
>
> 这份文档旨在阐述一套基于 Python 的轻量级图像显示系统的设计思路。该方案的核心目标是解决传统图形界面开发中“像素与物理尺寸脱节”的问题，利用 Python 标准库 **Tkinter** 结合轻量级第三方库 **screeninfo**，实现对指定显示器的精准控制。该设计摒弃了庞大的科学计算依赖，力求在保持极低资源占用的前提下，达成以“厘米”为单位的物理级图像渲染与定位。
>
> 整个系统的运行逻辑建立在一个关键的前置校准环节之上，这是实现“物理空间的精准映射”的基石。在首次运行或环境变更时，程序会启动校准脚本，在目标显示器上绘制一个标准参照物（如虚拟标尺）。用户需使用物理直尺测量该参照物的实际长度并回填数据。系统据此反向计算出该显示器当前的精确像素密度（PPI），并生成一个核心转换系数——**PPC（Pixels Per Centimeter，每厘米像素数）**。这一系数将作为后续所有渲染操作的数学基准，确保软件层面的逻辑尺寸能无缝映射到现实世界的物理尺寸。
>
> 在定位机制上，本方案采用绝对坐标系策略。我们定义指定显示器的物理左上角为原点 $(0, 0)$，所有的位置偏移量均以“厘米”为单位进行描述，而非传统的像素坐标。当用户输入位置参数（如 $10\text{cm}, 15\text{cm}$）时，系统会自动结合 PPC 系数，计算出从屏幕左上角向右偏移 10 厘米、向下偏移 15 厘米所需的具体像素数，并将图像的左上角精准投射至该坐标。这种设计极大地简化了多显示器环境下的定位逻辑，用户无需关心不同屏幕分辨率的差异，仅需关注物理布局即可。
>
> 针对图像的尺寸控制，系统设计了一套灵活的元组参数机制 `(Width, Height)`，以满足四种不同场景的显示需求：
>
> 1. **强制物理尺寸**（如 `(5, 10)`）：系统忽略图像原始比例，强制将其拉伸或压缩至宽 5cm、高 10cm 的矩形区域内。
> 2. **定宽等比缩放**（如 `(5, None)`）：锁定宽度为 5cm，高度根据原始图像长宽比自动计算，确保图像不失真。
> 3. **定高等比缩放**（如 `(None, 10)`）：锁定高度为 10cm，宽度按比例自适应。
> 4. **原始分辨率直出**（即 `(None, None)`）：跳过物理映射逻辑，直接以图像文件的原始像素尺寸进行点对点显示。
>
> 综上所述，该设计通过“校准-映射-渲染”的三步流程，在不引入重型依赖的前提下，成功将屏幕显示从“像素定义”提升至“物理定义”，为需要精确空间控制的应用场景提供了一套高效、可复用的解决方案。

