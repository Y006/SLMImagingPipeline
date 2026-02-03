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
python cli.py capture-psf

# 采集测量数据
python cli.py capture-measurement --config configs/single_shot_config.yaml --repeat 5

# 系统标定/预览
python cli.py calibrate --config configs/single_shot_config.yaml

# 查看 PNG 图片中嵌入的实验元数据
python cli.py inspect path/to/image.png

# ADMM 算法重建
python cli.py admm --config configs/single_shot_config.yaml --note "测试运行"
```

### 3. 配置说明

参数默认通过 `configs/single_shot_config.yaml` 配置。

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

## TODO List

- [x] 使用 loguru 更新日志打印方案。
- [x] 使用 Typer 创建 CLI 入口。
- [x] 修改显示器设计方案：支持使用厘米为单位的物理尺寸而不是像素尺寸对显示器显示的场景图片进行控制。
- [ ] 将 mask 的生成代码添加到 CLI 里面。
- [ ] 使用 Tkinter 或者 CustomTkinter 制作 GUI 界面实现相机的预览。
- [ ] 添加更多的重建算法支持和硬件设备控制。
- [ ] 使用 CustomTkinter 设计一个完整的 GUI 控制页面。