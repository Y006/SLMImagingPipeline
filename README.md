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
# 或在 Windows 上
.venv\Scripts\activate
```

### 2. 运行 CLI 命令

所有操作通过命令行界面完成，以下是常用命令：

#### 数据采集
```bash
# 采集 PSF（点扩展函数）
python cli.py capture-psf --config configs/single_shot_config.yaml --repeat 3

# 采集测量数据
python cli.py capture-measurement --config configs/single_shot_config.yaml --repeat 5

# 系统标定/预览
python cli.py calibrate --config configs/single_shot_config.yaml
```

#### 工具命令
```bash
# 查看 PNG 图片中嵌入的实验元数据
python cli.py inspect path/to/image.png
```

#### 图像重建
```bash
# ADMM 算法重建
python cli.py admm --config configs/single_shot_config.yaml --note "测试运行"

# Wiener 滤波重建（预览模式）
python cli.py wiener --config configs/single_shot_config.yaml
```

#### 查看所有命令
```bash
python cli.py --help
```

### 3. 配置说明

所有参数通过 `configs/single_shot_config.yaml` 配置：

```yaml
project:
  id: 2026-02-02-exp003          # 实验 ID
  root_dir: "./output"            # 输出目录

reconstruction:
  psf_path: "path/to/psf.jpg"    # PSF 图像路径
  measurement_path: "path/to/m.jpg"  # 测量图像路径
  downsample: 16                  # 下采样倍数
  iterations: 400                 # ADMM 迭代次数
  device: "cuda:0"                # 计算设备（"cpu" 或 "cuda:0"）
```

### 4. 输出结果

所有结果保存至 `output/{project_id}/` 目录：
- 原始采集数据保存为 PNG，并嵌入实验元数据
- 重建结果同样包含完整的配置和参数记录
- 可通过 `inspect` 命令查看嵌入的元数据
