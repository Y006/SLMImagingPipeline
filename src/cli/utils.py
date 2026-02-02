"""
CLI 公共工具函数

提供配置加载、图片保存等基础功能
"""

import json
import numpy as np
from pathlib import Path
from PIL import Image
from PIL.PngImagePlugin import PngInfo
from loguru import logger
import yaml
import typer
import re


def load_context(config_path_str: str):
    """
    加载配置并准备输出目录
    
    Args:
        config_path_str: 配置文件路径
        
    Returns:
        tuple: (config_dict, save_dir_path)
    """
    # 1. 读 YAML
    path = Path(config_path_str)
    if not path.is_absolute():
        path = Path.cwd() / path
    if not path.exists():
        logger.error(f"配置文件缺失: {path}")
        raise typer.Exit(1)
    
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # 2. 准备目录
    proj = cfg.get("project", {})
    root = Path(proj.get("root_dir", "./output"))
    if not root.is_absolute():
        root = Path.cwd() / root
    
    save_dir = root / proj.get("id", "default")
    save_dir.mkdir(parents=True, exist_ok=True)
    
    return cfg, save_dir


def save_meta_image(bmp_path: Path, metadata: dict):
    """
    保存带元数据的 PNG（从 BMP 转换）
    
    Args:
        bmp_path: BMP 文件路径
        metadata: 要嵌入的元数据字典
    """
    try:
        png_path = bmp_path.with_suffix(".png")
        img = Image.open(bmp_path)
        info = PngInfo()
        for k, v in metadata.items():
            val = json.dumps(v, ensure_ascii=False) if isinstance(v, (dict, list, int, float)) else str(v)
            info.add_text(k, val)
        img.save(png_path, "PNG", pnginfo=info)
        bmp_path.unlink(missing_ok=True)
        logger.success(f"已保存: {png_path.name}")
    except Exception as e:
        logger.error(f"保存失败: {e}")


def save_numpy_array_as_png(image_arr: np.ndarray, save_path: Path, metadata: dict):
    """
    将 Numpy 图像数组保存为带元数据的 PNG
    
    Args:
        image_arr: 图像数组 (H, W, C) 或 (H, W)
        save_path: 保存路径
        metadata: 要嵌入的元数据字典
    """
    try:
        # 1. 数据归一化与类型转换 (float -> uint8)
        if image_arr.dtype != np.uint8:
            image_arr = image_arr.astype(np.float32)
            image_arr = (image_arr - image_arr.min()) / (image_arr.max() - image_arr.min() + 1e-8)
            image_arr = (image_arr * 255).astype(np.uint8)

        # 2. 转换为 PIL Image
        if image_arr.ndim == 3 and image_arr.shape[2] == 1:
            image_arr = image_arr.squeeze(2)
            
        img = Image.fromarray(image_arr)

        # 3. 注入元数据
        info = PngInfo()
        for k, v in metadata.items():
            val = json.dumps(v, ensure_ascii=False) if isinstance(v, (dict, list, int, float, type(None))) else str(v)
            info.add_text(k, val)

        # 4. 保存
        img.save(save_path, "PNG", pnginfo=info)
        logger.success(f"重建结果已保存: {save_path}")
        
    except Exception as e:
        logger.error(f"保存图像失败: {e}")


def get_next_sequence_number(save_dir: Path, prefix: str = "psf") -> int:
    """
    扫描文件夹，获取下一个可用的编号
    例如：存在 001_psf.png, 002_psf.bmp，则返回 3
    
    支持 .bmp 和 .png 两种格式
    """
    max_num = 0
    # 匹配模式：数字_前缀.bmp 或 .png
    pattern = re.compile(rf"^(\d+)_{re.escape(prefix)}\.(bmp|png)$")
    
    # 扫描 .bmp 和 .png 两种格式
    for ext in ["bmp", "png"]:
        for file in save_dir.glob(f"*_{prefix}.{ext}"):
            match = pattern.match(file.name)
            if match:
                num = int(match.group(1))
                if num > max_num:
                    max_num = num
    
    logger.debug(f"[编号扫描] 目录: {save_dir}, 前缀: {prefix}, 最大编号: {max_num}")
    return max_num + 1