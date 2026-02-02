"""
重建算法相关逻辑

包含：
- run_admm_logic: ADMM 重建算法
- run_wiener_logic: Wiener 滤波重建算法
"""

import datetime
from pathlib import Path
from loguru import logger
import typer
import torch
from PIL import Image
import numpy as np

from .utils import load_context, save_numpy_array_as_png
from src.reconstruction.admm_recon import ADMMReconstructionPipeline
from src.reconstruction.winner_recon import process_one_pair


def run_admm_logic(config_path: str, note: str):
    """
    ADMM 重建算法 (完全基于 YAML 配置)
    
    Args:
        config_path: 配置文件路径
        note: 本次重建备注
    """
    # 1. 加载配置
    cfg, _ = load_context(config_path)
    
    # 2. 提取 reconstruction 节点
    recon_conf = cfg.get("reconstruction")
    if not recon_conf:
        logger.error("配置文件中缺少 'reconstruction' 模块")
        raise typer.Exit(1)

    # 3. 路径解析与检查
    def resolve_path(p_str):
        if not p_str: return None
        p = Path(p_str)
        return p if p.is_absolute() else Path.cwd() / p

    psf_p = resolve_path(recon_conf.get("psf_path"))
    meas_p = resolve_path(recon_conf.get("measurement_path"))
    gt_p = resolve_path(recon_conf.get("ground_truth_file"))

    if not psf_p or not psf_p.exists():
        logger.error(f"PSF 文件未找到: {psf_p}")
        raise typer.Exit(1)
    if not meas_p or not meas_p.exists():
        logger.error(f"测量文件未找到: {meas_p}")
        raise typer.Exit(1)

    # 4. 初始化重建管道
    logger.info(f"启动 ADMM 重建 | Iter: {recon_conf.get('iterations')} | Device: {recon_conf.get('device')}")
    
    pipeline = ADMMReconstructionPipeline(
        psf_path=str(psf_p),
        measurement_path=str(meas_p),
        ground_truth_file=str(gt_p) if gt_p else None,
        downsample=recon_conf.get("downsample", 1),
        iterations=recon_conf.get("iterations", 100),
        mu1_init=recon_conf.get("mu1_init", 1e-4),
        mu2_init=recon_conf.get("mu2_init", 1e-4),
        mu3_init=recon_conf.get("mu3_init", 1e-4),
        tau_init=recon_conf.get("tau_init", 2.0),
        device=recon_conf.get("device", "cpu"),
        save_dir=None,  # 禁止 Pipeline 内部保存
        save_name=None
    )

    try:
        # 5. 运行重建
        recon_image = pipeline.run()
        
        if recon_image is None:
            logger.error("重建返回结果为空 (None)")
            raise typer.Exit(1)

        # 6. 准备保存路径
        proj_conf = cfg.get("project", {})
        root_dir = Path(proj_conf.get("root_dir", "./output"))
        proj_id = proj_conf.get("id", "default_exp")
        
        if not root_dir.is_absolute():
            root_dir = Path.cwd() / root_dir
        save_dir = root_dir / proj_id / "reconstruction"
        save_dir.mkdir(parents=True, exist_ok=True)

        # 文件名
        save_name_conf = recon_conf.get("save_name", {})
        file_name = save_name_conf.get("recon", f"recon_{datetime.datetime.now().strftime('%H%M%S')}.png")
        save_path = save_dir / file_name

        # 7. 组装元数据
        meta = {
            "mode": "reconstruction",
            "algorithm": "ADMM",
            "timestamp": datetime.datetime.now().isoformat(),
            "note": note,
            "recon_params": {
                "iterations": recon_conf.get("iterations"),
                "downsample": recon_conf.get("downsample"),
                "mu_vals": [recon_conf.get("mu1_init"), recon_conf.get("mu2_init"), recon_conf.get("mu3_init")],
                "tau": recon_conf.get("tau_init"),
                "device": recon_conf.get("device")
            },
            "input_files": {
                "psf": str(psf_p.name),
                "meas": str(meas_p.name),
                "gt": str(gt_p.name) if gt_p else None
            },
            "project": proj_conf
        }

        # 8. 保存带 Meta 的 PNG
        save_numpy_array_as_png(recon_image, save_path, meta)

    except Exception as e:
        logger.exception(f"重建过程发生错误: {e}")
        raise typer.Exit(1)


def run_wiener_logic(config_path: str, note: str):
    """
    Wiener 滤波重建算法 (仅预览，不保存)
    
    Args:
        config_path: 配置文件路径
        note: 本次重建备注
    """
    # 1. 加载配置
    cfg, _ = load_context(config_path)
    
    # 2. 提取 reconstruction 节点
    recon_conf = cfg.get("reconstruction")
    if not recon_conf:
        logger.error("配置文件中缺少 'reconstruction' 模块")
        raise typer.Exit(1)

    # 3. 路径解析与检查
    def resolve_path(p_str):
        if not p_str: return None
        p = Path(p_str)
        return p if p.is_absolute() else Path.cwd() / p

    psf_p = resolve_path(recon_conf.get("psf_path"))
    meas_p = resolve_path(recon_conf.get("measurement_path"))

    if not psf_p or not psf_p.exists():
        logger.error(f"PSF 文件未找到: {psf_p}")
        raise typer.Exit(1)
    if not meas_p or not meas_p.exists():
        logger.error(f"测量文件未找到: {meas_p}")
        raise typer.Exit(1)

    # 4. 使用默认参数
    delta = 100000000  # 默认正则化参数
    
    logger.info(f"启动 Wiener 滤波重建 (仅预览) | Delta: {delta:.2e}")
    logger.info(f"PSF: {psf_p.name}")
    logger.info(f"测量: {meas_p.name}")

    try:
        # 5. 创建临时输出路径用于预览
        import tempfile
        temp_output = Path(tempfile.gettempdir()) / f"wiener_preview_{datetime.datetime.now().strftime('%H%M%S')}.png"
        
        # 6. 执行 Wiener 重建
        process_one_pair(
            psf_path=str(psf_p),
            blur_path=str(meas_p),
            delta=delta,
            output_path=str(temp_output)
        )
        
        # 7. 预览结果
        logger.info("正在显示预览...")
        preview_img = Image.open(temp_output)
        preview_img.show(title="Wiener 滤波重建预览")
        
        # 8. 清理临时文件
        temp_output.unlink(missing_ok=True)
        logger.success("预览完成")
        
        # TODO: 保存功能待实现
        # 可在此处添加保存逻辑

    except Exception as e:
        logger.exception(f"Wiener 重建过程发生错误: {e}")
        raise typer.Exit(1)
