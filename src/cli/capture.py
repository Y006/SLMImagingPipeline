"""
数据采集相关逻辑

包含：
- run_psf_logic: PSF 采集
- run_measurement_logic: Measurement 采集
- run_calibration_logic: 标定/预览模式
"""

import time
import datetime
import threading
from pathlib import Path
from loguru import logger
import typer

from .utils import load_context, save_meta_image, get_next_sequence_number

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


def run_psf_logic(config_path: str, repeat: int, note: str):
    """
    PSF 采集逻辑
    
    流程：加载配置 -> 路径检查 -> 初始化SLM -> 初始化相机 -> 拍摄
    
    Args:
        config_path: 配置文件路径
        repeat: 重复次数
        note: 备注信息
    """
    
    # 1. 加载上下文
    config, save_dir = load_context(config_path)
    raw_cap_conf = config.get("capture_settings", {})

    # --- 路径检查 ---
    slm_path_str = raw_cap_conf.get("slm_image_path")
    if not slm_path_str:
        logger.error("配置错误: PSF 模式必须指定 slm_image_path")
        raise typer.Exit(code=1)
    
    slm_path = Path(slm_path_str)
    if not slm_path.is_absolute():
        slm_path = Path.cwd() / slm_path
    
    if not slm_path.exists():
        logger.error(f"SLM 图片不存在: {slm_path}")
        raise typer.Exit(code=1)

    slm = None
    cam = None

    try:
        # --- 2. 初始化 SLM ---
        slm = SLM(verbose=True)
        if not slm.init(): 
            logger.error("SLM 初始化失败")
            raise typer.Exit(code=2)
            
        slm.img_show(str(slm_path))
        time.sleep(1)

        # --- 3. 初始化相机 ---
        cam = HikCamera(dev_index=0)
        if not cam.open():
            logger.error("相机打开失败")
            raise typer.Exit(code=3)

        # --- 4. 清洗配置参数 ---
        valid_setting_keys = ["exposure_us", "slm_image_path"]
        clean_settings = {
            k: raw_cap_conf.get(k) 
            for k in valid_setting_keys 
            if k in raw_cap_conf
        }

        full_phy = config.get("physical_setup", {})
        valid_phy_keys = [
            "sensor_mask_distance", 
            "mask_object_distance", 
            "point_light_brightness", 
            "mask_LED-light_distance"
        ]
        clean_physical = {
            k: full_phy.get(k) 
            for k in valid_phy_keys 
            if k in full_phy
        }

        # --- 5. 拍摄循环 ---
        exp = raw_cap_conf.get("exposure_us", 20000)
        timeout = raw_cap_conf.get("camera_timeout_ms", 5000)
        logger.info(f"开始采集 PSF | 曝光: {exp}us | 数量: {repeat}")

        start_idx = get_next_sequence_number(save_dir, "psf")
        
        for i in range(repeat):
            # 实现记忆性累加：起始编号 + 当前循环索引
            current_num = start_idx + i
            suffix = f"{current_num:03d}"
            # 最终路径：save_dir / "001_psf.bmp"
            tmp_bmp = save_dir / f"{suffix}_psf.bmp"

            if cam.snap(tmp_bmp, exposure_us=exp, timeout_ms=timeout, img_type=MV_Image_Bmp):
                meta = {
                    "mode": "psf",
                    "note": note,
                    "timestamp": datetime.datetime.now().isoformat(),
                    "repeat_idx": i + 1,
                    "settings": clean_settings, 
                    "physical": clean_physical,
                    "project": config.get("project")
                }
                save_meta_image(tmp_bmp, meta)
                
                if repeat > 1:
                    time.sleep(0.2)
            else:
                logger.warning(f"拍摄失败: {i+1}/{repeat}")

    except Exception as e:
        logger.exception(f"PSF 运行时错误: {e}")
        raise typer.Exit(code=4)

    finally:
        if cam: cam.close()


def run_measurement_logic(config_path: str, repeat: int, note: str):
    """
    Measurement 采集逻辑
    
    流程：加载配置 -> 路径检查(SLM+Display) -> 启动显示 -> 初始化SLM -> 初始化相机 -> 拍摄
    
    Args:
        config_path: 配置文件路径
        repeat: 重复次数
        note: 备注信息
    """
    
    # 1. 加载上下文
    config, save_dir = load_context(config_path)
    raw_cap_conf = config.get("capture_settings", {})

    # --- 路径检查 ---
    slm_path_str = raw_cap_conf.get("slm_image_path")
    if not slm_path_str:
        logger.error("配置错误: Measurement 模式必须指定 slm_image_path")
        raise typer.Exit(code=1)
    slm_path = Path(slm_path_str)
    if not slm_path.is_absolute(): slm_path = Path.cwd() / slm_path
    if not slm_path.exists():
        logger.error(f"SLM 图片不存在: {slm_path}")
        raise typer.Exit(code=1)

    disp_path_str = raw_cap_conf.get("display_image_path")
    if not disp_path_str:
        logger.error("配置错误: Measurement 模式必须指定 display_image_path")
        raise typer.Exit(code=1)
    disp_path = Path(disp_path_str)
    if not disp_path.is_absolute(): disp_path = Path.cwd() / disp_path
    if not disp_path.exists():
        logger.error(f"显示图片不存在: {disp_path}")
        raise typer.Exit(code=1)

    slm = None
    cam = None

    try:
        # --- 2. 启动显示器 ---
        mon_idx = raw_cap_conf.get("monitor_idx", 1)
        scale_f = raw_cap_conf.get("scale_factor", 1.0)
        
        logger.info(f"正在显示图片: {disp_path.name} (显示器 {mon_idx})")
        t = threading.Thread(
            target=display_image,
            args=(str(disp_path), mon_idx, scale_f),
            daemon=True
        )
        t.start()
        time.sleep(0.5)

        # --- 3. 初始化 SLM ---
        slm = SLM(verbose=True)
        if not slm.init(): 
            logger.error("SLM 初始化失败")
            raise typer.Exit(code=2)
            
        slm.img_show(str(slm_path))
        wait_slm = raw_cap_conf.get("slm_settle_time_s", 0.1) 
        time.sleep(wait_slm)

        # --- 4. 初始化相机 ---
        cam = HikCamera(dev_index=0)
        if not cam.open():
            logger.error("相机打开失败")
            raise typer.Exit(code=3)

        # --- 5. 清洗配置参数 ---
        valid_setting_keys = [
            "exposure_us", 
            "slm_image_path", 
            "display_image_path", 
            "monitor_idx", 
            "scale_factor"
        ]
        clean_settings = {
            k: raw_cap_conf.get(k) 
            for k in valid_setting_keys 
            if k in raw_cap_conf
        }

        full_phy = config.get("physical_setup", {})
        valid_phy_keys = [
            "object_name",
            "brightness", 
            "sensor_mask_distance", 
            "mask_object_distance"
        ]
        clean_physical = {
            k: full_phy.get(k) 
            for k in valid_phy_keys 
            if k in full_phy
        }

        # --- 6. 拍摄循环 ---
        exp = raw_cap_conf.get("exposure_us", 20000)
        timeout = raw_cap_conf.get("camera_timeout_ms", 5000)
        
        logger.info(f"开始采集 Measurement | 曝光: {exp}us | 数量: {repeat}")

        start_idx = get_next_sequence_number(save_dir, "m")

        for i in range(repeat):
            # 实现记忆性累加：起始编号 + 当前循环索引
            current_num = start_idx + i
            suffix = f"{current_num:03d}"
            # 最终路径：save_dir / "001_m.bmp"
            tmp_bmp = save_dir / f"{suffix}_m.bmp"

            if cam.snap(tmp_bmp, exposure_us=exp, timeout_ms=timeout, img_type=MV_Image_Bmp):
                meta = {
                    "mode": "measurement",
                    "note": note,
                    "timestamp": datetime.datetime.now().isoformat(),
                    "repeat_idx": i + 1,
                    "settings": clean_settings, 
                    "physical": clean_physical,
                    "project": config.get("project")
                }
                save_meta_image(tmp_bmp, meta)
                
                if repeat > 1:
                    time.sleep(0.2)
            else:
                logger.warning(f"拍摄失败: {i+1}/{repeat}")

    except Exception as e:
        logger.exception(f"Measurement 运行时错误: {e}")
        raise typer.Exit(code=4)

    finally:
        if cam: cam.close()


def run_calibration_logic(config_path: str):
    """
    标定/预览模式
    
    流程：加载配置 -> 启动显示器 -> 启动 SLM -> 挂起等待
    特点：完全没有相机代码
    
    Args:
        config_path: 配置文件路径
    """
    cfg, _ = load_context(config_path)
    cap = cfg.get("capture_settings", {})

    # --- 1. 显示器 ---
    disp_path = cap.get("display_image_path")
    if disp_path:
        logger.info(f"显示图片: {disp_path}")
        t = threading.Thread(
            target=display_image,
            args=(disp_path, cap.get("monitor_idx", 1), cap.get("scale_factor", 1.0)),
            daemon=True
        )
        t.start()
    
    # --- 2. SLM ---
    slm_path = cap.get("slm_image_path")
    if slm_path:
        logger.info(f"SLM 加载: {slm_path}")
        slm = SLM(verbose=True)
        slm.init()
        slm.img_show(slm_path)
    
    # --- 3. 挂起 ---
    logger.info(">>> 系统已就绪，正在显示图案... (按 Ctrl+C 退出) <<<")
    time.sleep(999999)
