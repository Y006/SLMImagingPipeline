# slm_display.py
import os
import sys
import time
import ctypes
import numpy as np
from PIL import Image
from loguru import logger

# 引入基类
from .base_display import BaseDisplayDevice, DisplayPayload

# 尝试设置 DPI 感知 (Windows)
try:
    ctypes.windll.shcore.SetProcessDpiAwareness(2)
except:
    try: ctypes.windll.user32.SetProcessDPIAware()
    except: pass

# --- SDK 路径配置 ---
# 请根据实际环境修改 SDK 路径
SDK_ROOT = r"C:\Program Files\HOLOEYE Photonics\SLM Display SDK (Python) v4.1.0"
if os.path.exists(SDK_ROOT):
    sys.path.append(os.path.join(SDK_ROOT, "examples"))
    try:
        import HEDS
        from HEDS.holoeye_slmdisplaysdk_types import *
        HAVE_SDK = True
    except ImportError:
        HAVE_SDK = False
else:
    HAVE_SDK = False

class SLMDisplay(BaseDisplayDevice):
    """
    HOLOEYE SLM 控制实现。
    特性：
    1. 严格像素检查（禁止缩放）
    2. 加载 LUT 校准
    3. 硬件稳定时间控制
    """
    def __init__(self, monitor_idx: int, lut_path: str = None, settle_time: float = 0.1):
        super().__init__(monitor_idx)
        self.lut_path = lut_path
        self.settle_time = settle_time
        self.slm_handle = None

    def initialize(self):
        if not HAVE_SDK:
            logger.warning("[SLM] SDK not found! Running in simulation mode.")
            self.is_active = True
            return

        try:
            # 1. SDK Init
            if HEDS.SDK.Init(4, 1) != HEDSERR_NoError:
                raise RuntimeError("SDK Init failed")
            
            # 2. SLM Open
            self.slm_handle = HEDS.SLM.Init()
            if self.slm_handle.errorCode() != HEDSERR_NoError:
                raise RuntimeError(f"SLM Open failed: {self.slm_handle.errorString()}")
            
            self.width_px = self.slm_handle.width_px
            self.height_px = self.slm_handle.height_px
            
            # 3. Load LUT
            if self.lut_path and os.path.exists(self.lut_path):
                err = self.slm_handle.loadCalibration(self.lut_path)
                if err != HEDSERR_NoError:
                    logger.error(f"[SLM] LUT load failed: {HEDS.SDK.ErrorString(err)}")
                else:
                    logger.info(f"[SLM] LUT loaded: {self.lut_path}")
            
            self.is_active = True
            logger.info(f"[SLM] Init Success. Res: {self.width_px}x{self.height_px}")
            
        except Exception as e:
            logger.critical(f"[SLM] Init Error: {e}")
            self.is_active = False

    def show(self, payload: DisplayPayload):
        """
        显示内容。
        payload.content: 支持 numpy.ndarray 或 图片路径
        """
        if not self.is_active: 
            logger.warning("[SLM] Device not active, skipping show.")
            return

        data = payload.content
        
        # 1. 数据解析与检查
        if isinstance(data, str):
            # 路径模式
            if not os.path.exists(data):
                logger.error(f"[SLM] Image not found: {data}")
                return
            
            if HAVE_SDK:
                # 使用 SDK 内置加载 (高效)
                err, data_handle = self.slm_handle.loadImageDataFromFile(data)
                if err != HEDSERR_NoError:
                    logger.error(f"[SLM] SDK Load failed: {err}")
                    return
                
                # [关键] 尺寸检查
                if data_handle.width_px != self.width_px or data_handle.height_px != self.height_px:
                    logger.error(f"[SLM] Size Mismatch! Img: {data_handle.width_px}x{data_handle.height_px}, Screen: {self.width_px}x{self.height_px}")
                    # 策略: 严格模式下直接拒绝，防止错误的相位调制
                    return 

                # 显示
                data_handle.show(HEDSSHF_PresentAutomatic)
                
        elif isinstance(data, np.ndarray):
            # 矩阵模式 (此处仅示意，需根据 SDK 具体 API 实现 `showData`)
            if data.shape[1] != self.width_px or data.shape[0] != self.height_px:
                 logger.error(f"[SLM] Matrix shape mismatch: {data.shape}")
                 return
            if HAVE_SDK:
                 # self.slm_handle.showData(data) # 伪代码
                 pass

        # 2. 硬件稳定等待 (重要!)
        wait = payload.exposure_wait if payload.exposure_wait > 0 else self.settle_time
        if wait > 0:
            time.sleep(wait)
            
        logger.info(f"[SLM] Shown: {payload.name}")

    def clear(self):
        # 显示全黑或关闭
        if self.is_active and HAVE_SDK:
            # HEDS SDK 具体清空指令，或显示一张全黑图
            # self.slm_handle.showBlack() # 伪代码
            pass

    def close(self):
        if HAVE_SDK:
            if self.slm_handle:
                self.slm_handle.close()
            HEDS.SDK.Close()
        self.is_active = False
        logger.info("[SLM] Closed.")