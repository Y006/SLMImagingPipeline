# -*- coding: utf-8 -*-
import os
import sys

# 让 Python 能看到 examples\HEDS\ 包
sdk_root = r"C:\Program Files\HOLOEYE Photonics\SLM Display SDK (Python) v4.1.0"
sys.path.append(os.path.join(sdk_root, "examples"))
import HEDS
from HEDS.holoeye_slmdisplaysdk_types import *  # 注意：不是 hedslib.heds_types


class SLM:
    """核心类：只保留两个功能
       1) 初始化 SDK + 打开 SLM
       2) 将一张图片显示到 SLM 上
    """

    def __init__(self, sdk_version=(4, 1), verbose=True):
        self.sdk_version = sdk_version
        self.verbose = verbose
        self.slm = None

    def init(self):
        """初始化 SDK + 打开 SLM"""
        if self.verbose:
            HEDS.SDK.PrintVersion()

        major, minor = self.sdk_version
        err = HEDS.SDK.Init(major, minor)
        if err != HEDSERR_NoError:
            raise RuntimeError(HEDS.SDK.ErrorString(err))
        if self.verbose:
            print(f"[SDK] HEDS Init v{major}.{minor} OK.")

        self.slm = HEDS.SLM.Init()
        if self.slm.errorCode() != HEDSERR_NoError:
            raise RuntimeError(HEDS.SDK.ErrorString(self.slm.errorCode()))

        if self.verbose:
            print("[SLM] Device opened successfully.")

    def img_show(self, img_path: str) -> bool:
        """将传入路径的图片显示到 SLM，返回是否成功"""
        if not os.path.isfile(img_path):
            if self.verbose:
                print(f"[SLM] Image not found: {img_path}")
            return False

        if self.slm is None:
            if self.verbose:
                print("[SLM] Not initialized. Call init() first.")
            return False

        err, dh = self.slm.loadImageDataFromFile(img_path)
        if err != HEDSERR_NoError or dh.errorCode() != HEDSERR_NoError:
            if self.verbose:
                print(f"[SLM] Failed to load image: {HEDS.SDK.ErrorString(err)}")
            return False

        err = dh.show(HEDSSHF_PresentAutomatic)
        if err != HEDSERR_NoError:
            if self.verbose:
                print(f"[SLM] Failed to show image: {HEDS.SDK.ErrorString(err)}")
            return False

        if self.verbose:
            print(f"[SLM] Showing image: {img_path}")

        return True


# ------------------ 用法示例 ------------------
if __name__ == "__main__":
    slm = SLM(verbose=True)
    slm.init()
   
    import time
    slm.img_show(r"D:\qjy\camera_slm_pipeline\data\fza_bin_gen_masked_r40\FZA_bin_R15.png")   # 替换为实际图片路径
    time.sleep(1000)  # 暂停60秒