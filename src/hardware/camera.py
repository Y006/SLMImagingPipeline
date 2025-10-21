# -*- coding: utf-8 -*-
import sys
from ctypes import *
from ctypes import byref, sizeof, memset, POINTER, c_ubyte, cast

from src.hardware.MvImport.MvCameraControl_class import *  # 你已确认有 SDK，这里直接使用

# ---- 简易错误映射与工具 ----
ERRMAP = {
    0x80000000: "MV_E_HANDLE(句柄/指针错误)",
    0x80000001: "MV_E_SUPPORT(不支持)",
    0x80000002: "MV_E_BUFOVER(缓冲区溢出/不足)",
    0x80000003: "MV_E_CALLORDER(调用顺序错误/状态不对)",
    0x80000004: "MV_E_PARAMETER(参数错误)",
    0x80000007: "MV_E_RESOURCE(资源申请失败)",
    0x80000103: "MV_E_NODATA(无数据)",
    0x80000105: "MV_E_TIMEOUT(超时)",
}
def explain(ret): return f"0x{ret:x} " + ERRMAP.get(ret, "（未知错误码）")
def OK(ret, what):
    if ret != MV_OK:
        print(f"[ERR] {what}: {explain(ret)}")
        return False
    return True

class HikCamera:
    """
    极简海康相机封装：
      - open(dev_index=0): 打开相机、设为连续采集、关闭自动曝光/增益
      - snap(save_path, exposure_us=20000, timeout_ms=1500, img_type=MV_Image_Jpeg) -> bool
      - close(): 释放
    仅在关键步骤打印日志，其他不做冗余配置。
    """
    def __init__(self, dev_index: int = 0):
        self.dev_index = dev_index
        self.cam = None
        self.payload = 0

    def open(self) -> bool:
        device_list = MV_CC_DEVICE_INFO_LIST()
        tlayer = MV_GIGE_DEVICE | MV_USB_DEVICE
        if not OK(MvCamera.MV_CC_EnumDevices(tlayer, device_list), "枚举设备"):
            return False
        if device_list.nDeviceNum == 0:
            print("[ERR] 未检测到相机")
            return False
        if self.dev_index >= device_list.nDeviceNum:
            print(f"[ERR] 设备索引 {self.dev_index} 超出范围 [0,{device_list.nDeviceNum-1}]")
            return False

        self.cam = MvCamera()
        dev_info = cast(device_list.pDeviceInfo[self.dev_index], POINTER(MV_CC_DEVICE_INFO)).contents
        if not OK(self.cam.MV_CC_CreateHandle(dev_info), "创建句柄"): return False
        if not OK(self.cam.MV_CC_OpenDevice(MV_ACCESS_Exclusive, 0), "打开设备"): return False

        # 基本工作模式：连续采集 + 关闭自动曝光/增益
        if not OK(self.cam.MV_CC_SetEnumValue("TriggerMode", MV_TRIGGER_MODE_OFF), "设置触发模式=Off"): return False
        self.cam.MV_CC_SetEnumValue("AcquisitionMode", 2)    # 连续（不同型号枚举值可能不同，不影响）
        self.cam.MV_CC_SetEnumValue("ExposureAuto", 0)       # Off
        self.cam.MV_CC_SetEnumValue("GainAuto", 0)           # Off

        # 记录负载大小用于分配取流缓冲
        val = MVCC_INTVALUE()
        memset(byref(val), 0, sizeof(val))
        if not OK(self.cam.MV_CC_GetIntValue("PayloadSize", val), "读取PayloadSize"): return False
        self.payload = val.nCurValue
        return True

    def snap(self, save_path: str, exposure_us: float = 20000.0,
             timeout_ms: int = 1500, img_type: int = MV_Image_Jpeg) -> bool:
        """
        抓拍一张并保存：
          - exposure_us: 曝光时间(微秒)，内部直接写 ExposureTime
          - timeout_ms : 取帧超时
          - img_type   : MV_Image_Jpeg 或 MV_Image_Bmp
        返回 True/False；失败时会打印关键错误原因。
        """
        if not self.cam:
            print("[ERR] 相机未打开")
            return False

        # 设置曝光（不同机型范围不同，如失败会打印但继续尝试抓拍）
        ret = self.cam.MV_CC_SetFloatValue("ExposureTime", float(exposure_us))
        if ret != MV_OK:
            print(f"[WARN] 设置曝光失败（可能超范围/不支持）：{explain(ret)}  已继续使用当前曝光。")

        if not OK(self.cam.MV_CC_StartGrabbing(), "开始取流"):
            return False

        # 取一帧
        frame_info = MV_FRAME_OUT_INFO_EX()
        memset(byref(frame_info), 0, sizeof(frame_info))
        buf = (c_ubyte * self.payload)()
        ret = self.cam.MV_CC_GetOneFrameTimeout(byref(buf), self.payload, frame_info, timeout_ms)
        if ret != MV_OK:
            self.cam.MV_CC_StopGrabbing()
            if ret == MV_E_TIMEOUT:
                print(f"[ERR] 取帧超时({timeout_ms}ms)。请检查曝光/触发/带宽。")
            else:
                print(f"[ERR] 取帧失败: {explain(ret)}")
            return False

        # 保存（官方建议输出缓冲至少 w*h*3 + 2048）
        out_size = frame_info.nWidth * frame_info.nHeight * 3 + 2048
        out_buf = (c_ubyte * out_size)()

        save_param = MV_SAVE_IMAGE_PARAM_EX()
        memset(byref(save_param), 0, sizeof(save_param))
        save_param.enImageType  = img_type
        save_param.enPixelType  = frame_info.enPixelType
        save_param.nWidth       = frame_info.nWidth
        save_param.nHeight      = frame_info.nHeight
        save_param.pData        = buf
        save_param.nDataLen     = frame_info.nFrameLen
        save_param.nJpgQuality  = 90
        save_param.pImageBuffer = out_buf
        save_param.nBufferSize  = out_size

        ret = self.cam.MV_CC_SaveImageEx2(save_param)
        if ret != MV_OK or save_param.nImageLen <= 0:
            self.cam.MV_CC_StopGrabbing()
            print("[ERR] 保存图像失败（可能是当前像素格式不支持直接保存）。"
                  "可尝试改用 BMP 或先转换到 BGR8 再保存。")
            print(f"Save ret={explain(ret)}, nImageLen={save_param.nImageLen}")
            return False

        try:
            with open(save_path, "wb") as f:
                f.write(string_at(save_param.pImageBuffer, save_param.nImageLen))
            # 成功
            self.cam.MV_CC_StopGrabbing()
            print(f"[OK] 已保存: {save_path}  尺寸: {frame_info.nWidth}x{frame_info.nHeight}")
            return True
        except Exception as e:
            self.cam.MV_CC_StopGrabbing()
            print(f"[ERR] 写文件失败: {e}")
            return False

    def close(self):
        if not self.cam:
            return
        try:
            self.cam.MV_CC_StopGrabbing()
        except Exception:
            pass
        try:
            self.cam.MV_CC_CloseDevice()
        except Exception as e:
            print(f"[WARN] 关闭设备异常: {e}")
        try:
            self.cam.MV_CC_DestroyHandle()
        except Exception as e:
            print(f"[WARN] 销毁句柄异常: {e}")
        self.cam = None

# ---- 示例 ----
if __name__ == "__main__":
    cam = HikCamera(dev_index=0)
    try:
        if not cam.open():
            sys.exit(1)
        ok = cam.snap("capture.jpg", exposure_us=500000.0, timeout_ms=2000, img_type=MV_Image_Jpeg)
        if not ok:
            print("[ERR] 抓拍失败")
    finally:
        cam.close()