# base_display.py
import abc
from dataclasses import dataclass, field
from typing import Any, Optional, Tuple, Dict, List

@dataclass
class DisplayPayload:
    """
    通用显示数据载荷
    """
    content: Any  # 对于 Monitor 是图片路径(str); 对于 SLM 是 矩阵(np.array) 或 路径(str)
    
    # --- 通用参数 ---
    name: str = "Task"
    
    # --- MonitorDisplay 专用参数 ---
    target_size_cm: Optional[Tuple[float, float]] = None  # (宽, 高) cm
    position: Tuple[float, float] = (0, 0)                # (x, y)
    anchor: str = "center"                                # 对齐方式
    
    # --- SLMDisplay 专用参数 ---
    exposure_wait: float = 0.0                            # 显示后强制等待时间(秒)
    
    extra_meta: Dict = field(default_factory=dict)        # 预留扩展

class BaseDisplayDevice(abc.ABC):
    """
    显示设备抽象基类
    """
    def __init__(self, monitor_idx: int):
        self.monitor_idx = monitor_idx
        self.width_px = 0
        self.height_px = 0
        self.is_active = False

    @abc.abstractmethod
    def initialize(self):
        """
        初始化设备。
        Monitor: 启动窗口线程/进程
        SLM: 初始化 SDK, 加载 LUT
        """
        pass

    @abc.abstractmethod
    def show(self, payload: DisplayPayload):
        """
        显示内容。
        payload: 包含数据和元数据的载荷对象
        """
        pass

    @abc.abstractmethod
    def clear(self):
        """
        清空显示（黑屏）
        """
        pass

    @abc.abstractmethod
    def close(self):
        """
        释放资源，关闭设备
        """
        pass

    def get_info(self) -> str:
        return f"Device[{self.monitor_idx}] Res:{self.width_px}x{self.height_px} Active:{self.is_active}"