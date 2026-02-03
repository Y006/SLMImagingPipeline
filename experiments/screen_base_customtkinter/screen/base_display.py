import abc
from dataclasses import dataclass, field
from typing import Any, Optional, Tuple, Dict

@dataclass
class DisplayPayload:
    """通用显示数据载荷"""
    content: Any  
    name: str = "Task"
    target_size_cm: Optional[Tuple[float, float]] = None 
    position: Tuple[float, float] = (0, 0) # 物理位置 (cm) 或像素             
    anchor: str = "center"                               
    exposure_wait: float = 0.0                           
    extra_meta: Dict = field(default_factory=dict)       

class BaseDisplayDevice(abc.ABC):
    """显示设备抽象基类，集成异步逻辑与物理参数"""
    def __init__(self, monitor_idx: int):
        self.monitor_idx = monitor_idx
        self.width_px = 0
        self.height_px = 0
        self.width_cm = 0.0
        self.ppcm = 0.0  # 每厘米像素数 (Pixels Per Centimeter)
        self.is_active = False

    @abc.abstractmethod
    def initialize(self):
        """初始化设备并启动渲染循环"""
        pass

    @abc.abstractmethod
    def show(self, payload: DisplayPayload):
        """发送显示指令到队列"""
        pass

    @abc.abstractmethod
    def clear(self):
        """清空显示"""
        pass

    @abc.abstractmethod
    def close(self):
        """释放资源"""
        pass