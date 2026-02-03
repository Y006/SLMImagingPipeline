# screen/__init__.py

from .base_display import BaseDisplayDevice, DisplayPayload
from .screen_display import MonitorDisplay  # 假设 screen_display.py 里类名是 MonitorDisplay
from .slm_display import SLMDisplay

# 暴露给外部调用的接口列表
__all__ = [
    "BaseDisplayDevice",
    "DisplayPayload",
    "MonitorDisplay",
    "SLMDisplay"
]