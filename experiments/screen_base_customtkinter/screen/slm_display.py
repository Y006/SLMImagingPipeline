import multiprocessing
import queue
import time
import os
import sys
from loguru import logger

# 导入基类
from .base_display import BaseDisplayDevice, DisplayPayload

def _slm_render_engine_process(monitor_idx, cmd_queue, is_simulator):
    """
    [独立进程逻辑]：SLM 渲染引擎。
    在子进程中运行，与主进程完全隔离。
    """
    try:
        import customtkinter as ctk
        from PIL import Image, ImageTk
        import tkinter as tk
        from screeninfo import get_monitors
    except ImportError as e:
        print(f"子进程缺少必要库: {e}")
        return

    root = None
    canvas = None
    # 必须维持对 PhotoImage 的引用，否则图片会闪烁或不显示
    cache = {"tk_img": None}

    if is_simulator:
        ctk.set_appearance_mode("Dark")
        root = ctk.CTk()
        root.title(f"SLM Simulator - Device {monitor_idx}")
        
        # --- 窗口样式配置 ---
        # 1. 保留软件边框 (设为 False)
        root.overrideredirect(False)
        
        # 2. 选择显示器并设置小窗口尺寸
        try:
            monitors = get_monitors()
            if monitor_idx < len(monitors):
                m = monitors[monitor_idx]
                # 设置窗口大小为 800x600，并偏移到目标显示器
                root.geometry(f"800x600+{(m.x + 100)}+{(m.y + 100)}")
            else:
                root.geometry("800x600+100+100")
        except Exception:
            root.geometry("800x600+100+100")
            
        root.attributes("-topmost", True)
        
        # SLM 逻辑画布 (通常 SLM 是 1920x1080)
        canvas = tk.Canvas(root, width=1920, height=1080, bg="black", highlightthickness=0)
        canvas.pack(fill="both", expand=True)
        logger.info(f"[SLM-Engine] 模拟器窗口已在进程 {os.getpid()} 中启动")

    def process_loop():
        """渲染循环：监听来自主进程的指令队列"""
        try:
            while not cmd_queue.empty():
                payload = cmd_queue.get_nowait()
                
                # 收到 None 信号，退出进程
                if payload is None:
                    root.quit()
                    return

                if is_simulator:
                    if payload.content == "BLACK" or payload.name == "CLEAR":
                        canvas.delete("all")
                        logger.info("[SLM-Engine] 屏幕已清空")
                    else:
                        # 模拟模式：加载图片并缩放到 SLM 标准分辨率
                        pil_img = Image.open(payload.content).resize((1920, 1080))
                        tk_img = ImageTk.PhotoImage(pil_img)
                        cache["tk_img"] = tk_img # 保持引用
                        
                        canvas.delete("all")
                        canvas.create_image(0, 0, anchor="nw", image=tk_img)
                        logger.info(f"[SLM-Engine] 模拟显示更新: {os.path.basename(payload.content)}")
                else:
                    # TODO: 此处未来接入 HOLOEYE HEDS SDK 硬件逻辑
                    # self.slm.loadImageDataFromFile(payload.content)
                    pass
                    
        except queue.Empty:
            pass
        except Exception as e:
            logger.error(f"[SLM-Engine] 渲染异常: {e}")
        
        # 每 ~16ms 检查一次指令 (约 60FPS)
        root.after(16, process_loop)

    if is_simulator:
        root.after(100, process_loop)
        root.mainloop()

class SLMDisplay(BaseDisplayDevice):
    """
    SLM 显示设备实现类
    """
    def __init__(self, monitor_idx: int, is_simulator: bool = True):
        super().__init__(monitor_idx)
        self.is_simulator = is_simulator
        
        # 针对 macOS 的 'spawn' 启动模式进行适配
        self.ctx = multiprocessing.get_context('spawn')
        self.cmd_queue = self.ctx.Queue(maxsize=10)
        self.process = None

    def initialize(self):
        """
        启动独立的渲染进程
        """
        if self.process and self.process.is_alive():
            return

        # 核心：target 指向顶层函数，避免 pickle 错误
        self.process = self.ctx.Process(
            target=_slm_render_engine_process,
            args=(self.monitor_idx, self.cmd_queue, self.is_simulator),
            daemon=True
        )
        self.process.start()
        self.is_active = True
        logger.success(f"SLM 进程已就绪 (PID: {self.process.pid})")

    def show(self, payload: DisplayPayload):
        """
        发送显示指令
        """
        if not self.is_active:
            logger.warning("SLM 进程尚未初始化")
            return
            
        try:
            self.cmd_queue.put_nowait(payload)
        except queue.Full:
            # 丢弃旧指令，确保实时性
            try:
                self.cmd_queue.get_nowait()
                self.cmd_queue.put_nowait(payload)
            except:
                pass

    def clear(self):
        """
        显示黑屏
        """
        self.show(DisplayPayload(content="BLACK", name="CLEAR"))

    # def close(self):
    #     """
    #     安全释放资源
    #     """
    #     if self.process:
    #         try:
    #             if self.process.is_alive():
    #                 self.cmd_queue.put(None) # 发送退出信号
    #                 self.process.join(timeout=1)
    #         except Exception:
    #             pass
                
    #         if self.process.is_alive():
    #             self.process.terminate()
            
    #         self.is_active = False
    #         logger.info("SLM 进程已关闭")

    def close(self):
        """安全释放资源，防止信号量泄露"""
        if self.process and self.process.is_alive():
            try:
                # 1. 尝试发送 None 退出信号
                self.cmd_queue.put(None, timeout=0.5)
                # 2. 给子进程一点时间处理退出
                self.process.join(timeout=1.0)
            except Exception:
                pass
            
            # 3. 强制终止并关闭队列 (这是消除 leaked semaphore 的关键)
            if self.process.is_alive():
                self.process.terminate()
            
            # 4. 显式关闭队列，通知资源管理器释放信号量
            self.cmd_queue.close()
            self.cmd_queue.join_thread() # 等待队列后台线程结束
            
            self.is_active = False
            logger.info("SLM 进程及通信队列已安全关闭")