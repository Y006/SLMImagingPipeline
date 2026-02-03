import multiprocessing
import queue
import time
import os
import sys
import platform
import tkinter as tk
from typing import Optional, List, Tuple
from loguru import logger
from PIL import Image, ImageTk

# 引入基类
from .base_display import BaseDisplayDevice, DisplayPayload

try:
    from screeninfo import get_monitors, Monitor
    import customtkinter as ctk
    ctk.set_appearance_mode("Dark")
except ImportError:
    pass

IS_MACOS = platform.system() == "Darwin"

# =========================================================================
# 1. 独立进程：显示器渲染循环 (保持不变)
# =========================================================================
def _run_monitor_process(monitor_idx: int, 
                         manual_width_cm: Optional[float], 
                         cmd_queue: multiprocessing.Queue, 
                         info_queue: multiprocessing.Queue,
                         custom_x, custom_y):
    try:
        # 获取屏幕信息
        monitors = get_monitors()
        if monitor_idx >= len(monitors):
            logger.error(f"[Process] Monitor index {monitor_idx} out of range")
            return

        m = monitors[monitor_idx]
        target_x, target_y = custom_x, custom_y
        target_w, target_h = m.width, m.height
        
        logger.debug(f"[子进程-硬件] 屏幕硬件起始坐标: ({m.x}, {m.y})")
        logger.debug(f"[子进程-设置] 自定义窗口起始坐标: ({target_x}, {target_y})")
        logger.debug(f"[子进程-设置] 窗口尺寸: {target_w}x{target_h}")
        
        # 计算 PPCM: 这里使用主进程传过来的、经过确认的 manual_width_cm
        if manual_width_cm and manual_width_cm > 0:
            ppcm = target_w / manual_width_cm
            logger.debug(f"[子进程-PPCM] 使用手动宽度: {manual_width_cm}cm -> PPCM={ppcm:.4f}")
        else:
            # 理论上主进程 init 已经处理了所有情况，这里只是兜底
            width_mm = m.width_mm if m.width_mm else 300.0
            ppcm = target_w / (width_mm / 10.0)
        
        info_queue.put({"width": target_w, "height": target_h, "ppcm": ppcm})

        # 创建窗口
        root = ctk.CTk()
        root.title(f"Monitor {monitor_idx} Output")
        root.configure(fg_color="black")
        
        # 定位与全屏
        geometry_str = f"{target_w}x{target_h}+{target_x}+{target_y}"
        logger.info(f"[子进程-窗口] 设置 geometry: {geometry_str}")
        root.geometry(geometry_str)
        root.update_idletasks()
        
        if IS_MACOS:
            root.attributes("-fullscreen", True)
            root.overrideredirect(True)
            root.geometry(f"{target_w}x{target_h}+{target_x}+{target_y}")
        else:
            root.overrideredirect(True)
            root.state('zoomed')
            root.attributes("-fullscreen", True)

        root.attributes("-topmost", True)
        root.update_idletasks()
        
        # 验证窗口实际位置（关键调试信息）
        actual_x = root.winfo_rootx()
        actual_y = root.winfo_rooty()
        logger.debug(f"[实时校验] 窗口当前物理位置: ({actual_x}, {actual_y})")
        actual_w = root.winfo_width()
        actual_h = root.winfo_height()
        logger.warning(f"[子进程-验证] 窗口实际位置: ({actual_x}, {actual_y}), 实际尺寸: {actual_w}x{actual_h}")
        
        if actual_x != target_x or actual_y != target_y:
            logger.error(f"[子进程-偏差] ⚠️ 位置偏移检测: 期望({target_x},{target_y}) vs 实际({actual_x},{actual_y}) | Δx={actual_x-target_x}, Δy={actual_y-target_y}")
        else:
            logger.success(f"[子进程-验证] ✓ 窗口位置精确对齐")
        
        canvas = tk.Canvas(root, width=target_w, height=target_h, bg="black", highlightthickness=0)
        canvas.pack(fill="both", expand=True)
        root.update_idletasks()
        
        # 验证画布坐标系统
        canvas_x = canvas.winfo_rootx()
        canvas_y = canvas.winfo_rooty()
        logger.debug(f"[实时校验] 画布绝对原点: ({canvas_x}, {canvas_y})")
        canvas_w = canvas.winfo_width()
        canvas_h = canvas.winfo_height()
        logger.info(f"[子进程-画布] 画布全局坐标: ({canvas_x}, {canvas_y}), 画布尺寸: {canvas_w}x{canvas_h}")
        logger.info(f"[子进程-画布] 画布逻辑原点 (0,0) 对应全局屏幕坐标: ({canvas_x}, {canvas_y})")
        
        context = {"current_tk_image": None}

        # 核心渲染逻辑 (复刻)
        def internal_update_display(cmd: dict):
            try:
                img_path = cmd.get("img_path")
                size = cmd.get("size")
                pos = cmd.get("pos", (0, 0))
                units = cmd.get("units", "cm")
                # 强制将 anchor 改为 nw (左上角) 以便进行精确的物理对齐验证
                anchor = cmd.get("anchor", "nw") 

                if img_path is None or not os.path.exists(img_path):
                    logger.error(f"[Display] 文件未找到: {img_path}")
                    return

                pil_img = Image.open(img_path)
                orig_w, orig_h = pil_img.size
                
                # 1. 尺寸计算
                if units == 'cm' and size is not None:
                    target_w_px, target_h_px = size[0] * ppcm, size[1] * ppcm
                else:
                    target_w_px, target_h_px = float(orig_w), float(orig_h)
                
                final_w, final_h = int(round(target_w_px)), int(round(target_h_px))
                if final_w != orig_w or final_h != orig_h:
                    pil_img = pil_img.resize((final_w, final_h), Image.Resampling.LANCZOS)
                
                # 2. 核心：强制同步窗口状态，获取最新的绝对位置
                root.update_idletasks()
                abs_x, abs_y = canvas.winfo_rootx(), canvas.winfo_rooty()

                # 3. 计算全局目标像素位置 (绝对物理坐标)
                global_target_x = pos[0] * ppcm
                global_target_y = pos[1] * ppcm

                # 4. 计算画布内部偏移量
                draw_x = global_target_x - abs_x
                draw_y = global_target_y - abs_y
                
                logger.info(f"═══════ 坐标系统详细追踪 ═══════")
                logger.info(f"[输入参数] 物理目标={pos}cm, 锚点={anchor}")
                logger.info(f"[系统反馈] 画布实际位置={abs_x, abs_y}px")
                logger.info(f"[计算结果] 画布绘图坐标={draw_x:.1f, draw_y:.1f}px")
                
                # 边界验证：现在由于 anchor 是 nw，Left 应该严格等于 pos[0]
                logger.info(f"[物理验证] 最终屏幕左边缘: { (abs_x + draw_x)/ppcm :.3f} cm")
                logger.info(f"═══════════════════════════════")

                tk_img = ImageTk.PhotoImage(pil_img)
                context["current_tk_image"] = tk_img
                
                # 强制使用左上角锚点进行位置测试
                tk_anchor = tk.NW 
                
                canvas.delete("all")
                canvas.create_image(draw_x, draw_y, image=tk_img, anchor=tk_anchor)
                
                logger.success(f"文件: {os.path.basename(img_path)} | 物理对齐点: {pos}cm")

            except Exception as e:
                logger.error(f"[Display] 渲染异常: {e}")

        def check_queue():
            try:
                task = None
                while not cmd_queue.empty():
                    t = cmd_queue.get_nowait()
                    if t["type"] in ["STOP", "CLEAR"]: task = t; break 
                    task = t 
                if task:
                    if task["type"] == "STOP": root.quit(); return
                    elif task["type"] == "CLEAR": canvas.delete("all"); context["current_tk_image"] = None
                    elif task["type"] == "UPDATE": internal_update_display(task)
            except Exception: pass
            root.after(16, check_queue)

        root.after(100, check_queue)
        root.mainloop()
        
    except Exception as e:
        logger.error(f"[Process] Crash: {e}")
    finally:
        sys.exit(0)


# =========================================================================
# 2. 独立进程：交互式校准逻辑 (保持样式复刻)
# =========================================================================
def _run_calibration_process(monitor_idx: int, initial_guess: float, result_queue: multiprocessing.Queue):
    try:
        monitors = get_monitors()
        if monitor_idx >= len(monitors): return
        m = monitors[monitor_idx]
        
        root = tk.Tk()
        root.title("Calibration")
        root.configure(bg="black")
        
        root.geometry(f"{m.width}x{m.height}+{m.x}+{m.y}")
        if IS_MACOS:
            root.attributes("-fullscreen", True)
            root.overrideredirect(True)
        else:
            root.overrideredirect(True)
            root.attributes("-fullscreen", True)
        root.attributes("-topmost", True)
        
        canvas = tk.Canvas(root, width=m.width, height=m.height, bg="black", highlightthickness=0)
        canvas.pack(fill="both", expand=True)
        
        state = {"width_cm": initial_guess}
        
        def draw():
            canvas.delete("all")
            current_w = state["width_cm"]
            if current_w <= 0: state["width_cm"] = 1.0
            ppcm = m.width / state["width_cm"]
            box_px = 10.0 * ppcm 
            cx, cy = m.width / 2, m.height / 2
            
            canvas.create_rectangle(cx - box_px/2, cy - box_px/4, cx + box_px/2, cy + box_px/4, outline='red', width=3)
            start_y = cy + box_px/4 + 10
            for i in range(-5, 6):
                x = cx + i * ppcm
                h = 30 if i % 5 == 0 else 15
                canvas.create_line(x, start_y, x, start_y + h, fill='white')
                if i % 5 == 0:
                    canvas.create_text(x, start_y + 45, text=str(abs(i)), fill='white', font=('Arial', 12))
            
            msg = f"Current width: {state['width_cm']:.2f} cm\n[Left/Right]: Fine | [Shift]: Coarse\n[Enter]: confirm | [Esc]: cancel"
            canvas.create_text(cx, cy - box_px/2 - 60, text=msg, fill="white", font=("Arial", 16), justify="center")

        def on_key(e):
            step = 0.01
            key = e.keysym.lower() # 统一使用 key
            
            # 检测修饰键
            is_shift = (e.state & 0x1) or 'shift' in key
            is_ctrl = (e.state & 0x4) or 'control' in key or 'command' in key
            
            if is_ctrl: step = 1.0
            elif is_shift: step = 0.1
            
            if key == 'left': 
                state["width_cm"] -= step
            elif key == 'right': 
                state["width_cm"] += step
            elif key == 'return': 
                result_queue.put(state["width_cm"])
                root.quit()
            elif key == 'escape': # 修正这里的变量名为 key
                result_queue.put(None)
                root.quit()
            
            draw()

        root.bind("<Key>", on_key)
        root.focus_force()
        root.after(100, draw)
        root.mainloop()
        
    except Exception as e:
        logger.error(f"[Calib] Error: {e}")
    finally:
        sys.exit(0)


# =========================================================================
# 3. 主控制类 (Client) - 已应用你的逻辑
# =========================================================================
class MonitorDisplay(BaseDisplayDevice):
    def __init__(self, 
                 monitor_idx: int, 
                 width_cm: Optional[float] = None, 
                 resolution_x: Optional[int] = None, 
                 bg_color: str = 'black',
                 pos_x: Optional[int] = None,
                 pos_y: Optional[int] = None):
        super().__init__(monitor_idx)
        self.manual_width_cm = None
        self.process = None
        self.cmd_queue = None
        self.info_queue = None
        self.ppcm = 1.0
        
        # --- 1. 获取显示器信息 ---
        try:
            all_monitors = get_monitors()
            if monitor_idx < 0 or monitor_idx >= len(all_monitors):
                raise ValueError(f"索引 {monitor_idx} 无效")
            self.target_monitor = all_monitors[monitor_idx]
        except Exception as e:
            logger.critical(f"硬件错误: {e}")
            raise

        self.target_pos_x = pos_x
        self.target_pos_y = pos_y

        if self.target_pos_x is None or self.target_pos_y is None:
            logger.info("-" * 40)
            logger.info(f"配置显示位置 (当前默认: {self.target_monitor.x}, {self.target_monitor.y})")
            pos_choice = input(">>> 是否手动输入起始坐标？(y/n, 默认n): ").strip().lower()
            
            if pos_choice == 'y':
                try:
                    self.target_pos_x = int(input(">>> 请输入 X 坐标: ").strip())
                    self.target_pos_y = int(input(">>> 请输入 Y 坐标: ").strip())
                except ValueError:
                    logger.warning("输入无效，将使用硬件默认位置。")
                    self.target_pos_x = self.target_monitor.x
                    self.target_pos_y = self.target_monitor.y
            else:
                self.target_pos_x = self.target_monitor.x
                self.target_pos_y = self.target_monitor.y

        # --- 2. 分辨率设置 ---
        if resolution_x:
            self.width_px = resolution_x
            logger.info(f"分辨率: 手动指定 {self.width_px}")
        else:
            self.width_px = self.target_monitor.width
            logger.info(f"分辨率: 自动检测 {self.width_px}x{self.target_monitor.height}")

        # --- 3. 物理宽度判定逻辑 (支持反复输入) ---
        final_width = None
        
        # A. 检查参数直接传入
        if width_cm is not None and width_cm > 0:
            final_width = width_cm
            logger.info(f"物理宽度: 使用参数值 {final_width} cm")
        
        # B. 检查硬件 EDID (避开 macOS)
        if final_width is None:
            detected_mm = self.target_monitor.width_mm
            if detected_mm and detected_mm > 0 and platform.system() != 'Darwin':
                final_width = detected_mm / 10.0
                logger.warning(f"物理宽度: 使用硬件报告值 {final_width} cm")

        # C. 交互式循环
        if final_width is None:
            while True:
                logger.info("-" * 40)
                logger.warning(f"当前显示器 {monitor_idx} 物理尺寸未知")
                print("  选项 A: 直接输入数值 (例如 28.65)")
                print("  选项 B: 输入 'c' 启动可视化标定工具")
                print("  选项 C: 输入 'q' 退出程序")
                
                user_input = input(">>> 请输入指令或数值: ").strip().lower()
                
                if user_input == 'c':
                    calib_result = self.run_calibration_ui(monitor_idx)
                    if calib_result is not None:
                        final_width = calib_result
                        logger.success(f"标定成功！宽度: {final_width:.2f} cm")
                        break # 成功标定，跳出循环
                    else:
                        logger.warning("标定工具已关闭，未保存结果。")
                        continue # 回到循环开始
                elif user_input == 'q':
                    logger.error("用户选择退出程序。")
                    sys.exit(0)
                else:
                    try:
                        val = float(user_input)
                        if val > 0:
                            final_width = val
                            logger.success(f"已手动设为: {final_width} cm")
                            break # 成功输入，跳出循环
                    except ValueError:
                        logger.error(f"无效输入: '{user_input}'，请重新输入数值或 'c'")
        
        self.manual_width_cm = final_width
        # 预计算 PPCM
        self.ppcm = self.width_px / self.manual_width_cm

    def initialize(self):
        if self.is_active: return
        self.cmd_queue = multiprocessing.Queue()
        self.info_queue = multiprocessing.Queue()
        # 修复：显式传递坐标 offset
        self.process = multiprocessing.Process(
            target=_run_monitor_process,
            args=(self.monitor_idx, self.manual_width_cm, self.cmd_queue, self.info_queue,
                  self.target_pos_x, self.target_pos_y),
            daemon=True
        )
        self.process.start()
        
        try:
            info = self.info_queue.get(timeout=5.0)
            self.width_px = info["width"]
            self.height_px = info["height"]
            self.ppcm = info["ppcm"] # 子进程会根据 manual_width_cm 重新计算精确 PPCM
            
            self.is_active = True
            logger.info(f"[Monitor] Active (PID: {self.process.pid}) | PPCM: {self.ppcm:.2f}")
        except queue.Empty:
            logger.error("[Monitor] Init Timeout.")
            self.close()

    def show(self, payload: DisplayPayload):
        if not self.is_active: return
        cmd = {
            "type": "UPDATE",
            "img_path": payload.content,
            "size": payload.target_size_cm,
            "pos": payload.position,
            "units": "cm",
            "anchor": payload.anchor
        }
        self.cmd_queue.put(cmd)

    def clear(self):
        if self.is_active: self.cmd_queue.put({"type": "CLEAR"})

    def close(self):
        if self.is_active:
            self.cmd_queue.put({"type": "STOP"})
            time.sleep(0.5)
            if self.process.is_alive(): self.process.terminate()
            self.is_active = False
            logger.info("[Monitor] Released.")

    @staticmethod
    def scan_monitors() -> List[Monitor]:
        return get_monitors()

    @staticmethod
    def run_calibration_ui(monitor_idx: int) -> Optional[float]:
        """[阻塞式] 启动校准进程"""
        logger.info(f"Starting Calibration on Monitor {monitor_idx}...")
        try:
            m = get_monitors()[monitor_idx]
            guess = m.width_mm / 10.0 if m.width_mm else 30.0
        except:
            guess = 30.0

        res_queue = multiprocessing.Queue()
        p = multiprocessing.Process(target=_run_calibration_process, args=(monitor_idx, guess, res_queue))
        p.start()
        p.join()
        
        if not res_queue.empty():
            return res_queue.get()
        return None