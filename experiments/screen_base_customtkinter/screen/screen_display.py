import threading
import queue
import time
import sys
import os
import gc
from dataclasses import dataclass
from typing import Optional, Tuple, List

# 立即加载所有依赖库
from loguru import logger
import sys
import threading
import queue
import time
import os
import gc
from dataclasses import dataclass
from typing import Optional, Tuple, List

from .base_display import BaseDisplayDevice, DisplayPayload

try:
    from screeninfo import get_monitors, Monitor
    import customtkinter as ctk
    import tkinter as tk
    from PIL import Image, ImageTk
    
    # 初始化配置
    ctk.set_appearance_mode("Dark")
    ctk.set_default_color_theme("blue")
    
except ImportError as e:
    print(f"缺少必要库 ({e})，请安装: pip install customtkinter pillow screeninfo loguru")
    sys.exit(1)

@dataclass
class DisplayCommand:
    cmd_type: str = "UPDATE"
    img_path: Optional[str] = None
    size: Optional[Tuple[Optional[float], Optional[float]]] = None
    pos: Tuple[float, float] = (0, 0)
    units: str = "cm"
    anchor: str = "center"

class MonitorDisplay(BaseDisplayDevice):
    """
    异步高精度显示控制器 (CustomTkinter 修复版)
    """
    _main_window_ref = None
    
    @staticmethod
    def scan_monitors() -> List[Monitor]:
        try:
            monitors_list = get_monitors()
            logger.info(f"检测到 {len(monitors_list)} 个显示器。")
            for i, m in enumerate(monitors_list):
                w_cm = f"{m.width_mm/10:.1f}cm" if m.width_mm else "未知"
                logger.info(f"  [{i}] {m.name} | 分辨率: {m.width}x{m.height} | 物理宽: {w_cm}")
            return monitors_list
        except Exception as e:
            logger.error(f"扫描屏幕失败: {e}")
            return []

    @staticmethod
    def run_calibration_ui(monitor_idx: int, initial_guess: float = 28.89) -> Optional[float]:
        """[阻塞式] 启动可视化校准界面 (全屏修复版)"""
        logger.info(f"正在启动校准界面... 目标屏幕: {monitor_idx}")
        
        try:
            target_mon = get_monitors()[monitor_idx]
            logic_w = target_mon.width
            logic_h = target_mon.height
            mon_x = target_mon.x
            mon_y = target_mon.y
        except:
            logic_w, logic_h = 1920, 1080
            mon_x, mon_y = 0, 0
            
        win = None
        current_width_cm: List[float] = [initial_guess]
        result: List[Optional[float]] = [None]
        state = {"running": True, "after_id": None}
        
        try:
            # [调试] 切换为原生 tk.Tk 以隔离 CustomTkinter 可能的后台任务干扰
            # 原理：ctk.CTk() 可能包含主题/缩放的后台轮询任务，在快速销毁时易引发 invalid command 错误
            logger.debug(f"[Trace] Creating Calibration Window (Native Tk)...")
            win = tk.Tk()
            win.title("Calibration")
            
            # 捕获 Tkinter 内部回调异常
            def report_callback_exception(exc, val, tb):
                import traceback
                logger.error(f"[TkError] {val}\n{''.join(traceback.format_tb(tb))}")
            win.report_callback_exception = report_callback_exception
            
            # === [核心修复] 强制全屏逻辑 ===
            # 1. 尝试使用操作系统级的全屏属性
            win.attributes("-fullscreen", True)
            
            # 2. 依然设置几何位置，确保在多屏系统中定位到正确的屏幕
            win.geometry(f"{logic_w}x{logic_h}+{mon_x}+{mon_y}")
            
            # 3. 辅助设置：去边框 + 置顶 (双重保险)
            win.overrideredirect(True)
            win.attributes("-topmost", True)
            
            # 4. 强制更新并抢占焦点
            win.update_idletasks()
            win.focus_force()
            # ===============================
            
            canvas = tk.Canvas(win, width=logic_w, height=logic_h, bg='black', highlightthickness=0)
            canvas.pack(fill="both", expand=True)
            
            def draw():
                if not state["running"]: return
                try:
                    if not win.winfo_exists(): return
                except: return

                try:
                    canvas.delete("all")
                    ppcm = logic_w / current_width_cm[0]
                    box_px = 10.0 * ppcm
                    
                    # 重新计算中心点 (使用 win.winfo_width 确保动态居中)
                    center_x = win.winfo_screenwidth() / 2 if win.winfo_screenwidth() > 0 else logic_w / 2
                    center_y = win.winfo_screenheight() / 2 if win.winfo_screenheight() > 0 else logic_h / 2
                    
                    # 绘制中心矩形
                    canvas.create_rectangle(
                        center_x - box_px/2, center_y - box_px/4,
                        center_x + box_px/2, center_y + box_px/4,
                        outline='red', width=3
                    )
                    
                    # 绘制刻度
                    range_cm = int(current_width_cm[0] / 2)
                    start_y = center_y + box_px/4 + 20
                    
                    for i in range(-range_cm, range_cm + 1):
                        x = center_x + (i * ppcm)
                        # 简单的裁剪优化
                        if x < 0 or x > logic_w: continue
                        
                        h = 30 if i % 5 == 0 else 15
                        canvas.create_line(x, start_y, x, start_y + h, fill='white')
                        if i % 5 == 0:
                            canvas.create_text(x, start_y + 45, text=str(abs(i)), fill='white', font=('Arial', 12))

                    info_str = (
                        f"Current width: {current_width_cm[0]:.2f} cm\n"
                        f"[←/→]: ±0.01 | [Shift]: ±0.1 | [Ctrl]: ±1.0 cm\n"
                        f"[Enter]: confirm  [Esc]: cancel"
                    )
                    canvas.create_text(center_x, center_y - box_px/2 - 60, text=info_str, fill='white', font=('Arial', 16), justify='center')
                except Exception:
                    pass

            def on_key(event):
                if not state["running"]: return

                key = event.keysym.lower()
                
                # [状态检测增强版]
                # 1. Shift 检测 (0x1)
                is_shift = (event.state & 0x1) or 'shift' in key
                
                # 2. 粗调检测: 支持 Control(0x4) 或 Command(0x8/0x10/0x20)
                # macOS 用户经常混用 Ctrl/Command，此处全部兼容以防万一
                # 注意: 0x8 只有在非 Shift/Ctrl 组合下才准确，这里做宽泛匹配
                state_mask = event.state
                is_coarse = (state_mask & 0x4) or (state_mask & 0x8) or (state_mask & 0x10) or 'control' in key or 'meta' in key or 'command' in key
                
                step = 0.01
                if is_coarse:
                    step = 1.0  # 粗调 1cm
                elif is_shift:
                    step = 0.1  # 中调 0.1cm
                
                if key == 'left':
                    current_width_cm[0] -= step
                    draw()
                elif key == 'right':
                    current_width_cm[0] += step
                    draw()
                elif key == 'return':
                    result[0] = current_width_cm[0]
                    state["running"] = False
                    # 取消尚未执行的定时任务
                    if state["after_id"]:
                        try: win.after_cancel(state["after_id"])
                        except: pass
                    win.quit()
                elif key == 'escape':
                    result[0] = None
                    state["running"] = False
                    # 取消尚未执行的定时任务
                    if state["after_id"]:
                        try: win.after_cancel(state["after_id"])
                        except: pass
                    win.quit()

            win.bind("<Key>", on_key)
            
            # 延时一小段时间再绘图，确保全屏切换完成
            state["after_id"] = win.after(100, draw)
            
            logger.info("校准界面已就绪，请按键盘...")
            logger.debug(f"[Trace] Entering mainloop...")
            win.mainloop()
            logger.debug(f"[Trace] Exited mainloop.")
            
            state["running"] = False
            
            # 必须确保在 destroy 之前处理掉所有 pending 的 idle tasks
            # 这一步可以防止销毁过程中触发排队的 update
            try:
                if win.winfo_exists():
                    win.update_idletasks()
            except: pass

            logger.debug(f"[Trace] Destroying window...")
            win.destroy()
            
            # 手动 pump 一次 events 确保 Tcl 处理完销毁消息 (缓解僵尸事件)
            try:
                from typing import cast
                _tk = cast(object, tk)
                if hasattr(_tk, "_default_root"):
                    _tk._default_root = None  # type: ignore[attr-defined]
            except Exception:
                pass
            
            logger.debug(f"[Trace] Window destroyed successfully.")
            return result[0]
                        
        except Exception as e:
            logger.error(f"校准界面异常: {e}")
            if win: 
                try: win.destroy()
                except: pass
            return None
        finally:
            gc.collect()
    
    
    def __init__(self, 
                 monitor_idx: int, 
                 width_cm: Optional[float] = None, 
                 resolution_x: Optional[int] = None, 
                 bg_color: str = 'black'):
        
        try:
            all_monitors = get_monitors()
            if monitor_idx < 0 or monitor_idx >= len(all_monitors):
                raise ValueError(f"索引 {monitor_idx} 无效")
            self.target_monitor = all_monitors[monitor_idx]
        except Exception as e:
            logger.critical(f"硬件错误: {e}")
            raise

        if resolution_x:
            self.res_x = resolution_x
            self.res_y = 1080 
            logger.info(f"分辨率: 手动指定 {self.res_x}")
        else:
            self.res_x = self.target_monitor.width
            self.res_y = self.target_monitor.height
            logger.info(f"分辨率: 自动检测 {self.res_x}x{self.res_y}")

        final_width = None
        if width_cm is not None and width_cm > 0:
            final_width = width_cm
            logger.info(f"物理宽度: 使用参数值 {final_width} cm")
        
        if final_width is None:
            detected_mm = self.target_monitor.width_mm
            import platform
            if detected_mm and detected_mm > 0 and platform.system() != 'Darwin':
                final_width = detected_mm / 10.0
                logger.warning(f"物理宽度: 使用硬件报告值 {final_width} cm")

        if final_width is None:
            logger.warning(f"无法自动获取显示器 {monitor_idx} 的物理尺寸。")
            logger.info("选项 A: 输入数值 (例如 28.65)")
            logger.info("选项 B: 输入 'c' 启动标定工具")
            
            while True:
                user_input = input(">>> 请输入指令或数值: ").strip().lower()
                
                if user_input == 'c':
                    calib_result = self.run_calibration_ui(monitor_idx)
                    if calib_result is not None:
                        final_width = calib_result
                        logger.success(f"标定完成！宽度: {final_width:.2f} cm")
                        break
                    else:
                        logger.warning("标定已取消")
                        continue
                else:
                    try:
                        val = float(user_input)
                        if val > 0:
                            final_width = val
                            break
                    except ValueError:
                        pass

        self.width_cm = final_width
        self.ppcm = self.res_x / self.width_cm
        self.bg_color = bg_color
        self.monitor_idx = monitor_idx
        self.win_x = self.target_monitor.x
        self.win_y = self.target_monitor.y
        
        # 设置基类所需的属性
        self.width_px = self.res_x
        self.height_px = self.res_y

        self.cmd_queue = queue.Queue(maxsize=5)
        self.stop_event = threading.Event()
        self.engine_ready = threading.Event()
        self.current_tk_image = None
        self._after_id = None
        
    def initialize(self):
        """初始化设备并启动渲染循环"""
        try:
            self._main_window_ref = ctk.CTk()
            self._main_window_ref.title("AsyncPrecisionDisplay")
            
            # === [核心修复] 这里的全屏逻辑需要与校准界面保持一致 ===
            # 1. 显式开启全屏属性 (覆盖任务栏的关键)
            self._main_window_ref.attributes("-fullscreen", True)
            
            # 2. 定位到指定屏幕
            geo_str = f"{self.res_x}x{self.res_y}+{self.win_x}+{self.win_y}"
            self._main_window_ref.geometry(geo_str)
            
            # 3. 辅助设置
            self._main_window_ref.update_idletasks() # 修复启动白屏
            self._main_window_ref.overrideredirect(True)
            self._main_window_ref.attributes("-topmost", True) # 确保渲染在最前
            self._main_window_ref.configure(fg_color=self.bg_color)
            
            # 4. 强制抢占焦点
            self._main_window_ref.focus_force()
            # ==================================================
            
            self.canvas = tk.Canvas(
                self._main_window_ref, 
                width=self.res_x, 
                height=self.res_y, 
                bg=self.bg_color, 
                highlightthickness=0
            )
            self.canvas.pack(fill="both", expand=True)

            logger.success(f"引擎启动 | PPCM: {self.ppcm:.2f}")
            self.engine_ready.set()
            
            self._main_window_ref.bind("<Escape>", lambda e: self.close())
            self._process_queue_loop()
            self._main_window_ref.mainloop()

        except Exception as e:
            logger.critical(f"渲染循环崩溃: {e}")
            import traceback
            logger.error(traceback.format_exc())
        finally:
            self.engine_ready.clear()
            if self._main_window_ref:
                try:
                    if self._after_id:
                        self._main_window_ref.after_cancel(self._after_id)
                except Exception:
                    pass
                
                try: self._main_window_ref.destroy()
                except: pass
                
            self._main_window_ref = None # 避免持有引用
            gc.collect()
            logger.info("引擎关闭")

    def _process_queue_loop(self):
        # 如果已设置停止标志，或者窗口已销毁，则停止循环
        if self.stop_event.is_set():
            try:
                if self._main_window_ref and self._main_window_ref.winfo_exists():
                    self._main_window_ref.quit()
            except Exception:
                pass
            return

        try:
            # 检查窗口是否还存在
            if not self._main_window_ref or not self._main_window_ref.winfo_exists():
                return
            
            latest_update_cmd = None
            needs_clear = False
            
            while not self.cmd_queue.empty():
                cmd = self.cmd_queue.get_nowait()
                if cmd.cmd_type == "UPDATE":
                    latest_update_cmd = cmd
                    needs_clear = False 
                elif cmd.cmd_type == "CLEAR":
                    needs_clear = True
                    latest_update_cmd = None

            if needs_clear:
                self.canvas.delete("all")
                self.current_tk_image = None
                
            if latest_update_cmd:
                self._update_display(latest_update_cmd)

            # 再次检查窗口状态并安排下一次执行
            if self._main_window_ref and self._main_window_ref.winfo_exists():
                self._after_id = self._main_window_ref.after(16, self._process_queue_loop)

        except queue.Empty:
            pass
        except Exception as e:
            # 忽略 Tcl/Tk 这里的特定错误，防止控制台刷屏
            if "invalid command name" not in str(e):
                logger.error(f"处理循环错误: {e}")

    def close(self):
        """释放资源"""
        self.stop_event.set()

    def show(self, payload: DisplayPayload):
        """发送显示指令到队列 (DisplayPayload)"""
        cmd = DisplayCommand("UPDATE", payload.content, payload.target_size_cm, payload.position, 'cm', payload.anchor)
        try:
            self.cmd_queue.put_nowait(cmd)
        except queue.Full:
            try:
                self.cmd_queue.get_nowait()
                self.cmd_queue.put_nowait(cmd)
            except: pass

    def show_image(self, 
                   img_path: str, 
                   size: Optional[Tuple[Optional[float], Optional[float]]] = None, 
                   pos: Tuple[float, float] = (0, 0), 
                   units: str = 'cm', 
                   anchor: str = 'center'):
        if not os.path.exists(img_path):
            logger.error(f"Missing file: {img_path}")
            return
        cmd = DisplayCommand("UPDATE", img_path, size, pos, units, anchor)
        try:
            self.cmd_queue.put_nowait(cmd)
        except queue.Full:
            try:
                self.cmd_queue.get_nowait()
                self.cmd_queue.put_nowait(cmd)
            except: pass

    def clear(self):
        """清空显示"""
        self.cmd_queue.put(DisplayCommand(cmd_type="CLEAR"))

    def _update_display(self, cmd: DisplayCommand):
        try:
            if cmd.img_path is None:
                return

            pil_img = Image.open(cmd.img_path)
            orig_w_px, orig_h_px = pil_img.size
            if orig_w_px == 0 or orig_h_px == 0: return

            w_px, h_px = float(orig_w_px), float(orig_h_px)
            
            if cmd.size is not None:
                req_w = cmd.size[0] if len(cmd.size) > 0 else None
                req_h = cmd.size[1] if len(cmd.size) > 1 else None
                
                target_w = req_w * self.ppcm if (req_w is not None and cmd.units == 'cm') else req_w
                target_h = req_h * self.ppcm if (req_h is not None and cmd.units == 'cm') else req_h
                
                aspect = orig_w_px / orig_h_px
                
                if target_w is not None and target_h is not None:
                    w_px, h_px = target_w, target_h
                elif target_w is not None:
                    w_px = target_w
                    h_px = w_px / aspect
                elif target_h is not None:
                    h_px = target_h
                    w_px = h_px * aspect
            
            # 使用四舍五入代替直接截断，提高尺寸转换精度
            final_w_px = int(round(w_px))
            final_h_px = int(round(h_px))
            
            # 计算量子化误差 (理论值 vs 实际整数像素值)
            err_w_um = (final_w_px - w_px) / self.ppcm * 10000 
            err_h_um = (final_h_px - h_px) / self.ppcm * 10000
            
            if final_w_px != orig_w_px or final_h_px != orig_h_px:
                pil_img = pil_img.resize((final_w_px, final_h_px), Image.Resampling.LANCZOS)

            # 优化日志显示：增加误差提示
            actual_w_cm = final_w_px/self.ppcm
            actual_h_cm = final_h_px/self.ppcm
            
            msg = (f"显示: {final_w_px}x{final_h_px}px ({actual_w_cm:.3f}x{actual_h_cm:.3f}cm) | "
                   f"误差: Δw={err_w_um:+.1f}um")
            logger.success(f"{msg} | {os.path.basename(cmd.img_path)}")
            
            # 坐标对齐逻辑修正：使用 winfo_rootx/y 获取画布实时绝对位置
            self._main_window_ref.update_idletasks()
            abs_x = self.canvas.winfo_rootx()
            abs_y = self.canvas.winfo_rooty()
            
            if cmd.units == 'cm':
                pos_x_px = (cmd.pos[0] * self.ppcm) - abs_x
                pos_y_px = (cmd.pos[1] * self.ppcm) - abs_y
            else:
                pos_x_px = cmd.pos[0] - abs_x
                pos_y_px = cmd.pos[1] - abs_y

            tk_img = ImageTk.PhotoImage(pil_img)
            self.current_tk_image = tk_img 
            
            tk_anchor = tk.CENTER
            if cmd.anchor == 'top-left': tk_anchor = tk.NW
            elif cmd.anchor == 'top-right': tk_anchor = tk.NE
            elif cmd.anchor == 'bottom-left': tk_anchor = tk.SW
            elif cmd.anchor == 'bottom-right': tk_anchor = tk.SE
            elif cmd.anchor == 'center': tk_anchor = tk.CENTER
            
            self.canvas.delete("all")
            self.canvas.create_image(pos_x_px, pos_y_px, image=tk_img, anchor=tk_anchor)
            
        except Exception as e:
            logger.error(f"Image Render Error: {e}")
            import traceback
            logger.error(traceback.format_exc())