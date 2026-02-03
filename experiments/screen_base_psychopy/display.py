import threading
import queue
import time
import sys
import os
import gc
from dataclasses import dataclass
from typing import Optional, Tuple, List

# 立即加载轻量级库
from loguru import logger
try:
    from screeninfo import get_monitors, Monitor
except ImportError:
    print("缺少必要库，请安装: pip install psychopy loguru screeninfo")
    sys.exit(1)

# 自定义 loguru 日志格式
# logger.remove()
# logger.add(sys.stderr, format="<green>{time:HH:mm:ss}</green> | <level>{message}</level>")

# 延迟加载占位符 (实现秒开的核心)
visual = None
core = None
event = None
monitors = None
psy_logging = None

# psychopy 的懒加载机制
def lazy_import_psychopy():
    """
    [核心机制] 只在真正需要显示时才加载 PsychoPy
    """
    global visual, core, event, monitors, psy_logging
    
    # 如果已经加载过，直接返回
    if visual is not None: 
        return

    logger.info("正在初始化显示引擎 (加载 PsychoPy)...")
    t0 = time.time()
    
    # [优化] 禁用音频驱动
    from psychopy import prefs
    prefs.hardware['audioLib'] = [] 
    
    # [优化] 屏蔽 Pygame 广告
    os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"
    
    # [加载]
    from psychopy import visual as v, core as c, event as e, monitors as m, logging as l
    
    # [静音] 只显示崩溃级错误
    l.console.setLevel(l.CRITICAL)
    
    # [赋值]
    visual, core, event, monitors, psy_logging = v, c, e, m, l
    
    logger.info(f"引擎加载完成，耗时 {time.time()-t0:.2f}s")

# 数据结构

@dataclass
class DisplayCommand:
    cmd_type: str = "UPDATE"
    img_path: Optional[str] = None
    size: Optional[Tuple[float, float]] = None
    pos: Optional[Tuple[float, float]] = (0, 0)
    units: str = "cm"
    anchor: str = "center"


# 核心类

class AsyncPrecisionDisplay:
    """
    异步高精度显示控制器 (Final Stable Version)
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
        """[阻塞式] 启动可视化校准界面"""
        
        # >>> 关键修复：确保进入 UI 前加载库 <<<
        lazy_import_psychopy()
        # ====================================

        logger.info(f"正在启动校准界面... 目标屏幕: {monitor_idx}")
        
        try:
            target_mon = get_monitors()[monitor_idx]
            logic_w = target_mon.width
            logic_h = target_mon.height
        except:
            logic_w, logic_h = 1920, 1080
            
        win = None
        try:
            # 创建临时配置消除警告
            temp_mon = monitors.Monitor(name='calib_temp')
            temp_mon.setWidth(initial_guess)
            temp_mon.setSizePix((logic_w, logic_h))
            temp_mon.save()

            win = visual.Window(
                screen=monitor_idx, fullscr=True, size=[logic_w, logic_h],
                monitor=temp_mon, color='black', units='pix',
                allowGUI=False, waitBlanking=True, useFBO=True, viewScale=None
            )
            
            real_w, real_h = win.size
            scale_factor = real_w / logic_w
            draw_logic_h = real_h / scale_factor
            
            current_width_cm = initial_guess
            
            while True:
                ppcm = logic_w / current_width_cm
                
                # 绘制刻度
                rulers = []
                box_px = 10.0 * ppcm
                start_y = - (box_px / 4) - 40
                
                rect = visual.Rect(win, width=box_px, height=box_px/2, pos=(0,0), lineColor='red', lineWidth=3)
                
                # 优化绘制：只画屏幕范围内的刻度
                range_cm = int(current_width_cm / 2)
                for i in range(-range_cm, range_cm + 1):
                    x = i * ppcm
                    if abs(x) > logic_w/2: continue 
                    
                    h = 30 if i % 5 == 0 else 15
                    line = visual.Line(win, start=(x, start_y), end=(x, start_y-h), lineColor='white')
                    rulers.append(line)
                    
                    if i % 5 == 0:
                        txt = visual.TextStim(win, text=str(abs(i)), pos=(x, start_y-50), height=20, font='Arial')
                        rulers.append(txt)

                info_str = (
                    f"Current width:{current_width_cm:.2f} cm\n"
                    f"[←/→]: ±0.01 cm  [Shift]: ±0.1 cm\n"
                    f"[Enter]: confirm  [Esc]: cancel"
                )
                msg = visual.TextStim(win, text=info_str, pos=(0, draw_logic_h/2 - 120), height=24, font='Arial')
                
                rect.draw()
                msg.draw()
                for r in rulers: r.draw()
                win.flip()
                
                keys = event.getKeys(modifiers=True)
                if not keys: continue
                
                for k, mods in keys:
                    step = 0.1 if mods['shift'] else 0.01
                    if k == 'left': current_width_cm -= step
                    elif k == 'right': current_width_cm += step
                    elif k == 'return':
                        win.close()
                        return current_width_cm
                    elif k == 'escape':
                        win.close()
                        return None
                        
        except Exception as e:
            logger.error(f"校准界面异常: {e}")
            if win: win.close()
            return None
        finally:
            if win: del win
            gc.collect()
            # 只有 core 加载成功了才调用 wait，防止 NoneType 错误
            if core: core.wait(0.5)

    def __init__(self, 
                 monitor_idx: int, 
                 width_cm: Optional[float] = None, 
                 resolution_x: Optional[int] = None, 
                 bg_color: str = 'black'):
        
        # 1. 硬件连接
        try:
            all_monitors = get_monitors()
            if monitor_idx < 0 or monitor_idx >= len(all_monitors):
                raise ValueError(f"索引 {monitor_idx} 无效")
            self.target_monitor = all_monitors[monitor_idx]
        except Exception as e:
            logger.critical(f"硬件错误: {e}")
            raise

        # 2. 分辨率
        if resolution_x:
            self.res_x = resolution_x
            self.res_y = 1080 
            logger.info(f"分辨率: 手动指定 {self.res_x}")
        else:
            self.res_x = self.target_monitor.width
            self.res_y = self.target_monitor.height
            logger.info(f"分辨率: 自动检测 {self.res_x}x{self.res_y}")

        # 3. 宽度决策
        final_width = None
        
        if width_cm is not None and width_cm > 0:
            final_width = width_cm
            logger.info(f"物理宽度: 使用参数值 {final_width} cm")
        
        # 硬件EDID (MacOS跳过)
        if final_width is None:
            detected_mm = self.target_monitor.width_mm
            import platform
            if detected_mm and detected_mm > 0 and platform.system() != 'Darwin':
                final_width = detected_mm / 10.0
                logger.warning(f"物理宽度: 使用硬件报告值 {final_width} cm")

        # 交互回退
        if final_width is None:
            logger.warning(f"无法自动获取显示器 {monitor_idx} 的物理尺寸。")
            logger.info("选项 A: 输入数值 (例如 28.65)")
            logger.info("选项 B: 输入 'c' 启动标定工具")
            
            while True:
                user_input = input(">>> 请输入指令或数值: ").strip().lower()
                
                if user_input == 'c':
                    # 调用静态标定方法
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
        self.win_w = self.res_x
        self.win_h = self.res_y 
        self.bg_color = bg_color
        self.monitor_idx = monitor_idx

        self.cmd_queue = queue.Queue(maxsize=5)
        self.stop_event = threading.Event()
        self.engine_ready = threading.Event()
        
    def start_loop(self):
        """主线程渲染循环"""
        
        # 确保启动循环前加载库
        lazy_import_psychopy()

        win = None
        current_stim = None
        
        try:
            temp_mon = monitors.Monitor(name='main_mon')
            temp_mon.setWidth(self.width_cm)
            temp_mon.setSizePix((self.res_x, self.res_y))
            temp_mon.save()

            win = visual.Window(
                screen=self.monitor_idx,
                fullscr=True,
                size=[self.res_x, self.res_y],
                monitor=temp_mon,
                color=self.bg_color,
                units='pix',
                allowGUI=False, waitBlanking=True, useFBO=True, viewScale=None
            )
            self._main_window_ref = win 
            
            real_w, real_h = win.size
            self.win_w = self.res_x 
            self.ppcm = self.res_x / self.width_cm 
            scale = real_w / self.res_x
            self.win_h = real_h / scale

            logger.success(f"引擎启动 | PPCM: {self.ppcm:.2f}")
            self.engine_ready.set()
            
            win.flip() 
            
            while not self.stop_event.is_set():
                try:
                    while not self.cmd_queue.empty():
                        cmd = self.cmd_queue.get_nowait()
                        if cmd.cmd_type == "UPDATE":
                            new_stim = self._create_stimulus(win, cmd)
                            if new_stim: current_stim = new_stim
                        elif cmd.cmd_type == "CLEAR":
                            current_stim = None
                except queue.Empty: pass

                if current_stim: current_stim.draw()
                win.flip()

                if 'escape' in event.getKeys(): self.stop_event.set()

        except Exception as e:
            logger.critical(f"渲染循环崩溃: {e}")
        finally:
            if win: 
                win.close()
                del win
            gc.collect()
            logger.info("引擎关闭")

    def close(self):
        self.stop_event.set()

    def show_image(self, 
                   img_path: str, 
                   size: Optional[Tuple[Optional[float], Optional[float]]] = None, 
                   pos: Tuple[float, float] = (0, 0), 
                   units: str = 'cm', 
                   anchor: str = 'center'):
        """
        请求显示图像 (非阻塞)。
        :param size: 目标尺寸 (width, height)。
                     - None 或 (None, None): 使用原始像素尺寸
                     - (W, None): 指定宽度，高度自适应 (保持宽高比)
                     - (None, H): 指定高度，宽度自适应 (保持宽高比)
                     - (W, H): 强制拉伸到指定尺寸
        :param units: 单位 ('cm' 或 'pix')，作用于 size 和 pos。
        """
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
        self.cmd_queue.put(DisplayCommand(cmd_type="CLEAR"))

    def _create_stimulus(self, win, cmd: DisplayCommand):
        """根据指令创建/更新 PsychoPy 刺激对象"""
        try:
            # 1. 先创建 Stimulus 对象 (PsychoPy 会自动加载图片并获取原始尺寸)
            # interpolate=True 保证缩放平滑，如果追求原始像素精准度可设为 False
            stim = visual.ImageStim(win=win, image=cmd.img_path, interpolate=True)
            
            # 获取原始像素尺寸 (PsychoPy 加载后的原始大小)
            orig_w_px, orig_h_px = stim.size
            if orig_w_px == 0 or orig_h_px == 0: return None # 避免除零

            # 2. 计算目标尺寸 (支持缺省单个维度)
            w_px, h_px = orig_w_px, orig_h_px # 默认使用原始尺寸
            
            if cmd.size is not None:
                # 安全获取用户请求的 w, h (允许 None)
                req_w = cmd.size[0] if len(cmd.size) > 0 else None
                req_h = cmd.size[1] if len(cmd.size) > 1 else None
                
                # 将 cm 转为 pix (如果是像素单位则保持不变)
                target_w = req_w * self.ppcm if (req_w is not None and cmd.units == 'cm') else req_w
                target_h = req_h * self.ppcm if (req_h is not None and cmd.units == 'cm') else req_h
                
                # 宽高比自适应计算
                aspect = orig_w_px / orig_h_px
                
                if target_w is not None and target_h is not None:
                    # 双维度指定 -> 强制拉伸
                    w_px, h_px = target_w, target_h
                elif target_w is not None:
                    # 指定宽 -> 高度自适应
                    w_px = target_w
                    h_px = w_px / aspect
                elif target_h is not None:
                    # 指定高 -> 宽度自适应
                    h_px = target_h
                    w_px = h_px * aspect
                
                # 应用尺寸
                stim.size = (w_px, h_px)

            # [调试日志] 打印最终计算出的尺寸
            logger.success(f"显示: {w_px:.0f}x{h_px:.0f}px ({w_px/self.ppcm:.2f}x{h_px/self.ppcm:.2f}cm) | {os.path.basename(cmd.img_path)}")
            
            # 3. 计算位置 (注意：如果 units='cm'，pos 是 cm，但 w_px 已经是像素了)
            if cmd.units == 'cm':
                pos_x_px = cmd.pos[0] * self.ppcm
                pos_y_px = cmd.pos[1] * self.ppcm
            else:
                pos_x_px = cmd.pos[0]
                pos_y_px = cmd.pos[1]

            # 4. 坐标变换 (User Top-Left -> PsychoPy Center)
            psy_x, psy_y = self._transform_coords((pos_x_px, pos_y_px), (w_px, h_px), cmd.anchor)
            stim.pos = (psy_x, psy_y)

            return stim
            
        except Exception as e:
            logger.error(f"Stimulus Error: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return None

    def _transform_coords(self, user_pos_px, size_px, anchor):
        u_x, u_y = user_pos_px
        w_px, h_px = size_px
        
        if anchor == 'top-left':
            center_u_x = u_x + w_px / 2
            center_u_y = u_y + h_px / 2
            psy_x = - (self.win_w / 2) + center_u_x
            psy_y = (self.win_h / 2) - center_u_y
        else:
            psy_x = - (self.win_w / 2) + u_x
            psy_y = (self.win_h / 2) - u_y
            
        return psy_x, psy_y
