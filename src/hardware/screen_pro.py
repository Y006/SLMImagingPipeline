"""
基于物理尺寸的轻量级图像显示系统

设计理念：
- 使用 Tkinter + screeninfo 实现轻量级显示控制
- 通过校准获取 PPC (Pixels Per Centimeter) 实现物理尺寸精准映射
- 采用厘米为单位的绝对坐标系，以显示器左上角为原点
- 支持四种灵活的尺寸控制模式

日期：2026-02-03
"""

import tkinter as tk
from PIL import Image, ImageTk
from PIL.Image import Resampling
from screeninfo import get_monitors
import json
import os
from pathlib import Path
from typing import Optional, Tuple
from loguru import logger


class ScreenPro:
    """基于物理尺寸的图像显示系统"""
    
    CALIBRATION_FILE = ".screen_calibration.json"
    
    def __init__(self, monitor_index: int = 0, bg: str = "black", skip_calibration: bool = False):
        """
        初始化显示系统
        
        Args:
            monitor_index: 目标显示器索引
            bg: 背景颜色
            skip_calibration: 是否跳过校准数据检查（用于校准流程）
        """
        self.monitor_index = monitor_index
        self.bg = bg
        
        # 创建 Tkinter 窗口
        self.root = tk.Tk()
        self.root.configure(bg=bg)
        self.root.overrideredirect(True)  # 无边框
        self.root.bind("<Escape>", lambda e: self.close())
        
        # 获取显示器几何信息
        self._init_monitor_geometry()
        
        # 创建 Canvas
        self.canvas = tk.Canvas(
            self.root, 
            width=self.screen_width, 
            height=self.screen_height,
            highlightthickness=0, 
            bg=bg
        )
        self.canvas.pack(fill="both", expand=True)
        
        self._tkimg = None  # 保存图像引用，防止 GC
        
        # 加载或初始化 PPC（每厘米像素数）
        if skip_calibration:
            # 跳过校准检查，用于校准流程
            self.ppc = None
        else:
            self.ppc = self._load_or_calibrate_ppc()
        
    def _init_monitor_geometry(self):
        """初始化显示器几何参数"""
        monitors = get_monitors()
        
        if not monitors:
            # 如果无法获取显示器信息，使用默认值
            self.screen_x = 0
            self.screen_y = 0
            self.screen_width = self.root.winfo_screenwidth()
            self.screen_height = self.root.winfo_screenheight()
            logger.warning("无法获取显示器信息，使用默认屏幕参数")
        else:
            # 确保索引有效
            idx = min(max(0, self.monitor_index), len(monitors) - 1)
            monitor = monitors[idx]
            
            self.screen_x = monitor.x
            self.screen_y = monitor.y
            self.screen_width = monitor.width
            self.screen_height = monitor.height
            
            logger.info(
                f"目标显示器 [{idx}]: "
                f"位置=({self.screen_x}, {self.screen_y}), "
                f"分辨率={self.screen_width}x{self.screen_height}"
            )
        
        # 设置窗口位置和大小
        self.root.geometry(
            f"{self.screen_width}x{self.screen_height}"
            f"+{self.screen_x}+{self.screen_y}"
        )
    
    def _get_calibration_path(self) -> Path:
        """获取校准文件路径"""
        # 保存在用户主目录下
        return Path.home() / f"{self.CALIBRATION_FILE}.monitor_{self.monitor_index}"
    
    def _load_or_calibrate_ppc(self) -> float:
        """加载校准数据，如果不存在则提示用户并退出"""
        calib_file = self._get_calibration_path()
        
        if calib_file.exists():
            try:
                with open(calib_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    ppc = data.get('ppc')
                    if ppc and ppc > 0:
                        logger.info(f"从配置文件加载 PPC: {ppc:.2f} 像素/厘米")
                        return ppc
            except Exception as e:
                logger.error(f"加载校准文件失败: {e}")
        
        # 如果没有校准数据，提示用户并退出
        logger.error("="*60)
        logger.error("错误：未找到校准数据！")
        logger.error(f"校准文件路径: {calib_file}")
        logger.error("")
        logger.error("请先运行校准流程：")
        logger.error("  screen = ScreenPro(monitor_index=0)")
        logger.error("  screen.calibrate_manual()")
        logger.error("")
        logger.error("或者直接设置 PPC 值：")
        logger.error("  screen = ScreenPro(monitor_index=0)")
        logger.error("  screen.set_ppc(your_ppc_value)")
        logger.error("="*60)
        raise RuntimeError("未找到校准数据，程序退出")
    
    def _calibrate(self) -> float:
        """
        校准流程：绘制标准参照物，用户测量后计算 PPC
        
        Returns:
            计算得到的 PPC 值（像素/厘米）
        """
        # 绘制一个 10cm 长的标尺（假设值，待用户校准）
        reference_length_cm = 10.0
        reference_length_px = 400  # 先绘制 400 像素长的参考线
        
        # 清空画布
        self.canvas.delete("all")
        
        # 绘制参考标尺
        center_x = self.screen_width // 2
        center_y = self.screen_height // 2
        
        line_start_x = center_x - reference_length_px // 2
        line_end_x = center_x + reference_length_px // 2
        
        # 绘制主标尺线
        self.canvas.create_line(
            line_start_x, center_y,
            line_end_x, center_y,
            fill="white", width=3
        )
        
        # 绘制起始和结束刻度
        tick_height = 30
        self.canvas.create_line(
            line_start_x, center_y - tick_height,
            line_start_x, center_y + tick_height,
            fill="white", width=3
        )
        self.canvas.create_line(
            line_end_x, center_y - tick_height,
            line_end_x, center_y + tick_height,
            fill="white", width=3
        )
        
        # 添加文字说明
        self.canvas.create_text(
            center_x, center_y - 80,
            text=f"校准标尺（标称长度: {reference_length_cm} cm）",
            fill="white", font=("Arial", 20)
        )
        self.canvas.create_text(
            center_x, center_y + 80,
            text="请用物理直尺测量上方标尺的实际长度",
            fill="yellow", font=("Arial", 16)
        )
        self.canvas.create_text(
            center_x, center_y + 110,
            text="测量完成后，关闭此窗口并在终端输入实测长度",
            fill="yellow", font=("Arial", 14)
        )
        
        self.root.update()
        
        # 等待用户测量（阻塞主线程，直到窗口关闭）
        logger.info("\n" + "="*60)
        logger.info("校准说明：")
        logger.info(f"1. 屏幕上显示了一个标称长度为 {reference_length_cm} cm 的白色标尺")
        logger.info("2. 请使用物理直尺测量该标尺的实际长度（厘米）")
        logger.info("3. 测量完成后，按 ESC 键关闭校准窗口")
        logger.info("="*60 + "\n")
        
        # 启动事件循环（用户按 ESC 会触发关闭）
        self.root.mainloop()
        
        # 窗口关闭后，提示用户输入实测长度
        while True:
            try:
                measured_cm = float(input(f"请输入实际测量长度（厘米）: "))
                if measured_cm <= 0:
                    logger.warning("长度必须大于 0，请重新输入")
                    continue
                break
            except ValueError:
                logger.warning("输入无效，请输入数字")
        
        # 计算 PPC
        ppc = reference_length_px / measured_cm
        
        logger.info(f"校准完成！计算得到 PPC = {ppc:.2f} 像素/厘米")
        
        # 保存校准数据
        self._save_calibration(ppc)
        
        # 重新创建窗口用于后续显示
        self._recreate_window()
        
        return ppc
    
    def calibrate_manual(self) -> float:
        """
        手动校准流程（公开接口）
        
        Returns:
            计算得到的 PPC 值（像素/厘米）
        """
        ppc = self._calibrate()
        self.ppc = ppc  # 更新实例的 PPC 值
        return ppc
    
    def set_ppc(self, ppc: float):
        """
        直接设置 PPC 值（用户已知 PPC 时使用）
        
        Args:
            ppc: 每厘米像素数
        """
        if ppc <= 0:
            raise ValueError("PPC 必须大于 0")
        
        self.ppc = ppc
        self._save_calibration(ppc)
        logger.info(f"PPC 已设置为: {ppc:.2f} 像素/厘米")
    
    def _save_calibration(self, ppc: float):
        """保存校准数据"""
        calib_file = self._get_calibration_path()
        data = {
            'ppc': ppc,
            'monitor_index': self.monitor_index,
            'screen_width': self.screen_width,
            'screen_height': self.screen_height
        }
        
        try:
            with open(calib_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2)
            logger.info(f"校准数据已保存至: {calib_file}")
        except Exception as e:
            logger.error(f"保存校准数据失败: {e}")
    
    def _recreate_window(self):
        """重新创建 Tkinter 窗口（校准后使用）"""
        self.root = tk.Tk()
        self.root.configure(bg=self.bg)
        self.root.overrideredirect(True)
        self.root.bind("<Escape>", lambda e: self.close())
        
        self.root.geometry(
            f"{self.screen_width}x{self.screen_height}"
            f"+{self.screen_x}+{self.screen_y}"
        )
        
        self.canvas = tk.Canvas(
            self.root,
            width=self.screen_width,
            height=self.screen_height,
            highlightthickness=0,
            bg=self.bg
        )
        self.canvas.pack(fill="both", expand=True)
        self._tkimg = None
    
    def cm_to_pixels(self, cm: float) -> int:
        """将厘米转换为像素"""
        return int(cm * self.ppc)
    
    def display_image(
        self,
        img_path: str,
        position_cm: Tuple[float, float] = (0, 0),
        size_cm: Tuple[Optional[float], Optional[float]] = (None, None)
    ) -> bool:
        """
        在指定物理位置显示指定物理尺寸的图像
        
        Args:
            img_path: 图像文件路径
            position_cm: 图像左上角的物理位置（厘米），相对于显示器左上角
                        例如 (10, 15) 表示向右 10cm，向下 15cm
            size_cm: 图像的物理尺寸（厘米），支持四种模式：
                    (w, h)      - 强制拉伸/压缩至指定尺寸
                    (w, None)   - 定宽等比缩放
                    (None, h)   - 定高等比缩放
                    (None, None) - 原始像素尺寸
        
        Returns:
            是否成功显示
        """
        try:
            # 打开图像
            pil_img = Image.open(img_path).convert("RGB")
            orig_w, orig_h = pil_img.size
            
            # 根据 size_cm 计算目标像素尺寸
            target_w_px, target_h_px = self._calculate_target_size(
                orig_w, orig_h, size_cm
            )
            
            # 调整图像尺寸
            if (target_w_px, target_h_px) != (orig_w, orig_h):
                pil_img = pil_img.resize(
                    (target_w_px, target_h_px),
                    Resampling.LANCZOS
                )
            
            # 创建背景画布
            canvas_img = Image.new("RGB", (self.screen_width, self.screen_height), self.bg)
            
            # 计算像素位置
            pos_x_px = self.cm_to_pixels(position_cm[0])
            pos_y_px = self.cm_to_pixels(position_cm[1])
            
            # 粘贴图像（左上角对齐）
            canvas_img.paste(pil_img, (pos_x_px, pos_y_px))
            
            # 显示
            self.canvas.delete("all")
            self._tkimg = ImageTk.PhotoImage(canvas_img)
            self.canvas.create_image(
                self.screen_width // 2,
                self.screen_height // 2,
                image=self._tkimg,
                anchor="center"
            )
            
            logger.info(
                f"图像显示成功: {img_path}\n"
                f"  原始尺寸: {orig_w}x{orig_h} px\n"
                f"  目标尺寸: {target_w_px}x{target_h_px} px\n"
                f"  物理位置: {position_cm} cm\n"
                f"  像素位置: ({pos_x_px}, {pos_y_px}) px"
            )
            
            return True
            
        except Exception as e:
            logger.error(f"显示图像失败: {img_path}, 错误: {e}")
            return False
    
    def _calculate_target_size(
        self,
        orig_w: int,
        orig_h: int,
        size_cm: Tuple[Optional[float], Optional[float]]
    ) -> Tuple[int, int]:
        """
        根据尺寸控制模式计算目标像素尺寸
        
        Args:
            orig_w: 原始图像宽度（像素）
            orig_h: 原始图像高度（像素）
            size_cm: 物理尺寸元组 (宽度_cm, 高度_cm)
        
        Returns:
            (目标宽度_px, 目标高度_px)
        """
        width_cm, height_cm = size_cm
        
        # 模式 4: 原始分辨率直出
        if width_cm is None and height_cm is None:
            return (orig_w, orig_h)
        
        # 模式 1: 强制物理尺寸
        if width_cm is not None and height_cm is not None:
            target_w = self.cm_to_pixels(width_cm)
            target_h = self.cm_to_pixels(height_cm)
            return (target_w, target_h)
        
        # 模式 2: 定宽等比缩放
        if width_cm is not None and height_cm is None:
            target_w = self.cm_to_pixels(width_cm)
            aspect_ratio = orig_h / orig_w
            target_h = int(target_w * aspect_ratio)
            return (target_w, target_h)
        
        # 模式 3: 定高等比缩放
        if width_cm is None and height_cm is not None:
            target_h = self.cm_to_pixels(height_cm)
            aspect_ratio = orig_w / orig_h
            target_w = int(target_h * aspect_ratio)
            return (target_w, target_h)
        
        # 不应该到达这里
        return (orig_w, orig_h)
    
    def clear(self):
        """清空显示内容"""
        self.canvas.delete("all")
        self.root.update()
    
    def start(self):
        """启动事件循环"""
        self.root.mainloop()
    
    def close(self):
        """关闭窗口"""
        try:
            self.root.destroy()
        except Exception:
            pass


def main():
    """示例用法"""
    # 创建显示系统（首次运行会自动进入校准流程）
    screen = ScreenPro(monitor_index=0, bg="black")
    
    # 示例 1: 强制物理尺寸（5cm x 10cm）
    # screen.display_image(
    #     "path/to/image.png",
    #     position_cm=(5, 5),
    #     size_cm=(5, 10)
    # )
    
    # 示例 2: 定宽等比缩放（宽度 8cm）
    # screen.display_image(
    #     "path/to/image.png",
    #     position_cm=(10, 5),
    #     size_cm=(8, None)
    # )
    
    # 示例 3: 定高等比缩放（高度 12cm）
    # screen.display_image(
    #     "path/to/image.png",
    #     position_cm=(0, 0),
    #     size_cm=(None, 12)
    # )
    
    # 示例 4: 原始分辨率
    # screen.display_image(
    #     "path/to/image.png",
    #     position_cm=(3, 3),
    #     size_cm=(None, None)
    # )
    
    logger.info("按 ESC 键关闭窗口")
    screen.start()


if __name__ == "__main__":
    main()
