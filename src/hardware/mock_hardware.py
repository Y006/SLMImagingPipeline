# -*- coding: utf-8 -*-
"""
Mock 硬件类 - 用于测试，不实际操作硬件，只打印日志

包含三个 Mock 类:
- MockHikCamera: 模拟海康相机
- MockScreen: 模拟显示屏
- MockSLM: 模拟空间光调制器
"""

import os
import time
import datetime
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from loguru import logger


class MockHikCamera:
    """
    Mock 海康相机类 - 模拟 HikCamera 的所有接口
    不实际操作硬件，仅打印日志用于测试
    """
    def __init__(self, dev_index: int = 0):
        self.dev_index = dev_index
        self.is_open = False
        self.payload = 1920 * 1200 * 3  # 模拟一个默认的 payload
        self.current_exposure = 20000.0  # 默认曝光时间（微秒）
        
        logger.info(f"[MockHikCamera] 初始化 - 设备索引: {dev_index}")

    def open(self) -> bool:
        """模拟打开相机"""
        logger.info(f"[MockHikCamera] 正在打开相机 (设备索引: {self.dev_index})...")
        logger.info(f"[MockHikCamera] 枚举设备...")
        logger.info(f"[MockHikCamera] 检测到 1 个设备")
        logger.info(f"[MockHikCamera] 创建句柄...")
        logger.info(f"[MockHikCamera] 打开设备...")
        logger.info(f"[MockHikCamera] 设置触发模式=Off")
        logger.info(f"[MockHikCamera] 设置连续采集模式")
        logger.info(f"[MockHikCamera] 关闭自动曝光")
        logger.info(f"[MockHikCamera] 关闭自动增益")
        logger.info(f"[MockHikCamera] 读取 PayloadSize: {self.payload}")
        
        self.is_open = True
        logger.success(f"[MockHikCamera] ✓ 相机已成功打开")
        return True

    def snap(self, save_path: str, exposure_us: float = 20000.0,
             timeout_ms: int = 1500, img_type: int = 1) -> bool:
        """
        模拟抓拍一张图像并保存
        生成一张包含参数信息的示意图
        
        Args:
            save_path: 保存路径
            exposure_us: 曝光时间(微秒)
            timeout_ms: 取帧超时(毫秒)
            img_type: 图像类型 (1=JPEG, 2=BMP)
        """
        if not self.is_open:
            logger.error("[MockHikCamera] ✗ 相机未打开")
            return False

        img_type_str = "JPEG" if img_type == 1 else "BMP"
        
        logger.info(f"[MockHikCamera] 准备抓拍...")
        logger.info(f"[MockHikCamera]   - 保存路径: {save_path}")
        logger.info(f"[MockHikCamera]   - 曝光时间: {exposure_us} μs ({exposure_us/1000:.2f} ms)")
        logger.info(f"[MockHikCamera]   - 超时设置: {timeout_ms} ms")
        logger.info(f"[MockHikCamera]   - 图像格式: {img_type_str}")
        
        # 模拟设置曝光
        if exposure_us != self.current_exposure:
            logger.info(f"[MockHikCamera] 设置曝光时间: {self.current_exposure:.1f} → {exposure_us:.1f} μs")
            self.current_exposure = exposure_us
        
        # 模拟开始取流
        logger.info(f"[MockHikCamera] 开始取流...")
        
        # 模拟取帧延迟（根据曝光时间）
        delay_s = min(exposure_us / 1000000.0 + 0.05, 0.5)  # 最多延迟 0.5 秒
        logger.info(f"[MockHikCamera] 等待曝光完成... ({delay_s*1000:.0f} ms)")
        time.sleep(delay_s)
        
        # 模拟取帧成功
        mock_width, mock_height = 1920, 1200
        logger.info(f"[MockHikCamera] ✓ 成功取得一帧")
        logger.info(f"[MockHikCamera]   - 图像尺寸: {mock_width}x{mock_height}")
        logger.info(f"[MockHikCamera]   - 帧大小: {mock_width * mock_height * 3} bytes")
        
        # 生成示意图
        logger.info(f"[MockHikCamera] 生成示意图像...")
        
        try:
            os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
            
            # 创建一张渐变背景的图像
            img = self._generate_mock_image(mock_width, mock_height, exposure_us, save_path)
            
            # 保存图像
            if img_type == 2:  # BMP
                img.save(save_path, "BMP")
            else:  # JPEG
                img.save(save_path, "JPEG", quality=90)
            
            logger.info(f"[MockHikCamera] 停止取流...")
            logger.success(f"[MockHikCamera] ✓ 已保存: {save_path}  尺寸: {mock_width}x{mock_height}")
            return True
        except Exception as e:
            logger.error(f"[MockHikCamera] ✗ 保存图像失败: {e}")
            return False
    
    def _generate_mock_image(self, width, height, exposure_us, filename):
        """
        生成一张包含信息的示意图
        
        Args:
            width: 图像宽度
            height: 图像高度
            exposure_us: 曝光时间（用于调整亮度）
            filename: 文件名（用于显示）
        
        Returns:
            PIL.Image: 生成的图像
        """
        # 根据曝光时间调整亮度（曝光越长，图像越亮）
        brightness = min(int(50 + exposure_us / 2000), 200)
        
        # 创建渐变背景
        img_array = np.zeros((height, width, 3), dtype=np.uint8)
        
        # 添加渐变效果（从左到右，从暗到亮）
        for x in range(width):
            intensity = int((x / width) * brightness)
            img_array[:, x, :] = [intensity, intensity, intensity]
        
        # 添加一些噪声（模拟相机噪声）
        noise = np.random.randint(-20, 20, (height, width, 3), dtype=np.int16)
        img_array = np.clip(img_array.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        
        # 添加一些圆形图案（模拟成像）
        center_x, center_y = width // 2, height // 2
        y_grid, x_grid = np.ogrid[:height, :width]
        
        # 添加几个同心圆
        for r in range(100, 400, 80):
            mask = ((x_grid - center_x) ** 2 + (y_grid - center_y) ** 2) < r ** 2
            outer_mask = ((x_grid - center_x) ** 2 + (y_grid - center_y) ** 2) < (r - 10) ** 2
            ring_mask = mask & ~outer_mask
            img_array[ring_mask] = [brightness + 30, brightness + 30, brightness + 30]
        
        # 转换为 PIL Image
        img = Image.fromarray(img_array)
        
        # 添加文字信息
        draw = ImageDraw.Draw(img)
        
        # 尝试使用系统字体，如果失败则使用默认字体
        try:
            # macOS 系统字体
            font_large = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 48)
            font_small = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 32)
        except:
            try:
                # 备选字体
                font_large = ImageFont.truetype("/Library/Fonts/Arial.ttf", 48)
                font_small = ImageFont.truetype("/Library/Fonts/Arial.ttf", 32)
            except:
                # 使用默认字体
                font_large = ImageFont.load_default()
                font_small = ImageFont.load_default()
        
        # 添加信息文字
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        filename_only = os.path.basename(filename)
        
        # 背景框（半透明）
        text_y = 50
        draw.rectangle([(20, text_y - 10), (width - 20, text_y + 250)], 
                      fill=(0, 0, 0, 180), outline=(255, 255, 255))
        
        # 添加文字
        draw.text((40, text_y), "MOCK CAMERA CAPTURE", fill=(0, 255, 0), font=font_large)
        draw.text((40, text_y + 60), f"Filename: {filename_only}", fill=(255, 255, 255), font=font_small)
        draw.text((40, text_y + 100), f"Timestamp: {timestamp}", fill=(255, 255, 255), font=font_small)
        draw.text((40, text_y + 140), f"Exposure: {exposure_us} μs ({exposure_us/1000:.2f} ms)", 
                 fill=(255, 255, 255), font=font_small)
        draw.text((40, text_y + 180), f"Resolution: {width}x{height}", 
                 fill=(255, 255, 255), font=font_small)
        
        # 添加水印
        draw.text((width - 280, height - 50), "Mock Hardware Mode", 
                 fill=(100, 100, 100), font=font_small)
        
        return img

    def close(self):
        """模拟关闭相机"""
        if not self.is_open:
            logger.info("[MockHikCamera] 相机未打开，无需关闭")
            return
        
        logger.info("[MockHikCamera] 停止取流...")
        logger.info("[MockHikCamera] 关闭设备...")
        logger.info("[MockHikCamera] 销毁句柄...")
        
        self.is_open = False
        logger.success("[MockHikCamera] ✓ 相机已关闭")


class MockScreen:
    """
    Mock 显示屏类 - 模拟 Screen 的所有接口
    不实际操作硬件，仅打印日志用于测试
    """
    def __init__(self, monitor_index: int = 0, bg: str = "black"):
        self.monitor_index = monitor_index
        self.bg = bg
        self.geom = (0, 0, 1920, 1080)  # 模拟显示器几何信息 (x, y, w, h)
        
        logger.info(f"[MockScreen] 初始化")
        logger.info(f"[MockScreen]   - 显示器索引: {monitor_index}")
        logger.info(f"[MockScreen]   - 背景色: {bg}")
        logger.info(f"[MockScreen]   - 模拟几何: {self.geom[2]}x{self.geom[3]} @ ({self.geom[0]}, {self.geom[1]})")
        logger.info(f"[MockScreen] 创建全屏窗口（模拟）")
        logger.info(f"[MockScreen] 按 Esc 键退出（模拟）")

    def show_image(self, img_path: str, scale_factor: float = 1.0) -> bool:
        """模拟居中显示图像"""
        logger.info(f"[MockScreen] 显示图像（居中）")
        logger.info(f"[MockScreen]   - 图像路径: {img_path}")
        logger.info(f"[MockScreen]   - 缩放因子: {scale_factor}")
        
        if not os.path.exists(img_path):
            logger.error(f"[MockScreen] ✗ 文件不存在: {img_path}")
            return False
        
        # 模拟图像处理
        x, y, w, h = self.geom
        mock_img_w, mock_img_h = 800, 600  # 假设原始图像尺寸
        new_w = int(mock_img_w * scale_factor)
        new_h = int(mock_img_h * scale_factor)
        offset_x = (w - new_w) // 2
        offset_y = (h - new_h) // 2
        
        logger.info(f"[MockScreen] 打开图像: {img_path}")
        logger.info(f"[MockScreen] 原始尺寸: {mock_img_w}x{mock_img_h}")
        logger.info(f"[MockScreen] 缩放后尺寸: {new_w}x{new_h}")
        logger.info(f"[MockScreen] 居中位置: ({offset_x}, {offset_y})")
        logger.info(f"[MockScreen] 创建背景画布: {w}x{h}, 颜色={self.bg}")
        logger.info(f"[MockScreen] 粘贴图像到画布...")
        logger.info(f"[MockScreen] 更新显示...")
        
        logger.success(f"[MockScreen] ✓ 成功显示: {img_path}")
        return True

    def show_image_at(self, img_path: str, position: tuple, scale_factor: float = 1.0) -> bool:
        """模拟在指定位置显示图像"""
        logger.info(f"[MockScreen] 显示图像（指定位置）")
        logger.info(f"[MockScreen]   - 图像路径: {img_path}")
        logger.info(f"[MockScreen]   - 缩放因子: {scale_factor}")
        logger.info(f"[MockScreen]   - 指定位置: {position}")
        
        if not os.path.exists(img_path):
            logger.error(f"[MockScreen] ✗ 文件不存在: {img_path}")
            return False
        
        # 模拟图像处理
        x, y, w, h = self.geom
        mock_img_w, mock_img_h = 800, 600
        new_w = int(mock_img_w * scale_factor)
        new_h = int(mock_img_h * scale_factor)
        pos_x, pos_y = position
        offset_x = pos_x - new_w // 2
        offset_y = pos_y - new_h // 2
        
        logger.info(f"[MockScreen] 打开图像: {img_path}")
        logger.info(f"[MockScreen] 原始尺寸: {mock_img_w}x{mock_img_h}")
        logger.info(f"[MockScreen] 缩放后尺寸: {new_w}x{new_h}")
        logger.info(f"[MockScreen] 显示位置: ({offset_x}, {offset_y})")
        logger.info(f"[MockScreen] 创建背景画布: {w}x{h}, 颜色={self.bg}")
        logger.info(f"[MockScreen] 粘贴图像到画布...")
        logger.info(f"[MockScreen] 更新显示...")
        
        logger.success(f"[MockScreen] ✓ 成功显示: {img_path}")
        return True

    def start(self):
        """模拟进入事件循环"""
        logger.info("[MockScreen] 进入事件循环（模拟）")
        logger.info("[MockScreen] 窗口将保持显示...")
        # 在实际测试中，这里不会阻塞
        logger.info("[MockScreen] (Mock 模式：立即返回，不阻塞)")

    def close(self):
        """模拟关闭窗口"""
        logger.info("[MockScreen] 销毁窗口...")
        logger.success("[MockScreen] ✓ 窗口已关闭")


# ================================
# Mock 常量定义（模拟海康相机 SDK）
# ================================
MV_Image_Bmp = 2  # Mock BMP 图像类型
MV_Image_Jpeg = 1  # Mock JPEG 图像类型


class MockSLM:
    """
    Mock 空间光调制器类 - 模拟 SLM 的所有接口
    不实际操作硬件，仅打印日志用于测试
    """
    def __init__(self, sdk_version=(4, 1), verbose=True):
        self.sdk_version = sdk_version
        self.verbose = verbose
        self.initialized = False
        self.current_image = None
        
        if self.verbose:
            logger.info(f"[MockSLM] 创建 SLM 实例")
            logger.info(f"[MockSLM]   - SDK 版本: {sdk_version[0]}.{sdk_version[1]}")
            logger.info(f"[MockSLM]   - 详细模式: {verbose}")

    def init(self):
        """模拟初始化 SDK + 打开 SLM"""
        if self.verbose:
            logger.info(f"[MockSLM] 打印 SDK 版本信息...")
            logger.info(f"[MockSLM] HOLOEYE SLM Display SDK v{self.sdk_version[0]}.{self.sdk_version[1]} (Mock)")
        
        major, minor = self.sdk_version
        logger.info(f"[MockSLM] 初始化 SDK v{major}.{minor}...")
        logger.success(f"[MockSLM] ✓ SDK 初始化成功")
        
        logger.info(f"[MockSLM] 初始化 SLM 设备...")
        logger.info(f"[MockSLM] 正在连接 SLM 硬件...")
        logger.info(f"[MockSLM] 检测到 SLM 设备: HOLOEYE GAEA-2 (Mock)")
        logger.info(f"[MockSLM] 设备分辨率: 1920x1080")
        logger.success(f"[MockSLM] ✓ 设备打开成功")
        
        self.initialized = True
        return True

    def img_show(self, img_path: str) -> bool:
        """模拟将图片显示到 SLM"""
        if not self.initialized:
            if self.verbose:
                logger.error("[MockSLM] ✗ SLM 未初始化，请先调用 init()")
            return False
        
        if not os.path.isfile(img_path):
            if self.verbose:
                logger.warning(f"[MockSLM] ✗ 图像文件不存在: {img_path}")
            return False
        
        logger.info(f"[MockSLM] 加载图像数据: {img_path}")
        logger.info(f"[MockSLM] 验证图像格式...")
        logger.info(f"[MockSLM] 图像尺寸: 1920x1080 (假设)")
        logger.info(f"[MockSLM] 创建数据句柄...")
        logger.success(f"[MockSLM] ✓ 图像加载成功")
        
        logger.info(f"[MockSLM] 显示图像到 SLM...")
        logger.info(f"[MockSLM] 使用模式: PresentAutomatic")
        logger.info(f"[MockSLM] 上传数据到显存...")
        logger.info(f"[MockSLM] 刷新 SLM 显示...")
        
        self.current_image = img_path
        
        if self.verbose:
            logger.success(f"[MockSLM] ✓ 正在显示: {img_path}")
        
        return True


# ================================
# Mock 显示函数（模拟 screen.py 的 display_image）
# ================================
def mock_display_image(display_image_path, monitor_idx, scale_factor):
    """
    Mock 版本的 display_image 函数
    模拟在新线程中显示图片的行为
    
    Args:
        display_image_path: 图片路径
        monitor_idx: 显示器索引
        scale_factor: 缩放因子
    """
    logger.info(f"[mock_display_image] 启动显示线程（模拟）")
    logger.info(f"[mock_display_image]   - 图片路径: {display_image_path}")
    logger.info(f"[mock_display_image]   - 显示器索引: {monitor_idx}")
    logger.info(f"[mock_display_image]   - 缩放因子: {scale_factor}")
    
    if not os.path.exists(display_image_path):
        logger.error(f"[mock_display_image] ✗ 图片不存在: {display_image_path}")
        return
    
    # 创建 Mock Screen 实例并显示
    screen = MockScreen(monitor_index=monitor_idx, bg="black")
    success = screen.show_image(display_image_path, scale_factor)
    
    if success:
        logger.info(f"[mock_display_image] 进入事件循环（模拟，daemon线程）")
        logger.info(f"[mock_display_image] 图片将持续显示...")
        # Mock 模式下不阻塞，立即返回
    else:
        logger.error(f"[mock_display_image] ✗ 显示失败")


# ================================
# 测试示例
# ================================
if __name__ == "__main__":
    logger.info("=" * 60)
    logger.info("Mock 硬件测试开始")
    logger.info("=" * 60)
    
    # 测试 MockHikCamera - 生成实际图片
    logger.info("\n" + "=" * 60)
    logger.info("测试 MockHikCamera - 生成示意图片")
    logger.info("=" * 60)
    cam = MockHikCamera(dev_index=0)
    cam.open()
    
    # 测试不同曝光时间
    cam.snap("test_capture_short.jpg", exposure_us=10000.0, timeout_ms=2000)
    cam.snap("test_capture_medium.bmp", exposure_us=50000.0, timeout_ms=2000, img_type=MV_Image_Bmp)
    cam.snap("test_capture_long.jpg", exposure_us=200000.0, timeout_ms=3000)
    
    cam.close()
    
    logger.info("\n✓ 已生成 3 张示意图片，可以查看 test_capture_*.jpg/bmp")
    
    # 测试 MockScreen
    logger.info("\n" + "=" * 60)
    logger.info("测试 MockScreen")
    logger.info("=" * 60)
    screen = MockScreen(monitor_index=1, bg="black")
    screen.show_image("test_image.jpg", scale_factor=0.5)
    screen.show_image_at("test_image2.jpg", position=(100, 200), scale_factor=0.3)
    screen.close()
    
    # 测试 MockSLM
    logger.info("\n" + "=" * 60)
    logger.info("测试 MockSLM")
    logger.info("=" * 60)
    slm = MockSLM(sdk_version=(4, 1), verbose=True)
    slm.init()
    slm.img_show("test_pattern.png")
    
    logger.info("\n" + "=" * 60)
    logger.info("Mock 硬件测试完成")
    logger.info("=" * 60)
