"""
文件: test_slideshow.py
功能: 导入 display 模块并测试自动轮播功能
运行方法: python test_slideshow.py
"""
import os
import time
import threading
import sys
from loguru import logger

# [关键] 从同级目录的 display.py 导入核心类
# 如果你的主文件名不是 display.py，请修改下面的名字
try:
    from display import AsyncPrecisionDisplay
except ImportError:
    print("错误：在当前目录下找不到 display.py，或者文件名不匹配。")
    sys.exit(1)

# 配置日志格式 (保持与主程序一致的风格)
logger.remove()
logger.add(sys.stderr, format="<green>{time:HH:mm:ss}</green> | <level>{message}</level>")

def run_auto_slideshow_logic(display_ctrl, folder_path: str, interval: float = 2.0):
    """
    [业务逻辑] 自动轮播模式 (软件定时触发)
    :param interval: 切换间隔 (秒)
    """
    logger.info(f"正在扫描文件夹: {folder_path}")
    valid_exts = ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')
    
    if not os.path.exists(folder_path):
        logger.error(f"文件夹不存在: {folder_path}")
        display_ctrl.close()
        return

    images = [f for f in os.listdir(folder_path) if f.lower().endswith(valid_exts) and not f.startswith('.')]
    images.sort()

    if not images:
        logger.error("文件夹为空或无有效图片")
        display_ctrl.close()
        return

    # 1. 关键：等待显示引擎初始化完毕
    logger.info("等待引擎就绪...")
    display_ctrl.engine_ready.wait()
    logger.success(f"引擎就绪！开始自动轮播 (共 {len(images)} 张，间隔 {interval}s)")

    import itertools
    img_iterator = itertools.cycle(images)

    # [核心技巧] 计算屏幕的物理中心坐标 (cm)
    # 1. 获取屏幕物理高度 (cm) = 像素高度 / PPCM
    screen_h_cm = display_ctrl.res_y / display_ctrl.ppcm
    
    # 2. 计算中心点 (宽的一半, 高的一半)
    # display_ctrl.width_cm 是我们在 main 里传入或标定的物理宽度
    center_x = display_ctrl.width_cm / 2
    center_y = screen_h_cm / 2
    
    logger.success(f"屏幕物理尺寸: {display_ctrl.width_cm:.2f} x {screen_h_cm:.2f} cm")
    logger.success(f"计算得中心点: ({center_x:.2f}, {center_y:.2f}) cm")
    logger.info(f"开始轮播 (间隔 {interval}s)...")
    
    # 2. 循环触发
    while not display_ctrl.stop_event.is_set():
        # A. 获取下一张图片
        current_file = next(img_iterator)
        full_path = os.path.join(folder_path, current_file)
        
        # B. 【软件触发核心】直接调用 show_image
        logger.info(f"软件触发 -> 切换至: {current_file}")
        
        display_ctrl.show_image(
            img_path=full_path, 
            size=(None, 20),  # 根据需要调整显示尺寸 (cm)
            pos=(center_x, center_y),         # 居中
            anchor='center',
            units='cm'
        )
        
        # C. 休眠 (控制切换频率)
        time.sleep(interval)

    logger.info("轮播结束")

if __name__ == "__main__":
    # 1. 扫描屏幕信息 (可选)
    logger.info(">>> 步骤1: 扫描屏幕...")
    AsyncPrecisionDisplay.scan_monitors()
    
    # 2. 配置参数
    # 请确认此路径存在
    TARGET_FOLDER = "/Users/qiujinyu/Pictures/switch2 游戏照片/aaa"
    
    # 这里的 index=0 是主屏，width_cm=60.05 是你之前标定的数值
    # 填入数值可以跳过交互标定环节
    MONITOR_IDX = 0
    WIDTH_CM = 60.05 
    INTERVAL = 0.5  # 0.5秒切换一次

    try:
        logger.info(f">>> 步骤2: 初始化显示器 (宽度设定: {WIDTH_CM}cm)...")
        display = AsyncPrecisionDisplay(monitor_idx=MONITOR_IDX, width_cm=WIDTH_CM)
        
        # 3. 启动业务逻辑线程
        t = threading.Thread(
            target=run_auto_slideshow_logic, 
            args=(display, TARGET_FOLDER, INTERVAL), 
            daemon=True
        )
        t.start()

        # 4. 主线程启动显示引擎 (这行代码会阻塞，直到窗口关闭)
        logger.info(">>> 步骤3: 启动渲染循环...")
        display.start_loop()
        
    except KeyboardInterrupt:
        logger.warning("用户强制停止")
    except Exception as e:
        logger.error(f"运行时错误: {e}")