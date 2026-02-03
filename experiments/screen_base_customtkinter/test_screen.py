"""
文件: test_slideshow.py
功能: 测试基于基类适配的 MonitorDisplay 自动轮播功能
"""
import os
import time
import threading
import sys
from loguru import logger

# 路径自动修复，确保能找到 screen 包
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from screen import MonitorDisplay, DisplayPayload
except ImportError:
    logger.error("无法找到 screen 模块，请检查目录结构。")
    sys.exit(1)

# 配置日志
logger.remove()
logger.add(sys.stderr, format="<green>{time:HH:mm:ss}</green> | <level>{message}</level>")

# def run_single_image_logic(display_ctrl, image_path: str):
#     """
#     业务逻辑：显示单张图片。
#     """
#     logger.info(f"准备显示单张图片: {image_path}")
    
#     if not os.path.exists(image_path):
#         logger.error(f"图片文件不存在: {image_path}")
#         display_ctrl.close()
#         return

#     # 1. 等待 UI 渲染引擎完全启动
#     logger.info("等待显示引擎就绪...")
#     display_ctrl.engine_ready.wait() 
#     logger.success("引擎就绪！开始显示图片")

#     # 2. 计算物理中心点
#     screen_h_cm = display_ctrl.height_px / display_ctrl.ppcm
#     center_x = display_ctrl.width_cm / 2
#     center_y = screen_h_cm / 2
    
#     try:
#         # 3. 封装为标准 DisplayPayload 发送
#         payload = DisplayPayload(
#             content=image_path,
#             target_size_cm=(3, None), 
#             position=(center_x, center_y),
#             anchor='center',
#         )
        
#         display_ctrl.show(payload)
#         logger.success("图片显示成功，按 Ctrl+C 退出")
        
#         # 持续显示，直到用户中断
#         while not display_ctrl.stop_event.is_set():
#             time.sleep(0.1)
            
#     except Exception as e:
#         logger.error(f"显示单张图片时出错: {e}")

def run_single_image_logic(display_ctrl, image_path: str, position_cm=(0,0), size_cm=(5, None)):
    """
    业务逻辑：显示单张图片。
    """
    logger.info(f"准备显示单张图片: {image_path}")
    
    if not os.path.exists(image_path):
        logger.error(f"图片文件不存在: {image_path}")
        display_ctrl.close()
        return

    # 1. 等待 UI 渲染引擎完全启动
    logger.info("等待显示引擎就绪...")
    display_ctrl.engine_ready.wait() 
    logger.success("引擎就绪！开始显示图片")
    
    try:
        # 3. 封装为标准 DisplayPayload 发送
        payload = DisplayPayload(
            content=image_path,
            target_size_cm=SIZE_CM, 
            position=position_cm,
            anchor='top-left',
        )
        
        display_ctrl.show(payload)
        logger.success("图片显示成功，按 Ctrl+C 退出")
        
        # 持续显示，直到用户中断
        while not display_ctrl.stop_event.is_set():
            time.sleep(0.1)
            
    except Exception as e:
        logger.error(f"显示单张图片时出错: {e}")

if __name__ == "__main__":
    # 图片地址
    IMAGE_PATH = "/Users/qiujinyu/Computational_Imaging/空间光调制器/系统控制代码/SLMSystem/img/2025112322240600_c.jpg"
    # 显示器编号
    MONITOR_IDX = 1
    # 显示器物理宽度 (cm)，设为 None 将触发交互式校准
    WIDTH_CM = 60.04
    # 显示位置 (cm)
    POSITION_CM = (5, 6)
    # 显示图片的大小 (cm)
    SIZE_CM = (5, None)

    # 1. 初始化显示器对象 (由于 AsyncPrecisionDisplay 的逻辑，初始化会处理校准)
    display = MonitorDisplay(
        monitor_idx=MONITOR_IDX, 
        width_cm=WIDTH_CM,
        bg_color='black'
    )

    try:
        logic_thread = threading.Thread(
                target=run_single_image_logic, 
                args=(display, IMAGE_PATH, POSITION_CM, SIZE_CM), 
                daemon=True
            )
        
        logic_thread.start()

        # 3. 启动渲染循环 (阻塞主线程，符合 macOS 规范)
        display.initialize()

    except KeyboardInterrupt:
        logger.warning("用户停止程序")
    finally:
        display.close()
        logger.info("程序已安全退出")