import tkinter as tk
from PIL import Image, ImageTk
from PIL.Image import Resampling
from screeninfo import get_monitors
import time
import os

class Screen:
    def __init__(self, monitor_index: int = 0, bg: str = "black"):
        """在 init 中选定显示器，并创建全屏 Tk 窗口与 Canvas。"""
        self.root = tk.Tk()
        self.root.configure(bg=bg)
        self.root.overrideredirect(True)     # 无边框
        self.root.bind("<Escape>", lambda e: self.close())

        # 选择目标显示器几何信息
        if get_monitors:
            mons = get_monitors()
        else:
            mons = []

        if not mons:
            x, y = 0, 0
            w = self.root.winfo_screenwidth()
            h = self.root.winfo_screenheight()
        else:
            m = mons[min(max(0, monitor_index), len(mons) - 1)]
            x, y, w, h = m.x, m.y, m.width, m.height

        self.geom = (x, y, w, h)
        self.root.geometry(f"{w}x{h}+{x}+{y}")

        self.canvas = tk.Canvas(self.root, width=w, height=h,
                                highlightthickness=0, bg=bg)
        self.canvas.pack(fill="both", expand=True)

        self._tkimg = None  # 保存引用，防止 GC
        self.bg = bg  # 保存背景色

    def show_image(self, img_path: str, scale_factor: float = 1.0) -> bool:
        """根据缩放因子缩放图像，按中心点显示，并填充背景色"""
        try:
            x, y, w, h = self.geom

            # 打开图片并转换为 RGB 模式
            pil = Image.open(img_path).convert("RGB")
            in_w, in_h = pil.size

            # 计算缩放后的新尺寸
            new_w = int(in_w * scale_factor)
            new_h = int(in_h * scale_factor)
            out = pil.resize((new_w, new_h), Resampling.LANCZOS)

            # 创建一个背景色填充的空白画布
            canvas_image = Image.new("RGB", (w, h), self.bg)

            # 计算居中显示的位置
            offset_x = (w - new_w) // 2
            offset_y = (h - new_h) // 2

            # 将缩放后的图像粘贴到空白画布上
            canvas_image.paste(out, (offset_x, offset_y))

            # 转换为 Tkinter 图像并显示
            self.canvas.delete("all")
            self._tkimg = ImageTk.PhotoImage(canvas_image)
            self.canvas.create_image(w // 2, h // 2, image=self._tkimg, anchor="center")

            print(f"[INFO] 成功显示: {img_path}")
            print(f"原始尺寸: {in_w}x{in_h}, 缩放后的尺寸: {new_w}x{new_h}")
            return True
        except Exception as e:
            print(f"[ERROR] 显示失败: {img_path}, 错误: {e}")
            return False

    def show_image_at(self, img_path: str, position: tuple, scale_factor: float = 1.0) -> bool:
        """根据缩放因子缩放图像，指定位置显示，并填充背景色"""
        try:
            x, y, w, h = self.geom

            # 打开图片并转换为 RGB 模式
            pil = Image.open(img_path).convert("RGB")
            in_w, in_h = pil.size

            # 计算缩放后的新尺寸
            new_w = int(in_w * scale_factor)
            new_h = int(in_h * scale_factor)
            out = pil.resize((new_w, new_h), Resampling.LANCZOS)

            # 创建一个背景色填充的空白画布
            canvas_image = Image.new("RGB", (w, h), self.bg)

            # 使用传入的 position 来控制图像的位置
            pos_x, pos_y = position
            offset_x = pos_x - new_w // 2
            offset_y = pos_y - new_h // 2

            # 将缩放后的图像粘贴到空白画布上
            canvas_image.paste(out, (offset_x, offset_y))

            # 转换为 Tkinter 图像并显示
            self.canvas.delete("all")
            self._tkimg = ImageTk.PhotoImage(canvas_image)
            self.canvas.create_image(w // 2, h // 2, image=self._tkimg, anchor="center")

            print(f"[INFO] 成功显示: {img_path}")
            print(f"原始尺寸: {in_w}x{in_h}, 缩放后的尺寸: {new_w}x{new_h}, 显示位置: {position}")
            return True
        except Exception as e:
            print(f"[ERROR] 显示失败: {img_path}, 错误: {e}")
            return False

    def start(self):
        """进入事件循环（只进一次）"""
        self.root.mainloop()

    def close(self):
        try:
            self.root.destroy()
        except Exception:
            pass

def display_image(display_image_path, monitor_idx, scale_factor):
    """在新线程中显示图片，保持显示器持续显示"""
    try:
        # 创建 Screen 对象，选择第一个显示器（默认选择第一个显示器）
        scr = Screen(monitor_index=monitor_idx, bg="black")
        
        # 显示图像（请确保图像路径正确）
        if not os.path.exists(display_image_path):
            print("[ERR] 显示器图片路径无效")
            exit(3)
        
        scr.show_image(display_image_path, scale_factor)
        scr.start()  # 启动 Tkinter 事件循环
    except Exception as e:
        print(f"[ERROR] 显示图像时出错: {e}")


def main():
    screen = Screen(monitor_index=2, bg="black")

    img_path1 = r"D:\qjy\camera_slm_pipeline\data\example\分辨率测试卡.JPG"
    scale_factor = 0.1
    success = screen.show_image(img_path1, scale_factor)
    if success:
        screen.root.update()  # 强制刷新窗口
        print("[INFO] 第一张图像已显示，开始事件循环...")
        # time.sleep(50000)
    else:
        print("[ERROR] 显示第一张图像失败")

    # img_path2 = r"D:\Lzy\dataset\dogvscat v2\1.png"
    # success = screen.show_image(img_path2, scale_factor)
    # if success:
    #     screen.root.update()
    #     print("[INFO] 第二张图像已显示，开始事件循环...")
    #     time.sleep(5)
    # else:
    #     print("[ERROR] 显示第二张图像失败")

    # scale_factor = 0.2

    # 在新的位置显示第三张图像
    img_path3 = r"D:\qjy\camera_slm_pipeline\data\example\分辨率测试卡.JPG"
    position = (1500, 700)  # 指定第三张图片显示的位置 (x, y)
    success = screen.show_image_at(img_path3, position, scale_factor)

    if success:
        print("[INFO] 第三张图像已显示，开始事件循环...")
        screen.start()  # 进入事件循环，保持显示
    else:
        print("[ERROR] 显示第三张图像失败")



if __name__ == "__main__":
    main()
