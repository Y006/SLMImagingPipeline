# -*- coding: utf-8 -*-
# ========= 参数区：按需修改 =========
FOLDER = r"D:\qjy\camera_slm_pipeline\data\fza_bin_gen"  # 输入图片文件夹路径
RADIUS = 40                      # 圆半径（像素）
# 若想自定义圆心，设置为 (x, y) 像素坐标；为 None 则使用图片中心
CENTER = None
# 圆外填充值：黑色为 0
FILL_VALUE = 0
# ===================================

import os
from pathlib import Path
from PIL import Image
import numpy as np

# 支持的图片扩展名
EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

folder = Path(FOLDER)
assert folder.exists() and folder.is_dir(), f"路径不存在或不是文件夹：{folder}"

# 输出目录：原文件夹名 + '_masked_r{半径}'
out_dir = folder.parent / f"{folder.name}_masked_r{RADIUS}"
out_dir.mkdir(parents=True, exist_ok=True)

# 收集图片
files = [p for p in sorted(folder.iterdir()) if p.suffix.lower() in EXTS]
assert files, f"文件夹中未找到图片：{folder}"

# 读取第一张图获取尺寸
with Image.open(files[0]) as im0:
    im0 = im0.convert("RGB")
    W, H = im0.size

# 圆心
if CENTER is None:
    cx, cy = W // 2, H // 2
else:
    cx, cy = CENTER
    assert 0 <= cx < W and 0 <= cy < H, "CENTER 超出图像范围"

# 生成圆形掩膜
yy, xx = np.ogrid[:H, :W]
mask = (xx - cx) * (xx - cx) + (yy - cy) * (yy - cy) <= (RADIUS * RADIUS)
mask_inv = ~mask

for p in files:
    with Image.open(p) as im:
        im = im.convert("RGB")
        arr = np.array(im)

        out = arr.copy()
        out[mask_inv] = FILL_VALUE

        out_img = Image.fromarray(out)
        out_path = out_dir / p.name

        save_kwargs = {}
        if out_path.suffix.lower() in {".jpg", ".jpeg"}:
            save_kwargs["quality"] = 95

        out_img.save(out_path, **save_kwargs)

print("处理完成！")
print(f"输入目录：{folder}")
print(f"输出目录：{out_dir}")
print(f"图像尺寸：{W}x{H}，圆心：({cx},{cy})，半径：{RADIUS}，圆外填充值：{FILL_VALUE}")
