import os
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
from mask import (
    CodedAperture,
    MultiLensArray,
    PhaseContour,
    FresnelZoneAperture,
    RandomBinaryMask
)

# 通用参数（SLM环境下的默认值）
DEFAULT_FEATURE_SIZE = 1e-5
DEFAULT_DISTANCE = 4e-3


# -----------------------------------------------------------
# 统一生成接口
# -----------------------------------------------------------
def generate_mask(mask_type, out_dir, **kwargs):
    """根据 mask_type 创建对应的掩膜，并保存 PNG"""

    if not os.path.exists(out_dir):
        print(f"❌ 错误：输出目录不存在：{out_dir}")
        sys.exit(1)
    print(f"✅ 输出目录存在：{out_dir}")

    mask_type = mask_type.lower()

    # ---- 根据类型创建不同掩膜 ----
    if mask_type == "random":
        mask_obj = RandomBinaryMask(
            fill_ratio=kwargs.get("fill_ratio", 0.5),
            resolution=kwargs.get("resolution", (512, 512)),
            feature_size=DEFAULT_FEATURE_SIZE,
            distance_sensor=DEFAULT_DISTANCE,
            seed=kwargs.get("seed", 42),
        )

    elif mask_type == "coded":
        mask_obj = CodedAperture(
            method=kwargs.get("method", "MLS"),
            n_bits=kwargs.get("n_bits", 8),
            resolution=kwargs.get("resolution", (512, 512)),
            feature_size=DEFAULT_FEATURE_SIZE,
            distance_sensor=DEFAULT_DISTANCE,
        )

    elif mask_type == "phase":
        mask_obj = PhaseContour(
            noise_period=kwargs.get("noise_period", (16, 16)),
            n_iter=kwargs.get("n_iter", 10),
            resolution=kwargs.get("resolution", (512, 512)),
            feature_size=DEFAULT_FEATURE_SIZE,
            distance_sensor=DEFAULT_DISTANCE,
        )

    elif mask_type == "fza":
        mask_obj = FresnelZoneAperture(
            radius=kwargs.get("radius", 0.56e-3),
            resolution=kwargs.get("resolution", (512, 512)),
            feature_size=DEFAULT_FEATURE_SIZE,
            distance_sensor=DEFAULT_DISTANCE,
        )

    elif mask_type == "multi":
        mask_obj = MultiLensArray(
            N=kwargs.get("N", 50),
            radius_range=kwargs.get("radius_range", (1e-4, 3e-4)),
            resolution=kwargs.get("resolution", (512, 512)),
            feature_size=DEFAULT_FEATURE_SIZE,
            distance_sensor=DEFAULT_DISTANCE,
        )

    else:
        raise ValueError(f"❌ 不支持的 mask 类型: {mask_type}")

    # ---- 保存 PNG ----
    mask_arr = mask_obj.height_map if hasattr(mask_obj, "height_map") and mask_obj.height_map is not None else mask_obj.mask
    mask_u8 = (mask_arr.astype(np.uint8) * 255)

    # ---- 动态生成文件名 ----
    filename_parts = [mask_type]
    # 动态拼接传入参数
    for key, val in kwargs.items():
        if val is None:
            continue
        # resolution 特殊处理：分成两个数字
        if key == "resolution":
            filename_parts.append(f"{key}={val[1]}x{val[0]}")
        elif isinstance(val, (list, tuple)):
            filename_parts.append(f"{key}={'x'.join(map(str, val))}")
        else:
            filename_parts.append(f"{key}={val}")

    filename = "_".join(filename_parts) + ".png"
    png_path = os.path.join(out_dir, filename)

    plt.imsave(png_path, mask_u8, cmap="gray", vmin=0, vmax=255)
    print(f"✅ 已保存 {mask_type} 掩膜: {png_path}")

    # ---- 可视化 ----
    plt.figure(figsize=(8, 6), dpi=150)
    plt.imshow(mask_arr, cmap="gray", interpolation="nearest")
    plt.axis("off")
    plt.title(f"{mask_type.upper()} Mask")
    plt.show()


# -----------------------------------------------------------
# 命令行接口
# -----------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="统一掩膜生成工具")
    parser.add_argument("--type", type=str, required=True, help="掩膜类型 (random, coded, phase, fza, multi)")
    parser.add_argument("--resolution", type=int, nargs=2, default=[768, 1024], help="掩膜分辨率 (height width)")
    parser.add_argument("--fill_ratio", type=float, help="随机掩膜填充比例")
    parser.add_argument("--method", type=str, help="coded 掩膜生成方法")
    parser.add_argument("--n_bits", type=int, help="coded 掩膜比特数")
    parser.add_argument("--noise_period", type=int, nargs=2, help="phase 掩膜噪声周期")
    parser.add_argument("--n_iter", type=int, help="phase 掩膜迭代次数")
    parser.add_argument("--radius", type=float, help="fresnel 掩膜半径")
    parser.add_argument("--seed", type=int, help="随机种子")

    args = parser.parse_args()

    out_dir = "/Users/qiujinyu/Computational_Imaging/无透镜相关代码仓库/SLMImagingPipeline/data/mask_patten_gen"

    # 将命令行参数动态传递给函数
    arg_dict = vars(args)
    mask_type = arg_dict.pop("type")

    generate_mask(mask_type, out_dir, **{k: v for k, v in arg_dict.items() if v is not None})
    
