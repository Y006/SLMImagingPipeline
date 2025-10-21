import glob
import os
from typing import Tuple, List
import numpy as np
import cv2
import matplotlib.pyplot as plt

# =========================
# CONFIG：这里改成你的路径/参数
# 方式1：用通配符批量读取
PATTERN = r"D:\qjy\camera_slm_pipeline\output\2025-10-15-exp003\*-psf-*.jpg"

# 方式2：列出具体文件（优先于 PATTERN；不想用就设为空列表）
PATHS = [
    r"D:\qjy\camera_slm_pipeline\output\2025-10-15-exp003\001-psf-Kis0.jpg",

]

PIXEL_UM = 3.45   # 像素尺寸(微米)。None -> 横轴用 cycles/pixel；否则自动换算到 lp/mm
ROI = 256         # 从PSF裁一个方形ROI做FFT
PAD = 6.0         # 零填充倍数(相对ROI边长)
NORM = "max"       # 'dc'->MTF(0)=1; 'max'->除以全局最大; 'none'->不归一化
SMOOTH = 9        # 对1D MTF做简单平滑（箱型平均窗口）
OUT_FIG = r"D:\qjy\camera_slm_pipeline\output\2025-10-15-exp003\mtf_compare.png"
# =========================


def read_gray(path: str) -> np.ndarray:
    """Read image as float32 grayscale in [0, 1]. (支持中文/空格路径)"""
    img = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(path)
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = img.astype(np.float32)
    if img.max() > 0:
        img /= img.max()
    return img


def estimate_background(img: np.ndarray, frac: float = 0.12) -> float:
    """用四角的中位数估计背景"""
    h, w = img.shape
    dy, dx = int(h * frac), int(w * frac)
    patches = np.concatenate([
        img[:dy, :dx].ravel(),
        img[:dy, -dx:].ravel(),
        img[-dy:, :dx].ravel(),
        img[-dy:, -dx:].ravel()
    ])
    return float(np.median(patches))


def extract_centered_roi(img: np.ndarray, roi: int = 256) -> np.ndarray:
    """围绕最亮像素裁ROI，并把峰值对到中心"""
    H, W = img.shape
    y0, x0 = np.unravel_index(np.argmax(img), img.shape)
    half = roi // 2
    y1, y2 = max(0, y0 - half), min(H, y0 + half)
    x1, x2 = max(0, x0 - half), min(W, x0 + half)
    roi_img = np.zeros((roi, roi), dtype=np.float32)
    y_dst1 = half - (y0 - y1)
    x_dst1 = half - (x0 - x1)
    roi_img[y_dst1:y_dst1 + (y2 - y1), x_dst1:x_dst1 + (x2 - x1)] = img[y1:y2, x1:x2]
    yy, xx = np.unravel_index(np.argmax(roi_img), roi_img.shape)
    dy = roi // 2 - yy
    dx = roi // 2 - xx
    roi_img = np.roll(np.roll(roi_img, dy, axis=0), dx, axis=1)
    return roi_img


def hann2d(h: int, w: int) -> np.ndarray:
    wy = np.hanning(h)
    wx = np.hanning(w)
    return np.outer(wy, wx).astype(np.float32)


def next_pow2(n: int) -> int:
    return 1 << (int(n - 1).bit_length())


def radial_average(img: np.ndarray, max_r: int = None) -> np.ndarray:
    """对已fftshift的方阵做径向平均，返回 r=0..max_r-1 的平均值"""
    H, W = img.shape
    assert H == W
    c0 = H // 2
    y, x = np.indices(img.shape)
    r = np.sqrt((y - c0) ** 2 + (x - c0) ** 2)
    r_int = np.floor(r).astype(np.int32).ravel()
    if max_r is None:
        max_r = H // 2
    flat = img.ravel().astype(np.float64)
    sums = np.bincount(r_int, weights=flat, minlength=max_r)
    counts = np.bincount(r_int, minlength=max_r)
    counts[counts == 0] = 1
    return (sums[:max_r] / counts[:max_r]).astype(np.float32)


def compute_mtf_from_psf(psf: np.ndarray,
                         roi: int = 256,
                         zero_pad_factor: float = 2.0,
                         apodize: bool = True,
                         norm: str = "dc") -> Tuple[np.ndarray, np.ndarray]:
    """从PSF计算1D MTF曲线，返回(frequency, mtf)，频率单位cycles/pixel"""
    bg = estimate_background(psf)
    psf = psf - bg
    psf[psf < 0] = 0
    psf = cv2.GaussianBlur(psf, (0, 0), sigmaX=0.4, sigmaY=0.4, borderType=cv2.BORDER_REPLICATE)

    roi_img = extract_centered_roi(psf, roi=roi)
    if apodize:
        roi_img = roi_img * hann2d(*roi_img.shape)

    pad = int(next_pow2(int(roi * zero_pad_factor)))
    pad = max(pad, roi)
    field = np.zeros((pad, pad), dtype=np.float32)
    s = (pad - roi) // 2
    field[s:s + roi, s:s + roi] = roi_img

    H = np.fft.fftshift(np.fft.fft2(field))
    mag = np.abs(H).astype(np.float32)

    if norm == "dc":
        dc = mag[pad // 2, pad // 2]
        if dc > 0:
            mag = mag / dc
    elif norm == "max":
        m = mag.max()
        if m > 0:
            mag = mag / m
    elif norm == "none":
        pass
    else:
        raise ValueError("norm must be one of ['dc','max','none']")

    mtf = radial_average(mag, max_r=pad // 2)
    f = np.arange(mtf.size, dtype=np.float32) / pad   # cycles/pixel
    keep = f <= 0.5 + 1e-9
    return f[keep], mtf[keep]


def plot_mtfs(paths: List[str],
              pixel_um: float = None,
              roi: int = 256,
              zero_pad_factor: float = 2.0,
              norm: str = "dc",
              smooth: int = 1,
              out: str = "mtf_compare.png"):
    plt.figure(figsize=(7.5, 4.6), dpi=140)
    for p in paths:
        img = read_gray(p)
        f_cyc_pix, mtf = compute_mtf_from_psf(img, roi=roi, zero_pad_factor=zero_pad_factor,
                                              apodize=True, norm=norm)
        if smooth and smooth > 1:
            k = int(smooth)
            mtf = np.convolve(mtf, np.ones(k) / k, mode="same")

        if pixel_um is not None:
            f = f_cyc_pix / (pixel_um / 1000.0)  # lp/mm
            x_label = "Frequency (lp/mm)"
            nyq = 0.5 / (pixel_um / 1000.0)
        else:
            f = f_cyc_pix
            x_label = "Frequency (cycles/pixel)"
            nyq = 0.5

        plt.plot(f, mtf, linewidth=1.5, label=os.path.basename(p))

    plt.xlabel(x_label)
    plt.ylabel("MTF (normalized)" if norm == "dc" else "MTF magnitude")
    plt.title("Modulation Transfer Functions")
    plt.grid(True, alpha=0.35)
    plt.legend(fontsize=8, loc="best")
    plt.axvline(nyq, linestyle="--", linewidth=1.0)  # 标出奈奎斯特
    plt.tight_layout()
    plt.savefig(out, bbox_inches="tight")
    print(f"Saved: {out}")
    plt.show()


def main():
    paths = list(PATHS) if PATHS else sorted(glob.glob(PATTERN))
    if not paths:
        raise SystemExit(f"No files found. Check PATTERN or PATHS in CONFIG.\nPATTERN={PATTERN}")
    plot_mtfs(paths, pixel_um=PIXEL_UM, roi=ROI,
              zero_pad_factor=PAD, norm=NORM, smooth=SMOOTH, out=OUT_FIG)


if __name__ == "__main__":
    main()