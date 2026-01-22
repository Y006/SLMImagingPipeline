import numpy as np
from PIL import Image
import torch
import os
from loguru import logger

# 自动设备选择
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = "cpu"
# ========== 工具函数 ==========

def padded_diffuser(path):
    logger.debug(f"[padded_diffuser] 加载PSF文件: {path}")
    psf = np.array(Image.open(path))/255.0
    logger.debug(f"[padded_diffuser] PSF形状: {psf.shape}, 值域: [{psf.min():.4f}, {psf.max():.4f}]")
    return psf

def ramp_padding(img,pad_width=((105, 105),(190,190),(0,0))):
    img = np.pad(img,pad_width,mode='linear_ramp')
    return img

def WieNer(blur, psf, delta):
    logger.info(f"[WieNer] 开始维纳滤波重建 - delta(正则化参数)={delta:.2e}")
    logger.debug(f"[WieNer] 输入模糊图像形状: {blur.shape}, 值域: [{blur.min():.4f}, {blur.max():.4f}]")
    logger.debug(f"[WieNer] 输入PSF形状: {psf.shape}, 值域: [{psf.min():.4f}, {psf.max():.4f}]")
    
    # FFT变换
    blur_fft = torch.fft.rfft2(blur)
    psf_fft = torch.fft.rfft2(psf)
    logger.debug(f"[WieNer] FFT变换完成 - blur_fft形状:{blur_fft.shape}, psf_fft形状:{psf_fft.shape}")
    
    # 计算维纳滤波器
    H_conj = torch.conj(psf_fft)
    H_abs = torch.abs(psf_fft) ** 2
    logger.debug(f"[WieNer] H_abs值域: [{H_abs.min():.4e}, {H_abs.max():.4e}]")
    
    wiener_filter = H_conj / (H_abs + delta)
    logger.debug(f"[WieNer] 维纳滤波器计算完成, 幅值范围:[{torch.abs(wiener_filter).min():.4e}, {torch.abs(wiener_filter).max():.4e}]")
    
    # 频域相乘并逆FFT
    out = torch.fft.irfft2(wiener_filter * blur_fft)
    logger.debug(f"[WieNer] 逆FFT完成 - 输出形状:{out.shape}, 值域:[{out.min():.4f}, {out.max():.4f}]")
    
    result = torch.fft.ifftshift(out, dim=(2, 3))
    logger.info(f"[WieNer] 维纳滤波重建完成 - 最终输出形状:{result.shape}")
    return result

def normalize_tensor_img(tensor):
    logger.debug(f"[normalize_tensor_img] 输入形状:{tensor.shape}, 值域:[{tensor.min():.4f}, {tensor.max():.4f}]")
    tensor = tensor - tensor.min()
    normalized = tensor / tensor.max()
    logger.debug(f"[normalize_tensor_img] 归一化后值域:[{normalized.min():.4f}, {normalized.max():.4f}]")
    return normalized

def save_tensor_img(tensor, path, crop_coords=None):
    logger.debug(f"[save_tensor_img] 开始保存图像到: {path}")
    logger.debug(f"[save_tensor_img] 输入tensor形状: {tensor.shape}")
    
    img_np = tensor.squeeze(0).permute(1, 2, 0).detach().cpu().numpy()
    logger.debug(f"[save_tensor_img] 转换为numpy后形状: {img_np.shape}")
    
    img_np = normalize_tensor_img(torch.tensor(img_np)).numpy()
    
    if crop_coords:
        y1, y2, x1, x2 = crop_coords
        logger.debug(f"[save_tensor_img] 裁剪区域: y[{y1}:{y2}], x[{x1}:{x2}]")
        img_np = img_np[y1:y2, x1:x2, :]
        logger.debug(f"[save_tensor_img] 裁剪后形状: {img_np.shape}")
    
    img_uint8 = np.clip(img_np * 255.0, 0, 255).astype(np.uint8)
    Image.fromarray(img_uint8).save(path)
    logger.info(f"[save_tensor_img] 图像已保存: {path}, 尺寸:{img_uint8.shape}")

# ========== 主流程函数 ==========

def process_one_pair(psf_path, blur_path, delta=80000, output_path="Test_Wiener.png"):
    logger.info(f"========== 开始处理一对图像 ==========")
    logger.info(f"[process_one_pair] PSF路径: {psf_path}")
    logger.info(f"[process_one_pair] 模糊图像路径: {blur_path}")
    logger.info(f"[process_one_pair] 输出路径: {output_path}")
    logger.info(f"[process_one_pair] delta参数: {delta:.2e}")
    
    # 加载PSF
    logger.debug(f"[process_one_pair] 步骤1: 加载PSF")
    psf_np = padded_diffuser(psf_path)

    psf_tensor = torch.tensor(psf_np).permute(2, 0, 1).sum(dim=0, keepdim=True).unsqueeze(0).to(device)
    logger.debug(f"[process_one_pair] PSF转换为tensor: {psf_tensor.shape}, 设备:{device}")
    logger.debug(f"[process_one_pair] PSF合并通道后值域: [{psf_tensor.min():.4f}, {psf_tensor.max():.4f}]")

    # 加载模糊图像
    logger.debug(f"[process_one_pair] 步骤2: 加载模糊图像")
    blur_np = np.array(Image.open(blur_path))/255.0
    logger.debug(f"[process_one_pair] 模糊图像形状: {blur_np.shape}, 值域: [{blur_np.min():.4f}, {blur_np.max():.4f}]")
    
    blur_tensor = torch.tensor(blur_np).permute(2, 0, 1).unsqueeze(0).to(device)
    logger.debug(f"[process_one_pair] 模糊图像转换为tensor: {blur_tensor.shape}, 设备:{device}")
    
    # Wiener 反卷积
    logger.debug(f"[process_one_pair] 步骤3: 执行维纳滤波")
    result = WieNer(blur_tensor, psf_tensor/psf_tensor.max(), delta)
    
    # 保存图像，裁剪区域可选
    logger.debug(f"[process_one_pair] 步骤4: 保存重建结果")
    save_tensor_img(result, output_path)
    logger.info(f"========== 处理完成 ==========\n")

# ========== 示例运行 ==========

if __name__ == '__main__':
    logger.info("维纳滤波重建程序启动")
    
    psf_path = r'D:\qjy\camera_slm_pipeline\output\exp020-1013-99c180\Image_20251013211130508.png'
    blur_path = r'D:\qjy\camera_slm_pipeline\output\exp020-1013-99c180\Image_20251013211809704.png'
    output_path = r'D:\qjy\camera_slm_pipeline\output\exp020-1013-99c180\Image_r.jpg'

    # folder_path = os.path.dirname(psf_path)
    # blur_filename = os.path.basename(blur_path) # 获取文件名，如 "m-lcd-1234.jpg"
    # lcd_part = blur_filename.split('-')[1] # 获取 "lcd"
    # psf_time = int(psf_path.split('-')[-1].split('.')[0]) # 取 psf 文件名中的四位数字
    # blur_time = int(blur_path.split('-')[-1].split('.')[0]) # 取 blur 文件名中的四位数字
    # time_part = max(psf_time, blur_time)
    # output_path = os.path.join(folder_path,f'r-{lcd_part}-{time_part:04d}.png')

    process_one_pair(
        psf_path=psf_path,
        blur_path=blur_path,
        delta=100000000,
        output_path=output_path
    )

    logger.info("程序执行完成！")