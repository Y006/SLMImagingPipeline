import numpy as np
from PIL import Image
import torch
import os

# 自动设备选择
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = "cpu"
# ========== 工具函数 ==========

def padded_diffuser(path):
    psf = np.array(Image.open(path))/255.0
    return psf

def ramp_padding(img,pad_width=((105, 105),(190,190),(0,0))):
    img = np.pad(img,pad_width,mode='linear_ramp')
    return img

def WieNer(blur, psf, delta):
    blur_fft = torch.fft.rfft2(blur)
    psf_fft = torch.fft.rfft2(psf)
    H_conj = torch.conj(psf_fft)
    H_abs = torch.abs(psf_fft) ** 2
    wiener_filter = H_conj / (H_abs + delta)
    out = torch.fft.irfft2(wiener_filter * blur_fft)
    return torch.fft.ifftshift(out, dim=(2, 3))
    # return out

def normalize_tensor_img(tensor):
    tensor = tensor - tensor.min()
    return tensor / tensor.max()

def save_tensor_img(tensor, path, crop_coords=None):
    img_np = tensor.squeeze(0).permute(1, 2, 0).detach().cpu().numpy()
    img_np = normalize_tensor_img(torch.tensor(img_np)).numpy()
    if crop_coords:
        y1, y2, x1, x2 = crop_coords
        img_np = img_np[y1:y2, x1:x2, :]
    
    img_uint8 = np.clip(img_np * 255.0, 0, 255).astype(np.uint8)
    Image.fromarray(img_uint8).save(path)

# ========== 主流程函数 ==========

def process_one_pair(psf_path, blur_path, delta=80000, output_path="Test_Wiener.png"):
    # 加载PSF
    psf_np = padded_diffuser(psf_path)

    psf_tensor = torch.tensor(psf_np).permute(2, 0, 1).sum(dim=0, keepdim=True).unsqueeze(0).to(device)

    # 加载模糊图像
    blur_np = np.array(Image.open(blur_path))/255.0
    blur_tensor = torch.tensor(blur_np).permute(2, 0, 1).unsqueeze(0).to(device)
    # Wiener 反卷积
    result = WieNer(blur_tensor, psf_tensor/psf_tensor.max(), delta)
    # 保存图像，裁剪区域可选
    save_tensor_img(result, output_path)

# ========== 示例运行 ==========

if __name__ == '__main__':
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
    print("Completed！")
