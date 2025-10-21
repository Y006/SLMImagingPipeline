import torch
import numpy as np

# ================================
# 基础图像操作
# ================================
def roll_n(X, axis, n):
    """沿指定轴循环平移 (类似 numpy.roll)，常用于周期边界条件。"""
    f_idx = tuple(slice(None, None, None) if i != axis else slice(0, n, None) for i in range(X.dim()))
    b_idx = tuple(slice(None, None, None) if i != axis else slice(n, None, None) for i in range(X.dim()))
    front = X[f_idx]
    back = X[b_idx]
    return torch.cat([back, front], axis)
def crop(model, x):
    """裁剪张量，去除之前的零填充，恢复到原始图像大小。"""
    C01 = model.PAD_SIZE0; C02 = model.PAD_SIZE0 + model.DIMS0              # Crop indices 
    C11 = model.PAD_SIZE1; C12 = model.PAD_SIZE1 + model.DIMS1              # Crop indices 
    return x[:, :, C01:C02, C11:C12]
def normalize_image(image):
    """归一化图像像素值，通常缩放到 [0,1] 或 [-1,1]。"""
    out_shape = image.shape
    image_flat = image.reshape((out_shape[0],out_shape[1]*out_shape[2]*out_shape[3]))
    image_max,_ = torch.max(image_flat,1)
    image_max_eye = torch.eye(out_shape[0], dtype = torch.float32, device=image.device)*1/image_max
    image_normalized = torch.reshape(torch.matmul(image_max_eye, image_flat), (out_shape[0],out_shape[1],out_shape[2],out_shape[3]))
    return image_normalized

# ================================
# FFT / 复数运算工具
# ================================
def complex_abs(t1):
    """计算复数张量的幅值 (|z|)。"""
    real1, imag1 = torch.unbind(t1, dim=2)
    return torch.sqrt(real1**2 + imag1**2)
def complex_multiplication(t1, t2):
    """执行复数乘法 (z1 * z2)，输入是 [real, imag] 格式的张量。"""
    real1, imag1 = torch.unbind(t1, dim=-1)
    real2, imag2 = torch.unbind(t2, dim=-1)
    return torch.stack([real1 * real2 - imag1 * imag2, real1 * imag2 + imag1 * real2], dim = -1)
def make_complex(r, i = 0):
    """把实部 r 和虚部 i 合并成复数张量 (real, imag)。"""
    if i==0:
        i = torch.zeros_like(r, dtype=torch.float32)
    return torch.stack((r, i), -1)
def make_real(c):
    """提取复数张量的实部 (通常取 ifft 结果的实数部分)。"""
    out_r, _ = torch.unbind(c,-1)
    return out_r
def batch_ifftshift2d(x):
    """对 2D 张量做 ifftshift（移动零频率到图像中心），支持批处理。"""
    real, imag = torch.unbind(x, -1)
    for dim in range(len(real.size()) - 1, 0, -1):
        real = roll_n(real, axis=dim, n=real.size(dim)//2)
        imag = roll_n(imag, axis=dim, n=imag.size(dim)//2)
    return torch.stack((real, imag), -1)  # last dim=2 (real&imag)

# ================================
# 正则化 / 梯度算子
# ================================
def make_laplacian(model):
    """构造拉普拉斯算子 (LtL)，用于平滑正则化。"""
    lapl = np.zeros([model.DIMS0*2,model.DIMS1*2])
    lapl[0,0] =4.; 
    lapl[0,1] = -1.; lapl[1,0] = -1.; 
    lapl[0,-1] = -1.; lapl[-1,0] = -1.; 

    LTL = np.abs(np.fft.fft2(lapl))
    return LTL
def Ltv_tf(a, b):
    """计算梯度算子的转置 (散度)，是 TV 正则化的伴随算子。"""
    return torch.cat([a[:,:, 0:1,:], a[:,:, 1:, :]-a[:,:, :-1, :], -a[:,:,-1:,:]],
                2) + torch.cat([b[:,:,:,0:1], b[:, :, :, 1:]-b[:, :, :,  :-1], -b[:,:, :,-1:]],3)
def L_tf(a):
    """计算图像的梯度 (∇s)，返回水平/垂直方向差分。"""
    xdiff = a[:,:, 1:, :]-a[:,:, :-1, :]
    ydiff = a[:,:, :, 1:]-a[:,:, :, :-1]
    return -xdiff, -ydiff
def soft_2d_gradient2_rgb(model, v,h,tau):
    """
    对梯度向量 (v,h) 执行 soft-thresholding 操作 (稀疏化)，
    这是 ADMM 中 TV 正则化的 u 更新步骤。
    """
    z0 = torch.tensor(0, dtype = torch.float32, device=model.cuda_device)
    z1 = torch.zeros(model.batch_size, 3, 1, model.DIMS1*2, dtype = torch.float32, device=model.cuda_device)
    z2 = torch.zeros(model.batch_size, 3, model.DIMS0*2, 1, dtype= torch.float32, device=model.cuda_device)

    vv = torch.cat([v, z1] , 2)
    hh = torch.cat([h, z2] , 3)
    
    mag = torch.sqrt(vv*vv + hh*hh)
    magt = torch.max(mag - tau, z0, out=None)
    mag = torch.max(mag - tau, z0, out=None) + tau
    
    mmult = magt/(mag)#+1e-5)
    if torch.any(mmult != mmult):
        print('here')
    if torch.any(v != v):
        print('there')

    return v*mmult[:,:, :-1,:], h*mmult[:,:, :,:-1]

# ================================
# 系统算子 (前向 H / 伴随 H*)
# ================================
def pad_zeros_torch(model, x):
    """对输入张量做零填充 (padding)，使其与 FFT 卷积尺寸一致。"""
    PADDING = (model.PAD_SIZE1, model.PAD_SIZE1, model.PAD_SIZE0, model.PAD_SIZE0)
    return torch.nn.functional.pad(x, PADDING, 'constant', 0)
def Hfor(model, x):
    """前向算子 H (卷积/成像)，计算 Hx。"""
    xc = torch.stack((x, torch.zeros_like(x, dtype=torch.float32)), -1)
    X = torch.view_as_real(torch.fft.fft2(torch.view_as_complex(xc)))
    HX = complex_multiplication(model.H, X)
    out = torch.view_as_real(torch.fft.ifft2(torch.view_as_complex(HX)))
    out_r, _ = torch.unbind(out, -1)
    return out_r
def Hadj(model, x):
    """伴随算子 H* (转置卷积)，计算 H^T x。"""
    xc = torch.stack((x, torch.zeros_like(x, dtype=torch.float32)), -1)
    X = torch.view_as_real(torch.fft.fft2(torch.view_as_complex(xc)))
    HX = complex_multiplication(model.Hconj, X)
    out = torch.view_as_real(torch.fft.ifft2(torch.view_as_complex(HX)))
    out_r, _ = torch.unbind(out, -1)
    return out_r

def admm(model, in_vars, alpha2k_1, alpha2k_2, CtC, Cty, n):  
    """
    单步 ADMM 更新 (RGB 版本)
    
    Args:
        model: 包含 mu1/mu2/mu3/tau 等参数的模型
        in_vars: [sk, alpha1k, alpha3k, Hskp]
        alpha2k_1, alpha2k_2: 对偶变量 (梯度方向)
        CtC, Cty: 系统矩阵相关 (零填充后的 H^T H, H^T y)
        n: 当前迭代索引

    Returns:
        out_vars: [skp, alpha1k_up, alpha3k_up, Hskp_up]
        alpha2k_1up, alpha2k_2up: 更新后的对偶变量
    """

    # === 取出输入变量 ===
    sk, alpha1k, alpha3k, Hskp = in_vars
    mu1, mu2, mu3 = model.mu1[n], model.mu2[n], model.mu3[n]
    tau = model.tau[n]

    # === 残差存储 (仅用于调试/日志) ===
    dual_resid_s, primal_resid_s = [], []
    dual_resid_u, primal_resid_u = [], []
    dual_resid_w, primal_resid_w = [], []

    # === 预计算矩阵系数 ===
    Smult = 1 / (mu1 * model.HtH + mu2 * model.LtL + mu3)  # 对应频域滤波器
    Vmult = 1 / (CtC + mu1)

    # ------------------------------------------------------------------
    # Step 1: 更新 u ← soft(∇s + α2/μ2, τ/μ2)
    # ------------------------------------------------------------------
    Lsk1, Lsk2 = L_tf(sk)  
    ukp_1, ukp_2 = soft_2d_gradient2_rgb(
        model, 
        Lsk1 + alpha2k_1 / mu2, 
        Lsk2 + alpha2k_2 / mu2, 
        tau
    )

    # ------------------------------------------------------------------
    # Step 2: 更新 v ← argmin 0.5||y - Hv||^2 + (μ1/2)||v - (...)||^2
    # ------------------------------------------------------------------
    vkp = Vmult * (mu1 * (alpha1k / mu1 + Hskp) + Cty)

    # ------------------------------------------------------------------
    # Step 3: 更新 w ← max(α3/μ3 + s, 0)
    # ------------------------------------------------------------------
    wkp = torch.maximum(alpha3k / mu3 + sk, torch.zeros_like(sk))

    # ------------------------------------------------------------------
    # Step 4: 更新 s ← FFT 解 (频域滤波)
    # ------------------------------------------------------------------
    skp_numer = (
        mu3 * (wkp - alpha3k / mu3)
        + mu1 * Hadj(model, vkp - alpha1k / mu1)
        + mu2 * Ltv_tf(ukp_1 - alpha2k_1 / mu2, ukp_2 - alpha2k_2 / mu2)
    )
    SKP_numer = torch.view_as_real(
    torch.fft.fft2(torch.view_as_complex(make_complex(skp_numer)))
    )
    skp = make_real(
        torch.view_as_real(
            torch.fft.ifft2(torch.view_as_complex(complex_multiplication(make_complex(Smult), SKP_numer)))
        )
    )
    # ------------------------------------------------------------------
    # Step 5: 更新拉格朗日乘子
    # ------------------------------------------------------------------
    # v 对偶
    Hskp_up = Hfor(model, skp)
    r_sv = Hskp_up - vkp
    dual_resid_s.append(mu1 * torch.norm(Hskp - Hskp_up))
    primal_resid_s.append(torch.norm(r_sv))
    alpha1kup = alpha1k + mu1 * r_sv

    # u 对偶
    Lskp1, Lskp2 = L_tf(skp)
    r_su_1, r_su_2 = Lskp1 - ukp_1, Lskp2 - ukp_2
    dual_resid_u.append(mu2 * torch.sqrt(torch.norm(Lsk1 - Lskp1)**2 + torch.norm(Lsk2 - Lskp2)**2))
    primal_resid_u.append(torch.sqrt(torch.norm(r_su_1)**2 + torch.norm(r_su_2)**2))
    alpha2k_1up, alpha2k_2up = alpha2k_1 + mu2 * r_su_1, alpha2k_2 + mu2 * r_su_2

    # w 对偶
    r_sw = skp - wkp
    dual_resid_w.append(mu3 * torch.norm(sk - skp))
    primal_resid_w.append(torch.norm(r_sw))
    alpha3kup = alpha3k + mu3 * r_sw

    # === 输出 ===
    out_vars = torch.stack([skp, alpha1kup, alpha3kup, Hskp_up])

    # Debug 打印 (可选)
    # print(f"[Iter {n}] s.shape={skp.shape}, v.shape={vkp.shape}, w.shape={wkp.shape}")
    # print(f"Residuals: primal_s={primal_resid_s[-1]:.3e}, dual_s={dual_resid_s[-1]:.3e}")

    return out_vars, alpha2k_1up, alpha2k_2up

class ADMM_Net(torch.nn.Module):
    def __init__(self, h, iterations, cuda_device,
                 mu1_init=1e-4, mu2_init=1e-4, mu3_init=1e-4, tau_init=2e-3):
        """
        初始化 ADMM 网络结构，可自定义 mu1, mu2, mu3, tau 初始值。

        Args:
            h (np.ndarray): PSF
            iterations (int): ADMM 迭代次数
            cuda_device: 运行设备
            mu1_init, mu2_init, mu3_init, tau_init (float): 可调超参数
        """
        super(ADMM_Net, self).__init__()
        
        # ======================
        # 基本参数
        # ======================
        self.iterations  = iterations
        self.batch_size  = 1
        self.cuda_device = cuda_device

        # 初始化 ADMM 学习参数 (mu1, mu2, mu3, tau)
        # self.initialize_learned_variables()
        self.initialize_learned_variables(mu1_init, mu2_init, mu3_init, tau_init)

        # ======================
        # 图像维度与填充
        # ======================
        self.DIMS0     = h.shape[0]  # 高度
        self.DIMS1     = h.shape[1]  # 宽度
        self.PAD_SIZE0 = self.DIMS0 // 2
        self.PAD_SIZE1 = self.DIMS1 // 2

        # ======================
        # PSF (点扩散函数)
        # ======================
        self.h_var = torch.nn.Parameter(
            torch.tensor(h, dtype=torch.float32, device=self.cuda_device),
            requires_grad=False
        )
            
        self.h_zeros = torch.nn.Parameter(
            torch.zeros(self.DIMS0*2, self.DIMS1*2, dtype=torch.float32, device=self.cuda_device),
            requires_grad=False
        )

        # 组合为复数形式 (实部=PSF，虚部=0)，再做 FFT
        self.h_complex = torch.stack(
            (pad_zeros_torch(self, self.h_var), self.h_zeros), 2
        ).unsqueeze(0)

        # ======================
        # 频域算子
        # ======================
        self.H = torch.view_as_real(
            torch.fft.fft2(torch.view_as_complex(batch_ifftshift2d(self.h_complex).squeeze()))
        )
        self.Hconj =  self.H* torch.tensor([1,-1], dtype = torch.float32, device=self.cuda_device) 
        self.HtH = complex_abs(complex_multiplication(self.H, self.Hconj))

        # 拉普拉斯算子 (TV 正则)
        self.LtL = torch.nn.Parameter(
            torch.tensor(make_laplacian(self), dtype=torch.float32, device=self.cuda_device),
            requires_grad=False
        )

    def initialize_learned_variables(self, mu1_init, mu2_init, mu3_init, tau_init):
            """初始化可学习或可调参数"""
            self.mu1 = torch.ones(self.iterations, dtype=torch.float32, device=self.cuda_device) * mu1_init
            self.mu2 = torch.ones(self.iterations, dtype=torch.float32, device=self.cuda_device) * mu2_init
            self.mu3 = torch.ones(self.iterations, dtype=torch.float32, device=self.cuda_device) * mu3_init
            self.tau = torch.ones(self.iterations, dtype=torch.float32, device=self.cuda_device) * tau_init

    def forward(self, inputs):    
        """
        Forward 推理流程 (ADMM 重建)
        
        Args:
            inputs: 已归一化的 Diffuser 图像 (y)
        
        Returns:
            x_outn: 归一化后的最终重建结果
        """
        # ------------------------------------------------------------------
        # Step 1: 输入准备
        # ------------------------------------------------------------------
        y = inputs

        # 构造零填充
        Cty = pad_zeros_torch(self, y)                      
        CtC = pad_zeros_torch(self, torch.ones_like(y))     

        # ------------------------------------------------------------------
        # Step 2: 初始化变量
        # ------------------------------------------------------------------
        in_vars, a2k_1_list, a2k_2_list = [], [], []

        sk      = torch.zeros_like(Cty, dtype=torch.float32)
        alpha1k = torch.zeros_like(Cty, dtype=torch.float32)
        alpha3k = torch.zeros_like(Cty, dtype=torch.float32)
        Hskp    = torch.zeros_like(Cty, dtype=torch.float32)

        # 注意 alpha2 是梯度方向，尺寸比 sk 少一行/列
        alpha2k_1 = torch.zeros_like(sk[:, :, :-1, :], dtype=torch.float32)  
        alpha2k_2 = torch.zeros_like(sk[:, :, :, :-1], dtype=torch.float32)

        # 保存初始化
        a2k_1_list.append(alpha2k_1)
        a2k_2_list.append(alpha2k_2)
        in_vars.append(torch.stack([sk, alpha1k, alpha3k, Hskp]))

        # ------------------------------------------------------------------
        # Step 3: 主循环 (ADMM 迭代)
        # ------------------------------------------------------------------
        for i in range(self.iterations):
            # print(f"\n--- Iteration {i+1}/{self.iterations} ---")
            # print(f"[Iter {i}] mu1={self.mu1[i].item():.2e}, mu2={self.mu2[i].item():.2e}, mu3={self.mu3[i].item():.2e}, tau={self.tau[i].item():.2e}")

            out_vars, alpha2k_1_up, alpha2k_2_up = admm(
                self, 
                in_vars[-1], 
                a2k_1_list[-1], 
                a2k_2_list[-1], 
                CtC, Cty, 
                i
            )

            # print(f"[ADMM] out_vars.shape = {out_vars.shape}")

            # 保存结果
            in_vars.append(out_vars)
            a2k_1_list.append(alpha2k_1_up)
            a2k_2_list.append(alpha2k_2_up)

            # 当前迭代输出 (裁剪+归一化)
            x_out = crop(self, in_vars[-1][0])
            x_outn = normalize_image(x_out)
            # print(f"[Iter {i+1}] x_out.shape = {x_out.shape}, x_outn.shape = {x_outn.shape}")

            # 保存中间结果 (可视化/调试用)
            self.in_list = in_vars

        return x_outn


