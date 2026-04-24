# 接受网络输出的退化图和目标图像，计算损失值
import torch.nn as nn
import torch
import torchvision.models as models
import torch.nn.functional as F
import torch.nn.functional as F
from loguru import logger
import numpy as np
# 使用预训练的 VGG19 模型作为感知损失的基础
class PerceptualLoss(nn.Module):
    def __init__(self, layers=['relu2_2', 'relu3_3', 'relu4_3']):
        super(PerceptualLoss, self).__init__()
        
        # Load VGG19 model and modify the first layer for single-channel input
        vgg = models.vgg19(pretrained=True).features
        vgg[0] = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1)  # Modify input channels from 3 to 1
        self.layers = layers
        self.vgg = nn.Sequential(*list(vgg.children())[:36]).eval()  # Keep layers up to relu4_3
        
        for param in self.vgg.parameters():
            param.requires_grad = False  # Freeze VGG parameters
        
        self.layer_mapping = {
            'relu1_2': 3,
            'relu2_2': 8,
            'relu3_3': 17,
            'relu4_3': 26
        }
    
    def forward(self, x, y):
        x = x.to('cuda')
        y = y.to('cuda')
        x_vgg = self.extract_features(x)
        y_vgg = self.extract_features(y)
        loss = 0.0
        for layer in self.layers:
            loss += F.mse_loss(x_vgg[layer], y_vgg[layer])
        return loss
    
    def extract_features(self, x):
        features = {}
        for name, module in self.vgg._modules.items():
            x = module(x)
            if int(name) in self.layer_mapping.values():
                layer_name = list(self.layer_mapping.keys())[list(self.layer_mapping.values()).index(int(name))]
                features[layer_name] = x
        return features



# PSNR计算
def psnr(img1, img2, max_val=255.0):
    """
    计算两个灰度图像张量之间的 PSNR
    :param img1: 第一个灰度图像张量，形状为 (N, 1, H, W) 或 (1, H, W)，值范围为 0-255
    :param img2: 第二个灰度图像张量，形状与 img1 相同，值范围为 0-255
    :param max_val: 图像的最大像素值（默认 255.0）
    :return: PSNR 值
    """
    if img1.ndim == 3:  # 如果是 (1, H, W)，需要扩展为 (N, 1, H, W)
        img1 = img1.unsqueeze(0)
        img2 = img2.unsqueeze(0)
    
    mse = F.mse_loss(img1, img2, reduction='mean')  # 计算均方误差
    psnr = 10 * torch.log10(max_val**2 / mse)  # 根据公式计算 PSNR
    return psnr.cpu()

# SSIM计算
def ssim(img1, img2, max_val=1.0, window_size=11, sigma=1.5):
    """
    计算两个灰度图像张量之间的 SSIM
    :param img1: 第一个灰度图像张量，形状为 (N, 1, H, W) 或 (1, H, W)
    :param img2: 第二个灰度图像张量，形状与 img1 相同
    :param max_val: 图像的最大像素值（默认 1.0）
    :param window_size: 高斯窗口大小
    :param sigma: 高斯分布的标准差
    :return: SSIM 值
    """
    if img1.ndim == 3:  # 如果是 (1, H, W)，需要扩展为 (N, 1, H, W)
        img1 = img1.unsqueeze(0)
        img2 = img2.unsqueeze(0)

    # 生成高斯核
    channels = 1
    coords = torch.arange(window_size, dtype=torch.float32) - window_size // 2
    gauss = torch.exp(-(coords**2) / (2 * sigma**2))
    gauss = gauss / gauss.sum()
    kernel = gauss[:, None] * gauss[None, :]
    kernel = kernel.expand(channels, 1, window_size, window_size).to(img1.device)

    # 计算均值
    mu1 = F.conv2d(img1, kernel, padding=window_size // 2, groups=channels)
    mu2 = F.conv2d(img2, kernel, padding=window_size // 2, groups=channels)
    mu1_sq, mu2_sq = mu1.pow(2), mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    # 计算方差
    sigma1_sq = F.conv2d(img1 * img1, kernel, padding=window_size // 2, groups=channels) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, kernel, padding=window_size // 2, groups=channels) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, kernel, padding=window_size // 2, groups=channels) - mu1_mu2

    # SSIM 常量
    C1 = (0.01 * max_val) ** 2
    C2 = (0.03 * max_val) ** 2

    # 计算 SSIM
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    ssim = ssim_map.mean()
    return ssim.cpu()


def huber_pde_loss(residual, delta=0.1):
    abs_res = torch.abs(residual)
    quadratic = torch.clamp(abs_res, max=delta)
    linear = abs_res - quadratic
    return torch.mean(0.5 * quadratic**2 + delta * linear)


def data(device, output, label):
    U=output[:,0:1]  # 无量纲x方向速度
    V=output[:,1:2]  # 无量纲y方向速度
    W=output[:,2:3]  # 无量纲z方向速度
    P=output[:,3:4]  # 无量纲压强
    T=torch.clamp(output[:,4:5],min=1e-4)  # 无量纲静温
    K=torch.clamp(output[:,5:6],min=1e-10)  # 无量纲湍流动能
    Omega=torch.clamp(output[:,6:7],min=1e-4) # 无量纲比耗散率

    U_t=label[:,0:1]  # 无量纲x方向速度
    V_t=label[:,1:2]  # 无量纲y方向速度
    W_t=label[:,2:3]  # 无量纲z方向速度
    P_t=label[:,3:4]  # 无量纲压强
    T_t=label[:,4:5]  # 无量纲压强
    K_t=label[:,5:6]  # 无量纲压强
    Omega_t=label[:,6:7]  # 无量纲压强

    MSE = nn.MSELoss().to(device)
    L1 = nn.L1Loss().to(device)
    loss_U=MSE(U, U_t)+L1(U, U_t)
    loss_V=MSE(V, V_t)+L1(V, V_t)
    loss_W=MSE(W, W_t)+L1(W, W_t)
    loss_P=MSE(P, P_t)+L1(P, P_t)
    loss_T=MSE(T, T_t)+L1(T, T_t)
    loss_K=MSE(K, K_t)+L1(K, K_t)
    loss_Omega=MSE(Omega, Omega_t)+L1(Omega, Omega_t)

    data_loss=loss_U+loss_V+loss_W+loss_P+loss_T+5*loss_K+10*loss_Omega

    return data_loss
    




# MSE损失
def loss_MSE(device, output, label):
    MSE = nn.MSELoss().to(device)
    loss = MSE(output, label)
    return loss

# L1损失
def loss_L1(device, output, label):
    L1 = nn.L1Loss().to(device)
    loss = L1(output, label)
    return loss

# 感知损失
def loss_criterion(device, output, label):
    criterion = PerceptualLoss().to(device)
    loss = criterion(output, label)
    return loss


def loss_PDE_and_bon(L,M0,T0,P0,input,output,data_min,data_max):
    PDE=RANS_PDE(L,M0,T0,P0,input,output,data_min,data_max)
    # 获取原始残差
    res_cont,res_mx,res_my,res_mz,res_energy,res_k,res_omega=PDE.rans_res()
    # 残差计算MSE（只算单项，不加权，权重在外面动态加）
    loss_cont=huber_pde_loss(res_cont)
    loss_mom=huber_pde_loss(res_mx+res_my+res_mz)
    loss_energy=huber_pde_loss(res_energy)
    loss_k=huber_pde_loss(res_k)
    loss_omega=huber_pde_loss(res_omega)
    # 边界损失
    res_bon=PDE.bon_res()
    loss_bon=huber_pde_loss(res_bon)

    # 返回所有单项损失和残差，方便外面动态加权
    return loss_cont,loss_mom,loss_energy,loss_k,loss_omega,loss_bon,res_cont,res_mx,res_my,res_mz,res_energy,res_k,res_omega


# 反归一化函数
def denormalize_for_pde(output_raw, data_min, data_max):
    """
    将网络输出的 [0, 1] 归一化值还原回物理空间，用于计算 PDE 残差。
    全程使用 Torch 操作，保证计算图不中断，可反向传播。
    
    参数:
        output_raw: [Batch, 7] 网络直接输出 (Sigmoid 后 [0, 1])
        input_min: [11] 归一化参数数组 (Torch Tensor)
        input_max: [11] 归一化参数数组 (Torch Tensor)
        
    返回:
        output_final: [Batch, 7] 物理空间无量纲值
    """
    device = output_raw.device
    
    # 确保统计量在同一设备上
    dm = data_min.to(device)
    dmx = data_max.to(device)
    
    # 提取输出部分的统计量 (后7维: indices 4 to 10)
    out_min = dm[4:11]
    out_max = dmx[4:11]
    
    # ==========================================
    # 第一步：通用线性反归一化 (所有变量)
    # ==========================================
    # 此时 out_linear 的形状为 [Batch, 7]
    # 其中:
    # out_linear[:, 0:4] 是 U, V, W, P, T 的物理值
    # out_linear[:, 5] 是 ln(K)
    # out_linear[:, 6] 是 ln(Omega)
    out_linear = output_raw * (out_max - out_min + 1e-8) + out_min
    
    # ==========================================
    # 第二步：分别处理
    # ==========================================
    
    # 1. 提取前 5 个变量 (U, V, W, P, T)，它们已经是最终物理值了
    out_uvwpt = out_linear[:, 0:5]
    
    # 2. 处理 K (索引 5)
    # 先钳位防止 exp 爆炸，然后 exp 还原
    k_ln = out_linear[:, 5:6]
    k_ln_clamped = torch.clamp(k_ln, min=-20.0, max=20.0) # 安全范围
    k = torch.exp(k_ln_clamped)
    
    # 3. 处理 Omega (索引 6)
    omega_ln = out_linear[:, 6:7]
    omega_ln_clamped = torch.clamp(omega_ln, min=-20.0, max=20.0)
    omega = torch.exp(omega_ln_clamped)
    
    # ==========================================
    # 第三步：安全拼接 (不使用 in-place 操作，保护计算图)
    # ==========================================
    output_final = torch.cat([out_uvwpt, k, omega], dim=1)

    logger.info(f"   反归一化之后的U范围: {output_final[:,0:1].min().item():.6f} ~ {output_final[:,0:1].max().item():.6f}")
    logger.info(f"   反归一化之后的V范围: {output_final[:,1:2].min().item():.6f} ~ {output_final[:,1:2].max().item():.6f}")
    logger.info(f"   反归一化之后的W范围: {output_final[:,2:3].min().item():.6f} ~ {output_final[:,2:3].max().item():.6f}")
    logger.info(f"   反归一化之后的P范围: {output_final[:,3:4].min().item():.6f} ~ {output_final[:,3:4].max().item():.6f}")
    logger.info(f"   反归一化之后的T范围: {output_final[:,4:5].min().item():.6f} ~ {output_final[:,4:5].max().item():.6f}")
    logger.info(f"   反归一化之后的K范围: {output_final[:,5:6].min().item():.6f} ~ {output_final[:,5:6].max().item():.6f}")
    logger.info(f"   反归一化之后的Omega范围: {output_final[:,6:7].min().item():.6f} ~ {output_final[:,6:7].max().item():.6f}")
    
    return output_final

def sigmoid_schedule(t, T, start_val, end_val):
    """
    平滑的 Sigmoid 过渡：两头慢，中间快
    """
    if t >= T:
        return end_val
    
    # 将 t/T 映射到 [-6, 6]，sigmoid 在这个区间内从 ~0 变到 ~1
    x = 12 * (t / T - 0.5) 
    sigma = 1 / (1 + np.exp(-x))
    
    return start_val + (end_val - start_val) * sigma

# 训练集总损失（epoch476+ 精准微调版）
def train_loss_TOTAL(epoch,PDEloss_start_epoch,device, L,M0,T0,P0,input,output_raw,label,data_min,data_max):
    
    loss_data=data(device, output_raw, label)
    # 获取所有单项损失

    output_final=denormalize_for_pde(output_raw,data_min,data_max)

    loss_cont,loss_mom,loss_energy,loss_k,loss_omega,loss_bon,res_cont,res_mx,res_my,res_mz,res_energy,res_k,res_omega = loss_PDE_and_bon(L,M0,T0,P0,input,output_final,data_min,data_max)

    w_data = 1 
    #w_pde = 0.0001    
      
    if epoch < PDEloss_start_epoch: # 第一阶段 纯数据拟合
        w_pde = 0
        w_cont = 0.0
        w_mom = 0.0
        w_energy = 0.0
        w_k = 0.0
        w_omega = 0.0
    
    elif PDEloss_start_epoch<=epoch < 2*PDEloss_start_epoch:  # 第二阶段 加入连续、动量、能量方程
        w_pde = sigmoid_schedule(epoch-PDEloss_start_epoch,PDEloss_start_epoch,1e-3,1e-1)
        w_cont = 1.0
        w_mom = 2.0
        w_energy = 1.5
        w_k = 0
        w_omega = 0
    
    elif 2*PDEloss_start_epoch<=epoch < 3*PDEloss_start_epoch : # 第三阶段 加入K方程
        w_pde=sigmoid_schedule(epoch-PDEloss_start_epoch,PDEloss_start_epoch,1e-1,1e0)
        w_cont = 1.0
        w_mom = 2.0
        w_energy = 1.5
        w_k = 10
        w_omega = 0

    else:
        w_pde=sigmoid_schedule(epoch-3*PDEloss_start_epoch,PDEloss_start_epoch,1e0,1e1)
        w_cont = 1.0
        w_mom = 2.0
        w_energy = 1.5
        w_k = 10      
        w_omega = 0.0002    

    # 计算加权后的PDE总损失
    loss_pde_unweighted = (
        w_cont * loss_cont +
        w_mom * loss_mom +
        w_energy * loss_energy +
        w_k * loss_k +
        w_omega * loss_omega
    )

    loss_pde = w_pde * loss_pde_unweighted

    # 总损失计算
    total_loss = w_data * loss_data + w_pde*loss_pde
    logger.info(f"当前数据拟合损失权重：{w_data}")
    logger.info(f"当前PDE损失权重:{w_pde}")
    logger.info(f"当前数据拟合损失: {w_data * loss_data.item():.6e} ")
    logger.info(f"当前PDE残差损失: {loss_pde.item():.6e} ")

    return total_loss,res_cont.mean(),res_mx.mean(),res_my.mean(),res_mz.mean(),res_energy.mean(),res_k.mean(),res_omega.mean()

# 验证集总损失
def val_loss_TOTAL(device,output,label):
    mse_loss = loss_MSE(device, output, label)
    #l1_loss = loss_L1(device, output, label)
    total_loss=mse_loss
    return total_loss

# 硬约束函数
def hard_consrain(input_d,output,output_sym):
    # 对称面所有条件硬约束 分别构建偶函数和奇函数
    U = 0.5 * (output[:, 0:1] + output_sym[:, 0:1])
    V = 0.5 * (output[:, 1:2] + output_sym[:, 1:2])
    W = 0.5 * (output[:, 2:3] - output_sym[:, 2:3])
    P = 0.5 * (output[:, 3:4] + output_sym[:, 3:4])
    T = 0.5 * (output[:, 4:5] + output_sym[:, 4:5])
    K = 0.5 * (output[:, 5:6] + output_sym[:, 5:6])
    Omega = 0.5 * (output[:, 6:7] + output_sym[:, 6:7])

    # 壁面无滑移硬约束
    U=input_d*U
    V=input_d*V
    W=input_d*W
    K=input_d*K

    # 结果合并
    output=torch.cat([U, V, W, P, T, K, Omega], dim=1)
    return output


# 残差损失
class RANS_PDE():
    def __init__(self,L,M0,T0,P0,input,output,data_min,data_max):  # (网络输出（无量纲化），网路输出（归一化），网络输入最大值，网络输入最小值)
        # 无量纲化参数一、先明确两个概念的区别（避免混淆）
        self.L=L        # 特征长度
        self.M0=M0      # 来流马赫数
        self.T0=T0      # 来流静温
        self.P0=P0      # 来流静压
        self.U0=M0*(1.4*287*T0)**0.5   # 来流速度
        self.Rou0=P0/(287*T0)       # 来流密度
        self.Miu0=(1.7894e-5)*((self.T0/288.15)**(1.5))*(288.15+110.4)/(self.T0+110.4)  # 来流动力粘度
        self.Re0=(self.Rou0*self.U0*L)/self.Miu0  # 来流雷诺数

        # 近壁k-ω组常数
        self.alpha1 = 5/9
        self.beta1 = 0.075
        self.sigma_k1 = 0.85
        self.sigma_omega1 = 0.5
        # 远场k-ε组常数
        self.alpha2 = 0.44
        self.beta2 = 0.0828
        self.sigma_k2 = 1.0
        self.sigma_omega2 = 0.856
        # 全局固定常数
        self.beta_star = 0.09
        self.a1 = 0.31
        self.Pr = 0.72     # 普朗特数
        self.Pr_t = 0.9    # 湍流普朗特数
        self.gamma = 1.4   # 空气比热比

        
        self.input=input
        self.output=output  # 无量纲输出 
        min_d = data_min[3]  # 改个变量名，避免和内置函数 min 冲突
        max_d = data_max[3]
        
        self.input_min_d = min_d.clone().detach().to("cuda").float()
        self.input_max_d = max_d.clone().detach().to("cuda").float()
        self.d_norm=torch.clamp(self.input[:,3:4] , min=1e-4)    # 归一化壁面距离d加防0保护
        self.d = self.d_norm * (self.input_max_d - self.input_min_d + 1e-8) + self.input_min_d # 反归一化为无量纲d
        self.d =torch.clamp(self.d,min=1e-4)           # 无量纲d加防0保护
        
        min_xyzd=data_min[0:4]
        max_xyzd=data_max[0:4]
        self.input_min_xyzd = min_xyzd.clone().detach().to("cuda").float()
        self.input_max_xyzd = max_xyzd.clone().detach().to("cuda").float()
        self.scale = (self.input_max_xyzd - self.input_min_xyzd + 1e-8).reshape(1,4) # 缩放因子
        #print(self.scale )
    # 求导函数
    def grad(self,y,x):
        out=torch.autograd.grad(y.sum(),x,create_graph=True,retain_graph=True,allow_unused=True) [0]# 无量纲输出对无量纲输入的导数
        out=out/self.scale   # 代入反归一化
        return out
    
    def rans_res(self): # 输入input形状：[无量纲x，无量纲y，无量纲z，无量纲壁面距离d]
        # 网络输出无量纲参数
        U=self.output[:,0:1]  # 无量纲x方向速度
        V=self.output[:,1:2]  # 无量纲y方向速度
        W=self.output[:,2:3]  # 无量纲z方向速度
        P=self.output[:,3:4]  # 无量纲压强
        T=torch.clamp(self.output[:,4:5],min=1e-4)  # 无量纲静温
        K=torch.clamp(self.output[:,5:6],min=1e-10)  # 无量纲湍流动能
        Omega=torch.clamp(self.output[:,6:7],min=1e-4) # 无量纲比耗散率
        #print(f"Omega:{Omega}")

        # 由网路输出导出的无量纲参数
        gamma_M0_sq = self.gamma * (self.M0 ** 2)
        Rou = (1.0 + gamma_M0_sq * P) / T
        Miu=(T**1.5)*((1+110.4/self.T0)/(T+110.4/self.T0)) # 无量纲动力粘度
        
        ###################################基础导数
        # U关于x、y、z的导数
        grad_U=self.grad(U,self.input)
        dU_dX=grad_U[:,0:1]
        dU_dY=grad_U[:,1:2]
        dU_dZ=grad_U[:,2:3]

        # V关于x、y、z的导数
        grad_V=self.grad(V,self.input)
        dV_dX=grad_V[:,0:1]
        dV_dY=grad_V[:,1:2]
        dV_dZ=grad_V[:,2:3]

        # W关于x、y、z的导数
        grad_W=self.grad(W,self.input)
        dW_dX=grad_W[:,0:1]
        dW_dY=grad_W[:,1:2]
        dW_dZ=grad_W[:,2:3]

        # Rou*U关于x的导数
        grad_Rou_U=self.grad(Rou*U,self.input)
        dRou_U_dX=grad_Rou_U[:,0:1]

        # Rou*V关于y的导数
        grad_Rou_V=self.grad(Rou*V,self.input)
        dRou_V_dY=grad_Rou_V[:,1:2]

        # Rou*W关于z的导数
        grad_Rou_W=self.grad(Rou*W,self.input)
        dRou_W_dZ=grad_Rou_W[:,2:3]

        # P关于x、y、z的导数
        grad_P=self.grad(P,self.input)
        dP_dX=grad_P[:,0:1]
        dP_dY=grad_P[:,1:2]
        dP_dZ=grad_P[:,2:3]

        # T关于x、y、z的导数
        grad_T=self.grad(T,self.input)
        dT_dX=grad_T[:,0:1]
        dT_dY=grad_T[:,1:2]
        dT_dZ=grad_T[:,2:3]

        # K关于x、y、z的导数
        grad_K=self.grad(K,self.input)
        dK_dX=grad_K[:,0:1]
        dK_dY=grad_K[:,1:2]
        dK_dZ=grad_K[:,2:3]

        # Omega关于x、y、z的导数
        grad_Omega=self.grad(Omega,self.input)
        dOmega_dX=grad_Omega[:,0:1]
        dOmega_dY=grad_Omega[:,1:2]
        dOmega_dZ=grad_Omega[:,2:3]

        ################################## 方程搭建
        
        
        # 能量方程
        Omu=((dW_dY-dV_dZ)**2+(dU_dZ-dW_dX)**2+(dV_dX-dU_dY)**2+1e-12)**0.5            # 涡量幅值,开防负保护
        arg2_1=(2*K**0.5)/(self.beta_star*Omega*self.d+1e-12) # 防0保护
        arg2_2=(500)/(self.Re0*Rou*self.d**2*Omega+1e-12) # 防0保护
        arg2=torch.maximum(arg2_1,arg2_2)     # 混合函数F2
        F2=torch.tanh(arg2**2)
        Miu_t=(Rou*self.a1*K)/(self.Re0*(torch.maximum(self.a1*Omega,Omu*F2))+1e-12)       # 湍流粘度 分母开防0保护
        Miu_t=torch.clamp(Miu_t,min=1e-20,max=1e4) # 防数值爆炸
        Phi=((Miu+Miu_t)/self.Re0)*(2*(dU_dX)**2+(dV_dY)**2+2*(dW_dZ)**2+\
            (dU_dY+dV_dX)**2+(dU_dZ+dW_dX)**2+(dV_dZ+dW_dY)**2-(2/3)*(dU_dX+dV_dY+dW_dZ)**2)         # 耗散函数
        Res_E=Rou*(U*(dT_dX)+V*(dT_dY)+W*(dT_dZ))-\
            (1/(self.Re0*self.Pr))*((self.grad((Miu+(self.Pr/self.Pr_t)*Miu_t)*dT_dX,self.input)[:,0:1])+\
            (self.grad((Miu+(self.Pr/self.Pr_t)*Miu_t)*dT_dY,self.input)[:,1:2])+\
            (self.grad((Miu+(self.Pr/self.Pr_t)*Miu_t)*dT_dZ,self.input)[:,2:3]))-\
            0.5*(self.gamma-1)*self.M0**2*(U*dP_dX+V*dP_dY+W*dP_dZ+Phi)
        
        # 湍流动能K输运方程
        CD_k_omega=torch.maximum(((2*Rou*self.sigma_omega2)/(Omega+1e-12))*(dK_dX*dOmega_dX+dK_dY*dOmega_dY+dK_dZ*dOmega_dZ),torch.tensor(1e-20,device="cuda"))
        arg1_1=(K**0.5)/(self.beta_star*Omega*self.d+1e-12)
        arg1_2=(500)/(self.Re0*Rou*self.d**2*Omega+1e-12)
        arg1_3=(4*Rou*self.sigma_omega2*K)/(CD_k_omega*self.d**2+1e-12)
        arg1=torch.minimum(torch.maximum(arg1_1,arg1_2),arg1_3)
        F1=torch.tanh(arg1**4)
        sigma_k=F1*(self.sigma_k1)+(1-F1)*self.sigma_k2
        P_k=torch.minimum(Miu_t*self.Re0*(2*dU_dX**2+2*dV_dY**2+2*dW_dZ**2+(dV_dX+dU_dY)**2+(dU_dZ+dW_dX)**2+(dV_dZ+dW_dY)**2-(2/3)*(dU_dX+dV_dY+dW_dZ)**2),(20*self.beta_star*Rou*K*Omega))
        Res_K=Rou*(U*dK_dX+V*dK_dY+W*dK_dZ)-P_k+(self.beta_star*Rou*K*Omega)-\
              (1/self.Re0)*(self.grad(((Miu+sigma_k*Miu_t)*dK_dX),self.input)[:,0:1]+\
              self.grad(((Miu+sigma_k*Miu_t)*dK_dY),self.input)[:,1:2]+\
              self.grad(((Miu+sigma_k*Miu_t)*dK_dZ),self.input)[:,2:3])
        
        # 比耗散率Omega输运方程
        sigma_omega=F1*self.sigma_omega1+(1-F1)*self.sigma_omega2
        alpha=F1*self.alpha1+(1-F1)*self.alpha2
        beta=F1*self.beta1+(1-F1)*self.beta2
        prod_term = alpha * ((Rou * P_k * self.Re0) / (Miu_t + 1e-12))
        prod_term = torch.clamp(prod_term, min=-1e3, max=1e3) # 加限制
        Res_Omega=Rou*(U*dOmega_dX+V*dOmega_dY+W*dOmega_dZ)-prod_term+\
                  beta*Rou*Omega**2-((1/self.Re0)*\
                  (self.grad((Miu+sigma_omega*Miu_t)*dOmega_dX,self.input)[:,0:1])+\
                  (self.grad((Miu+sigma_omega*Miu_t)*dOmega_dY,self.input)[:,1:2])+\
                  (self.grad((Miu+sigma_omega*Miu_t)*dOmega_dZ,self.input)[:,2:3]))-\
                  2*(1-F1)*((Rou*self.sigma_omega2)/(Omega+1e-12))*\
                  (dK_dX*dOmega_dX+dK_dY*dOmega_dY+dK_dZ*dOmega_dZ)

        # 连续方程
        Res_C=dRou_U_dX+dRou_V_dY+dRou_W_dZ

        # x方向动量方程
        Res_MX=Rou*(U*dU_dX+V*dU_dY+W*dU_dZ)+dP_dX-\
               (1/self.Re0)*self.grad((Miu+Miu_t)*(2*dU_dX-(2/3)*(dU_dX+dV_dY+dW_dZ)),self.input)[:,0:1]-\
               (1/self.Re0)*self.grad((Miu+Miu_t)*(dU_dY+dV_dX),self.input)[:,1:2]-\
               (1/self.Re0)*self.grad((Miu+Miu_t)*(dU_dZ+dW_dX),self.input)[:,2:3]
        
        # Y方向动量方程
        Res_MY=Rou*(U*dV_dX+V*dV_dY+W*dV_dZ)+dP_dY-\
               (1/self.Re0)*self.grad((Miu+Miu_t)*(dV_dX+dU_dY),self.input)[:,0:1]-\
               (1/self.Re0)*self.grad((Miu+Miu_t)*(2*dV_dY-(2/3)*(dU_dX+dV_dY+dW_dZ)),self.input)[:,1:2]-\
               (1/self.Re0)*self.grad((Miu+Miu_t)*(dV_dZ+dW_dY),self.input)[:,2:3]
        
        # Z方向动量方程
        Res_MZ=Rou*(U*dW_dX+V*dW_dY+W*dW_dZ)+dP_dZ-\
               (1/self.Re0)*self.grad((Miu+Miu_t)*(dW_dX+dU_dZ),self.input)[:,0:1]-\
               (1/self.Re0)*self.grad((Miu+Miu_t)*(dW_dY+dV_dZ),self.input)[:,1:2]-\
               (1/self.Re0)*self.grad((Miu+Miu_t)*(2*dW_dZ-(2/3)*(dU_dX+dV_dY+dW_dZ)),self.input)[:,2:3]
        return Res_C,Res_MX,Res_MY,Res_MZ,Res_E,Res_K,Res_Omega

    # 绝热壁面残差
    def bon_res(self):
    # ✅ 修复：把 mask 变成一维 [1024]，而不是二维 [1024, 1]
        wall_points_id = self.d[:, 0] < 1e-4
        
        # 这行你定义了但后面没用到？如果需要的话可以保留
        wall_points = self.input[wall_points_id]
        
        T = self.output[:, 4:5]
        dT_dd = self.grad(T, self.input)[:, 3:4]
        return dT_dd
    