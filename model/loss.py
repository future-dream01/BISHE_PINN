# 接受网络输出的退化图和目标图像，计算损失值
import torch.nn as nn
import torch
import torchvision.models as models
import torch.nn.functional as F
import torch.nn.functional as F

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

# 总损失
def loss_TOTAL(device, L,M0,T0,P0,input,output,label,input_max,input_min):
    mse_loss = loss_MSE(device, output, label)
    l1_loss = loss_L1(device, output, label)
    #criterion_loss=loss_criterion(device, output, label)
    PDE=RANS_PDE(L,M0,T0,P0,input,output,input_max,input_min)
    PDE_loss=PDE.rans_res()
    total_loss =0.4* mse_loss  + 0.2*l1_loss +0.4*
    return total_loss

# 残差损失
class RANS_PDE():
    def __init__(self,L,M0,T0,P0,input,output,input_max,input_min):  # (网络输出（无量纲化），网路输出（归一化），网络输入最大值，网络输入最小值)
        # 无量纲化参数
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
        self.input_max=input_max
        self.input_min=input_min
        self.output=output  # 无量纲输出 
        self.input=((input + 1) / 2) * (self.input_max - self.input_min) + self.input_min  # 无量纲输入
        self.XYZ=self.input[:,0:3]   # 无量纲坐标xyz
        self.d=self.input[:,3:4]     # 无量纲壁面距离d

    # 求导函数
    def grad(self,y,x):
        out=torch.autograd.grad(y.sum(),x,create_graph=True,retain_graph=True)[0]  # 无量纲输出对无量纲输入的导数
        return out
    
    def rans_res(self): # 输入input形状：[无量纲x，无量纲y，无量纲z，无量纲壁面距离d]
        N=input.shape[0]    # 批次数

        # 网络输出无量纲参数
        U=self.output[:,0:1]  # 无量纲x方向速度
        V=self.output[:,1:2]  # 无量纲y方向速度
        W=self.output[:,2:3]  # 无量纲z方向速度
        P=self.output[:,3:4]  # 无量纲压强
        T=self.output[:,4:5]  # 无量纲静温
        K=self.output[:,5:6]  # 无量纲湍流动能
        Omega=self.output[:,6:7] # 无量纲比耗散率
        Omega=torch.clamp(Omega,min=1e-8) # 设置下限，壁面除0

        # 由网路输出导出的无量纲参数
        Rou=(P*self.gamma*(self.M0)**2)/T # 无量纲密度
        Miu=(T**1.5)*((1+110.4/self.T0)/(T+110.4/self.T0)) # 无量纲动力粘度
        
        ###################################基础导数
        # U关于x、y、z的导数
        grad_U=self.grad(U,self.XYZ)
        dU_dX=grad_U[:,0:1]
        dU_dY=grad_U[:,1:2]
        dU_dZ=grad_U[:,2:3]

        # V关于x、y、z的导数
        grad_V=self.grad(V,self.XYZ)
        dV_dX=grad_V[:,0:1]
        dV_dY=grad_V[:,1:2]
        dV_dZ=grad_V[:,2:3]

        # W关于x、y、z的导数
        grad_W=self.grad(V,self.XYZ)
        dW_dX=grad_V[:,0:1]
        dW_dY=grad_V[:,1:2]
        dW_dZ=grad_V[:,2:3]

        # Rou*U关于x的导数
        grad_Rou_U=self.grad(Rou*U,self.XYZ)
        dRou_U_dX=grad_Rou_U[:,0:1]

        # Rou*V关于y的导数
        grad_Rou_V=self.grad(Rou*V,self.XYZ)
        dRou_V_dY=grad_Rou_V[:,1:2]

        # Rou*W关于z的导数
        grad_Rou_W=self.grad(Rou*W,self.XYZ)
        dRou_W_dZ=grad_Rou_W[:,2:3]
  
        # P关于x、y、z的导数
        grad_P=self.grad(P,self.XYZ)
        dP_dX=grad_P[:,0:1]
        dP_dY=grad_P[:,1:2]
        dP_dZ=grad_P[:,2:3]

        # T关于x、y、z的导数
        grad_T=self.grad(T,self.XYZ)
        dT_dX=grad_T[:,0:1]
        dT_dY=grad_T[:,1:2]
        dT_dZ=grad_T[:,2:3]

        # K关于x、y、z的导数
        grad_K=self.grad(K,self.XYZ)
        dK_dX=grad_K[:,0:1]
        dK_dY=grad_K[:,1:2]
        dK_dZ=grad_K[:,2:3]

        # Omega关于x、y、z的导数
        grad_Omega=self.grad(Omega,self.XYZ)
        dOmega_dX=grad_K[:,0:1]
        dOmega_dY=grad_K[:,1:2]
        dOmega_dZ=grad_K[:,2:3]

        ################################## 方程搭建
        # 连续方程
        Res_C=dRou_U_dX+dRou_V_dY+dRou_W_dZ

        # x方向动量方程
        Res_MX=Rou*(U*dU_dX+V*dV_dY+W*dW_dZ)+dP_dX-(1/self.Re0)*self.grad()

        # 能量方程
        Omu=((dW_dY-dV_dZ)**2+(dU_dZ-dW_dX)**2+(dV_dX-dU_dY)**2)**0.5            # 涡量幅值
        arg2=torch.max((2*K**0.5)/(self.beta_star*Omega*self.d),(500)/(self.Re0*Rou*self.d**2*Omega))     # 混合函数F2
        F2=torch.tanh(arg2**2)
        Miu_t=(Rou*self.a1*K)/(self.Re0*(torch.max(self.a1*Omega,Omu*F2)))       # 湍流粘度
        Phi=((Miu+Miu_t)/self.Re0)*(2*(dU_dX)**2+(dV_dY)**2+2*(dW_dZ)**2+(dU_dY+dV_dX)**2+(dU_dZ+dW_dX)**2+(dV_dZ+dW_dY)**2-(2/3)*(dU_dX+dV_dY+dW_dZ)**2)         # 耗散函数
        Res_E=Rou*(U*(dT_dX)+V*(dT_dY)+W(dT_dZ))-(1/(self.Re0*self.Pr))*((self.grad((Miu+(self.Pr/self.Pr_t)*Miu_t)*dT_dX)[:,0:1])+(self.grad((Miu+(self.Pr/self.Pr_t)*Miu_t)*dT_dY)[:,1:2])+(self.grad((Miu+(self.Pr/self.Pr_t)*Miu_t)*dT_dZ)[:,2:3]))-(self.gamma-1)*self.M0**2*(U*dP_dX+V*dP_dY+W*dP_dZ+Phi)




