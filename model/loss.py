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
def loss_TOTAL(device, output, label,input_max,input_min):
    mse_loss = loss_MSE(device, output, label)
    l1_loss = loss_L1(device, output, label)
    #criterion_loss=loss_criterion(device, output, label)
    PDE=RANS_PDE(input_max,input_min)
    PDE_loss=PDE.rans_res()
    total_loss =0.4* mse_loss  + 0.2*l1_loss +0.4*
    return total_loss

# 残差损失
class RANS_PDE():
    def __init__(self,net_output,net_input,input_max,input_min):  # (网络输出（无量纲化），网路输出（归一化），网络输入最大值，网络输入最小值)
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
        self.Pr_t = 0.9    # 湍流普朗特数
        self.gamma = 1.4   # 空气比热比
        self.Ma = 0.5      # 来流马赫数
        self.Re = 1e6      # 雷诺数
        self.net_output=net_output
        self.net_input=net_input
        self.input_max=input_max
        self.input_min=input_min

    # 求导函数
    def grad(self,y,x):
        out=torch.autograd.grad(y.sum(),x,create_graph=True,retain_graph=True)[0]  # 无量纲输出对归一化输入的导数
        out=out*(2/(self.input_max-self.input_min))  # 乘上归一化输入对无量纲输入的导数，得无量纲输出对无量纲输入的导数
        return out
    
    def rans_res(self): # 输入input形状：[无量纲x，无量纲y，无量纲z，无量纲壁面距离d]
        N=input.shape[0]    # 批次数
        U=self.net_output[:,0:1]
        V=self.net_output[:,1:2]
        W=self.net_output[:,2:3]
        P=self.net_output[:,3:4]
        T=self.net_output[:,4:5]
        K=self.net_output[:,5:6]
        omega=self.net_output[:,6:7]
        omega=torch.clamp(omega,min=1e-8) # 设置下限，壁面除0

        # U关于x、y、z的导数
        grad_U=self.grad(U,self.net_input)
        dU_dX=grad_U[:,0:1]
        dU_dY=grad_U[:,1:2]
        dU_dZ=grad_U[:,2:3]
        # V关于x、y、z的导数
        grad_V=self.grad(V,self.net_input)
        dV_dX=grad_V[:,0:1]
        dV_dY=grad_V[:,1:2]
        dV_dZ=grad_V[:,2:3]
        # W关于x、y、z的导数
        grad_W=self.grad(V,self.net_input)
        dV_dX=grad_V[:,0:1]
        dV_dY=grad_V[:,1:2]
        dV_dZ=grad_V[:,2:3]


