import torch
import torch.nn as nn
from .net import *

class PINN(nn.Module):
    def __init__(self, device, data_min, data_max):
        super(PINN, self).__init__()
        # 输出7个变量：U, V, W, P, T, K, Omega (data_min[4:11])
        min_val = data_min[4:11]
        max_val = data_max[4:11]

        self.output_min = min_val.clone().detach().to(device).float()
        self.output_max = max_val.clone().detach().to(device).float()
        
        self.backbone = Backbone_Lin()

    def forward(self, x):
        out = self.backbone(x)
        
        # ===================== 1. 通用线性反归一化（所有变量共用） =====================
        out = out * (self.output_max - self.output_min + 1e-8) + self.output_min

        # ===================== 2. 单独对 Omega 做对数反归一化（核心适配+防爆炸） =====================
        # 提取前6个变量 (U,V,W,P,T,K)
        out_6 = out[:, 0:6]
        
        # 单独处理Omega（此时out[:,6:7]是反归一化后的 ln(Omega)）
        omega_ln = out[:, 6:7]
        
        # 🚀 关键1：exp前钳位，彻底杜绝inf！
        # 限制 ln(Omega) 范围：exp(-10)=4.5e-5，exp(20)=4.8e8（远低于GPU上限）
        omega_ln_clamped = torch.clamp(omega_ln, min=-10, max=20)
        
        # 🚀 关键2：指数还原为物理无量纲Omega
        omega = torch.exp(omega_ln_clamped)
        
        # 🚀 关键3：二次钳位，最终限制Omega范围
        omega_clamped = torch.clamp(omega, min=1e-4, max=1e6)

        # 🚀 关键4：安全拼接（替代inplace赋值，不破坏计算图）
        out_final = torch.cat([out_6, omega_clamped], dim=1)

        return out_final