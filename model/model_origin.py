import torch.nn as nn
import torch.nn.functional as F
from .net import *

class PINN(nn.Module):
    def __init__(self,device,data_min,data_max):
        super(PINN,self).__init__()
        min_val = data_min[4:11]  # 改个变量名，避免和内置函数 min 冲突
        max_val = data_max[4:11]

        # ✅ 核心修复：直接保存张量，不要用 torch.tensor()！
        self.output_min = min_val.clone().detach().to(device).float()
        self.output_max = max_val.clone().detach().to(device).float()

        # print(f"self.output_min:{self.output_min}")
        # print(f"self.output_max:{self.output_max}")
        
        self.backbone=Backbone_Lin()

    def forward(self,x):
        out=self.backbone(x)
        # 反归一化公式保持不变
        out = out * (self.output_max - self.output_min + 1e-8) + self.output_min
        return out