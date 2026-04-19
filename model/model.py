# 模型总体结构代码

import torch.nn as nn
import torch.nn.functional as F     # 函数模块
from .net import *

class PINN(nn.Module):
    def __init__(self,device,data_min,data_max):
        super(PINN,self).__init__()
        min = data_min[4:11]  # 从第4位开始取后7个
        max = data_max[4:11]

        # 转张量 + 送到GPU
        self.output_min = torch.tensor(min, dtype=torch.float32).to(device)
        self.output_max = torch.tensor(max, dtype=torch.float32).to(device)
        #self.out=[]
        self.backbone=Backbone_Lin()

    def forward(self,x):
        out=self.backbone(x)
        # 适配 0~1 归一化的反归一化公式
        out = out * (self.output_max - self.output_min + 1e-8) + self.output_min # 返回网络的最终的无量纲输出
        return out