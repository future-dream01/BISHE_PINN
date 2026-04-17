# 模型总体结构代码

import torch.nn as nn
import torch.nn.functional as F     # 函数模块
from .net import *

class PINN(nn.Module):
    def __init__(self,input_min,input_max):
        super(PINN,self).__init__()
        self.input_min=input_min
        self.input_max=input_max
        self.out=[]
        self.backbone=Backbone_Lin()

    def forward(self,x):
        self.out=self.backbone(x)
        # 适配 0~1 归一化的反归一化公式
        self.out = self.out * (self.input_max - self.input_min + 1e-8) + self.input_min # 返回网络的最终的无量纲输出
        return self.out