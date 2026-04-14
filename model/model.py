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
        # 反归一化
        self.out=((self.out+1)*(self.input_max-self.input_min))/2+self.input_min
        # 返回计算损失函数
        return self.out