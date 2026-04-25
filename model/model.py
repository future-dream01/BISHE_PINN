# 整体网络模型
import torch
import torch.nn as nn
from .net import *
from loguru import logger
class PINN_XYZD(nn.Module):
    def __init__(self):
        super(PINN, self).__init__()        
        self.backbone = Backbone_Lin()

    def forward(self, x):
        out = self.backbone(x)
        logger.info(f"   网络输出的归一化U范围: {out[:,0:1].min().item():.6f} ~ {out[:,0:1].max().item():.6f}")
        logger.info(f"   网络输出的归一化V范围: {out[:,1:2].min().item():.6f} ~ {out[:,1:2].max().item():.6f}")
        logger.info(f"   网络输出的归一化W范围: {out[:,2:3].min().item():.6f} ~ {out[:,2:3].max().item():.6f}")
        logger.info(f"   网络输出的归一化P范围: {out[:,3:4].min().item():.6f} ~ {out[:,3:4].max().item():.6f}")
        logger.info(f"   网络输出的归一化T范围: {out[:,4:5].min().item():.6f} ~ {out[:,4:5].max().item():.6f}")
        logger.info(f"   网络输出的归一化K范围: {out[:,5:6].min().item():.6f} ~ {out[:,5:6].max().item():.6f}")
        logger.info(f"   网络输出的归一化Omega范围: {out[:,6:7].min().item():.6f} ~ {out[:,6:7].max().item():.6f}")
        return out
    
class 