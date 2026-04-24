import torch
import torch.nn as nn
from .net import *
from loguru import logger
class PINN(nn.Module):
    def __init__(self):
        super(PINN, self).__init__()
        # # 输出7个变量：U, V, W, P, T, K, Omega (data_min[4:11])
        # min_val = data_min[4:11]
        # max_val = data_max[4:11]

        # self.output_min = min_val.clone().detach().to(device).float()
        # self.output_max = max_val.clone().detach().to(device).float()
        
        self.backbone = Backbone_Lin()

    def forward(self, x):
        out = self.backbone(x)
        logger.info(f"   网络输出的U范围: {out[:,0:1].min().item():.6f} ~ {out[:,0:1].max().item():.6f}")
        logger.info(f"   网络输出的的V范围: {out[:,1:2].min().item():.6f} ~ {out[:,1:2].max().item():.6f}")
        logger.info(f"   网络输出的的W范围: {out[:,2:3].min().item():.6f} ~ {out[:,2:3].max().item():.6f}")
        logger.info(f"   网络输出的的P范围: {out[:,3:4].min().item():.6f} ~ {out[:,3:4].max().item():.6f}")
        logger.info(f"   网络输出的的T范围: {out[:,4:5].min().item():.6f} ~ {out[:,4:5].max().item():.6f}")
        logger.info(f"   网络输出的的K范围: {out[:,5:6].min().item():.6f} ~ {out[:,5:6].max().item():.6f}")
        logger.info(f"   网络输出的的Omega范围: {out[:,6:7].min().item():.6f} ~ {out[:,6:7].max().item():.6f}")
        
        

        return out