# 整体网络模型
import torch
import torch.nn as nn
from .net import *
from loguru import logger

# X，Y，Z，D网络分支
class PINN_XYZD(nn.Module):
    def __init__(self):
        super(PINN_XYZD, self).__init__()        
        self.backbone = Backbone_Lin_XYZD()

    def forward(self, x):
        out = self.backbone(x)
        return out

# Ma、Pr网络分支
class PINN_MaPr(nn.Module):
    def __init__(self):
        super(PINN_MaPr,self).__init__()
        self.backbone=Backbone_Lin_MaPr()

    def forward(self,x):
        out=self.backbone(x)
        return out
    
class PINN(nn.Module):
    def __init__(self):
        super(PINN,self).__init__()
        self.pinn_xyzd=PINN_XYZD()
        self.pinn_mapr=PINN_MaPr()
        self.norm=nn.LayerNorm(14)
        self.sigmoid=nn.Sigmoid()
        self.lin1=nn.Linear(14,7)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                # ✅ 关键修改2：用Xavier初始化，配合Sigmoid
                nn.init.xavier_uniform_(m.weight, gain=1.0)
                if m.bias is not None:
                    # ✅ 关键修改3：偏置初始化为小的随机值，不是0
                    nn.init.uniform_(m.bias, -0.1, 0.1)

    def forward(self,x):
        out1=self.pinn_xyzd(x[:,0:4]) # X、Y、Z、D分支
        out2=self.pinn_mapr(x[:,4:6]) # Ma、Pr分支
        out=torch.concat([out1,out2],dim=1)
        out=self.lin1(out)
        out = out * 1.4
        out=self.sigmoid(out)
        # logger.info(f"   网络输出的归一化U范围: {out[:,0:1].min().item():.6f} ~ {out[:,0:1].max().item():.6f}")
        # logger.info(f"   网络输出的归一化V范围: {out[:,1:2].min().item():.6f} ~ {out[:,1:2].max().item():.6f}")
        # logger.info(f"   网络输出的归一化W范围: {out[:,2:3].min().item():.6f} ~ {out[:,2:3].max().item():.6f}")
        # logger.info(f"   网络输出的归一化P范围: {out[:,3:4].min().item():.6f} ~ {out[:,3:4].max().item():.6f}")
        # logger.info(f"   网络输出的归一化T范围: {out[:,4:5].min().item():.6f} ~ {out[:,4:5].max().item():.6f}")
        # logger.info(f"   网络输出的归一化K范围: {out[:,5:6].min().item():.6f} ~ {out[:,5:6].max().item():.6f}")
        logger.info(f"   网络输出的归一化Omega范围: {out[:,6:7].min().item():.6f} ~ {out[:,6:7].max().item():.6f}")
        return out

