from .model import PINN_XYZD
from .dataload import data_prepare
from .loss import train_loss_TOTAL,val_loss_TOTAL,ssim,psnr,hard_consrain,denormalize_for_pde
from .graph import allgraph