# 绘制各类性能曲线
import matplotlib.pyplot as plt
import numpy as np
import os,sys
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), './..'))
sys.path.append(project_root)

def allgraph(current_datetime,epoches,train_losses,res_cont_epoches,res_mx_epoches,res_my_epoches,res_mz_epoches,res_energy_epoches,res_k_epoches,res_omega_epoches,val_losses):
    # 训练集损失曲线
    plt.figure(figsize=(20, 15))
    plt.subplot(1, 3, 1)
    # ===================== 训练损失：对数坐标+绝对值 =====================
    plt.semilogy(epoches, np.abs(train_losses), 'b-', label='Training Loss', linewidth=2)
    plt.title('Training set Loss over Epochs (Log Scale)')
    plt.xlabel('Epochs')    
    plt.ylabel('Loss (Log Scale)')
    plt.legend()
    plt.grid(alpha=0.3, which="both")

    # 训练集所有残差（已优化）
    plt.subplot(1, 3, 2)
    plt.semilogy(epoches, np.abs(res_cont_epoches), 'b-', label='Res Cont', linewidth=2)
    plt.semilogy(epoches, np.abs(res_mx_epoches), 'r-', label='Res Mx', linewidth=2)
    plt.semilogy(epoches, np.abs(res_my_epoches), 'g-', label='Res My', linewidth=2)
    plt.semilogy(epoches, np.abs(res_mz_epoches), 'y-', label='Res Mz', linewidth=2)
    plt.semilogy(epoches, np.abs(res_energy_epoches), 'm-', label='Res Energy', linewidth=2)
    plt.semilogy(epoches, np.abs(res_k_epoches), 'c-', label='Res K', linewidth=2)
    plt.semilogy(epoches, np.abs(res_omega_epoches), 'k-', label='Res Omega', linewidth=2)

    plt.title('Training Set All 7 Residuals (Log Scale)')
    plt.xlabel('Epochs')
    plt.ylabel('Residual Absolute Value (Log Scale)')
    plt.legend(fontsize=10)
    plt.grid(alpha=0.3, which="both")

    # ===================== 验证损失：对数坐标+绝对值 =====================
    plt.subplot(1, 3, 3)
    plt.semilogy(epoches, np.abs(val_losses), 'g-', label='Validation Loss', linewidth=2)
    plt.title('Validation set loss over Epochs (Log Scale)')
    plt.xlabel('Epochs')    
    plt.ylabel('Validation Loss (Log Scale)')
    plt.legend()
    plt.grid(alpha=0.3, which="both")

    # 统一布局调整
    plt.tight_layout()
    
    plt.savefig(f'{project_root}/outputs/训练与性能情况/{current_datetime}/参数曲线.png')
    plt.clf()