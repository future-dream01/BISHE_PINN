# 绘制各类性能曲线
import matplotlib.pyplot as plt
import os,sys
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), './..'))
sys.path.append(project_root)



def allgraph(current_datetime,epoches,train_losses,res_cont_epoches,res_mx_epoches,res_my_epoches,res_mz_epoches,res_energy_epoches,res_k_epoches,res_omega_epoches,val_losses):
    # 训练集损失曲线
    plt.figure(figsize=(20, 15))
    plt.subplot(1, 3, 1)
    plt.plot(epoches, train_losses, 'b', label='loss')
    plt.title('Training set Loss over Epochs')
    plt.xlabel('Epochs')    
    plt.ylabel('Loss')
    plt.legend()

    # 训练集连续方程残差
    plt.subplot(1, 3, 2)
    plt.plot(epoches, res_cont_epoches, 'b-', label='Res Cont', linewidth=2)    # 连续方程
    plt.plot(epoches, res_mx_epoches, 'r-', label='Res Mx', linewidth=2)        # x动量
    plt.plot(epoches, res_my_epoches, 'g-', label='Res My', linewidth=2)        # y动量
    plt.plot(epoches, res_mz_epoches, 'y-', label='Res Mz', linewidth=2)        # z动量
    plt.plot(epoches, res_energy_epoches, 'm-', label='Res Energy', linewidth=2)# 能量方程
    plt.plot(epoches, res_k_epoches, 'c-', label='Res K', linewidth=2)          # k输运方程
    plt.plot(epoches, res_omega_epoches, 'k-', label='Res Omega', linewidth=2)  # omega输运方程

    plt.title('Training Set All 7 Residuals ')
    plt.xlabel('Epochs')
    plt.ylabel('Residual Value')
    plt.legend()          # 显示图例（必须有）
    plt.grid(alpha=0.3)   # 加网格更清晰
    

    # 效果值power_data曲线
    plt.subplot(1, 3, 3)
    plt.plot(epoches, val_losses, 'g', label='power_data')
    plt.title('Verification set loss over Epochs')
    plt.xlabel('Epochs')    
    plt.ylabel('Verification Loss')
    plt.legend()

    plt.savefig(f'{project_root}/outputs/训练与性能情况/{current_datetime}/参数曲线.png')  # 保存训练损失图像
    plt.clf()