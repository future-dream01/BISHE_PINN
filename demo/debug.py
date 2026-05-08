import os, sys
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib.tri as tri
from loguru import logger
from datetime import datetime
import random

# ===================== 项目路径 =====================
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), './..'))
sys.path.append(project_root)

# ===================== 导入训练模块 =====================
from model import PINN, data_prepare, denormalize_for_pde

# ===================== 参数 =====================
L = 0.095
M0 = 0.42
T0 = 249.15
P0 = 47181
GAMMA = 1.4
R_gas = 287
U0 = M0 * (GAMMA * R_gas * T0) ** 0.5
q0 = GAMMA * P0 * M0**2

BATCHSIZE = 1024
weight_name = "best_29weights"
weight_PATH = f'{project_root}/outputs/weights/{weight_name}.pth'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 输出路径
current_datetime = datetime.now().strftime("%m-%d_%H-%M")
save_root = f'{project_root}/outputs/推理结果/DEBUG_{current_datetime}'
os.makedirs(save_root, exist_ok=True)

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def debug():
    set_seed(42)
    
    # 1. 加载数据
    logger.info("Loading data...")
    train_dataloader, val_dataloader, data_min_np, data_max_np = data_prepare(BATCHSIZE)
    data_min = torch.tensor(data_min_np, dtype=torch.float32).to(device)
    data_max = torch.tensor(data_max_np, dtype=torch.float32).to(device)

    # 2. 加载模型
    model = PINN().to(device)
    if os.path.isfile(weight_PATH):
        checkpoint = torch.load(weight_PATH, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'], strict=True)
    model.eval()

    # 3. 只取第一批数据做测试
    input_batch, target_batch = next(iter(val_dataloader))
    input_batch = input_batch.to(device)
    target_batch = target_batch.to(device)

    with torch.no_grad():
        output_norm = model(input_batch)
        
        # 🔴 关键诊断：这里必须分别反归一化！
        # 注意：如果 denormalize_for_pde 会修改 input，我们需要 clone()
        # 为了保险，我这里使用 .clone()
        output_pred_phys = denormalize_for_pde(device, output_norm.clone(), data_min, data_max)
        output_true_phys = denormalize_for_pde(device, target_batch.clone(), data_min, data_max)

    # 转 Numpy
    pred_np = output_pred_phys.cpu().numpy()
    true_np = output_true_phys.cpu().numpy()

    # ==========================================
    # 检查 1: 打印数值看是不是真的一样
    # ==========================================
    print("\n" + "="*60)
    print("🔍 数值对比 (前5个样本)")
    print("="*60)
    print(f"{'Index':<6} | {'Pred_U':<10} | {'True_U':<10} | {'Equal?':<10}")
    print("-" * 60)
    for i in range(min(5, len(pred_np))):
        p = pred_np[i, 0]
        t = true_np[i, 0]
        print(f"{i:<6} | {p:<10.6f} | {t:<10.6f} | {np.isclose(p, t):<10}")
    
    # 检查整个数组
    if np.allclose(pred_np, true_np):
        print("\n❌ 严重警告：预测值数组 和 真值数组 完全相等！")
        print("   问题出在训练代码/反归一化代码，不在绘图。")
    else:
        print("\n✅ 数据正常：预测值和真值有差异。")

    # ==========================================
    # 检查 2: 简单绘图 (只画 U 速度)
    # ==========================================
    # 提取坐标 (简单反归一化)
    in_min_4 = data_min[:4].cpu().numpy()
    in_max_4 = data_max[:4].cpu().numpy()
    in_range_4 = in_max_4 - in_min_4 + 1e-8
    input_np = input_batch.cpu().numpy()
    input_denorm_4 = input_np[:, :4] * in_range_4 + in_min_4
    Y = input_denorm_4[:, 1]
    Z = input_denorm_4[:, 2]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # 直接散点图，不做三角剖分，最直观
    sc1 = ax1.scatter(Y, Z, c=pred_np[:, 0], cmap='jet', s=2)
    ax1.set_title("Prediction (Scatter)")
    plt.colorbar(sc1, ax=ax1)
    
    sc2 = ax2.scatter(Y, Z, c=true_np[:, 0], cmap='jet', s=2)
    ax2.set_title("Ground Truth (Scatter)")
    plt.colorbar(sc2, ax=ax2)
    
    plt.savefig(f"{save_root}/DEBUG_Scatter.png")
    print(f"\n🖼️  诊断图已保存至: {save_root}/DEBUG_Scatter.png")
    print("   如果这两张图看起来不一样，那就是之前 PyVista 代码的索引逻辑错了。")

if __name__ == "__main__":
    debug()