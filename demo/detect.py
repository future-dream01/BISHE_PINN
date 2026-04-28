import os, sys
import torch
import numpy as np
import matplotlib.pyplot as plt
from loguru import logger
from datetime import datetime
from matplotlib import cm
from scipy.cluster.hierarchy import fclusterdata
import matplotlib.tri as tri
import random

# ===================== 项目路径 =====================
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), './..'))
sys.path.append(project_root)

# ===================== 导入训练模块 =====================
from model import PINN, data_prepare, denormalize_for_pde

# ===================== 【1:1 复制训练代码的参数】 =====================
L = 0.095        # 特征长度
M0 = 0.42        # 来流马赫数
T0 = 249.15      # 来流静温
P0 = 47181       # 来流静压
GAMMA = 1.4      # 比热比
R_gas = 287      # 气体常数
U0 = M0 * (GAMMA * R_gas * T0) ** 0.5
q0 = GAMMA * P0 * M0**2

# ===================== 推理配置 =====================
BATCHSIZE = 1024

# ==========================================
# 【修改这里】切换到你新训的权重路径
# ==========================================
weight_name = "04-27_12-53/best_1476weights"  # 🔧 改成你新的文件夹名
weight_PATH = f'{project_root}/outputs/weights/{weight_name}.pth'

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 输出路径
current_datetime = datetime.now().strftime("%m-%d_%H-%M")
save_root = f'{project_root}/outputs/推理结果/{current_datetime}'
os.makedirs(save_root, exist_ok=True)
log_file_path = f'{save_root}/inference.log'
logger.add(log_file_path, rotation="500 MB", level="INFO")

# 绘图参数
PLOT_VAR_LIST = ["Velocity", "Static_P", "Mach", "Total_P"]
X_TOLERANCE = 1e-4
WALL_DIST_MAX = 0.6
ALPHA_PARAM = 1.0

# ===================== 随机种子 =====================
def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# ===================== 掩码函数 (保留用于绘图) =====================
def get_combined_mask(points_y, points_z, wall_dist, triang):
    triangles = triang.triangles
    wd_tri = wall_dist[triangles]
    mean_wd = np.mean(wd_tri, axis=1)
    mask_wall = (mean_wd > WALL_DIST_MAX) | np.isnan(mean_wd) | (mean_wd < 0)

    pts = np.column_stack([points_y, points_z])
    y_tri = pts[triangles, 0]
    z_tri = pts[triangles, 1]
    a = np.sqrt((y_tri[:, 1] - y_tri[:, 0])**2 + (z_tri[:, 1] - z_tri[:, 0])**2)
    b = np.sqrt((y_tri[:, 2] - y_tri[:, 1])**2 + (z_tri[:, 2] - z_tri[:, 1])**2)
    c = np.sqrt((y_tri[:, 0] - y_tri[:, 2])**2 + (z_tri[:, 0] - z_tri[:, 2])**2)
    s = (a + b + c) / 2.0
    area = np.sqrt(np.clip(s * (s - a) * (s - b) * (s - c), 0, None))
    area[area < 1e-12] = 1e-12
    scale = np.max([np.max(points_y) - np.min(points_y), np.max(points_z) - np.min(points_z)])
    circum_r = (a * b * c) / (4.0 * area)
    mask_alpha = circum_r > (scale / ALPHA_PARAM)

    return mask_wall | mask_alpha

# ===================== 主推理函数 =====================
def infer_all_dataset():
    set_seed(42)
    logger.info("="*60)
    logger.info(f"🔥 开始推理，权重文件: {weight_name}")
    logger.info(f"🔧 模式: 数据拟合 + K/Omega 对数反归一化")
    logger.info("="*60)

    # 1. 加载数据
    logger.info("正在加载数据...")
    train_dataloader, val_dataloader, data_min_np, data_max_np = data_prepare(BATCHSIZE)
    
    data_min = torch.tensor(data_min_np, dtype=torch.float32).to(device)
    data_max = torch.tensor(data_max_np, dtype=torch.float32).to(device)
    
    in_min_4 = data_min[:4].cpu().numpy()
    in_max_4 = data_max[:4].cpu().numpy()
    in_range_4 = in_max_4 - in_min_4 + 1e-8

    # 2. 加载模型
    model = PINN().to(device)
    if os.path.isfile(weight_PATH):
        checkpoint = torch.load(weight_PATH, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'], strict=True)
        logger.info(f"✅ 权重加载成功 (Epoch {checkpoint['epoch']})")
    else:
        logger.error(f"❌ 权重文件未找到: {weight_PATH}")
        return
    model.eval()
    for param in model.parameters():
        param.requires_grad = False

    # 全局存储
    all_X, all_Y, all_Z, all_WALL_D = [], [], [], []
    # 预测值 (物理空间无量纲)
    all_U_pred, all_V_pred, all_W_pred, all_P_pred, all_T_pred, all_K_pred, all_Omega_pred = [], [], [], [], [], [], []
    # 真实值 (物理空间无量纲)
    all_U_true, all_V_true, all_W_true, all_P_true, all_T_true, all_K_true, all_Omega_true = [], [], [], [], [], [], []

    # 3. 推理循环
    logger.info("开始推理...")
    first_batch = True
    with torch.no_grad():
        for input_batch, target_batch in val_dataloader:
            input_batch = input_batch.to(device)
            target_batch = target_batch.to(device)

            # ==============================================
            # 【核心修改 1/2】同时对 预测值 和 真值 做反归一化
            # ==============================================
            
            # 1. 网络前向 (得到 [0,1] 归一化值)
            output_norm = model(input_batch)
            
            # 2. 预测值反归一化 -> 物理空间无量纲
            output_pred_phys = denormalize_for_pde(device,output_norm, data_min, data_max)
            
            # 3. 真值反归一化 -> 物理空间无量纲 (必须和预测值用完全一样的逻辑!)
            output_true_phys = denormalize_for_pde(device,target_batch, data_min, data_max)

            # 打印第一批数据确认
            if first_batch:
                logger.info("\n" + "="*50)
                logger.info("🔍 第一批数据诊断 (物理空间无量纲):")
                logger.info(f"   Input shape: {input_batch.shape}")
                logger.info(f"   Output shape: {output_pred_phys.shape}")
                logger.info(f"   Input[0] (x,y,z,d): {input_batch[0, :].cpu().numpy()}")
                logger.info(f"   --- 预测值 ---")
                logger.info(f"   Pred[0] (U,V,W,P,T,K,Omega): {output_pred_phys[0, :].cpu().numpy()}")
                logger.info(f"   --- 真值 ---")
                logger.info(f"   True[0] (U,V,W,P,T,K,Omega): {output_true_phys[0, :].cpu().numpy()}")
                logger.info("="*50 + "\n")
                first_batch = False

            # 提取数据 (转为 numpy)
            input_np = input_batch.cpu().numpy()
            pred_np = output_pred_phys.cpu().numpy()
            true_np = output_true_phys.cpu().numpy()

            # 反归一化坐标
            input_denorm = input_np * in_range_4 + in_min_4
            X, Y, Z, WALL_D = input_denorm[:, 0], input_denorm[:, 1], input_denorm[:, 2], input_denorm[:, 3]

            # ==============================================
            # 【核心修改 2/2】保存所有7个变量
            # ==============================================
            all_X.extend(X); all_Y.extend(Y); all_Z.extend(Z); all_WALL_D.extend(WALL_D)
            
            # 预测值
            all_U_pred.extend(pred_np[:, 0]); all_V_pred.extend(pred_np[:, 1]); all_W_pred.extend(pred_np[:, 2])
            all_P_pred.extend(pred_np[:, 3]); all_T_pred.extend(pred_np[:, 4])
            all_K_pred.extend(pred_np[:, 5]); all_Omega_pred.extend(pred_np[:, 6])
            
            # 真实值
            all_U_true.extend(true_np[:, 0]); all_V_true.extend(true_np[:, 1]); all_W_true.extend(true_np[:, 2])
            all_P_true.extend(true_np[:, 3]); all_T_true.extend(true_np[:, 4])
            all_K_true.extend(true_np[:, 5]); all_Omega_true.extend(true_np[:, 6])

    # 转为数组
    all_X = np.array(all_X); all_Y = np.array(all_Y); all_Z = np.array(all_Z); all_WALL_D = np.array(all_WALL_D)
    
    # 预测值
    all_U_pred = np.array(all_U_pred); all_V_pred = np.array(all_V_pred); all_W_pred = np.array(all_W_pred)
    all_P_pred = np.array(all_P_pred); all_T_pred = np.array(all_T_pred)
    all_K_pred = np.array(all_K_pred); all_Omega_pred = np.array(all_Omega_pred)
    
    # 真实值
    all_U_true = np.array(all_U_true); all_V_true = np.array(all_V_true); all_W_true = np.array(all_W_true)
    all_P_true = np.array(all_P_true); all_T_true = np.array(all_T_true)
    all_K_true = np.array(all_K_true); all_Omega_true = np.array(all_Omega_true)

    # ==============================================
    # 打印数值范围统计 (包含 K 和 Omega)
    # ==============================================
    logger.info("\n" + "="*80)
    logger.info("📊 网络输出 (物理空间无量纲)")
    logger.info("="*80)
    logger.info(f"{'变量':<10} | {'Min(Pred)':<12} | {'Max(Pred)':<12} | {'Min(True)':<12} | {'Max(True)':<12}")
    logger.info("-" * 80)
    logger.info(f"{'U':<10} | {all_U_pred.min():<12.6f} | {all_U_pred.max():<12.6f} | {all_U_true.min():<12.6f} | {all_U_true.max():<12.6f}")
    logger.info(f"{'V':<10} | {all_V_pred.min():<12.6f} | {all_V_pred.max():<12.6f} | {all_V_true.min():<12.6f} | {all_V_true.max():<12.6f}")
    logger.info(f"{'W':<10} | {all_W_pred.min():<12.6f} | {all_W_pred.max():<12.6f} | {all_W_true.min():<12.6f} | {all_W_true.max():<12.6f}")
    logger.info(f"{'P':<10} | {all_P_pred.min():<12.6f} | {all_P_pred.max():<12.6f} | {all_P_true.min():<12.6f} | {all_P_true.max():<12.6f}")
    logger.info(f"{'T':<10} | {all_T_pred.min():<12.6f} | {all_T_pred.max():<12.6f} | {all_T_true.min():<12.6f} | {all_T_true.max():<12.6f}")
    logger.info(f"{'K':<10} | {all_K_pred.min():<12.2e} | {all_K_pred.max():<12.2e} | {all_K_true.min():<12.2e} | {all_K_true.max():<12.2e}")
    logger.info(f"{'Omega':<10} | {all_Omega_pred.min():<12.2e} | {all_Omega_pred.max():<12.2e} | {all_Omega_true.min():<12.2e} | {all_Omega_true.max():<12.2e}")

    # 有量纲化 (只对平均流场做，K和Omega保持无量纲用于分析)
    Vel_pred = np.sqrt(all_U_pred**2 + all_V_pred**2 + all_W_pred**2) * U0
    P_pred = all_P_pred * q0 + P0
    T_pred = all_T_pred * T0
    c_pred = np.sqrt(GAMMA * R_gas * T_pred)
    Ma_pred = Vel_pred / (c_pred + 1e-12)
    TP_pred = P_pred * (1 + 0.5*(GAMMA-1)*Ma_pred**2) ** (GAMMA/(GAMMA-1))

    Vel_true = np.sqrt(all_U_true**2 + all_V_true**2 + all_W_true**2) * U0
    P_true = all_P_true * q0 + P0
    T_true = all_T_true * T0
    c_true = np.sqrt(GAMMA * R_gas * T_true)
    Ma_true = Vel_true / (c_true + 1e-12)
    TP_true = P_true * (1 + 0.5*(GAMMA-1)*Ma_true**2) ** (GAMMA/(GAMMA-1))

    X_dim = all_X * L

    logger.info("\n" + "="*80)
    logger.info("📊 最终物理量 (Dimensional)")
    logger.info(f"   参考值: U0={U0:.2f} m/s, P0={P0:.2f} Pa, T0={T0:.2f} K")
    logger.info("="*80)
    logger.info(f"{'变量':<15} | {'Min(Pred)':<15} | {'Max(Pred)':<15} | {'Min(True)':<15} | {'Max(True)':<15} | {'单位':<10}")
    logger.info("-" * 110)
    logger.info(f"{'Velocity':<15} | {Vel_pred.min():<15.4f} | {Vel_pred.max():<15.4f} | {Vel_true.min():<15.4f} | {Vel_true.max():<15.4f} | {'m/s':<10}")
    logger.info(f"{'Static_P':<15} | {P_pred.min():<15.2f} | {P_pred.max():<15.2f} | {P_true.min():<15.2f} | {P_true.max():<15.2f} | {'Pa':<10}")
    logger.info(f"{'Temperature':<15} | {T_pred.min():<15.4f} | {T_pred.max():<15.4f} | {T_true.min():<15.4f} | {T_true.max():<15.4f} | {'K':<10}")
    logger.info(f"{'Mach':<15} | {Ma_pred.min():<15.4f} | {Ma_pred.max():<15.4f} | {Ma_true.min():<15.4f} | {Ma_true.max():<15.4f} | {'-':<10}")
    logger.info("="*80 + "\n")

    # 绘图
    logger.info("正在绘图...")
    clusters = fclusterdata(X_dim.reshape(-1,1), t=X_TOLERANCE, criterion='distance')
    cluster_ids = np.unique(clusters)
    
    for cid in cluster_ids:
        mask = clusters == cid
        X_sec = X_dim[mask]; Y_sec = all_Y[mask]; Z_sec = all_Z[mask]; WALL_D_sec = all_WALL_D[mask]
        
        if len(X_sec) < 50: continue
        
        # 物理过滤
        fluid_mask = (WALL_D_sec <= WALL_DIST_MAX) & (WALL_D_sec >= 0) & (~np.isnan(WALL_D_sec))
        Y_fluid = Y_sec[fluid_mask]; Z_fluid = Z_sec[fluid_mask]; WALL_D_fluid = WALL_D_sec[fluid_mask]
        
        # 提取数据
        Vel_pred_fluid = Vel_pred[mask][fluid_mask]
        Vel_true_fluid = Vel_true[mask][fluid_mask]
        P_pred_fluid = P_pred[mask][fluid_mask]
        P_true_fluid = P_true[mask][fluid_mask]
        Ma_pred_fluid = Ma_pred[mask][fluid_mask]
        Ma_true_fluid = Ma_true[mask][fluid_mask]
        
        if len(Y_fluid) < 50: continue

        # 正常绘图
        data = {
            "Velocity": [Vel_pred_fluid, Vel_true_fluid],
            "Static_P": [P_pred_fluid, P_true_fluid],
            "Mach": [Ma_pred_fluid, Ma_true_fluid]
        }
        
        for name in PLOT_VAR_LIST:
            if name not in data: continue
            val_pred, val_true = data[name]
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), dpi=150)
            triang = tri.Triangulation(Y_fluid, Z_fluid)
            triang.set_mask(get_combined_mask(Y_fluid, Z_fluid, WALL_D_fluid, triang))
            
            # 绘制预测值
            cf1 = ax1.tricontourf(triang, val_pred, 80, cmap=cm.jet)
            ax1.set_title(f"Prediction | {name}")
            ax1.set_aspect('equal')
            
            # 绘制真值
            cf2 = ax2.tricontourf(triang, val_true, 80, cmap=cm.jet)
            ax2.set_title(f"Ground Truth | {name}")
            ax2.set_aspect('equal')
            
            # 添加图例
            plt.colorbar(cf1, ax=ax1, fraction=0.046, pad=0.04)
            plt.colorbar(cf2, ax=ax2, fraction=0.046, pad=0.04)
            
            plt.tight_layout()
            
            plt.savefig(f"{save_root}/X_{np.mean(X_sec):.4f}m_{name}_Compare.png")
            plt.close()

    logger.info(f"\n🎉 推理完成！结果保存在: {save_root}")

if __name__ == "__main__":
    infer_all_dataset()