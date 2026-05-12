import os, sys
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from loguru import logger
from datetime import datetime

# 项目根路径
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), './..'))
sys.path.append(project_root)

# 仅导入必要的工具函数
from model import PINN, data_prepare, denormalize_for_pde
from model.utils_zx import set_seed, L, X_TOLERANCE, MA_TOLERANCE, WALL_DIST_MAX

# ===================== 核心配置 =====================
BATCHSIZE = 1024
weight_name = "mapr_best_697weights"
weight_PATH = f'{project_root}/outputs/weights/mapr_best/{weight_name}.pth'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 输出路径
current_datetime = datetime.now().strftime("%m-%d_%H-%M")
save_root = f'{project_root}/outputs/推理结果/{current_datetime}_Pure_Dimensionless'
os.makedirs(save_root, exist_ok=True)
log_file_path = f'{save_root}/inference.log'
logger.add(log_file_path, rotation="500 MB", level="INFO")

# ===================== 散点图配置 =====================
SCATTER_SAMPLE_NUM = 4000  # 每个散点图采样点数
SCATTER_VAR_COLORS = {
    "U": "#1f77b4",        # 蓝色
    "V": "#ff7f0e",        # 橙色
    "W": "#2ca02c",        # 绿色
    "P_dimless": "#d62728",# 红色
    "T_dimless": "#9467bd",# 紫色
    "K": "#8c564b",        # 棕色
    "Omega": "#e377c2"     # 粉色
}

# ===================== 标准误差计算函数 =====================
def calculate_metrics(y_true, y_pred):
    """
    纯无量纲量误差计算，无任何额外转换
    参数:
        y_true: 反归一化后的物理无量纲真值
        y_pred: 反归一化后的物理无量纲预测值
    返回:
        rmse: 无量纲RMSE
        r2: R²决定系数
    """
    # 过滤无效值
    mask = ~np.isnan(y_true) & ~np.isnan(y_pred) & ~np.isinf(y_true) & ~np.isinf(y_pred)
    y_true = y_true[mask]
    y_pred = y_pred[mask]
    
    if len(y_true) < 2:
        return 0.0, 0.0
    
    # 计算R²
    y_mean = np.mean(y_true)
    sst = np.sum((y_true - y_mean) ** 2)
    sse = np.sum((y_true - y_pred) ** 2)
    r2 = 1 - (sse / sst) if sst != 0 else 1.0
    
    # 计算无量纲RMSE
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    
    return rmse, r2

# ===================== 单变量散点图绘制函数 =====================
def plot_single_var_scatter(y_true, y_pred, var_name, x_sec_mean, ma_sec_mean, pr_sec_mean, save_root):
    """
    纯无量纲量散点图，与误差计算使用完全相同的数据
    """
    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['font.size'] = 12
    
    fig, ax = plt.subplots(figsize=(8, 8), dpi=150)
    
    # 过滤无效值
    valid_mask = ~np.isnan(y_true) & ~np.isnan(y_pred) & ~np.isinf(y_true) & ~np.isinf(y_pred)
    y_true = y_true[valid_mask]
    y_pred = y_pred[valid_mask]
    
    if len(y_true) < 10:
        logger.warning(f"⚠️ 变量 {var_name} 有效点数不足，跳过散点图")
        plt.close()
        return
    
    # 随机采样
    if len(y_true) > SCATTER_SAMPLE_NUM:
        idx = np.random.choice(len(y_true), SCATTER_SAMPLE_NUM, replace=False)
        y_true = y_true[idx]
        y_pred = y_pred[idx]
    
    # 绘制散点和对角线
    ax.scatter(y_true, y_pred, color=SCATTER_VAR_COLORS[var_name], alpha=0.7, s=20, edgecolors='none')
    min_val = min(np.min(y_true), np.min(y_pred))
    max_val = max(np.max(y_true), np.max(y_pred))
    range_val = max_val - min_val
    min_val -= range_val * 0.05
    max_val += range_val * 0.05
    ax.plot([min_val, max_val], [min_val, max_val], 'k--', linewidth=2)
    
    # 图形设置
    ax.set_title(f'{var_name}\nX={x_sec_mean:.4f}m, Ma={ma_sec_mean:.4f}, Pr={pr_sec_mean:.4f}', 
                 fontsize=16, fontweight='bold', pad=20)
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.set_aspect('equal', adjustable='box')
    ax.set_xlim(min_val, max_val)
    ax.set_ylim(min_val, max_val)
    
    # 保存
    scatter_filename = f"X_{x_sec_mean:.4f}m_Ma_{ma_sec_mean:.4f}_Pr_{pr_sec_mean:.4f}_{var_name}_Scatter.png"
    plt.savefig(os.path.join(save_root, scatter_filename), bbox_inches='tight', dpi=150)
    plt.close()
    
    logger.info(f"✅ 散点图已保存: {scatter_filename}")

# ===================== 主函数 =====================
def main():
    set_seed(42)
    logger.info(f"🔥 开始纯无量纲量推理+误差分析+散点图生成")
    
    # 1. 加载数据集
    logger.info("加载数据集...")
    train_dataloader, val_dataloader, data_min_np, data_max_np = data_prepare(BATCHSIZE)
    data_min = torch.tensor(data_min_np, dtype=torch.float32).to(device)
    data_max = torch.tensor(data_max_np, dtype=torch.float32).to(device)
    
    # 反归一化参数
    in_min_4 = data_min[:4].cpu().numpy()  # x,y,z,d
    in_max_4 = data_max[:4].cpu().numpy()
    in_range_4 = in_max_4 - in_min_4 + 1e-8
    in_min_Ma = data_min[4].cpu().numpy()   # Ma
    in_max_Ma = data_max[4].cpu().numpy()
    in_range_Ma = in_max_Ma - in_min_Ma + 1e-8
    in_min_Pr = data_min[5].cpu().numpy() if len(data_min) > 5 else 1.0
    in_max_Pr = data_max[5].cpu().numpy() if len(data_max) > 5 else 1.0
    in_range_Pr = in_max_Pr - in_min_Pr + 1e-8

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

    # 3. 推理并存储纯无量纲量
    logger.info("正在推理...")
    buffer = {
        'X': [], 'Y': [], 'Z': [], 'WD': [], 
        'Ma': [], 'Pr': [],
        # 仅存储反归一化后的物理无量纲量
        'Up': [], 'Vp': [], 'Wp': [], 'Pp': [], 'Tp': [], 'Kp': [], 'Op': [],
        'Ut': [], 'Vt': [], 'Wt': [], 'Pt': [], 'Tt': [], 'Kt': [], 'Ot': []
    }

    with torch.no_grad():
        for input_batch, target_batch in val_dataloader:
            input_batch = input_batch.to(device)
            target_batch = target_batch.to(device)

            # 网络推理 + 反归一化到物理无量纲空间
            output_norm = model(input_batch)
            pred_phys = denormalize_for_pde(device, output_norm, data_min, data_max)
            true_phys = denormalize_for_pde(device, target_batch, data_min, data_max)

            # 转numpy
            in_np = input_batch.cpu().numpy()
            p_np = pred_phys.cpu().numpy()
            t_np = true_phys.cpu().numpy()

            # 坐标和工况反归一化
            in_denorm_4 = in_np[:, :4] * in_range_4 + in_min_4
            Ma_denorm = in_np[:, 4] * in_range_Ma + in_min_Ma
            Pr_denorm = in_np[:, 5] * in_range_Pr + in_min_Pr if in_np.shape[1] > 5 else np.ones_like(Ma_denorm)

            # 存入buffer（纯无量纲量，无任何后续处理）
            buffer['X'].extend(in_denorm_4[:, 0])
            buffer['Y'].extend(in_denorm_4[:, 1])
            buffer['Z'].extend(in_denorm_4[:, 2])
            buffer['WD'].extend(in_denorm_4[:, 3])
            buffer['Ma'].extend(Ma_denorm)
            buffer['Pr'].extend(Pr_denorm)
            
            # 网络输出的原始物理无量纲量
            buffer['Up'].extend(p_np[:, 0]); buffer['Vp'].extend(p_np[:, 1]); buffer['Wp'].extend(p_np[:, 2])
            buffer['Pp'].extend(p_np[:, 3]); buffer['Tp'].extend(p_np[:, 4]); buffer['Kp'].extend(p_np[:, 5]); buffer['Op'].extend(p_np[:, 6])
            
            # 真值的原始物理无量纲量
            buffer['Ut'].extend(t_np[:, 0]); buffer['Vt'].extend(t_np[:, 1]); buffer['Wt'].extend(t_np[:, 2])
            buffer['Pt'].extend(t_np[:, 3]); buffer['Tt'].extend(t_np[:, 4]); buffer['Kt'].extend(t_np[:, 5]); buffer['Ot'].extend(t_np[:, 6])

    # 转为numpy数组
    for k in buffer: buffer[k] = np.array(buffer[k])
    X_dim = buffer['X'] * L

    # 4. 按(X + Ma)分组
    logger.info("正在按(X, Ma)分组...")
    X_rounded = np.round(X_dim / X_TOLERANCE) * X_TOLERANCE
    Ma_rounded = np.round(buffer['Ma'] / MA_TOLERANCE) * MA_TOLERANCE
    x_ma_pairs = np.column_stack([X_rounded, Ma_rounded])
    unique_pairs, inverse_indices = np.unique(x_ma_pairs, axis=0, return_inverse=True)
    logger.info(f"✅ 共找到 {len(unique_pairs)} 组切片，每组生成7张散点图，总计 {len(unique_pairs)*7} 张")

    # 存储所有误差指标
    all_metrics = []

    # 5. 遍历切片计算误差并绘制散点图
    logger.info("开始处理切片...")
    for i, (x_sec_rounded, ma_sec_rounded) in enumerate(unique_pairs):
        mask = inverse_indices == i

        # 提取切片数据
        x_loc = X_dim[mask]
        wd_loc = buffer['WD'][mask]
        ma_slice = buffer['Ma'][mask]
        pr_slice = buffer['Pr'][mask]
        
        x_sec_mean = np.mean(x_loc)
        ma_sec_mean = np.mean(ma_slice)
        pr_sec_mean = np.mean(pr_slice)
        
        if len(x_loc) < 50:
            logger.warning(f"⚠️ 切片 X={x_sec_mean:.4f}m, Ma={ma_sec_mean:.4f} 点数不足，跳过")
            continue

        # 流体域过滤（与训练时保持一致）
        fluid_mask = (wd_loc <= WALL_DIST_MAX + 0.2) & (wd_loc >= -0.1) & (~np.isnan(wd_loc))
        
        # 提取过滤后的纯无量纲量
        Up_f = buffer['Up'][mask][fluid_mask]
        Vp_f = buffer['Vp'][mask][fluid_mask]
        Wp_f = buffer['Wp'][mask][fluid_mask]
        Pp_f = buffer['Pp'][mask][fluid_mask]
        Tp_f = buffer['Tp'][mask][fluid_mask]
        Kp_f = buffer['Kp'][mask][fluid_mask]
        Op_f = buffer['Op'][mask][fluid_mask]
        
        Ut_f = buffer['Ut'][mask][fluid_mask]
        Vt_f = buffer['Vt'][mask][fluid_mask]
        Wt_f = buffer['Wt'][mask][fluid_mask]
        Pt_f = buffer['Pt'][mask][fluid_mask]
        Tt_f = buffer['Tt'][mask][fluid_mask]
        Kt_f = buffer['Kt'][mask][fluid_mask]
        Ot_f = buffer['Ot'][mask][fluid_mask]

        if len(Up_f) < 50:
            logger.warning(f"⚠️ 切片 X={x_sec_mean:.4f}m, Ma={ma_sec_mean:.4f} 流体域点数不足，跳过")
            continue

        # 计算误差并绘制散点图
        logger.info(f"\n📊 切片 X={x_sec_mean:.4f}m, Ma={ma_sec_mean:.4f}, Pr={pr_sec_mean:.4f} 误差分析:")
        logger.info("-" * 60)
        logger.info(f"{'变量':<12} | {'无量纲RMSE':<12} | {'R²':<12}")
        logger.info("-" * 60)
        
        slice_vars = [
            ("U", Up_f, Ut_f),
            ("V", Vp_f, Vt_f),
            ("W", Wp_f, Wt_f),
            ("P_dimless", Pp_f, Pt_f),
            ("T_dimless", Tp_f, Tt_f),
            ("K", Kp_f, Kt_f),
            ("Omega", Op_f, Ot_f)
        ]
        
        for var_name, y_pred, y_true in slice_vars:
            rmse, r2 = calculate_metrics(y_true, y_pred)
            
            all_metrics.append({
                "X(m)": round(x_sec_mean, 4),
                "Ma": round(ma_sec_mean, 4),
                "Pr": round(pr_sec_mean, 4),
                "Variable": var_name,
                "RMSE(Dimensionless)": rmse,
                "R²": r2
            })
            
            logger.info(f"{var_name:<12} | {rmse:<12.6f} | {r2:<12.4f}")
            
            # 绘制散点图
            plot_single_var_scatter(y_true, y_pred, var_name, x_sec_mean, ma_sec_mean, pr_sec_mean, save_root)
        
        logger.info("-" * 60)

    # 6. 计算全局误差
    logger.info("\n" + "="*80)
    logger.info("📊 全局模型预测精度（所有工况流体域平均）")
    logger.info("="*80)
    logger.info(f"{'变量':<12} | {'无量纲RMSE':<12} | {'R²':<12}")
    logger.info("-" * 40)

    global_fluid_mask = (buffer['WD'] <= WALL_DIST_MAX + 0.2) & (buffer['WD'] >= -0.1) & (~np.isnan(buffer['WD']))
    
    global_vars = [
        ("U", buffer['Up'][global_fluid_mask], buffer['Ut'][global_fluid_mask]),
        ("V", buffer['Vp'][global_fluid_mask], buffer['Vt'][global_fluid_mask]),
        ("W", buffer['Wp'][global_fluid_mask], buffer['Wt'][global_fluid_mask]),
        ("P_dimless", buffer['Pp'][global_fluid_mask], buffer['Pt'][global_fluid_mask]),
        ("T_dimless", buffer['Tp'][global_fluid_mask], buffer['Tt'][global_fluid_mask]),
        ("K", buffer['Kp'][global_fluid_mask], buffer['Kt'][global_fluid_mask]),
        ("Omega", buffer['Op'][global_fluid_mask], buffer['Ot'][global_fluid_mask])
    ]
    
    for var_name, y_pred_all, y_true_all in global_vars:
        rmse, r2 = calculate_metrics(y_true_all, y_pred_all)
        
        all_metrics.append({
            "X(m)": "Global",
            "Ma": "All",
            "Pr": "All",
            "Variable": var_name,
            "RMSE(Dimensionless)": rmse,
            "R²": r2
        })
        
        logger.info(f"{var_name:<12} | {rmse:<12.6f} | {r2:<12.4f}")

    logger.info("="*80 + "\n")

    # 7. 保存误差报告
    metrics_df = pd.DataFrame(all_metrics)
    csv_path = f"{save_root}/dimensionless_error_analysis.csv"
    metrics_df.to_csv(csv_path, index=False, float_format="%.6f")
    logger.info(f"✅ 完整误差分析报告已保存至: {csv_path}")

    logger.info(f"\n🎉 所有任务完成！结果保存在: {save_root}")
    logger.info("✅ 所有计算均基于反归一化后的原始物理无量纲量")
    logger.info(f"✅ 已生成 {len(unique_pairs)*7} 张单变量散点图")
    logger.info("✅ 已生成完整的分切片+全局误差报告")

if __name__ == "__main__":
    main()