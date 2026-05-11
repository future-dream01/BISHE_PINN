import os, sys
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from loguru import logger
from datetime import datetime
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), './..'))
sys.path.append(project_root)
from model import PINN, data_prepare, denormalize_for_pde
from model.utils_zx import (
    set_seed, L,
    compute_physical_quantities,
    plot_slice_comparison,
    X_TOLERANCE, MA_TOLERANCE, WALL_DIST_MAX
)

# ===================== 配置 =====================
BATCHSIZE = 1024
weight_name = "mapr_best_200weights"    # 目前最好的是697
weight_PATH = f'{project_root}/outputs/weights/mapr_best/{weight_name}.pth'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 输出路径
current_datetime = datetime.now().strftime("%m-%d_%H-%M")
save_root = f'{project_root}/outputs/推理结果/{current_datetime}_ErrorAnalysis_Scatter'
os.makedirs(save_root, exist_ok=True)
log_file_path = f'{save_root}/inference.log'
logger.add(log_file_path, rotation="500 MB", level="INFO")

# ===================== ✅ 新增：散点图配置 =====================
SCATTER_SAMPLE_NUM = 1500  # 每个散点图的采样点数
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
def calculate_metrics(y_true, y_pred, ref_value=None):
    """
    计算标准学术误差指标：无量纲RMSE和R²
    参数:
        y_true: 真实值 (numpy.ndarray)
        y_pred: 预测值 (numpy.ndarray)
        ref_value: 物理参考量（用于无量纲化RMSE）
                  本身无量纲的变量传None或1.0
    返回:
        rmse_dimless: 无量纲RMSE
        r2: R²决定系数
    """
    # 过滤所有无效值
    mask = ~np.isnan(y_true) & ~np.isnan(y_pred) & ~np.isinf(y_true) & ~np.isinf(y_pred)
    y_true = y_true[mask]
    y_pred = y_pred[mask]
    
    if len(y_true) < 2:
        return 0.0, 0.0
    
    # 计算R²（线性变换不变，用原始值即可）
    y_mean = np.mean(y_true)
    sst = np.sum((y_true - y_mean) ** 2)
    sse = np.sum((y_true - y_pred) ** 2)
    r2 = 1 - (sse / sst) if sst != 0 else 1.0
    
    # 计算无量纲RMSE
    if ref_value is None or ref_value == 0:
        # 本身就是无量纲量（如马赫数、已无量纲化的原始输出）
        rmse_dimless = np.sqrt(np.mean((y_true - y_pred) ** 2))
    else:
        rmse_dimless = np.sqrt(np.mean(((y_true - y_pred) / ref_value) ** 2))
    
    return rmse_dimless, r2

# ===================== ✅ 新增：单变量散点图绘制函数 =====================
def plot_single_var_scatter(y_true, y_pred, var_name, x_sec_mean, ma_sec_mean, pr_sec_mean, save_root):
    """
    为单个原始无量纲变量绘制独立的散点图
    参数:
        y_true: 无量纲真值
        y_pred: 无量纲预测值
        var_name: 变量名
        x_sec_mean: 截面X坐标(m)
        ma_sec_mean: 平均马赫数
        pr_sec_mean: 平均压比
        save_root: 保存根目录
    """
    # 设置全局字体（学术论文标准）
    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['font.size'] = 12
    
    # 创建图形
    fig, ax = plt.subplots(figsize=(8, 8), dpi=150)
    
    # 过滤无效值
    valid_mask = ~np.isnan(y_true) & ~np.isnan(y_pred) & ~np.isinf(y_true) & ~np.isinf(y_pred)
    y_true = y_true[valid_mask]
    y_pred = y_pred[valid_mask]
    
    if len(y_true) < 10:
        logger.warning(f"⚠️ 变量 {var_name} 有效点数不足，跳过散点图")
        plt.close()
        return
    
    # 随机采样，避免点太多导致图面混乱
    if len(y_true) > SCATTER_SAMPLE_NUM:
        idx = np.random.choice(len(y_true), SCATTER_SAMPLE_NUM, replace=False)
        y_true = y_true[idx]
        y_pred = y_pred[idx]
    
    # 获取该变量的颜色
    color = SCATTER_VAR_COLORS[var_name]
    
    # 绘制散点
    ax.scatter(y_true, y_pred, color=color, alpha=0.7, s=20, edgecolors='none')
    
    # 绘制完美预测对角线
    min_val = min(np.min(y_true), np.min(y_pred))
    max_val = max(np.max(y_true), np.max(y_pred))
    # 扩展一点范围，让点不贴边
    range_val = max_val - min_val
    min_val -= range_val * 0.05
    max_val += range_val * 0.05
    
    ax.plot([min_val, max_val], [min_val, max_val], 'k--', linewidth=2)
    
    # 设置图形属性（精简版，无坐标轴标签）
    ax.set_title(f'{var_name}\nX={x_sec_mean:.4f}m, Ma={ma_sec_mean:.4f}, Pr={pr_sec_mean:.4f}', 
                 fontsize=16, fontweight='bold', pad=20)
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.set_aspect('equal', adjustable='box')
    ax.set_xlim(min_val, max_val)
    ax.set_ylim(min_val, max_val)
    
    # 保存图片（文件名包含完整工况信息）
    scatter_filename = f"X_{x_sec_mean:.4f}m_Ma_{ma_sec_mean:.4f}_Pr_{pr_sec_mean:.4f}_{var_name}_Scatter.png"
    scatter_save_path = os.path.join(save_root, scatter_filename)
    plt.savefig(scatter_save_path, bbox_inches='tight', dpi=150)
    plt.close()
    
    logger.info(f"✅ 散点图已保存: {scatter_filename}")

# ===================== 主推理函数 =====================
def main():
    set_seed(42)
    logger.info(f"🔥 开始多工况推理+误差分析+散点图生成")
    import pyvista as pv
    pv.set_plot_theme("document")
    os.environ["PYVISTA_OFF_SCREEN"] = "true"

    logger.info("加载数据集")
    train_dataloader, val_dataloader, data_min_np, data_max_np = data_prepare(BATCHSIZE)
    data_min = torch.tensor(data_min_np, dtype=torch.float32).to(device)
    data_max = torch.tensor(data_max_np, dtype=torch.float32).to(device)
    
    # 反归一化参数
    in_min_4 = data_min[:4].cpu().numpy()  # x,y,z,d
    in_max_4 = data_max[:4].cpu().numpy()
    in_range_4 = in_max_4 - in_min_4 + 1e-8
    
    # 马赫数反归一化参数
    in_min_Ma = data_min[4].cpu().numpy()   # Ma
    in_max_Ma = data_max[4].cpu().numpy()
    in_range_Ma = in_max_Ma - in_min_Ma + 1e-8
    
    # 压比 Pr 的反归一化参数
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

    logger.info("正在进行推理...")
    buffer = {
        'X': [], 'Y': [], 'Z': [], 'WD': [], 
        'Ma': [], 'Pr': [], # 保存每个点的工况参数
        # 预测值 (物理空间无量纲)
        'Up': [], 'Vp': [], 'Wp': [], 'Pp': [], 'Tp': [], 'Kp': [], 'Op': [],
        # 真值 (物理空间无量纲)
        'Ut': [], 'Vt': [], 'Wt': [], 'Pt': [], 'Tt': [], 'Kt': [], 'Ot': []
    }

    with torch.no_grad():
        for batch_idx, (input_batch, target_batch) in enumerate(val_dataloader):
            input_batch = input_batch.to(device)
            target_batch = target_batch.to(device)

            # 推理
            output_norm = model(input_batch)
            pred_phys = denormalize_for_pde(device, output_norm.clone(), data_min.clone(), data_max.clone())
            true_phys = denormalize_for_pde(device, target_batch.clone(), data_min.clone(), data_max.clone())

            # 转 Numpy
            in_np = input_batch.cpu().numpy()
            p_np = pred_phys.cpu().numpy()
            t_np = true_phys.cpu().numpy()

            # 坐标、d反归一化
            in_denorm_4 = in_np[:, :4] * in_range_4 + in_min_4

            # 马赫数反归一化
            Ma_denorm = in_np[:, 4] * in_range_Ma + in_min_Ma
            
            # 压比 Pr 反归一化
            if in_np.shape[1] > 5:
                Pr_denorm = in_np[:, 5] * in_range_Pr + in_min_Pr
            else:
                Pr_denorm = np.ones_like(Ma_denorm)

            # 存入 Buffer
            buffer['X'].extend(in_denorm_4[:, 0])
            buffer['Y'].extend(in_denorm_4[:, 1])
            buffer['Z'].extend(in_denorm_4[:, 2])
            buffer['WD'].extend(in_denorm_4[:, 3])
            buffer['Ma'].extend(Ma_denorm)
            buffer['Pr'].extend(Pr_denorm)
            
            # 预测值
            buffer['Up'].extend(p_np[:, 0]); buffer['Vp'].extend(p_np[:, 1]); buffer['Wp'].extend(p_np[:, 2])
            buffer['Pp'].extend(p_np[:, 3]); buffer['Tp'].extend(p_np[:, 4]); buffer['Kp'].extend(p_np[:, 5]); buffer['Op'].extend(p_np[:, 6])
            
            # 真值
            buffer['Ut'].extend(t_np[:, 0]); buffer['Vt'].extend(t_np[:, 1]); buffer['Wt'].extend(t_np[:, 2])
            buffer['Pt'].extend(t_np[:, 3]); buffer['Tt'].extend(t_np[:, 4]); buffer['Kt'].extend(t_np[:, 5]); buffer['Ot'].extend(t_np[:, 6])

    # 转为 Numpy 数组
    for k in buffer: buffer[k] = np.array(buffer[k])
    X_dim = buffer['X'] * L

    # ==========================================
    # 分组 (按 X + Ma)
    # ==========================================
    logger.info("正在按(X, Ma)分组...")
    X_rounded = np.round(X_dim / X_TOLERANCE) * X_TOLERANCE
    Ma_rounded = np.round(buffer['Ma'] / MA_TOLERANCE) * MA_TOLERANCE
    x_ma_pairs = np.column_stack([X_rounded, Ma_rounded])
    unique_pairs, inverse_indices = np.unique(x_ma_pairs, axis=0, return_inverse=True)
    logger.info(f"✅ 共找到 {len(unique_pairs)} 组切片 (X+Ma组合)")
    logger.info(f"✅ 每个切片将生成 7 张单变量散点图，总计 {len(unique_pairs)*7} 张")

    # ==========================================
    # 定义需要计算误差和绘制散点图的原始变量
    # 格式: (变量名, 预测值数组, 真实值数组, 无量纲化参考量)
    # ==========================================
    raw_variables = [
        ("U", buffer['Up'], buffer['Ut'], 1.0),
        ("V", buffer['Vp'], buffer['Vt'], 1.0),
        ("W", buffer['Wp'], buffer['Wt'], 1.0),
        ("P_dimless", buffer['Pp'], buffer['Pt'], 1.0),
        ("T_dimless", buffer['Tp'], buffer['Tt'], 1.0),
        ("K", buffer['Kp'], buffer['Kt'], 1.0),  # 已为物理无量纲
        ("Omega", buffer['Op'], buffer['Ot'], 1.0) # 已为物理无量纲
    ]

    # 存储所有切片的误差指标
    all_metrics = []

    # ==========================================
    # 主循环：遍历切片 -> 计算误差 -> 绘制散点图 -> 绘图 -> 保存特定切片
    # ==========================================
    logger.info("开始处理切片并计算误差...")

    num_cont=0
    
    for i, (x_sec_rounded, ma_sec_rounded) in enumerate(unique_pairs):
        mask = inverse_indices == i

        # 提取该切片的所有数据
        x_loc = X_dim[mask]
        y_loc = buffer['Y'][mask]
        z_loc = buffer['Z'][mask]
        wd_loc = buffer['WD'][mask]
        
        # 提取该切片所有点的 Ma 和 Pr
        ma_slice = buffer['Ma'][mask]
        pr_slice = buffer['Pr'][mask]
        
        # 计算平均工况 (用于标题和误差记录)
        x_sec_mean = np.mean(x_loc)
        ma_sec_mean = np.mean(ma_slice)
        pr_sec_mean = np.mean(pr_slice)
        
        if len(x_loc) < 50:
            logger.warning(f"⚠️ 切片 X={x_sec_mean:.4f}m, Ma={ma_sec_mean:.4f} 点数不足，跳过")
            continue

        # ✅ 流体域过滤（误差计算、散点图、云图使用完全相同的点）
        fluid_mask = (wd_loc <= WALL_DIST_MAX + 0.2) & (wd_loc >= -0.1) & (~np.isnan(wd_loc))
        
        # 提取最终数据（应用流体域过滤）
        y_f = y_loc[fluid_mask]
        z_f = z_loc[fluid_mask]
        wd_f = wd_loc[fluid_mask]
        
        # 提取无量纲原始变量（应用流体域过滤）
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
        
        # 提取对应点的 Ma 和 Pr (应用流体域过滤)
        ma_f = ma_slice[fluid_mask]
        pr_f = pr_slice[fluid_mask]

        if len(y_f) < 50:
            logger.warning(f"⚠️ 切片 X={x_sec_mean:.4f}m, Ma={ma_sec_mean:.4f} 流体域点数不足，跳过")
            continue

        # ===================== 计算该切片的原始变量误差 =====================
        logger.info(f"\n📊 切片 X={x_sec_mean:.4f}m, Ma={ma_sec_mean:.4f}, Pr={pr_sec_mean:.4f} 误差分析:")
        logger.info("-" * 60)
        logger.info(f"{'变量':<12} | {'无量纲RMSE':<12} | {'R²':<12}")
        logger.info("-" * 60)
        
        # 逐个计算原始变量误差并绘制散点图
        slice_vars = [
            ("U", Up_f, Ut_f, 1.0),
            ("V", Vp_f, Vt_f, 1.0),
            ("W", Wp_f, Wt_f, 1.0),
            ("P_dimless", Pp_f, Pt_f, 1.0),
            ("T_dimless", Tp_f, Tt_f, 1.0),
            ("K", Kp_f, Kt_f, 1.0),
            ("Omega", Op_f, Ot_f, 1.0)
        ]
        
        for var_name, y_pred, y_true, ref in slice_vars:
            rmse, r2 = calculate_metrics(y_true, y_pred, ref_value=ref)
            
            # 添加到全局指标列表
            all_metrics.append({
                "X(m)": round(x_sec_mean, 4),
                "Ma": round(ma_sec_mean, 4),
                "Pr": round(pr_sec_mean, 4),
                "Variable": var_name,
                "RMSE(Dimensionless)": rmse,
                "R²": r2
            })
            
            # 打印该变量的误差
            logger.info(f"{var_name:<12} | {rmse:<12.6f} | {r2:<12.4f}")
            
            # ✅ 新增：绘制该变量的散点图
            plot_single_var_scatter(
                y_true, y_pred,
                var_name,
                x_sec_mean, ma_sec_mean, pr_sec_mean,
                save_root
            )
        
        logger.info("-" * 60)

        # 有量纲化（用于绘图和保存特定切片）
        pred_dict, true_dict = compute_physical_quantities(
            Up_f, Vp_f, Wp_f, Pp_f, Tp_f,
            Ut_f, Vt_f, Wt_f, Pt_f, Tt_f,
            ma_f, pr_f
        )

        num_cont+=1
        if num_cont==4:
            logger.info(f"找到目标切片,开始写入网络预测的csv文件")
            Pressure_pred=pred_dict["Static_P"]
            Total_Pressure_pred=pred_dict["Total_P"]
            Mach_pred=pred_dict["Mach"]
            data=np.column_stack([x_loc[fluid_mask],y_f,z_f,Pressure_pred,Total_Pressure_pred,Mach_pred])
            np.savetxt(f"{save_root}/0515_042_pred.csv",data,delimiter=" ")
            logger.info(f"✅ 网络预测文件写入完成")

            logger.info(f"找到目标切片,开始写入真值的csv文件")
            Pressure_true=true_dict["Static_P"]
            Total_Pressure_true=true_dict["Total_P"]
            Mach_true=true_dict["Mach"]
            data=np.column_stack([x_loc[fluid_mask],y_f,z_f,Pressure_true,Total_Pressure_true,Mach_true])
            np.savetxt(f"{save_root}/0515_042_true.csv",data,delimiter=" ")
            logger.info(f"✅ 真值文件写入完成")

        if num_cont==5:
            logger.info(f"找到目标切片,开始写入网络预测的csv文件")
            Pressure_pred=pred_dict["Static_P"]
            Total_Pressure_pred=pred_dict["Total_P"]
            Mach_pred=pred_dict["Mach"]
            data=np.column_stack([x_loc[fluid_mask],y_f,z_f,Pressure_pred,Total_Pressure_pred,Mach_pred])
            np.savetxt(f"{save_root}/0515_090_pred.csv",data,delimiter=" ")
            logger.info(f"✅ 网络预测文件写入完成")

            logger.info(f"找到目标切片,开始写入真值的csv文件")
            Pressure_true=true_dict["Static_P"]
            Total_Pressure_true=true_dict["Total_P"]
            Mach_true=true_dict["Mach"]
            data=np.column_stack([x_loc[fluid_mask],y_f,z_f,Pressure_true,Total_Pressure_true,Mach_true])
            np.savetxt(f"{save_root}/0515_090_true.csv",data,delimiter=" ")
            logger.info(f"✅ 真值文件写入完成")

        # 绘图
        plot_slice_comparison(
            pred_dict, true_dict, y_f, z_f, wd_f,
            x_sec_mean, ma_sec_mean, pr_sec_mean,
            save_root
        )

    # ==============================================
    # 计算全局误差指标（所有流体域点）
    # ==============================================
    logger.info("\n" + "="*80)
    logger.info("📊 全局模型预测精度（所有工况流体域平均）")
    logger.info("="*80)
    logger.info(f"{'变量':<12} | {'无量纲RMSE':<12} | {'R²':<12}")
    logger.info("-" * 40)

    # 全局流体域mask
    global_fluid_mask = (buffer['WD'] <= WALL_DIST_MAX + 0.2) & (buffer['WD'] >= -0.1) & (~np.isnan(buffer['WD']))

    for var_name, y_pred_all, y_true_all, ref in raw_variables:
        y_pred_global = y_pred_all[global_fluid_mask]
        y_true_global = y_true_all[global_fluid_mask]
        
        rmse, r2 = calculate_metrics(y_true_global, y_pred_global, ref_value=ref)
        
        # 添加全局指标到列表
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

    # ==============================================
    # 保存完整误差报告为CSV
    # ==============================================
    metrics_df = pd.DataFrame(all_metrics)
    csv_path = f"{save_root}/multi_case_raw_variables_error_analysis.csv"
    metrics_df.to_csv(csv_path, index=False, float_format="%.6f")
    logger.info(f"✅ 完整多工况误差分析报告已保存至: {csv_path}")

    logger.info(f"\n🎉 推理完成！所有结果保存在: {save_root}")
    logger.info("✅ 已生成多工况分切片原始变量RMSE/R²误差报告")
    logger.info(f"✅ 已生成 {len(unique_pairs)*7} 张单变量散点图")
    logger.info("✅ 已保存目标切片的预测值和真值CSV文件")
    logger.info("✅ 已生成所有切片的云图对比")

if __name__ == "__main__":
    main()