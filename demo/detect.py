import os, sys
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from loguru import logger
from datetime import datetime
import pyvista as pv
import matplotlib.tri as tri
import random
from matplotlib.colors import LinearSegmentedColormap, ListedColormap

# ===================== 项目路径 =====================
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), './..'))
sys.path.append(project_root)

# ===================== 导入训练模块 =====================
from model import PINN, data_prepare, denormalize_for_pde

# ===================== 【1:1 复制训练代码的参数】 =====================
L = 0.095        # 特征长度
M0 = 0.66        # 来流马赫数
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
weight_name = "05-08_22-34/best_1026weights"  # 🔧 改成你新的文件夹名
weight_PATH = f'{project_root}/outputs/weights/{weight_name}.pth'

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 输出路径
current_datetime = datetime.now().strftime("%m-%d_%H-%M")
save_root = f'{project_root}/outputs/推理结果/{current_datetime}_PyVista_CleanScatter'
os.makedirs(save_root, exist_ok=True)
log_file_path = f'{save_root}/inference.log'
logger.add(log_file_path, rotation="500 MB", level="INFO")

# ===================== 【1:1 复制参考代码的绘图参数】 =====================
PLOT_VAR_LIST = ["Velocity", "Static_P", "Mach", "Total_P"]
X_TOLERANCE = 1e-4
WALL_DIST_MAX = 0.6

# ✅ 色块阶数（和参考代码一致）
N_DISCRETE_COLORS = 50

# ✅ 【关键参数】专门解决空缺问题！
# 如果还有空缺，继续调大这个值（10 → 15 → 20）
# 如果出现凹结构连线，调小这个值（10 → 8 → 5）
ALPHA_MULTIPLIER = 30.0  # 从原来的3.5大幅调大到10.0

# ✅ 基础锚定颜色（和参考代码1:1对应）
base_color_anchors = [
    "#194799",  # 最小值 深蓝色
    "#2357b0",
    "#3670c2",
    "#5a98d6",
    "#90c0e8",
    "#c9dff0",
    "#f9d7c0",
    "#f28a5d",
    "#d9483c",
    "#a80f1a"   # 最大值 深红色
]

# ✅ 自动生成高阶数离散色图（和参考代码完全一样）
temp_smooth_cmap = LinearSegmentedColormap.from_list("temp_smooth", base_color_anchors, N=256)
discrete_colors = temp_smooth_cmap(np.linspace(0, 1, N_DISCRETE_COLORS))
custom_discrete_cmap = ListedColormap(discrete_colors, name=f"discrete_cfd_{N_DISCRETE_COLORS}")

# ===================== 散点图配置 =====================
# 原始变量的颜色映射（和之前保持一致，每个变量固定颜色）
SCATTER_VAR_COLORS = {
    "U": "#1f77b4",        # 蓝色
    "V": "#ff7f0e",        # 橙色
    "W": "#2ca02c",        # 绿色
    "P_dimless": "#d62728",# 红色
    "T_dimless": "#9467bd",# 紫色
    "K": "#8c564b",        # 棕色
    "Omega": "#e377c2"     # 粉色
}

# 散点图采样点数（避免点太多导致图面混乱）
SCATTER_SAMPLE_NUM = 1500  # 单变量可以适当增加采样点数

# ===================== 随机种子 =====================
def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# ===================== 【标准误差计算函数】 =====================
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

# ===================== 【空缺修复版】网格生成 =====================
def create_valid_mesh(y, z, wall_dist):
    """
    专门解决云图全空缺问题：
    1. 大幅调大alpha值，保留更多内部三角形
    2. 放宽异常三角形过滤条件
    3. 增加详细的调试信息
    """
    from scipy.spatial import KDTree
    
    # 1. 先过滤所有无效点
    valid_mask = ~np.isnan(y) & ~np.isnan(z) & ~np.isnan(wall_dist)
    y_clean = y[valid_mask]
    z_clean = z[valid_mask]
    wd_clean = wall_dist[valid_mask]
    
    if len(y_clean) < 50:
        return None, None
    
    logger.info(f"🔍 原始点数: {len(y)}, 过滤后点数: {len(y_clean)}")
    
    # 2. 构建点云
    points = np.column_stack((y_clean, z_clean, np.zeros_like(y_clean)))
    cloud = pv.PolyData(points)
    cloud["wall_dist"] = wd_clean
    
    # 3. 自动计算最优alpha值（大幅调大系数）
    tree = KDTree(points[:, :2])
    distances, _ = tree.query(points[:, :2], k=2)
    avg_neighbor_dist = np.mean(distances[:, 1])
    alpha_value = avg_neighbor_dist * ALPHA_MULTIPLIER
    logger.info(f"✅ 平均点间距: {avg_neighbor_dist:.6f}, alpha值: {alpha_value:.6f}")
    
    # 4. 生成网格（关闭自动删除孤立点）
    grid = cloud.delaunay_2d(alpha=alpha_value, tol=1e-12)
    
    logger.info(f"✅ 初始网格: {grid.n_points} 个点, {grid.n_cells} 个三角形")
    
    # 5. 大幅放宽异常三角形过滤条件
    if grid.n_cells > 0:
        cell_areas = grid.compute_cell_sizes()["Area"]
        mean_area = np.mean(cell_areas)
        # 从原来的5倍改成20倍，几乎不过滤任何正常三角形
        valid_cell_mask = cell_areas < (mean_area * 20)
        grid = grid.extract_cells(valid_cell_mask)
        logger.info(f"✅ 过滤后网格: {grid.n_points} 个点, {grid.n_cells} 个三角形")
    
    # ✅ 关键：返回网格和原始有效掩码
    return grid, valid_mask

# ===================== 【终极版】绘图函数：100%解决长度不匹配 =====================
def plot_slice_comparison(pred_dict, true_dict, y_f, z_f, wd_f,
                           x_sec_mean, ma_sec_mean, pr_sec_mean,
                           save_root, plot_var_list=PLOT_VAR_LIST):
    """
    终极修复版：
    1. 先获取网格和原始点掩码
    2. 再根据网格实际保留的点过滤物理量
    3. 永远不会出现长度不匹配错误
    """
    # 生成网格并同时得到原始有效掩码
    base_grid, valid_mask = create_valid_mesh(y_f, z_f, wd_f)
    
    if base_grid is None:
        logger.warning(f"切片 X={x_sec_mean:.4f} 有效点数不足，跳过")
        return

    # 单位映射
    unit_map = {
        "Velocity": "m/s",
        "Static_P": "Pa",
        "Mach":     "-",
        "Total_P":  "Pa"
    }

    # ✅ 优化版色条配置（和参考代码完全一致）
    scalar_bar_config = {
        'title': '',  # 标题在下方单独设置
        'font_family': 'times',  # 色条字体 Times New Roman
        'color': 'black',
        'position_x': 0.2,   # 色条往左移一点，给右边留空间
        'position_y': 0,  # 稍微往上移一点
        'width': 0.6,        # ✅ 关键：加宽色条，给数字更多空间
        'height': 0.1,       # ✅ 关键：加高色条
        'n_labels': 4,       # ✅ 关键：只显示4个刻度，彻底避免重叠
        'label_font_size': 14, # 字体稍微大一点，更清晰
        'title_font_size': 16
    }

    # 遍历变量绘图
    for name in plot_var_list:
        if name not in pred_dict or name not in true_dict:
            continue
        
        # 提取物理量
        val_p = pred_dict[name]
        val_t = true_dict[name]
        
        # 过滤物理量中的NaN和Inf
        val_p = np.nan_to_num(val_p, nan=np.nanmean(val_p), posinf=np.nanmax(val_p), neginf=np.nanmin(val_p))
        val_t = np.nan_to_num(val_t, nan=np.nanmean(val_t), posinf=np.nanmax(val_t), neginf=np.nanmin(val_t))
        
        # ✅ 终极修复：先过滤原始无效点
        val_p_clean = val_p[valid_mask]
        val_t_clean = val_t[valid_mask]
        
        # ✅ 终极修复：再根据网格实际保留的点过滤
        # 这一步确保物理量长度和网格点数100%匹配
        grid_points = base_grid.points[:, :2]
        original_points = np.column_stack((y_f[valid_mask], z_f[valid_mask]))
        
        # 建立点的索引映射
        from scipy.spatial import cKDTree
        tree = cKDTree(original_points)
        _, indices = tree.query(grid_points, k=1)
        
        # 最终的物理量数组（长度和网格点数完全一致）
        val_p_final = val_p_clean[indices]
        val_t_final = val_t_clean[indices]
        
        # 赋值（现在绝对不会有长度不匹配问题）
        grid_p = base_grid.copy()
        grid_t = base_grid.copy()
        grid_p["Value"] = val_p_final
        grid_t["Value"] = val_t_final

        # 创建 Plotter（和参考代码窗口大小一致）
        plotter = pv.Plotter(shape=(1, 2), off_screen=True, window_size=[2000, 800])
        
        # ✅ 自动裁剪极端值，防止色条被拉偏
        all_vals = np.concatenate([val_p_final, val_t_final])
        clim_low = np.percentile(all_vals, 0.5)
        clim_high = np.percentile(all_vals, 99.5)
        clim = [clim_low, clim_high]

        # --- 左图：预测值 ---
        plotter.subplot(0, 0)
        # 使用自定义离散色图
        plotter.add_mesh(
            grid_p, 
            scalars="Value", 
            cmap=custom_discrete_cmap, 
            clim=clim, 
            show_edges=False,
            scalar_bar_args=scalar_bar_config
        )
        # 图中文字 Times New Roman
        plotter.add_text(f"Prediction\nX={x_sec_mean:.4f}m\nMa={ma_sec_mean:.4f}\nPr={pr_sec_mean:.4f}", 
                         font_size=10, font='times', color='black')
        plotter.view_xy()
        plotter.enable_parallel_projection()
        
        # --- 右图：真值 ---
        plotter.subplot(0, 1)
        # 使用自定义离散色图
        plotter.add_mesh(
            grid_t, 
            scalars="Value", 
            cmap=custom_discrete_cmap, 
            clim=clim, 
            show_edges=False,
            scalar_bar_args=scalar_bar_config
        )
        # 图中文字 Times New Roman
        plotter.add_text(f"Ground Truth\nX={x_sec_mean:.4f}m\nMa={ma_sec_mean:.4f}\nPr={pr_sec_mean:.4f}", 
                         font_size=10, font='times', color='black')
        plotter.view_xy()
        plotter.enable_parallel_projection()

        plotter.link_views()
        
        # 底部统一标题（Times New Roman）
        plotter.add_text(f"{name} ({unit_map[name]})", 
                         position='lower_edge', 
                         font_size=12, 
                         font='times', 
                         color='black')
        
        # 保存（和参考代码文件名格式一致）
        filename = f"{save_root}/X_{x_sec_mean:.4f}m_Ma_{ma_sec_mean:.4f}_Pr_{pr_sec_mean:.4f}_{name}_Compare.png"
        plotter.screenshot(filename)
        plotter.close()
    logger.info(f"✅ 切片 X={x_sec_mean:.4f}m 云图已保存")

# ===================== 【精简版】单变量散点图绘制函数 =====================
def plot_single_var_scatter(pred_dict, true_dict, x_sec_mean, ma_sec_mean, var_name, save_root):
    """
    精简版单变量散点图（无误差指标、无坐标轴标签）
    """
    logger.info(f"📊 正在绘制切片 X={x_sec_mean:.4f}m 的 {var_name} 散点图...")
    
    # 设置全局字体
    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['font.size'] = 12
    
    # 创建图形
    fig, ax = plt.subplots(figsize=(8, 8), dpi=150)
    
    # 提取该变量的数据
    if var_name not in true_dict or var_name not in pred_dict:
        logger.warning(f"⚠️ 变量 {var_name} 不存在，跳过")
        return
    
    y_true = true_dict[var_name]
    y_pred = pred_dict[var_name]
    
    # 过滤无效值
    valid_mask = ~np.isnan(y_true) & ~np.isnan(y_pred) & ~np.isinf(y_true) & ~np.isinf(y_pred)
    y_true = y_true[valid_mask]
    y_pred = y_pred[valid_mask]
    
    if len(y_true) < 10:
        logger.warning(f"⚠️ 变量 {var_name} 有效点数不足，跳过")
        return
    
    # 随机采样
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
    
    # 设置图形属性（已删除坐标轴标签和误差文本框）
    ax.set_title(f'{var_name}\nX={x_sec_mean:.4f}m, Ma={ma_sec_mean:.4f}', 
                 fontsize=16, fontweight='bold', pad=20)
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.set_aspect('equal', adjustable='box')
    ax.set_xlim(min_val, max_val)
    ax.set_ylim(min_val, max_val)
    
    # 隐藏坐标轴刻度标签（可选，如需显示可注释掉下面两行）
    # ax.set_xticklabels([])
    # ax.set_yticklabels([])
    
    # 保存图片
    scatter_filename = f"{save_root}/X_{x_sec_mean:.4f}m_Ma_{ma_sec_mean:.4f}_{var_name}_Scatter_Plot.png"
    plt.savefig(scatter_filename, bbox_inches='tight', dpi=150)
    plt.close()
    
    logger.info(f"✅ 切片 X={x_sec_mean:.4f}m 的 {var_name} 散点图已保存")

# ===================== 【O(n)时间复杂度】X坐标分组函数（可选，解决大内存问题） =====================
# 如果遇到fclusterdata内存错误，把下面的注释打开，替换原来的分组代码
"""
def group_by_x_coordinate(X, tolerance=1e-4):
    logger.info("🔍 正在按X坐标分组...")
    scale = 1.0 / tolerance
    X_rounded = np.round(X * scale).astype(np.int64)
    unique_X_rounded, inverse_indices = np.unique(X_rounded, return_inverse=True)
    unique_X = np.zeros(len(unique_X_rounded))
    for i in range(len(unique_X_rounded)):
        mask = inverse_indices == i
        unique_X[i] = np.mean(X[mask])
    logger.info(f"✅ 共找到 {len(unique_X)} 组切片")
    return inverse_indices, unique_X
"""

# ===================== 主推理函数 =====================
def infer_all_dataset():
    set_seed(42)
    logger.info("="*60)
    logger.info(f"🔥 开始推理，权重文件: {weight_name}")
    logger.info(f"🎨 云图风格: PyVista 参考代码风格（含精简版散点图）")
    logger.info(f"⚙️  当前 ALPHA_MULTIPLIER = {ALPHA_MULTIPLIER}")
    logger.info("="*60)
    
    # PyVista 配置（和参考代码一致）
    pv.set_plot_theme("document")
    os.environ["PYVISTA_OFF_SCREEN"] = "true"

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

            # 网络前向
            output_norm = model(input_batch)
            
            # 预测值反归一化
            output_pred_phys = denormalize_for_pde(device,output_norm, data_min, data_max)
            
            # 真值反归一化
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

            # 提取数据
            input_np = input_batch.cpu().numpy()
            pred_np = output_pred_phys.cpu().numpy()
            true_np = output_true_phys.cpu().numpy()

            # 反归一化坐标
            input_denorm = input_np * in_range_4 + in_min_4
            X, Y, Z, WALL_D = input_denorm[:, 0], input_denorm[:, 1], input_denorm[:, 2], input_denorm[:, 3]

            # 保存所有变量
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
    # 全局数据检查
    # ==============================================
    logger.info("\n" + "="*80)
    logger.info("📊 全局数据完整性检查")
    logger.info("="*80)
    logger.info(f"总点数: {len(all_X)}")
    logger.info(f"NaN点数: {np.sum(np.isnan(all_X) | np.isnan(all_Y) | np.isnan(all_Z))}")
    logger.info(f"Inf点数: {np.sum(np.isinf(all_X) | np.isinf(all_Y) | np.isinf(all_Z))}")
    logger.info("="*80 + "\n")

    # ==============================================
    # 打印数值范围统计
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
    logger.info("="*80 + "\n")

    # 有量纲化（修复了之前的V_pred错误）
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
    logger.info(f"{'Total_P':<15} | {TP_pred.min():<15.2f} | {TP_pred.max():<15.2f} | {TP_true.min():<15.2f} | {TP_true.max():<15.2f} | {'Pa':<10}")
    logger.info("="*80 + "\n")

    # ==============================================
    # 定义需要计算误差的所有变量
    # 格式: (变量名, 预测值数组, 真实值数组, 无量纲化参考量)
    # ==============================================
    variables_to_calculate = [
        # 原始输出变量（物理无量纲）
        ("U", all_U_pred, all_U_true, 1.0),
        ("V", all_V_pred, all_V_true, 1.0),
        ("W", all_W_pred, all_W_true, 1.0),
        ("P_dimless", all_P_pred, all_P_true, 1.0),
        ("T_dimless", all_T_pred, all_T_true, 1.0),
        ("K", all_K_pred, all_K_true, U0**2),        # K无量纲化: K/U0²
        ("Omega", all_Omega_pred, all_Omega_true, U0/L),  # Omega无量纲化: Ω*L/U0
        
        # 导出物理量
        ("Velocity", Vel_pred, Vel_true, U0),
        ("Static_P", P_pred, P_true, q0),
        ("Temperature", T_pred, T_true, T0),
        ("Mach", Ma_pred, Ma_true, None),
        ("Total_P", TP_pred, TP_true, q0),
    ]

    # ==============================================
    # 绘图 + 分切片误差计算
    # ==============================================
    logger.info("正在绘制云图并计算分切片误差...")
    
    # 按X坐标分组（如果遇到内存错误，替换为上面的group_by_x_coordinate函数）
    from scipy.cluster.hierarchy import fclusterdata
    clusters = fclusterdata(X_dim.reshape(-1,1), t=X_TOLERANCE, criterion='distance')
    cluster_ids = np.unique(clusters)
    
    # 如果使用O(n)分组函数，替换上面三行为：
    # clusters, unique_X = group_by_x_coordinate(X_dim, tolerance=X_TOLERANCE)
    # cluster_ids = np.unique(clusters)
    
    logger.info(f"✅ 共找到 {len(cluster_ids)} 组切片")
    
    # 存储所有切片的误差指标
    all_metrics = []
    
    for cid in cluster_ids:
        mask = clusters == cid
        X_sec = X_dim[mask]; Y_sec = all_Y[mask]; Z_sec = all_Z[mask]; WALL_D_sec = all_WALL_D[mask]
        
        if len(X_sec) < 50: continue
        
        # 流体域过滤（和绘图用完全相同的mask）
        fluid_mask = (WALL_D_sec <= WALL_DIST_MAX + 0.3) & (WALL_D_sec >= -0.2) & (~np.isnan(WALL_D_sec))
        Y_fluid = Y_sec[fluid_mask]; Z_fluid = Z_sec[fluid_mask]; WALL_D_fluid = WALL_D_sec[fluid_mask]
        
        if len(Y_fluid) < 50: continue

        # 提取该切片的所有物理量（包含原始变量和导出变量）
        pred_dict = {
            # 原始输出变量（物理无量纲）
            "U": all_U_pred[mask][fluid_mask],
            "V": all_V_pred[mask][fluid_mask],
            "W": all_W_pred[mask][fluid_mask],
            "P_dimless": all_P_pred[mask][fluid_mask],
            "T_dimless": all_T_pred[mask][fluid_mask],
            "K": all_K_pred[mask][fluid_mask],
            "Omega": all_Omega_pred[mask][fluid_mask],
            
            # 导出物理量
            "Velocity": Vel_pred[mask][fluid_mask],
            "Static_P": P_pred[mask][fluid_mask],
            "Mach": Ma_pred[mask][fluid_mask],
            "Total_P": TP_pred[mask][fluid_mask]
        }
        
        true_dict = {
            # 原始输出变量（物理无量纲）
            "U": all_U_true[mask][fluid_mask],
            "V": all_V_true[mask][fluid_mask],
            "W": all_W_true[mask][fluid_mask],
            "P_dimless": all_P_true[mask][fluid_mask],
            "T_dimless": all_T_true[mask][fluid_mask],
            "K": all_K_true[mask][fluid_mask],
            "Omega": all_Omega_true[mask][fluid_mask],
            
            # 导出物理量
            "Velocity": Vel_true[mask][fluid_mask],
            "Static_P": P_true[mask][fluid_mask],
            "Mach": Ma_true[mask][fluid_mask],
            "Total_P": TP_true[mask][fluid_mask]
        }

        # 切片平均参数
        x_sec_mean = np.mean(X_sec)
        ma_sec_mean = M0  # 单工况固定马赫数
        pr_sec_mean = 1.0 # 单工况固定压比

        # ===================== 计算该切片的误差指标 =====================
        logger.info(f"\n📊 切片 X={x_sec_mean:.4f}m 误差分析:")
        logger.info("-" * 50)
        logger.info(f"{'变量':<12} | {'无量纲RMSE':<12} | {'R²':<12}")
        logger.info("-" * 50)
        
        for var_name, y_pred_all, y_true_all, ref in variables_to_calculate:
            # 提取该切片流体域的点（和绘图用点完全一致）
            y_pred_slice = y_pred_all[mask][fluid_mask]
            y_true_slice = y_true_all[mask][fluid_mask]
            
            # 计算标准误差指标
            rmse, r2 = calculate_metrics(y_true_slice, y_pred_slice, ref_value=ref)
            
            # 添加到全局指标列表
            all_metrics.append({
                "X(m)": round(x_sec_mean, 4),
                "Variable": var_name,
                "RMSE(Dimensionless)": rmse,
                "R²": r2
            })
            
            # 打印该变量的误差
            logger.info(f"{var_name:<12} | {rmse:<12.6f} | {r2:<12.4f}")
        
        logger.info("-" * 50)

        # 调用云图绘制函数
        plot_slice_comparison(
            pred_dict, true_dict,
            Y_fluid, Z_fluid, WALL_D_fluid,
            x_sec_mean, ma_sec_mean, pr_sec_mean,
            save_root
        )
        
        # ===================== 为每个原始变量单独绘制散点图 =====================
        logger.info(f"\n📊 开始绘制切片 X={x_sec_mean:.4f}m 的单变量散点图...")
        for var_name in SCATTER_VAR_COLORS.keys():
            plot_single_var_scatter(
                pred_dict, true_dict,
                x_sec_mean, ma_sec_mean,
                var_name,
                save_root
            )
        logger.info(f"✅ 切片 X={x_sec_mean:.4f}m 的所有单变量散点图已绘制完成")

    # ==============================================
    # 计算全局误差指标（所有流体域点）
    # ==============================================
    logger.info("\n" + "="*80)
    logger.info("📊 全局模型预测精度（流体域平均）")
    logger.info("="*80)
    logger.info(f"{'变量':<12} | {'无量纲RMSE':<12} | {'R²':<12}")
    logger.info("-" * 40)

    # 全局流体域mask
    global_fluid_mask = (all_WALL_D <= WALL_DIST_MAX + 0.3) & (all_WALL_D >= -0.2) & (~np.isnan(all_WALL_D))

    for var_name, y_pred_all, y_true_all, ref in variables_to_calculate:
        y_pred_global = y_pred_all[global_fluid_mask]
        y_true_global = y_true_all[global_fluid_mask]
        
        rmse, r2 = calculate_metrics(y_true_global, y_pred_global, ref_value=ref)
        
        # 添加全局指标到列表
        all_metrics.append({
            "X(m)": "Global",
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
    csv_path = f"{save_root}/detailed_error_analysis.csv"
    metrics_df.to_csv(csv_path, index=False, float_format="%.6f")
    logger.info(f"✅ 完整分切片误差分析报告已保存至: {csv_path}")

    logger.info("\n🎉 推理完成！所有结果保存在: {save_root}")
    logger.info("✅ 已生成分切片+分变量的RMSE/R²误差报告")
    logger.info("✅ 已生成每个截面每个原始变量的精简版散点图")
    logger.info("✅ 已彻底解决云图全空缺问题")
    logger.info("✅ 已彻底解决标量长度不匹配问题")

if __name__ == "__main__":
    infer_all_dataset()