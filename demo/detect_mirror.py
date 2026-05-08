import os, sys
import torch
import numpy as np
from loguru import logger
from datetime import datetime
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), './..'))
sys.path.append(project_root)
from model import PINN, data_prepare, denormalize_for_pde

# ===================== 下面是整合后的 utils.py 功能 =====================
import pyvista as pv
import matplotlib.tri as tri
import random
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, ListedColormap

# ===================== 固定物理常数 =====================
L = 0.095        # 特征长度 (仅用于坐标有量纲化)
T0 = 249.15      # 来流静温
GAMMA = 1.4      # 比热比
R_gas = 287      # 气体常数
P0 = 47181.0 # 参考静压 (根据你的实际情况修改)

# ===================== 绘图参数 =====================
PLOT_VAR_LIST = ["Velocity", "Static_P", "Mach", "Total_P"]
X_TOLERANCE = 1e-4
MA_TOLERANCE = 1e-3
WALL_DIST_MAX = 0.6

# ✅ 【关键参数1】在这里调节色块阶数！
N_DISCRETE_COLORS = 50

# ✅ 【关键参数2】在这里调节凹结构挖空程度！
# 如果凹的地方还有连线，把这个值调小（比如 0.005, 0.003）
# 如果把流道内部挖空了，把这个值调大（比如 0.02, 0.03）
ALPHA_VALUE = 0.012 / L  # 初始值，你可以微调

# 基础锚定颜色
base_color_anchors = [
    "#194799", "#2357b0", "#3670c2", "#5a98d6", "#90c0e8",
    "#c9dff0", "#f9d7c0", "#f28a5d", "#d9483c", "#a80f1a"
]

# 色图生成
temp_smooth_cmap = LinearSegmentedColormap.from_list("temp_smooth", base_color_anchors, N=256)
discrete_colors = temp_smooth_cmap(np.linspace(0, 1, N_DISCRETE_COLORS))
custom_discrete_cmap = ListedColormap(discrete_colors, name=f"discrete_cfd_{N_DISCRETE_COLORS}")

# ===================== 工具函数 =====================
def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# ===================== 网格生成（修复凹结构+修复比例版）=====================
def create_valid_mesh(x, y, wall_dist):
    """
    修复版：
    1. 先归一化坐标，保证X/Y比例1:1，不被挤压
    2. 使用可调的ALPHA_VALUE，更好地挖空凹结构
    """
    # ========== 修复1：坐标归一化，保证比例正常 ==========
    x_min, x_max = x.min(), x.max()
    y_min, y_max = y.min(), y.max()
    
    # 把X和Y都缩放到 [0, 1] 区间
    x_norm = (x - x_min) / (x_max - x_min + 1e-12)
    y_norm = (y - y_min) / (y_max - y_min + 1e-12)
    
    # ========== 修复2：使用归一化后的坐标做三角剖分 ==========
    points = np.column_stack((x_norm, y_norm, np.zeros_like(x_norm)))
    cloud = pv.PolyData(points)
    
    # 使用可调的ALPHA_VALUE
    grid = cloud.delaunay_2d(alpha=ALPHA_VALUE)
    
    return grid

# ===================== 模块1：有量纲化物理量计算 =====================
def compute_physical_quantities(Up_f, Vp_f, Wp_f, Pp_f, Tp_f,
                                 Ut_f, Vt_f, Wt_f, Pt_f, Tt_f,
                                 ma_f, pr_f):
    U0 = ma_f * np.sqrt(GAMMA * R_gas * T0)
    q0 = GAMMA * P0 * ma_f**2

    # 预测值
    Vel_p = np.sqrt(Up_f**2 + Vp_f**2 + Wp_f**2) * U0
    P_p = Pp_f * q0 + P0
    T_p = Tp_f * T0
    c_p = np.sqrt(GAMMA * R_gas * T_p)
    Ma_p = Vel_p / (c_p + 1e-12)
    TP_p = P_p * (1 + 0.5*(GAMMA-1)*Ma_p**2) ** (GAMMA/(GAMMA-1))

    # 真值
    Vel_t = np.sqrt(Ut_f**2 + Vt_f**2 + Wt_f**2) * U0
    P_t = Pt_f * q0 + P0
    T_t = Tt_f * T0
    c_t = np.sqrt(GAMMA * R_gas * T_t)
    Ma_t = Vel_t / (c_t + 1e-12)
    TP_t = P_t * (1 + 0.5*(GAMMA-1)*Ma_t**2) ** (GAMMA/(GAMMA-1))

    pred_dict = {
        "Velocity": Vel_p, "Static_P": P_p, "Mach": Ma_p, "Total_P": TP_p
    }
    true_dict = {
        "Velocity": Vel_t, "Static_P": P_t, "Mach": Ma_t, "Total_P": TP_t
    }
    return pred_dict, true_dict

# ===================== 模块2：切片云图对比绘图 =====================
def plot_slice_comparison(pred_dict, true_dict, x_f, y_f, wd_f,
                           x_sec_mean, ma_sec_mean, pr_sec_mean,
                           save_root, plot_var_list=PLOT_VAR_LIST):
    try:
        # 注意：现在传入的是 x_f 和 y_f
        base_grid = create_valid_mesh(x_f, y_f, wd_f)
    except Exception as e:
        logger.warning(f"网格生成失败: {e}")
        return

    unit_map = {
        "Velocity": "m/s", "Static_P": "Pa", "Mach": "-", "Total_P": "Pa"
    }

    # 优化版色条配置
    scalar_bar_config = {
        'title': '', 'font_family': 'times', 'color': 'black',
        'position_x': 0.2, 'position_y': 0, 'width': 0.6, 'height': 0.1,
        'n_labels': 4, 'label_font_size': 14, 'title_font_size': 16
    }

    for name in plot_var_list:
        if name not in pred_dict or name not in true_dict:
            continue
        
        val_p = pred_dict[name]
        val_t = true_dict[name]
        unit = unit_map[name]

        grid_p = base_grid.copy()
        grid_t = base_grid.copy()
        grid_p["Value"] = val_p
        grid_t["Value"] = val_t

        plotter = pv.Plotter(shape=(1, 2), off_screen=True, window_size=[2000, 800])
        clim = [np.min([val_p.min(), val_t.min()]), np.max([val_p.max(), val_t.max()])]

        # 左图：预测值
        plotter.subplot(0, 0)
        plotter.add_mesh(grid_p, scalars="Value", cmap=custom_discrete_cmap, clim=clim, 
                         show_edges=False, scalar_bar_args=scalar_bar_config)
        plotter.add_text(f"Prediction\nZ=0.001m\nMa={ma_sec_mean:.4f}\nPr={pr_sec_mean:.4f}", 
                         font_size=10, font='times', color='black')
        plotter.view_xy()
        plotter.enable_parallel_projection()
        
        # 右图：真值
        plotter.subplot(0, 1)
        plotter.add_mesh(grid_t, scalars="Value", cmap=custom_discrete_cmap, clim=clim, 
                         show_edges=False, scalar_bar_args=scalar_bar_config)
        plotter.add_text(f"Ground Truth\nZ=0.001m\nMa={ma_sec_mean:.4f}\nPr={pr_sec_mean:.4f}", 
                         font_size=10, font='times', color='black')
        plotter.view_xy()
        plotter.enable_parallel_projection()

        plotter.link_views()
        plotter.add_text(f"{name} ({unit})", position='lower_edge', font_size=12, font='times', color='black')
        
        filename = f"{save_root}/ZSlice_{name}_Compare.png"
        plotter.screenshot(filename)
        plotter.close()
    logger.info(f"✅ 云图已保存")

# ===================== 上面是整合后的 utils.py 功能 =====================

# ===================== 配置 =====================
BATCHSIZE = 1024
weight_name = "mapr_best_213weights"
weight_PATH = f'{project_root}/outputs/weights/mapr_best/{weight_name}.pth'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 输出路径
current_datetime = datetime.now().strftime("%m-%d_%H-%M")
save_root = f'{project_root}/outputs/推理结果/{current_datetime}_SingleZSlice_Fixed'
os.makedirs(save_root, exist_ok=True)
log_file_path = f'{save_root}/inference.log'
logger.add(log_file_path, rotation="500 MB", level="INFO")

# ===================== 主推理函数（单Z截面版）=====================
def main():
    set_seed(42)
    logger.info("="*60)
    logger.info("开始推理 - 单Z截面模式（已修复比例和凹结构）")
    logger.info("="*60)
    
    pv.set_plot_theme("document")
    os.environ["PYVISTA_OFF_SCREEN"] = "true"

    # 1. 加载数据集
    logger.info("加载数据集...")
    train_dataloader, val_dataloader, data_min_np, data_max_np = data_prepare(BATCHSIZE)
    data_min = torch.tensor(data_min_np, dtype=torch.float32).to(device)
    data_max = torch.tensor(data_max_np, dtype=torch.float32).to(device)
    
    # 反归一化参数
    in_min_4 = data_min[:4].cpu().numpy()
    in_max_4 = data_max[:4].cpu().numpy()
    in_range_4 = in_max_4 - in_min_4 + 1e-8
    
    in_min_Ma = data_min[4].cpu().numpy()
    in_max_Ma = data_max[4].cpu().numpy()
    in_range_Ma = in_max_Ma - in_min_Ma + 1e-8
    
    in_min_Pr = data_min[5].cpu().numpy() if len(data_min) > 5 else 1.0
    in_max_Pr = data_max[5].cpu().numpy() if len(data_max) > 5 else 1.0
    in_range_Pr = in_max_Pr - in_min_Pr + 1e-8

    # 2. 加载模型
    logger.info(f"加载模型权重: {weight_name}")
    model = PINN().to(device)
    if os.path.isfile(weight_PATH):
        checkpoint = torch.load(weight_PATH, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'], strict=True)
        logger.info(f"✅ 权重加载成功 (Epoch {checkpoint['epoch']})")
    else:
        logger.error(f"❌ 权重文件未找到: {weight_PATH}")
        return
    model.eval()

    # 3. 推理并收集数据
    logger.info("正在进行推理...")
    buffer = {
        'X': [], 'Y': [], 'Z': [], 'WD': [],
        'Ma': [], 'Pr': [],
        'Up': [], 'Vp': [], 'Wp': [], 'Pp': [], 'Tp': [], 'Kp': [], 'Op': [],
        'Ut': [], 'Vt': [], 'Wt': [], 'Pt': [], 'Tt': [], 'Kt': [], 'Ot': []
    }

    with torch.no_grad():
        for input_batch, target_batch in val_dataloader:
            input_batch = input_batch.to(device)
            target_batch = target_batch.to(device)

            output_norm = model(input_batch)
            pred_phys = denormalize_for_pde(device, output_norm, data_min, data_max)
            true_phys = denormalize_for_pde(device, target_batch, data_min, data_max)

            in_np = input_batch.cpu().numpy()
            p_np = pred_phys.cpu().numpy()
            t_np = true_phys.cpu().numpy()

            in_denorm_4 = in_np[:, :4] * in_range_4 + in_min_4
            Ma_denorm = in_np[:, 4] * in_range_Ma + in_min_Ma
            Pr_denorm = in_np[:, 5] * in_range_Pr + in_min_Pr if in_np.shape[1] > 5 else np.ones_like(Ma_denorm)

            buffer['X'].extend(in_denorm_4[:, 0])
            buffer['Y'].extend(in_denorm_4[:, 1])
            buffer['Z'].extend(in_denorm_4[:, 2])
            buffer['WD'].extend(in_denorm_4[:, 3])
            buffer['Ma'].extend(Ma_denorm)
            buffer['Pr'].extend(Pr_denorm)

            buffer['Up'].extend(p_np[:,0]); buffer['Vp'].extend(p_np[:,1]); buffer['Wp'].extend(p_np[:,2])
            buffer['Pp'].extend(p_np[:,3]); buffer['Tp'].extend(p_np[:,4]); buffer['Kp'].extend(p_np[:,5]); buffer['Op'].extend(p_np[:,6])
            buffer['Ut'].extend(t_np[:,0]); buffer['Vt'].extend(t_np[:,1]); buffer['Wt'].extend(t_np[:,2])
            buffer['Pt'].extend(t_np[:,3]); buffer['Tt'].extend(t_np[:,4]); buffer['Kt'].extend(t_np[:,5]); buffer['Ot'].extend(t_np[:,6])

    # 转为Numpy数组
    for k in buffer: buffer[k] = np.array(buffer[k])
    X_dim = buffer['X'] * L

    # ==========================================
    # 核心：直接处理整个数据集（单Z截面）
    # ==========================================
    logger.info("处理Z截面数据...")
    
    x_loc = X_dim
    y_loc = buffer['Y']
    z_loc = buffer['Z']
    wd_loc = buffer['WD']
    ma_slice = buffer['Ma']
    pr_slice = buffer['Pr']

    x_sec_mean = np.mean(x_loc)
    ma_sec_mean = np.mean(ma_slice)
    pr_sec_mean = np.mean(pr_slice)
    z_sec_mean = np.mean(z_loc)

    logger.info(f"✅ 截面平均Z坐标: {z_sec_mean:.6f} m")
    logger.info(f"✅ 截面总点数: {len(x_loc)}")

    # 流体域过滤
    fluid_mask = (wd_loc <= WALL_DIST_MAX + 0.2) & (wd_loc >= -0.1) & (~np.isnan(wd_loc))
    
    # 提取有效数据
    xx = x_loc[fluid_mask]
    yy = y_loc[fluid_mask]
    ww = wd_loc[fluid_mask]

    # 物理量
    Up_f = buffer['Up'][fluid_mask]
    Vp_f = buffer['Vp'][fluid_mask]
    Wp_f = buffer['Wp'][fluid_mask]
    Pp_f = buffer['Pp'][fluid_mask]
    Tp_f = buffer['Tp'][fluid_mask]

    Ut_f = buffer['Ut'][fluid_mask]
    Vt_f = buffer['Vt'][fluid_mask]
    Wt_f = buffer['Wt'][fluid_mask]
    Pt_f = buffer['Pt'][fluid_mask]
    Tt_f = buffer['Tt'][fluid_mask]

    ma_f = ma_slice[fluid_mask]
    pr_f = pr_slice[fluid_mask]

    if len(yy) < 50:
        logger.error("❌ 有效点数不足！")
        return

    # 4. 有量纲化
    logger.info("计算有量纲化物理量...")
    pred_dict, true_dict = compute_physical_quantities(
        Up_f, Vp_f, Wp_f, Pp_f, Tp_f,
        Ut_f, Vt_f, Wt_f, Pt_f, Tt_f,
        ma_f, pr_f
    )

    # 5. 保存CSV
    logger.info("保存CSV...")
    Pressure_pred = pred_dict["Static_P"]
    Total_Pressure_pred = pred_dict["Total_P"]
    Mach_pred = pred_dict["Mach"]
    data_pred = np.column_stack([xx, yy, z_loc[fluid_mask], Pressure_pred, Total_Pressure_pred, Mach_pred])
    np.savetxt(f"{save_root}/ZSlice_Pred.csv", data_pred, delimiter=" ", header="X Y Z Static_P Total_P Mach", comments="")
    
    Pressure_true = true_dict["Static_P"]
    Total_Pressure_true = true_dict["Total_P"]
    Mach_true = true_dict["Mach"]
    data_true = np.column_stack([xx, yy, z_loc[fluid_mask], Pressure_true, Total_Pressure_true, Mach_true])
    np.savetxt(f"{save_root}/ZSlice_True.csv", data_true, delimiter=" ", header="X Y Z Static_P Total_P Mach", comments="")
    logger.info(f"✅ CSV已保存")

    # 6. 绘图
    logger.info("绘制云图...")
    logger.info(f"✅ 当前 ALPHA_VALUE = {ALPHA_VALUE:.6f} (如果凹结构还有连线，请调小这个值)")
    
    plot_slice_comparison(
        pred_dict, true_dict,
        xx, yy, ww,  # 传入真实的 X 和 Y
        x_sec_mean, ma_sec_mean, pr_sec_mean,
        save_root
    )

    logger.info("="*60)
    logger.info(f"✅ 所有任务完成！保存在: {save_root}")
    logger.info("="*60)

if __name__ == "__main__":
    main()