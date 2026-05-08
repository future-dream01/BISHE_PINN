import os
import numpy as np
import pyvista as pv
import matplotlib.tri as tri
from loguru import logger
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

# ✅ 【关键参数】在这里调节色块阶数！
# 数值越大，色块越多，过渡越细腻（但依然是分段色块，不是连续渐变）
# 建议范围：10 ~ 100
N_DISCRETE_COLORS = 50  # 你可以改成 20, 50, 100 试试

# 基础锚定颜色（和你提供的色条1:1对应，不需要改）
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

# ✅ 自动生成高阶数离散色图（无光滑过渡）
# 1. 先在锚定颜色之间做平滑插值
temp_smooth_cmap = LinearSegmentedColormap.from_list("temp_smooth", base_color_anchors, N=256)
# 2. 提取指定数量的离散颜色
discrete_colors = temp_smooth_cmap(np.linspace(0, 1, N_DISCRETE_COLORS))
# 3. 生成最终的纯离散色图（彻底消除光滑过渡）
custom_discrete_cmap = ListedColormap(discrete_colors, name=f"discrete_cfd_{N_DISCRETE_COLORS}")

# ===================== 工具函数 =====================
def set_seed(seed):
    import torch
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# ===================== 网格生成（已去掉所有过滤）=====================
# ===================== 网格生成（修复凹结构版）=====================
def create_valid_mesh(y, z, wall_dist):
    """
    修复版：解决凹结构出现多余直线的问题
    使用 PyVista 的 Delaunay2D + Alpha 过滤
    """
    # 1. 构建点云
    points = np.column_stack((y, z, np.zeros_like(y)))
    cloud = pv.PolyData(points)

    # 2. 生成 Delaunay 三角网格
    # 关键参数 alpha：控制“挖空”程度，越小越容易挖掉外部
    # 你可以根据你的流道大小调整这个值，建议范围 0.001 ~ 0.01 (无量纲)
    alpha_value = 0.03 / L  # 自动适配你的特征长度
    
    grid = cloud.delaunay_2d(alpha=alpha_value)
    
    # 3. 把物理量附加上去（虽然这里只用了坐标，但保留接口）
    return grid
# ===================== 模块1：有量纲化物理量计算 =====================
def compute_physical_quantities(Up_f, Vp_f, Wp_f, Pp_f, Tp_f,
                                 Ut_f, Vt_f, Wt_f, Pt_f, Tt_f,
                                 ma_f, pr_f):
    """
    核心有量纲化计算模块
    输入：无量纲物理量 + 工况参数 (Ma, Pr)
    输出：两个字典，分别包含预测值和真值的所有有量纲化物理量
    """
    # 1. 确定 P0 (固定为参考静压，如需随 Pr 变化请修改此处)
    #P0 = P0 * np.ones_like(ma_f)
    
    # 2. 逐点计算参考速度 U0 和动压 q0
    U0 = ma_f * np.sqrt(GAMMA * R_gas * T0)
    q0 = GAMMA * P0 * ma_f**2  # 移除了 0.5

    
    # 3. 有量纲化预测值
    Vel_p = np.sqrt(Up_f**2 + Vp_f**2 + Wp_f**2) * U0
    P_p = Pp_f * q0 + P0
    T_p = Tp_f * T0
    c_p = np.sqrt(GAMMA * R_gas * T_p)
    Ma_p = Vel_p / (c_p + 1e-12)
    TP_p = P_p * (1 + 0.5*(GAMMA-1)*Ma_p**2) ** (GAMMA/(GAMMA-1))
    Roup_f=(1+GAMMA*(ma_f**2)*Pp_f)/Tp_f      # 无量纲化的密度
    Rou_p=Roup_f*(P0/(287*T0))    # 有量纲的密度


    # 4. 有量纲化真值 (使用完全相同的参考值)
    Vel_t = np.sqrt(Ut_f**2 + Vt_f**2 + Wt_f**2) * U0
    P_t = Pt_f * q0 + P0
    T_t = Tt_f * T0
    c_t = np.sqrt(GAMMA * R_gas * T_t)
    Ma_t = Vel_t / (c_t + 1e-12)
    TP_t = P_t * (1 + 0.5*(GAMMA-1)*Ma_t**2) ** (GAMMA/(GAMMA-1))
    Roup_t=(1+GAMMA*(ma_f**2)*Pt_f)/Tt_f      # 无量纲化的密度
    Rou_p=Roup_t*(P0/(287*T0))              # 有量纲的密度

    # 5. 打包成字典
    pred_dict = {
        "Velocity": Vel_p,
        "Static_P": P_p,
        "Mach":     Ma_p,
        "Total_P":  TP_p
    }
    
    true_dict = {
        "Velocity": Vel_t,
        "Static_P": P_t,
        "Mach":     Ma_t,
        "Total_P":  TP_t
    }

    return pred_dict, true_dict

# ===================== 模块2：切片云图对比绘图（完整功能版）=====================
def plot_slice_comparison(pred_dict, true_dict, y_f, z_f, wd_f,
                           x_sec_mean, ma_sec_mean, pr_sec_mean,
                           save_root, plot_var_list=PLOT_VAR_LIST):
    """
    核心绘图模块（兼容旧版 PyVista + 可调节阶数离散色条）
    输入：预测值字典、真值字典、坐标信息、切片工况、保存路径
    输出：保存云图对比图
    """
    # 生成网格（已无过滤）
    try:
        base_grid = create_valid_mesh(y_f, z_f, wd_f)
    except Exception as e:
        logger.warning(f"切片 X={x_sec_mean:.4f}, Ma={ma_sec_mean:.4f} 网格生成失败: {e}")
        return

    # 单位映射
    unit_map = {
        "Velocity": "m/s",
        "Static_P": "Pa",
        "Mach":     "-",
        "Total_P":  "Pa"
    }

        # ✅ 优化版色条配置：彻底解决数字重叠
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
        
        val_p = pred_dict[name]
        val_t = true_dict[name]
        unit = unit_map[name]

        # 赋值
        grid_p = base_grid.copy()
        grid_t = base_grid.copy()
        grid_p["Value"] = val_p
        grid_t["Value"] = val_t

        # 创建 Plotter
        plotter = pv.Plotter(shape=(1, 2), off_screen=True, window_size=[2000, 800])
        clim = [np.min([val_p.min(), val_t.min()]), np.max([val_p.max(), val_t.max()])]

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
        plotter.add_text(f"{name} ({unit})", 
                         position='lower_edge', 
                         font_size=12, 
                         font='times', 
                         color='black')
        
        # 保存
        filename = f"{save_root}/X_{x_sec_mean:.4f}m_Ma_{ma_sec_mean:.4f}_Pr_{pr_sec_mean:.4f}_{name}_Compare.png"
        plotter.screenshot(filename)
        plotter.close()