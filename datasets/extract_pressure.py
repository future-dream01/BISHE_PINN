import pyvista as pv
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt

# ==========================================
# 【参数设置区域】
# ==========================================
# --- 通用文件与物理参数 ---
cas_path = "066D_A6_hermites11_banmo_062_106.cas"
zone_id = 1  # 【关键】先运行一次，看打印的Zone列表，改成正确的流体域ID
L = 0.095       # 特征长度
M0 = 0.62       # 来流马赫数
T0 = 249.15     # 来流静温
P0 = 47181      # 来流静压（Pa）
P_ref = 61143   # ✅ 新增：压力无量纲化参考压力（Pa）

# --- 进气道流道范围（强制过滤用）---
x_inlet = 0.297183   # 入口X坐标
x_outlet = 0.515712  # 出口X坐标

# --- ✅ Z截面坐标设置 ---
z_section = 0.0002  # 【核心】修改这里可以提取任意Z截面的壁面压力

# --- ✅ 轴向采样间隔（最关键参数）---
x_step = 0.001  # 轴向X方向每隔多少米取一个点
# 例如：0.001m = 1mm间隔，整个进气道约220个点；0.0005m = 0.5mm间隔，约440个点

# --- 壁面压力提取参数 ---
plot_sampling_points = True  # 是否绘制取样点验证图
plot_pressure_curve = True   # 是否绘制沿程压力变化曲线

# ==========================================
# 【自动生成带Z坐标的输出文件名】
# ==========================================
z_str = f"{z_section:.4f}".replace('.', '_').replace('-', 'n')
output_upper_wall = f"upper_wall_pressure_nondim_z={z_str}.csv"   # 上壁面输出文件（无量纲）
output_lower_wall = f"lower_wall_pressure_nondim_z={z_str}.csv"   # 下壁面输出文件（无量纲）

# ==========================================
# 【核心函数区域】
# ==========================================

# 打印所有Zone信息，帮你找到正确的流体域
def print_mesh_info(mesh):
    print("\n" + "="*70)
    print("Cas文件包含的所有Zone信息（请找到你的进气道流体域ID）：")
    print("="*70)
    for i, block in enumerate(mesh):
        try:
            name = block.get('Name', f'Zone_{i}')
        except:
            name = f'Zone_{i}'
        
        n_cells = block.n_cells
        n_points = block.n_points
        bounds = block.bounds
        
        print(f"✅ Zone ID: {i}")
        print(f"  名称: {name}")
        print(f"  网格数: {n_cells} | 节点数: {n_points}")
        print(f"  X范围: {bounds[0]:.6f} ~ {bounds[1]:.6f} m")
        print(f"  Y范围: {bounds[2]:.6f} ~ {bounds[3]:.6f} m")
        print(f"  Z范围: {bounds[4]:.6f} ~ {bounds[5]:.6f} m")
        print("-" * 50)

# ✅ 已修改：压力自动除以P_ref=61143进行无量纲化
def extract_wall_pressure(target_block, x_inlet, x_outlet, z_section, x_step, P_ref):
    print(f"\n正在提取 Z={z_section:.4f}m 截面...")
    
    # 第一步：先提取整个Z=常数的对称面
    symmetry_plane = target_block.slice(normal=[0, 0, 1], origin=[0, 0, z_section])
    
    if symmetry_plane.n_points == 0:
        print(f"❌ 错误：Z={z_section:.4f}m 截面无数据！")
        print(f"   请检查Z坐标是否在流道范围内：{target_block.bounds[4]:.6f} ~ {target_block.bounds[5]:.6f} m")
        return None, None, None, None
    
    # 确保所有数据都在节点上
    symmetry_plane = symmetry_plane.point_data_to_cell_data(True).cell_data_to_point_data()
    
    # 第二步：生成等间隔的轴向X采样点
    x_samples = np.arange(x_inlet, x_outlet + x_step/2, x_step)
    print(f"✅ 轴向采样间隔：{x_step}m")
    print(f"✅ 共生成 {len(x_samples)} 个轴向采样位置")
    print(f"✅ 压力无量纲化：P_nondim = P / {P_ref} Pa")
    
    upper_wall_data = []
    lower_wall_data = []
    
    # 第三步：遍历每个X位置，切竖线并取上下极值点
    print("\n🔍 正在逐个提取轴向截面的上下壁面点...")
    for i, x in enumerate(x_samples):
        # 在Z=0对称面上，切X=常数的竖线（垂直于X轴的平面）
        x_slice = symmetry_plane.slice(normal=[1, 0, 0], origin=[x, 0, 0])
        
        if x_slice.n_points == 0:
            print(f"   跳过 X={x:.6f}m：无数据")
            continue
        
        # 提取这条竖线上的所有点的Y坐标和压力（自动无量纲化）
        y_coords = x_slice.points[:, 1]
        pressure = x_slice["PRESSURE"] / P_ref  # ✅ 核心修改：压力除以参考压力
        
        # 过滤无效点
        valid_mask = pressure > 0
        y_valid = y_coords[valid_mask]
        p_valid = pressure[valid_mask]
        
        if len(y_valid) == 0:
            print(f"   跳过 X={x:.6f}m：无有效数据")
            continue
        
        # ===================== 核心逻辑 =====================
        # 上壁面：这条竖线上Y最大的点
        upper_idx = np.argmax(y_valid)
        upper_y = y_valid[upper_idx]
        upper_p = p_valid[upper_idx]
        
        # 下壁面：这条竖线上Y最小的点
        lower_idx = np.argmin(y_valid)
        lower_y = y_valid[lower_idx]
        lower_p = p_valid[lower_idx]
        # ===================================================
        
        upper_wall_data.append({
            "X(m)": x,
            "Y(m)": upper_y,
            "Pressure(nondim)": upper_p
        })
        
        lower_wall_data.append({
            "X(m)": x,
            "Y(m)": lower_y,
            "Pressure(nondim)": lower_p
        })
        
        # 打印进度
        if (i + 1) % 20 == 0:
            print(f"   已处理 {i+1}/{len(x_samples)} 个位置")
    
    # 转换为DataFrame并按X严格排序
    upper_df = pd.DataFrame(upper_wall_data).sort_values("X(m)").reset_index(drop=True)
    lower_df = pd.DataFrame(lower_wall_data).sort_values("X(m)").reset_index(drop=True)
    
    print(f"\n✅ 上壁面提取到 {len(upper_df)} 个点")
    print(f"✅ 下壁面提取到 {len(lower_df)} 个点")
    
    # 强制验证：确保上下壁面完全分离
    upper_y_min = upper_df["Y(m)"].min()
    lower_y_max = lower_df["Y(m)"].max()
    y_gap = upper_y_min - lower_y_max
    
    print(f"\n✅ 上下壁面强制验证：")
    print(f"   上壁面Y范围：{upper_df['Y(m)'].min():.6f} ~ {upper_df['Y(m)'].max():.6f} m")
    print(f"   下壁面Y范围：{lower_df['Y(m)'].min():.6f} ~ {lower_df['Y(m)'].max():.6f} m")
    print(f"   上下壁面最小间隙：{y_gap:.6f} m")
    
    if y_gap < 0:
        print("❌ 严重错误：上下壁面Y值重叠！请检查cas文件和Zone ID")
    else:
        print("✅ 验证通过：上下壁面100%分离，无任何混淆")
    
    # 返回所有数据用于绘图验证
    return upper_df, lower_df, symmetry_plane.points[:, 0], symmetry_plane.points[:, 1]

# ✅ 取样验证图（无变化）
def plot_wall_sampling(x_valid, y_valid, upper_df, lower_df, x_inlet, x_outlet, z_section):
    print("\n正在绘制取样点验证图...")
    
    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['font.size'] = 12
    
    fig, ax = plt.subplots(figsize=(12, 6), dpi=150)
    
    # 绘制Z=0截面所有点（浅灰色背景）
    ax.scatter(x_valid, y_valid, c='#f0f0f0', s=1, alpha=0.3, label='Z=0截面所有点')
    
    # 绘制上壁面最终点（红色）
    ax.scatter(upper_df["X(m)"], upper_df["Y(m)"], c='#d62728', s=20, alpha=1.0, label='上壁面采样点')
    
    # 绘制下壁面最终点（蓝色）
    ax.scatter(lower_df["X(m)"], lower_df["Y(m)"], c='#1f77b4', s=20, alpha=1.0, label='下壁面采样点')
    
    # 设置图形属性
    ax.set_xlabel('X (m)', fontsize=14)
    ax.set_ylabel('Y (m)', fontsize=14)
    ax.set_title(f'Z={z_section:.4f}m 截面壁面采样点分布', fontsize=16, fontweight='bold', pad=20)
    ax.set_xlim(x_inlet - 0.01, x_outlet + 0.01)
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.legend(fontsize=12, loc='upper right')
    ax.set_aspect('equal', adjustable='box')
    
    # 保存图片（带Z坐标）
    z_str = f"{z_section:.4f}".replace('.', '_').replace('-', 'n')
    plt.savefig(f"wall_sampling_points_z={z_str}.png", bbox_inches='tight', dpi=150)
    plt.show()
    
    print(f"✅ 取样点验证图已保存为：wall_sampling_points_z={z_str}.png")

# ✅ 已修改：压力曲线使用无量纲坐标
def plot_pressure_distribution(upper_df, lower_df, x_inlet, x_outlet, P0, P_ref, z_section):
    print("\n正在绘制沿程压力变化图...")
    
    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['font.size'] = 12
    
    fig, ax = plt.subplots(figsize=(12, 6), dpi=150)
    
    # 绘制上壁面压力曲线（红色实线+圆点）
    ax.plot(upper_df["X(m)"], upper_df["Pressure(nondim)"], color='#d62728', linewidth=2, linestyle='-', marker='o', markersize=3, label='上壁面')
    
    # 绘制下壁面压力曲线（蓝色实线+圆点）
    ax.plot(lower_df["X(m)"], lower_df["Pressure(nondim)"], color='#1f77b4', linewidth=2, linestyle='-', marker='o', markersize=3, label='下壁面')
    
    # 绘制来流静压参考线（黑色虚线，无量纲）
    P0_nondim = P0 / P_ref
    ax.axhline(y=P0_nondim, color='black', linestyle='--', linewidth=1.5, label=f'来流静压 P0/P_ref={P0_nondim:.4f}')
    
    # 设置图形属性
    ax.set_xlabel('X (m)', fontsize=14)
    ax.set_ylabel('Static Pressure (P / 61143)', fontsize=14)  # ✅ 修改纵坐标标签
    ax.set_title(f'Z={z_section:.4f}m 截面上下壁面沿程静压分布（无量纲）', fontsize=16, fontweight='bold', pad=20)
    ax.set_xlim(x_inlet, x_outlet)
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.legend(fontsize=12)
    
    # 自动调整Y轴范围
    all_pressures = np.concatenate([upper_df["Pressure(nondim)"], lower_df["Pressure(nondim)"]])
    y_min = np.min(all_pressures) * 0.95
    y_max = np.max(all_pressures) * 1.05
    ax.set_ylim(y_min, y_max)
    
    # 保存图片（带Z坐标）
    z_str = f"{z_section:.4f}".replace('.', '_').replace('-', 'n')
    plt.savefig(f"wall_pressure_distribution_nondim_z={z_str}.png", bbox_inches='tight', dpi=150)
    plt.show()
    
    print(f"✅ 沿程压力变化图已保存为：wall_pressure_distribution_nondim_z={z_str}.png")

# ✅ 已修改：保存无量纲压力CSV
def save_wall_csv(df, file_path, wall_name):
    if df.empty:
        print(f"⚠️  {wall_name}无数据，跳过保存")
        return
    
    # 只保留X和无量纲压力列
    df_to_save = df[["X(m)", "Pressure(nondim)"]].copy()
    
    df_to_save.to_csv(file_path, index=False, float_format='%.10f')
    print(f"✅ {wall_name}无量纲压力数据已保存至：{file_path}")
    print(f"   无量纲压力范围：{df_to_save['Pressure(nondim)'].min():.6f} ~ {df_to_save['Pressure(nondim)'].max():.6f}")

# ==========================================
# 【主程序】
# ==========================================
def main():
    # 1. 读取网格
    print("\n正在读取 Fluent 文件...")
    mesh = pv.read(cas_path)
    
    # 打印所有Zone信息，帮你找到正确的ID
    print_mesh_info(mesh)
    
    # 2. 选取指定的Zone
    print(f"\n正在选取 Zone ID = {zone_id} ...")
    target_block = mesh[zone_id]
    
    if target_block is None or target_block.n_points == 0:
        print(f"❌ 错误：Zone ID {zone_id} 无有效网格，请检查上面的Zone列表！")
        return
    
    print(f"✅ 已选中 Zone ID {zone_id}")
    print(f"   总节点数: {target_block.n_points}")
    print(f"   原始X范围: {target_block.bounds[0]:.6f} ~ {target_block.bounds[1]:.6f} m")
    print(f"   原始Z范围: {target_block.bounds[4]:.6f} ~ {target_block.bounds[5]:.6f} m")

    # 3. 按轴向间隔提取上下壁面无量纲压力
    upper_df, lower_df, x_valid, y_valid = extract_wall_pressure(
        target_block, x_inlet, x_outlet, z_section, x_step, P_ref
    )
    
    if upper_df is None or lower_df is None:
        return

    # 4. 保存CSV文件（无量纲压力）
    save_wall_csv(upper_df, output_upper_wall, "上壁面")
    save_wall_csv(lower_df, output_lower_wall, "下壁面")

    # 5. 绘制取样点验证图
    if plot_sampling_points:
        plot_wall_sampling(
            x_valid, y_valid, upper_df, lower_df, x_inlet, x_outlet, z_section
        )

    # 6. 绘制沿程压力变化图（无量纲）
    if plot_pressure_curve and not upper_df.empty and not lower_df.empty:
        plot_pressure_distribution(
            upper_df, lower_df, x_inlet, x_outlet, P0, P_ref, z_section
        )

    print("\n" + "="*70)
    print(f"所有任务完成！当前处理截面：Z={z_section:.4f}m")
    print(f"✅ 轴向采样间隔：{x_step}m")
    print(f"✅ 压力无量纲化：P / {P_ref} Pa")
    print(f"✅ 壁面区分方法：每个X截面取Y最大/最小值")
    print(f"✅ 上壁面无量纲压力：{output_upper_wall}")
    print(f"✅ 下壁面无量纲压力：{output_lower_wall}")
    if plot_sampling_points:
        z_str = f"{z_section:.4f}".replace('.', '_').replace('-', 'n')
        print(f"✅ 取样点验证图：wall_sampling_points_z={z_str}.png")
    if plot_pressure_curve:
        print(f"✅ 沿程压力变化图：wall_pressure_distribution_nondim_z={z_str}.png")
    print("="*70)

if __name__ == "__main__":
    main()