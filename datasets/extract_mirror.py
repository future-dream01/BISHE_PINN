import pyvista as pv
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt

# ==========================================
# 【参数设置区域】
# ==========================================
# --- 通用文件与物理参数 ---
a=90
b=113

M0 = a/100 # 来流马赫数（直接用这个）
Pr = b/100  # 压比（直接用这个）

cas_path = f"05D_A0_hermites08_banmo_0{a}_{b}.cas"
zone_id = 2  # 【关键】先运行一次，看打印的Zone列表，改成正确的流体域ID
L = 0.095       # 特征长度
T0 = 249.15     # 来流静温
P0 = 47181      # 来流静压
U0 = M0 * (1.4 * 287 * T0) ** 0.5
Rou0 = P0 / (287 * T0)
q0 = 1.4 * P0 * M0**2  # 来流动压，标准无量纲化

# --- 进气道流道范围（强制过滤用）---
x_inlet = 0.297183   # 入口X坐标（依然保留，过滤Z截面里的无效X范围）
x_outlet = 0.515712  # 出口X坐标

# --- ✅ 新增：单Z截面参数 ---
z_target = 0.001     # 目标Z截面位置（你要的 z=0.001）
output_single_csv = f"05D_A0_0{a}_{b}_Z{z_target:.6f}.csv"  # 输出文件名包含Z坐标
samples_single_section = 30000  # 这个截面的总采样点数（你可以自由控制）
near_wall_ratio = 0.7    # 近壁面采样比例（和原来一样）
near_wall_thresh = 0.005 # 近壁面阈值（有量纲，单位m）

# ✅ 自动创建图片保存文件夹（不存在则创建）
SAVE_PIC_DIR = "pictures"
os.makedirs(SAVE_PIC_DIR, exist_ok=True)

# ==========================================
# 【核心函数区域】
# ==========================================

# 【核心修复1】新增：统一过滤逻辑函数
# 保证算极值和采数据用的是同一套标准
def filter_valid_points(points, point_data, x_inlet, x_outlet, T0, P0):
    """
    严格过滤有效点，和 get_global_stats 里的逻辑完全一致
    """
    # 1. X坐标范围过滤（加1e-8容忍浮点误差）
    x_coords = points[:, 0]
    in_x_range = (x_coords >= x_inlet - 1e-8) & (x_coords <= x_outlet + 1e-8)
    
    # 2. 物理量合理性过滤
    temp_valid = point_data["TEMPERATURE"] > 0.8 * T0  # 温度不低于来流的80%
    press_valid = point_data["PRESSURE"] > 0            # 压力不能为负
    tke_valid = point_data["TKE"] >= 0                  # 湍动能不能为负
    wall_dist_valid = point_data["WALL_DIST"] >= 0      # 壁面距离不能为负
    
    # 合并所有过滤条件
    valid_mask = in_x_range & temp_valid & press_valid & tke_valid & wall_dist_valid
    
    return valid_mask

# 统一无量纲化函数，直接使用外部 M0 和 Pr
def nondimensionalize_data(points, point_data, L, U0, T0, P0, q0):
    df = pd.DataFrame({
        "X": (points[:, 0]) / L,
        "Y": (points[:, 1]) / L,
        "Z": (points[:, 2]) / L,
        "壁面距离": (point_data["WALL_DIST"]) / L,
        # 直接使用外部定义的 M0 和 Pr
        "Ma": M0,
        "Pr": Pr,
        "U": (point_data["X_VELOCITY"]) / U0,
        "V": (point_data["Y_VELOCITY"]) / U0,
        "W": (point_data["Z_VELOCITY"]) / U0,
        "静压": ((point_data["PRESSURE"]) - P0) / q0,
        "静温": (point_data["TEMPERATURE"]) / T0,
        "湍流动能": (point_data["TKE"]) / (U0 ** 2),
        "比耗散率": (point_data["SDR"]) * (L / U0),
    })
    return df

# 提取全局极值（依然保留，因为CSV需要表头的MIN/MAX行）
def get_global_stats(target_block, L, U0, T0, P0, q0, x_inlet, x_outlet):
    print(f"\n正在提取【进气道内部有效点】的【无量纲化后】全局极值...")
    
    # 确保数据在节点上
    data = target_block.point_data_to_cell_data(True).cell_data_to_point_data()
    
    # ==========================================
    # 使用统一过滤逻辑
    # ==========================================
    valid_mask = filter_valid_points(
        data.points, data.point_data, 
        x_inlet, x_outlet, T0, P0
    )
    
    # 提取有效点
    valid_points = data.points[valid_mask]
    valid_point_data = {}
    for key in data.point_data.keys():
        valid_point_data[key] = data.point_data[key][valid_mask]
    
    print(f"   总网格点数：{len(data.points)} | 进气道内部有效点数：{len(valid_points)}")
    if len(valid_points) == 0:
        print("❌ 错误：没有有效点，请检查Zone ID和x_inlet/x_outlet范围！")
        exit()
    
    # 先对有效点做无量纲化
    full_nondim_df = nondimensionalize_data(
        points=valid_points,
        point_data=valid_point_data,
        L=L, U0=U0, T0=T0, P0=P0, q0=q0
    )
    
    # 从无量纲化后的有效点里提取MIN/MAX
    global_min = full_nondim_df.min().to_dict()
    global_max = full_nondim_df.max().to_dict()
    
    # 打印原始有量纲极值，方便你核对
    print("\n✅ 原始有量纲极值（有效点）：")
    print(f"   X范围：{valid_points[:,0].min():.6f} ~ {valid_points[:,0].max():.6f} m")
    print(f"   静温范围：{valid_point_data['TEMPERATURE'].min():.2f} ~ {valid_point_data['TEMPERATURE'].max():.2f} K")
    print(f"   壁面距离范围：{valid_point_data['WALL_DIST'].min():.6f} ~ {valid_point_data['WALL_DIST'].max():.6f} m")
    
    print("\n✅ 无量纲化后的全局极值（有效点，含Ma/Pr）：")
    for col in full_nondim_df.columns:
        print(f"   {col:8s} | 无量纲MIN: {global_min[col]:.6f} | 无量纲MAX: {global_max[col]:.6f}")
    
    return global_min, global_max, full_nondim_df.columns.tolist()

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

# CSV保存函数（和原来完全一样，保证格式兼容）
def save_csv_with_header(df, file_path, global_min, global_max, columns):
    with open(file_path, 'w', encoding='utf-8-sig', newline='') as f:
        f.write(','.join(columns) + '\n')
        min_vals = [f"{global_min[col]:.10f}" for col in columns]
        f.write(','.join(min_vals) + '\n')
        max_vals = [f"{global_max[col]:.10f}" for col in columns]
        f.write(','.join(max_vals) + '\n')
        df.to_csv(f, header=False, index=False, float_format='%.10f')
    
    print(f"✅ 文件已保存：{file_path}")
    print(f"   格式：第1行=表头 | 第2行=无量纲全局MIN | 第3行=无量纲全局MAX | 第4行起=无量纲采样数据")

# 【核心修复2】修改后的分层采样函数（和原来逻辑完全一致）
def stratified_sampling(slice_data, near_thresh, total_samples, near_ratio, L, U0, T0, P0, q0, x_inlet, x_outlet):
    # 先做统一过滤，只保留有效点
    valid_mask = filter_valid_points(
        slice_data.points, 
        slice_data.point_data, 
        x_inlet, x_outlet, T0, P0
    )
    
    # 提取过滤后的有效点
    valid_points = slice_data.points[valid_mask]
    valid_point_data = {}
    for key in slice_data.point_data.keys():
        valid_point_data[key] = slice_data.point_data[key][valid_mask]
    
    if len(valid_points) == 0:
        return pd.DataFrame()
    
    # 只对有效点做无量纲化
    df = nondimensionalize_data(
        points=valid_points,
        point_data=valid_point_data,
        L=L, U0=U0, T0=T0, P0=P0, q0=q0
    )
    
    # 后面的采样逻辑保持不变
    near_thresh_nondim = near_thresh / L
    near_wall = df[df["壁面距离"] < near_thresh_nondim].copy()
    far_wall = df[df["壁面距离"] >= near_thresh_nondim].copy()

    near_num = int(total_samples * near_ratio)
    far_num = total_samples - near_num

    near_sampled = near_wall.sample(n=min(near_num, len(near_wall)), random_state=42) if len(near_wall) > 0 else pd.DataFrame()
    far_sampled = far_wall.sample(n=min(far_num, len(far_wall)), random_state=42) if len(far_wall) > 0 else pd.DataFrame()
    
    sampled_df = pd.concat([near_sampled, far_sampled], ignore_index=True)
    return sampled_df

# 可视化函数（空白无坐标轴 + 自动保存，适配Z截面）
def visualize_sampling_points(slice_points, sampled_points, z_pos, save_path):
    """
    可视化采样点（空白无坐标轴背景）+ 自动保存图片
    注意：Z截面画的是 X-Y 平面
    """
    # 创建画布
    plt.figure(figsize=(9, 8), facecolor='white')
    
    # 绘制点：X轴和Y轴
    plt.scatter(slice_points[:, 0]/L, slice_points[:, 1]/L, 
                c="lightgray", s=4, alpha=0.3)
    plt.scatter(sampled_points["X"], sampled_points["Y"], 
                c="crimson", s=6, alpha=0.9)

    # 空白无坐标轴
    plt.axis('off')
    plt.axis("equal")
    plt.tight_layout(pad=0)

    # 保存图片
    plt.savefig(save_path, 
                dpi=150,        
                bbox_inches='tight',
                facecolor='white')
    
    plt.close()

# ==========================================
# 【主程序】（核心修改：只处理单个Z截面）
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

    # 3. 提取进气道内部有效点的无量纲全局极值（依然保留，CSV需要）
    global_min, global_max, columns = get_global_stats(
        target_block, L, U0, T0, P0, q0, x_inlet, x_outlet
    )

    # 4. ✅ 核心修改：切单个Z截面
    print(f"\n" + "="*70)
    print(f"开始处理【单个Z截面】 Z = {z_target:.6f}")
    print("="*70)
    
    # 切Z平面：normal=[0,0,1]，origin=[0, 0, z_target]
    slice_plane = target_block.slice(normal=[0,0,1], origin=[0, 0, z_target])
    if slice_plane.n_points == 0:
        print(f"❌ 错误：Z={z_target:.6f} 位置无数据，请检查z_target是否在网格Z范围内！")
        return
    
    slice_plane = slice_plane.point_data_to_cell_data(True).cell_data_to_point_data()
    print(f"   Z截面原始点数：{slice_plane.n_points}")
    
    # 5. 分层采样（控制点数）
    df_single = stratified_sampling(
        slice_plane, near_wall_thresh, samples_single_section, near_wall_ratio,
        L, U0, T0, P0, q0, x_inlet, x_outlet
    )
    
    if df_single.empty:
        print(f"❌ 错误：Z截面采样后无有效数据！")
        return
    
    print(f"   完成：实际采样 {len(df_single)} 点")
    
    # 6. 可视化采样点
    pic_name = f"sample_Z_{z_target:.6f}.png"
    pic_save_path = os.path.join(SAVE_PIC_DIR, pic_name)
    visualize_sampling_points(slice_plane.points, df_single, z_target, pic_save_path)
    print(f"   采样点图片已保存：{pic_save_path}")

    # 7. 最终检查（和原来一样，确保数据在极值范围内）
    print("\n" + "="*70)
    print("【最终检查】采样数据范围 vs CSV极值范围")
    print("="*70)
    all_ok = True
    for col in columns:
        data_min = df_single[col].min()
        data_max = df_single[col].max()
        
        # 允许1e-6的浮点误差
        is_ok = (data_min >= global_min[col] - 1e-6) and (data_max <= global_max[col] + 1e-6)
        status = "✅" if is_ok else "❌"
        
        if not is_ok:
            all_ok = False
        
        print(f"   {status} {col:8s} | 采样: {data_min:.6f}~{data_max:.6f} | 极值: {global_min[col]:.6f}~{global_max[col]:.6f}")
    
    print("="*70)
    if all_ok:
        print("✅ 所有采样数据均在极值范围内！")
    else:
        print("⚠️  警告：部分数据超出范围，请检查！")
    
    # 8. 保存CSV
    save_csv_with_header(df_single, output_single_csv, global_min, global_max, columns)

    print("\n" + "="*70)
    print("所有任务完成！")
    print(f"✅ 目标Z截面：Z = {z_target:.6f}")
    print(f"✅ 总采样点数：{len(df_single)}")
    print(f"✅ CSV文件：{output_single_csv}")
    print(f"✅ 采样点图片：{pic_save_path}")
    print(f"✅ 核心保证：所有逻辑和原来完全一致，只是改成了单Z截面")
    print("="*70)

if __name__ == "__main__":
    main()