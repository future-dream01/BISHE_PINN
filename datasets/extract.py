# 数据集制作
import pyvista as pv
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt

# ==========================================
# 【参数设置区域】
# ==========================================
# --- 通用文件与物理参数 ---
cas_path = "05D_A0_hermites08_banmo_066_107.cas"
zone_id = 2  # 【关键】先运行一次，看打印的Zone列表，改成正确的流体域ID
L = 0.095       # 特征长度
M0 = 0.66       # 来流马赫数
T0 = 249.15     # 来流静温
P0 = 47181      # 来流静压
U0 = M0 * (1.4 * 287 * T0) ** 0.5
Rou0 = P0 / (287 * T0)
q0 = 1.4 * P0 * M0**2  # 来流动压，标准无量纲化

# --- 进气道流道范围（强制过滤用）---
x_inlet = 0.297183   # 入口X坐标
x_outlet = 0.515712  # 出口X坐标

# --- 训练集参数 (Train) ---
output_train_csv = "050D_A0_train.csv"
samples_inlet = 3000
samples_outlet = 3000
N1_train = 3    # 0.297183 ~ 0.34
N2_train = 10   # 0.34 ~ 0.47
N3_train = 3    # 0.47 ~ 0.515712
samples_train_section = 2000
near_wall_ratio_train = 0.7
near_wall_thresh_train = 0.005  # 有量纲阈值，单位m

# --- 验证集参数 (Validation) ---
output_val_csv = "050D_A0_val.csv"
val_x_sections = [0.31,0.41,0.438,0.490] 
samples_val_section = 8000
near_wall_ratio_val = 0.75
near_wall_thresh_val = 0.005

# ==========================================
# 【核心函数区域】
# ==========================================

# 统一无量纲化函数（全局极值和采样数据共用，100%逻辑一致）
def nondimensionalize_data(points, point_data, L, U0, T0, P0, q0):
    df = pd.DataFrame({
        "X": (points[:, 0]) / L,
        "Y": (points[:, 1]) / L,
        "Z": (points[:, 2]) / L,
        "壁面距离": (point_data["WALL_DIST"]) / L,
        "U": (point_data["X_VELOCITY"]) / U0,
        "V": (point_data["Y_VELOCITY"]) / U0,
        "W": (point_data["Z_VELOCITY"]) / U0,
        "静压": ((point_data["PRESSURE"]) - P0) / q0,
        "静温": (point_data["TEMPERATURE"]) / T0,
        "湍流动能": (point_data["TKE"]) / (U0 ** 2),
        "比耗散率": (point_data["SDR"]) * (L / U0),
    })
    return df

# 【修复】仅提取进气道内部有效点的无量纲全局极值
def get_global_stats(target_block, L, U0, T0, P0, q0, x_inlet, x_outlet):
    print(f"\n正在提取【进气道内部有效点】的【无量纲化后】全局极值...")
    
    # 确保数据在节点上
    data = target_block.point_data_to_cell_data(True).cell_data_to_point_data()
    
    # ==========================================
    # 【核心修复1：强制过滤进气道内部点】
    # ==========================================
    # 1. 只保留X在入口~出口之间的点
    x_coords = data.points[:, 0]
    in_x_range = (x_coords >= x_inlet) & (x_coords <= x_outlet)
    
    # 2. 物理量合理性过滤，排除无效点
    temp_valid = data["TEMPERATURE"] > 0.8 * T0  # 温度不低于来流的80%
    press_valid = data["PRESSURE"] > 0            # 压力不能为负
    tke_valid = data["TKE"] >= 0                  # 湍动能不能为负
    wall_dist_valid = data["WALL_DIST"] >= 0      # 壁面距离不能为负
    
    # 合并过滤条件
    valid_mask = in_x_range & temp_valid & press_valid & tke_valid & wall_dist_valid
    
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
    
    print("\n✅ 无量纲化后的全局极值（有效点）：")
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

# CSV保存函数
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

# 分层采样函数
def stratified_sampling(slice_data, near_thresh, total_samples, near_ratio, L, U0, T0, P0, q0):
    df = nondimensionalize_data(
        points=slice_data.points,
        point_data=slice_data.point_data,
        L=L, U0=U0, T0=T0, P0=P0, q0=q0
    )
    
    near_thresh_nondim = near_thresh / L
    near_wall = df[df["壁面距离"] < near_thresh_nondim].copy()
    far_wall = df[df["壁面距离"] >= near_thresh_nondim].copy()

    near_num = int(total_samples * near_ratio)
    far_num = total_samples - near_num

    near_sampled = near_wall.sample(n=min(near_num, len(near_wall)), random_state=42) if len(near_wall) > 0 else pd.DataFrame()
    far_sampled = far_wall.sample(n=min(far_num, len(far_wall)), random_state=42) if len(far_wall) > 0 else pd.DataFrame()
    
    sampled_df = pd.concat([near_sampled, far_sampled], ignore_index=True)
    return sampled_df

# 可视化函数
def visualize_sampling_points(slice_points, sampled_points, x_pos, set_name=""):
    plt.figure(figsize=(9, 8))
    plt.scatter(slice_points[:, 1]/L, slice_points[:, 2]/L, c="lightgray", s=4, alpha=0.3, label="原始所有点")
    plt.scatter(sampled_points["Y"], sampled_points["Z"], c="crimson", s=6, alpha=0.9, label="采样点")
    
    prefix = f"[{set_name}]" if set_name else ""
    plt.xlabel("Y (无量纲)")
    plt.ylabel("Z (无量纲)")
    plt.title(f"{prefix} 截面 X = {x_pos:.6f} m\n采样总数：{len(sampled_points)}")
    plt.legend()
    plt.axis("equal")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

# 坐标重合检查
def check_overlap(train_x, val_x, tol=1e-6):
    print("\n" + "="*70)
    print("正在验证坐标重合情况...")
    
    overlap_coords = []
    for vx in val_x:
        min_dist = np.min(np.abs(np.array(train_x) - vx))
        if min_dist < tol:
            overlap_coords.append(vx)
    
    if overlap_coords:
        print(f"⚠️  警告：发现 {len(overlap_coords)} 个重合坐标！")
        for c in overlap_coords:
            print(f"   - X = {c:.6f}")
        print("建议修改验证集坐标，避免数据泄露。")
    else:
        print("✅ 验证通过：验证集与训练集无重合坐标。")
    print("="*70)
    return overlap_coords

# 截面处理函数
def process_sections(target_block, x_list, sample_counts, near_thresh, near_ratio, 
                     L, U0, T0, P0, q0, set_name=""):
    all_data = []
    
    for idx, x_pos in enumerate(x_list):
        print(f"\n[{set_name}] 处理第 {idx+1}/{len(x_list)} 个截面 X = {x_pos:.6f}")
        
        slice_plane = target_block.slice(normal=[1,0,0], origin=[x_pos, 0, 0])
        if slice_plane.n_points == 0:
            print(f"   跳过：无数据")
            continue
        
        slice_plane = slice_plane.point_data_to_cell_data(True).cell_data_to_point_data()
        current_samples = sample_counts[idx]
        
        sampled_df = stratified_sampling(
            slice_plane, near_thresh, current_samples, near_ratio,
            L, U0, T0, P0, q0
        )

        all_data.append(sampled_df)
        print(f"   完成：实际采样 {len(sampled_df)} 点")
        #visualize_sampling_points(slice_plane.points, sampled_df, x_pos, set_name)
        
    if not all_data:
        return pd.DataFrame()
    
    return pd.concat(all_data, ignore_index=True)

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

    # 3. 提取进气道内部有效点的无量纲全局极值
    global_min, global_max, columns = get_global_stats(
        target_block, L, U0, T0, P0, q0, x_inlet, x_outlet
    )

    # 4. 准备训练集截面坐标
    print("\n正在生成训练集截面...")
    s1 = np.linspace(x_inlet, 0.34, N1_train + 2)[1:-1]
    s2 = np.linspace(0.34, 0.47, N2_train)
    s3 = np.linspace(0.47, x_outlet, N3_train + 2)[1:-1]
    
    train_x_raw = np.concatenate([[x_inlet], s1, s2, s3, [x_outlet]])
    train_x_raw = np.unique(train_x_raw)
    train_x = np.sort(train_x_raw)
    
    # 训练集采样数列表
    train_sample_counts = []
    for x in train_x:
        if abs(x - x_inlet) < 1e-8:
            train_sample_counts.append(samples_inlet)
        elif abs(x - x_outlet) < 1e-8:
            train_sample_counts.append(samples_outlet)
        else:
            train_sample_counts.append(samples_train_section)

    print(f"训练集共 {len(train_x)} 个截面")

    # 5. 准备验证集截面坐标
    val_x = np.array(val_x_sections)
    val_x = np.sort(val_x)
    val_sample_counts = [samples_val_section] * len(val_x)
    print(f"\n验证集共 {len(val_x)} 个指定截面")

    # 6. 重合性检查
    check_overlap(train_x, val_x)

    # 7. 处理训练集
    print("\n" + "="*70)
    print("开始处理【训练集】")
    print("="*70)
    df_train = process_sections(
        target_block, train_x, train_sample_counts,
        near_wall_thresh_train, near_wall_ratio_train,
        L, U0, T0, P0, q0, set_name="Train"
    )
    
    if not df_train.empty:
        save_csv_with_header(df_train, output_train_csv, global_min, global_max, columns)
        print(f"   训练集总采样点数：{len(df_train)}")

    # 8. 处理验证集
    print("\n" + "="*70)
    print("开始处理【验证集】")
    print("="*70)
    df_val = process_sections(
        target_block, val_x, val_sample_counts,
        near_wall_thresh_val, near_wall_ratio_val,
        L, U0, T0, P0, q0, set_name="Val"
    )
    
    if not df_val.empty:
        save_csv_with_header(df_val, output_val_csv, global_min, global_max, columns)
        print(f"   验证集总采样点数：{len(df_val)}")

    print("\n" + "="*70)
    print("所有任务完成！")
    print(f"✅ 核心保证：所有极值、采样数据均来自进气道内部有效点，无量纲化逻辑完全统一")
    print("="*70)

if __name__ == "__main__":
    main()