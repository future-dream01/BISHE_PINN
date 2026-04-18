import pyvista as pv
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt

# ==========================================
# 【参数设置区域】
# ==========================================
# --- 通用文件与物理参数 ---
cas_path = "066D_A3_hermites08_banmo_042_101.cas"
zone_id = 1
L = 0.095       # 特征长度
M0 = 0.42       # 来流马赫数
T0 = 249.15     # 来流静温
P0 = 47181      # 来流静压
U0 = M0 * (1.4 * 287 * T0) ** 0.5
Rou0 = P0 / (287 * T0)

# --- 训练集参数 (Train) ---
output_train_csv = "066D_A3_train.csv"
x_inlet = 0.297183
x_outlet = 0.515712
samples_inlet = 5000
samples_outlet = 5000
N1_train = 5    # 0.297183 ~ 0.34
N2_train = 15   # 0.34 ~ 0.47
N3_train = 5    # 0.47 ~ 0.515712
samples_train_section = 2000
near_wall_ratio_train = 0.7
near_wall_thresh_train = 0.005

# --- 验证集参数 (Validation) ---
output_val_csv = "066D_A3_val.csv"
val_x_sections = [0.31, 0.38, 0.42, 0.49] 
samples_val_section = 2000
near_wall_ratio_val = 0.5
near_wall_thresh_val = 0.005

# ==========================================
# 【核心函数区域】
# ==========================================

# 【新增】提取整个流体域所有点的全局极值（用于CSV表头）
def get_global_stats(target_block, L, U0, T0, P0, Rou0):
    """
    读取整个流体域的所有网格点，计算所有变量的全局MIN/MAX
    返回：包含MIN和MAX的两个字典，以及列名列表
    """
    print("\n正在提取整个进气道流域的全局极值（所有网格点）...")
    # 确保数据在节点上
    data = target_block.point_data_to_cell_data(True).cell_data_to_point_data()
    
    # 对整个流场做无量纲化（和采样函数逻辑完全一致）
    full_df = pd.DataFrame({
        "X": (data.points[:, 0])/L,
        "Y": (data.points[:, 1])/L,
        "Z": (data.points[:, 2])/L,
        "壁面距离": (data["WALL_DIST"])/L,
        "U": (data["X_VELOCITY"])/U0,
        "V": (data["Y_VELOCITY"])/U0,
        "W": (data["Z_VELOCITY"])/U0,
        "静压": ((data["PRESSURE"])-P0)/(Rou0*U0*U0),
        "静温": (data["TEMPERATURE"])/T0,
        "湍流动能": (data["TKE"])/(U0*U0),
        "比耗散率": (data["SDR"])*(L/U0),
    })
    
    # 计算全局MIN和MAX
    global_min = full_df.min().to_dict()
    global_max = full_df.max().to_dict()
    
    print("✅ 全局极值提取完成：")
    for col in full_df.columns:
        print(f"   {col:8s} | MIN: {global_min[col]:.6f} | MAX: {global_max[col]:.6f}")
    
    return global_min, global_max, full_df.columns.tolist()

# 【新增】带全局极值头的CSV保存函数
def save_csv_with_header(df, file_path, global_min, global_max, columns):
    """
    保存CSV：
    第1行：原始表头
    第2行：全局MIN
    第3行：全局MAX
    第4行起：采样数据
    """
    # 构造MIN/MAX行
    min_row = pd.DataFrame([global_min], columns=columns)
    max_row = pd.DataFrame([global_max], columns=columns)
    
    # 拼接：MIN -> MAX -> 采样数据
    final_df = pd.concat([min_row, max_row, df], ignore_index=True)
    
    # 保存（注意：此时第0行是MIN，第1行是MAX，第2行起是数据，但表头依然是正确的）
    # 为了让第1行视觉上是表头，我们手动写入
    with open(file_path, 'w', encoding='utf-8-sig', newline='') as f:
        # 1. 写入表头
        f.write(','.join(columns) + '\n')
        # 2. 写入MIN行（标记一下方便识别，或者直接写数值）
        min_vals = [f"{global_min[col]:.10f}" for col in columns]
        f.write(','.join(min_vals) + '\n')
        # 3. 写入MAX行
        max_vals = [f"{global_max[col]:.10f}" for col in columns]
        f.write(','.join(max_vals) + '\n')
        # 4. 写入采样数据
        df.to_csv(f, header=False, index=False, float_format='%.10f')
    
    print(f"✅ 文件已保存：{file_path}")
    print(f"   格式：第1行=表头 | 第2行=全局MIN | 第3行=全局MAX | 第4行起=数据")

def stratified_sampling(slice_data, near_thresh, total_samples, near_ratio, L, U0, T0, P0, Rou0):
    df = pd.DataFrame({
        "X": (slice_data.points[:, 0])/L,
        "Y": (slice_data.points[:, 1])/L,
        "Z": (slice_data.points[:, 2])/L,
        "壁面距离": (slice_data["WALL_DIST"])/L,
        "U": (slice_data["X_VELOCITY"])/U0,
        "V": (slice_data["Y_VELOCITY"])/U0,
        "W": (slice_data["Z_VELOCITY"])/U0,
        "静压": ((slice_data["PRESSURE"])-P0)/(Rou0*U0*U0),
        "静温": (slice_data["TEMPERATURE"])/T0,
        "湍流动能": (slice_data["TKE"])/(U0*U0),
        "比耗散率": (slice_data["SDR"])*(L/U0),
    })
    
    near_wall = df[df["壁面距离"] < near_thresh].copy()
    far_wall = df[df["壁面距离"] >= near_thresh].copy()

    near_num = int(total_samples * near_ratio)
    far_num = total_samples - near_num

    near_sampled = near_wall.sample(n=min(near_num, len(near_wall)), random_state=42) if len(near_wall) > 0 else pd.DataFrame()
    far_sampled = far_wall.sample(n=min(far_num, len(far_wall)), random_state=42) if len(far_wall) > 0 else pd.DataFrame()
    
    sampled_df = pd.concat([near_sampled, far_sampled], ignore_index=True)
    return sampled_df

def visualize_sampling_points(slice_points, sampled_points, x_pos, set_name=""):
    plt.figure(figsize=(9, 8))
    plt.scatter(slice_points[:, 1], slice_points[:, 2], c="lightgray", s=4, alpha=0.3, label="原始所有点")
    plt.scatter(sampled_points["Y"], sampled_points["Z"], c="crimson", s=6, alpha=0.9, label="采样点")
    
    prefix = f"[{set_name}]" if set_name else ""
    plt.xlabel("Y (m)")
    plt.ylabel("Z (m)")
    plt.title(f"{prefix} 截面 X = {x_pos:.6f} m\n采样总数：{len(sampled_points)}")
    plt.legend()
    plt.axis("equal")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

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

def process_sections(target_block, x_list, sample_counts, near_thresh, near_ratio, 
                     L, U0, T0, P0, Rou0, set_name=""):
    all_data = []
    far_ratio = 1 - near_ratio
    
    for idx, x_pos in enumerate(x_list):
        print(f"\n[{set_name}] 处理第 {idx+1}/{len(x_list)} 个截面 X = {x_pos:.6f}")
        
        slice_plane = target_block.slice(normal=[1,0,0], origin=[x_pos, 0, 0])
        if slice_plane.n_points == 0:
            print(f"   跳过：无数据")
            continue
        
        slice_plane = slice_plane.point_data_to_cell_data(True).cell_data_to_point_data()
        current_samples = sample_counts[idx]
        
        sampled_df = stratified_sampling(slice_plane, near_thresh, current_samples, near_ratio,
                                          L, U0, T0, P0, Rou0)

        all_data.append(sampled_df)
        print(f"   完成：实际采样 {len(sampled_df)} 点")
        visualize_sampling_points(slice_plane.points, sampled_df, x_pos, set_name)
        
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
    target_block = mesh[zone_id]
    if target_block is None or target_block.n_points == 0:
        print("错误：无有效网格")
        return
    print(f"✅ 已加载流体域 Zone {zone_id}")

    # 【关键步骤1：训练前提取整个流场的全局极值】
    # 注意：训练集和验证集使用【同一套】全局极值，保证归一化基准一致
    global_min, global_max, columns = get_global_stats(target_block, L, U0, T0, P0, Rou0)

    # 2. 准备训练集截面坐标
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

    # 3. 准备验证集截面坐标
    val_x = np.array(val_x_sections)
    val_x = np.sort(val_x)
    val_sample_counts = [samples_val_section] * len(val_x)
    print(f"\n验证集共 {len(val_x)} 个指定截面")

    # 4. 重合性检查
    check_overlap(train_x, val_x)

    # 5. 处理训练集
    print("\n" + "="*70)
    print("开始处理【训练集】")
    print("="*70)
    df_train = process_sections(
        target_block, train_x, train_sample_counts,
        near_wall_thresh_train, near_wall_ratio_train,
        L, U0, T0, P0, Rou0, set_name="Train"
    )
    
    if not df_train.empty:
        # 【关键修改】使用新的保存函数，带上全局极值头
        save_csv_with_header(df_train, output_train_csv, global_min, global_max, columns)
        print(f"   训练集采样点数：{len(df_train)}")

    # 6. 处理验证集
    print("\n" + "="*70)
    print("开始处理【验证集】")
    print("="*70)
    df_val = process_sections(
        target_block, val_x, val_sample_counts,
        near_wall_thresh_val, near_wall_ratio_val,
        L, U0, T0, P0, Rou0, set_name="Val"
    )
    
    if not df_val.empty:
        # 【关键修改】验证集也使用【同一套】全局极值
        save_csv_with_header(df_val, output_val_csv, global_min, global_max, columns)
        print(f"   验证集采样点数：{len(df_val)}")

    print("\n" + "="*70)
    print("所有任务完成！")
    print("重要提示：训练集和验证集使用了完全相同的全局极值（来自整个流场），保证归一化一致性。")
    print("="*70)

if __name__ == "__main__":
    main()