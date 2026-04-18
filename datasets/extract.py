import pyvista as pv
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt

# ==========================================
# 【参数设置区域】请在此处修改配置
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
# 强制包含的进/出口坐标
x_inlet = 0.297183
x_outlet = 0.515712
# 进/出口截面【单独】采样数设置
samples_inlet = 5000
samples_outlet = 5000
# 三个区域的截面数 (不含强制的进/出口，程序会自动补上)
N1_train = 5    # 0.297183 ~ 0.34
N2_train = 15   # 0.34 ~ 0.47
N3_train = 5    # 0.47 ~ 0.515712
# 训练集【普通截面】采样参数
samples_train_section = 2000
near_wall_ratio_train = 0.7
near_wall_thresh_train = 0.005

# --- 验证集参数 (Validation) ---
output_val_csv = "066D_A3_val.csv"
# 【关键】在此处指定验证集的任意X坐标列表
val_x_sections = [0.31, 0.38, 0.42, 0.49] 
# 验证集采样参数
samples_val_section = 2000
near_wall_ratio_val = 0.5
near_wall_thresh_val = 0.005

# ==========================================
# 【核心函数区域】
# ==========================================

def stratified_sampling(slice_data, near_thresh, total_samples, near_ratio, L, U0, T0, P0, Rou0):
    """
    分层采样函数（为了通用性，将物理常数作为参数传入）
    """
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
    """可视化函数（增加了set_name参数区分训练/验证）"""
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
    """
    检查验证集坐标是否与训练集重合
    """
    print("\n" + "="*70)
    print("正在验证坐标重合情况...")
    
    overlap_coords = []
    for vx in val_x:
        # 计算与所有训练集坐标的最小距离
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
    """
    通用处理流程：处理一组截面，返回DataFrame
    """
    all_data = []
    far_ratio = 1 - near_ratio
    
    for idx, x_pos in enumerate(x_list):
        print(f"\n[{set_name}] 处理第 {idx+1}/{len(x_list)} 个截面 X = {x_pos:.6f}")
        
        # 切片
        slice_plane = target_block.slice(normal=[1,0,0], origin=[x_pos, 0, 0])
        if slice_plane.n_points == 0:
            print(f"   跳过：无数据")
            continue
        
        # 数据格式统一
        slice_plane = slice_plane.point_data_to_cell_data(True).cell_data_to_point_data()
        
        # 获取当前截面的采样数
        current_samples = sample_counts[idx]
        
        # 采样
        sampled_df = stratified_sampling(slice_plane, near_thresh, current_samples, near_ratio,
                                          L, U0, T0, P0, Rou0)
        
        all_data.append(sampled_df)
        print(f"   完成：实际采样 {len(sampled_df)} 点")
        
        # 可视化
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

    # ==========================================
    # 2. 准备训练集截面坐标
    # ==========================================
    print("\n正在生成训练集截面...")
    # 生成三个区域的内部截面
    s1 = np.linspace(x_inlet, 0.34, N1_train + 2)[1:-1] # 去掉首尾（首尾是进/出口）
    s2 = np.linspace(0.34, 0.47, N2_train)
    s3 = np.linspace(0.47, x_outlet, N3_train + 2)[1:-1]
    
    # 合并并强制加入进/出口
    train_x_raw = np.concatenate([[x_inlet], s1, s2, s3, [x_outlet]])
    train_x_raw = np.unique(train_x_raw) # 去重
    train_x = np.sort(train_x_raw)       # 排序
    
    # 生成训练集对应的采样数列表
    train_sample_counts = []
    for x in train_x:
        if abs(x - x_inlet) < 1e-8:
            train_sample_counts.append(samples_inlet)
        elif abs(x - x_outlet) < 1e-8:
            train_sample_counts.append(samples_outlet)
        else:
            train_sample_counts.append(samples_train_section)

    print(f"训练集共 {len(train_x)} 个截面")
    print(f"   包含强制入口 X={x_inlet} (采样{samples_inlet})")
    print(f"   包含强制出口 X={x_outlet} (采样{samples_outlet})")

    # ==========================================
    # 3. 准备验证集截面坐标
    # ==========================================
    val_x = np.array(val_x_sections)
    val_x = np.sort(val_x)
    val_sample_counts = [samples_val_section] * len(val_x)
    print(f"\n验证集共 {len(val_x)} 个指定截面")

    # ==========================================
    # 4. 重合性检查
    # ==========================================
    check_overlap(train_x, val_x)

    # ==========================================
    # 5. 处理训练集
    # ==========================================
    print("\n" + "="*70)
    print("开始处理【训练集】")
    print("="*70)
    df_train = process_sections(
        target_block, train_x, train_sample_counts,
        near_wall_thresh_train, near_wall_ratio_train,
        L, U0, T0, P0, Rou0, set_name="Train"
    )
    
    if not df_train.empty:
        df_train.to_csv(output_train_csv, index=False, encoding="utf-8-sig")
        print(f"\n✅ 训练集保存至：{output_train_csv}")
        print(f"   总点数：{len(df_train)}")

    # ==========================================
    # 6. 处理验证集
    # ==========================================
    print("\n" + "="*70)
    print("开始处理【验证集】")
    print("="*70)
    df_val = process_sections(
        target_block, val_x, val_sample_counts,
        near_wall_thresh_val, near_wall_ratio_val,
        L, U0, T0, P0, Rou0, set_name="Val"
    )
    
    if not df_val.empty:
        df_val.to_csv(output_val_csv, index=False, encoding="utf-8-sig")
        print(f"\n✅ 验证集保存至：{output_val_csv}")
        print(f"   总点数：{len(df_val)}")

    print("\n" + "="*70)
    print("所有任务完成！")
    print("="*70)

if __name__ == "__main__":
    main()