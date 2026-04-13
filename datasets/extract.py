import pyvista as pv
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt

# 参数设置+无量纲化
cas_path = "066D_A3_hermites08_banmo_042_101.cas"   # cas dat文件名
zone_id = 1                                         # 流体域Zone ID
output_total_csv = "066D_A3_hermites08_banmo_042_101.csv"  # 总输出文件
N1 = 5    # 0.297183 ~ 0.34内的截面数
N2 = 15   # 0.34 ~ 0.47内的截面数
N3 = 5    # 0.47 ~ 0.515712内的截面数
total_samples_per_section = 1000  # 每个截面的采样点总数
near_wall_ratio = 0.65            # 近壁区采样点占比
near_wall_threshold = 0.005       # 近壁区阈值：<0.005m
L=0.095    # 特征长度
M0=0.42    # 来流马赫数
T0=249.15  # 来流静温
P0=47181   # 来流静压
U0=(1.4*287*T0)**0.5 # 来流速度
Rou0=P0/(287*T0)     # 来流密度




# 采样、合并函数
def stratified_sampling(slice_data, near_thresh, total_samples, near_ratio):
    """
    分层采样：固定总采样数，近壁多采、远壁少采，占比和为100%
    返回采样后的DataFrame
    """
    # 转为DataFrame
    df = pd.DataFrame({
        "X": (slice_data.points[:, 0])/L,
        "Y": (slice_data.points[:, 1])/L,
        "Z": (slice_data.points[:, 2])/L,
        "壁面距离": (slice_data["WALL_DIST"])/L,
        "U": (slice_data["X_VELOCITY"])/U0,
        "V": (slice_data["Y_VELOCITY"])/U0,
        "W": (slice_data["Z_VELOCITY"])/U0,
        "静压": ((slice_data["PRESSURE"])-P0)/(0.5*Rou0*U0*U0),
        "静温": (slice_data["TEMPERATURE"])/T0,
        "湍流动能": (slice_data["TKE"])/(U0*U0),
        "比耗散率": (slice_data["SDR"])*(L/U0),
    })
    
    # 划分近壁/远壁
    near_wall = df[df["壁面距离"] < near_thresh].copy()
    far_wall = df[df["壁面距离"] >= near_thresh].copy()

    # 计算近壁/远壁采样数量（固定总数，占比和为100%）
    near_num = int(total_samples * near_ratio)
    far_num = total_samples - near_num

    # 安全采样（防止点数不足，自动补齐）
    near_sampled = near_wall.sample(n=min(near_num, len(near_wall)), random_state=42) if len(near_wall) > 0 else pd.DataFrame()
    far_sampled = far_wall.sample(n=min(far_num, len(far_wall)), random_state=42) if len(far_wall) > 0 else pd.DataFrame()
    
    # 合并采样结果
    sampled_df = pd.concat([near_sampled, far_sampled], ignore_index=True)
    return sampled_df

# 可视化函数
def visualize_sampling_points(slice_points, sampled_points, x_pos):
    """可视化：所有截面的原始点 + 采样点分布"""
    plt.figure(figsize=(9, 8))
    # 原始截面点（灰色）
    plt.scatter(slice_points[:, 1], slice_points[:, 2], c="lightgray", s=4, alpha=0.3, label="原始所有点")
    # 采样点（红色）
    plt.scatter(sampled_points["Y"], sampled_points["Z"], c="crimson", s=6, alpha=0.9, label="采样点")
    
    plt.xlabel("Y (m)")
    plt.ylabel("Z (m)")
    plt.title(f"截面 X = {x_pos:.6f} m\n采样总数：{len(sampled_points)} | 近壁占比：{near_wall_ratio:.0%}")
    plt.legend()
    plt.axis("equal")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

# 主函数
def main():
    # 读取FLUENT网格
    print("\n正在读取 Fluent 文件...")
    mesh = pv.read(cas_path)
    print(f"总 Zone 数量: {len(mesh)}")

    # 选取流体域
    target_block = mesh[zone_id]
    if target_block is None or target_block.n_points == 0:
        print("无有效网格")
        exit()
    print(f"已加载流体域 Zone {zone_id}")
    all_sampled_data = []  # 存储所有截面的采样数据
    # 截面切片设置
    x_start1, x_end1 = 0.297183, 0.34
    x_start2, x_end2 = 0.34, 0.47
    x_start3, x_end3 = 0.47, 0.515712
    section1 = np.linspace(x_start1, x_end1, N1)
    section2 = np.linspace(x_start2, x_end2, N2)
    section3 = np.linspace(x_start3, x_end3, N3)
    all_x_sections = np.concatenate([section1, section2, section3])
    all_x_sections = np.unique(all_x_sections)  # 去重
    all_x_sections = np.sort(all_x_sections)
    far_wall_ratio = 1 - near_wall_ratio
    print(f"共生成 {len(all_x_sections)} 个截面")
    print(f"截面位置 X = {[f'{x:.6f}' for x in all_x_sections]}")
    print(f"每个截面固定采样：{total_samples_per_section} 个点")
    print(f"近壁区占比：{near_wall_ratio:.0%} | 远壁区占比：{far_wall_ratio:.0%}（总和100%）")
    for idx, x_pos in enumerate(all_x_sections):
        print(f"\n────────── 正在处理第{idx+1}个截面 X = {x_pos:.6f} m ──────────")
        
        # 切片
        slice_plane = target_block.slice(normal=[1,0,0], origin=[x_pos, 0, 0])
        if slice_plane.n_points == 0:
            print(f"截面 X={x_pos} 无数据，跳过")
            continue
        
        # 数据格式统一
        slice_plane = slice_plane.point_data_to_cell_data(True).cell_data_to_point_data()
        
        # 分层采样（固定总数）
        sampled_df = stratified_sampling(slice_plane, near_wall_threshold, total_samples_per_section, near_wall_ratio)
        
        # 保存到总列表
        all_sampled_data.append(sampled_df)
        print(f"✅ 采样完成：目标{total_samples_per_section}点 → 实际{len(sampled_df)}点")
        
        # 生成【所有截面】的可视化图像
        #visualize_sampling_points(slice_plane.points, sampled_df, x_pos)

    # 合并、保存数据
    if len(all_sampled_data) == 0:
        print("\n无有效采样数据")
        exit()

    final_df = pd.concat(all_sampled_data, ignore_index=True)
    final_df.to_csv(output_total_csv, index=False, encoding="utf-8-sig")

    # 输出日志
    print("\n" + "="*70)
    print(f"全部处理完成！")
    print(f"总截面数：{len(all_x_sections)}")
    print(f"单个截面采样：{total_samples_per_section} 个点")
    print(f"全流道总采样点数：{len(final_df)}")
    print(f"总数据已保存至：{output_total_csv}")
    print(f"近壁区占比：{near_wall_ratio:.0%} | 远壁区占比：{far_wall_ratio:.0%}")
    print("="*70)

if __name__=="__main__":
    main()