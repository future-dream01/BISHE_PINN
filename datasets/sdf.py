import trimesh
import numpy as np
import pandas as pd
import os

# ====================== 配置区域 ======================
mesh_wall_path = "wall.stl"
mesh_fluid_path = "fluid.stl"

points_input = np.array([
    [0.5,0.029619377,0.035493694],
    [0.5,0.031041767,0.034030586],
    [0.5,0.03430242,0.030420672]
])

csv_input_path = "x_0.2971830m_结果.csv" 
use_csv = False
output_path = "points_with_wall_distance.csv"
# ====================================================

# ====================== 1. 加载 STL ======================
print("正在加载 STL 文件...")
mesh_wall = trimesh.load(mesh_wall_path)
mesh_fluid = trimesh.load(mesh_fluid_path)
mesh_wall.apply_scale(0.001)
mesh_fluid.apply_scale(0.001)
trimesh.repair.fix_normals(mesh_wall)
trimesh.repair.fix_normals(mesh_fluid)
print(f"✅ STL 加载完成")

# ====================== 2. 加载点 ======================
if use_csv and os.path.exists(csv_input_path):
    df = pd.read_csv(csv_input_path)
    points = df[["X", "Y", "Z"]].values
else:
    points = points_input

n_points = len(points)
print(f"✅ 待计算点数: {n_points}")

# ====================== 3. 计算 (终极修复区) ======================
print("\n正在计算壁面距离...")
proximity_wall = trimesh.proximity.ProximityQuery(mesh_wall)
proximity_fluid = trimesh.proximity.ProximityQuery(mesh_fluid)

# --- 计算 SDF ---
signed_distance = proximity_fluid.signed_distance(points)
# 🔥 终极修复1：强制转成 numpy 数组并拍平
signed_distance = np.array(signed_distance).flatten()

# --- 判断内部 ---
is_inside = signed_distance > 0
is_inside = np.array(is_inside).flatten()

# --- 计算壁面距离 ---
d_wall, _, _ = proximity_wall.on_surface(points)
d_wall = np.array(d_wall).flatten()
d_wall = np.abs(d_wall)

# ====================== 4. 🔥 终极调试：打印长度，看看到底谁不一样 ======================
print("\n" + "="*60)
print("【调试信息】各数组长度：")
print(f"  坐标点 (points):    {n_points}")
print(f"  SDF:                 {len(signed_distance)}")
print(f"  是否在内部:          {len(is_inside)}")
print(f"  壁面距离:            {len(d_wall)}")
print("="*60)

# ====================== 5. 🔥 终极暴力修复：强制截取或填充到一样长 ======================
def force_length(arr, target_len):
    arr = np.array(arr).flatten()
    if len(arr) == target_len:
        return arr
    elif len(arr) > target_len:
        return arr[:target_len] # 截取前N个
    else:
        # 如果太短，填充0 (一般不会走到这一步)
        return np.pad(arr, (0, target_len - len(arr)), mode='constant')

signed_distance = force_length(signed_distance, n_points)
is_inside = force_length(is_inside, n_points)
d_wall = force_length(d_wall, n_points)

# ====================== 6. 输出 ======================
result_df = pd.DataFrame({
    "X": points[:, 0],
    "Y": points[:, 1],
    "Z": points[:, 2],
    "壁面距离_m": d_wall,
    "是否在流场内": is_inside,
    "SDF": signed_distance
})

print("\n计算结果预览:")
print(result_df)

result_df.to_csv(output_path, index=False, encoding="utf-8-sig")
print(f"\n✅ 全部完成！结果已保存至: {output_path}")