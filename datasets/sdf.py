import trimesh
import numpy as np

# ====================== 1. 加载STL ======================
mesh_fluid = trimesh.load("fluid.stl")
mesh_wall = trimesh.load("wall.stl")

# 坐标缩放
mesh_fluid.apply_scale(0.001)
mesh_wall.apply_scale(0.001)

# 修复壁面STL法向（解决壁距负数！）
trimesh.repair.fix_normals(mesh_wall)

# 验证流体域封闭
if not mesh_fluid.is_watertight:
    trimesh.repair.fill_holes(mesh_fluid)
    trimesh.repair.fix_normals(mesh_fluid)
print(f"✅ 流体域STL验证通过，是否封闭：{mesh_fluid.is_watertight}，三角面数量：{len(mesh_fluid.faces)}")

# ====================== 2. 初始化查询器 ======================
proximity_fluid = trimesh.proximity.ProximityQuery(mesh_fluid)
proximity_wall = trimesh.proximity.ProximityQuery(mesh_wall)

# ====================== 3. 生成随机点 ======================
bounds = mesh_fluid.bounds
x_min, y_min, z_min = bounds[0]
x_max, y_max, z_max = bounds[1]

num_points = 100000
points = np.random.uniform(
    low=[x_min, y_min, z_min],
    high=[x_max, y_max, z_max],
    size=(num_points, 3)
)

# ====================== 4. 计算 SDF（判断内外） ======================
signed_distance = proximity_fluid.signed_distance(points)
inside_mask = signed_distance > 0

# 过滤内部点
points_inside = points[inside_mask]
signed_inside = signed_distance[inside_mask]

# ====================== 5. 计算壁面距离（强制转一维！！） ======================
d_wall_inside, _, _ = proximity_wall.on_surface(points_inside)
d_wall_inside = np.ravel(d_wall_inside)  # 关键：转成一维数组

# 强制取绝对值（彻底杜绝负数！）
d_wall_inside = np.abs(d_wall_inside)

# ====================== 6. 打印信息 ======================
print("\n" + "="*60)
print(f"生成总点数：{num_points}")
print(f"内部点数：{len(points_inside)}，接受率：{len(points_inside)/num_points:.2%}")
print(f"内部点SDF范围：[{signed_inside.min():.4f}, {signed_inside.max():.4f}]")
print(f"壁面距离范围：[{d_wall_inside.min():.6f}, {d_wall_inside.max():.6f}] m")
print("="*60)

# ====================== 7. 测试点输出（彻底修复报错） ======================
print("\n测试点验证：")
test_indices = [0, 10, 20, 30]
for idx in test_indices:
    if idx < len(points_inside):
        p = points_inside[idx]
        sd = signed_inside[idx]
        dw = d_wall_inside[idx]
        
        # 转成普通数字再打印，绝对不报错
        p_str = f"[{p[0]:.3f},{p[1]:.3f},{p[2]:.3f}]"
        print(f"点{idx}: 坐标={p_str}, SDF={float(sd):.4f}, 壁距={float(dw):.6f}m")

# ====================== 8. 保存 ======================
np.savez("pinn_sdf_data.npz",
    points=points_inside,
    sdf=signed_inside,
    d_wall=d_wall_inside
)

print("\n✅ 全部完成！数据已保存到 pinn_sdf_data.npz")