import pyvista as pv
import pandas as pd

# ====================== 文件输入输出 ======================
cas_path = "05D_A0_hermites08_banmo_042_101.cas"
output_csv = "x_0.2971830m_结果.csv"
# ====================================================

# 1. 读取并合并网格
print("正在读取 Fluent 数据...")
mesh = pv.read(cas_path)
if hasattr(mesh, "combine"):
    mesh = mesh.combine()

# 无量纲化
# 2. 截取截面
slice_plane = mesh.slice(normal=[1,0,0], origin=[0.2971830, 0, 0])
if slice_plane.n_points == 0:
    print("截面无数据！")
    exit()

# 将所有物理量插值到网格节点（统一长度）
slice_plane = slice_plane.point_data_to_cell_data(pass_point_data=True).cell_data_to_point_data()

# 3. 提取数据（变量名完全匹配你的文件）
data = {
    "X": slice_plane.points[:, 0],
    "Y": slice_plane.points[:, 1],
    "Z": slice_plane.points[:, 2],
    "U": slice_plane["X_VELOCITY"],
    "V": slice_plane["Y_VELOCITY"],
    "W": slice_plane["Z_VELOCITY"],
    "密度": slice_plane["DENSITY"],
    "静压": slice_plane["PRESSURE"],
    "静温": slice_plane["TEMPERATURE"],
    "湍流动能": slice_plane["TKE"],
    "比耗散率": slice_plane["SDR"]
}

# 4. 保存CSV
df = pd.DataFrame(data)
df.to_csv(output_csv, index=False, encoding="utf-8-sig")

print(f"\n✅ 提取完成！共 {len(df)} 个点")
print(f"✅ 文件已保存：{output_csv}")