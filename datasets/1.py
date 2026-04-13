import pyvista as pv
cas_path = "05D_A0_hermites08_banmo_042_101.cas"
mesh = pv.read(cas_path)
# 打印所有变量
print("你的文件里所有可用变量：")
print(mesh[2].array_names)  # 看第一个zone的变量