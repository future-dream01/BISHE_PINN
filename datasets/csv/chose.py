import pandas as pd
import numpy as np

# ==========================================
# 【配置区域】
# ==========================================
csv_files = [
    "05D_A0_042_101.csv",
    "05D_A0_046_102.csv",
    "05D_A0_050_103.csv",
    "05D_A0_054_104.csv",
    "05D_A0_058_105.csv",
    "05D_A0_062_106.csv",
    "05D_A0_066_107.csv",
    "05D_A0_070_108.csv",
    "05D_A0_074_109.csv",
    "05D_A0_078_110.csv",
]

# ==========================================
# 【核心处理逻辑】
# ==========================================
if not csv_files:
    print("❌ 错误：没有找到CSV文件！")
    exit()

print(f"✅ 找到 {len(csv_files)} 个CSV文件")

# 初始化变量
columns_ref = None
all_min_values = []  # 存所有文件的第2行
all_max_values = []  # 存所有文件的第3行

for file_path in csv_files:
    try:
        # 【核心修改1】只读取前3行，且不把任何行作为表头（header=None）
        # 这样读取后：
        # df.iloc[0] = 原文件第1行（表头）
        # df.iloc[1] = 原文件第2行（MIN行）
        # df.iloc[2] = 原文件第3行（MAX行）
        df = pd.read_csv(file_path, header=None, nrows=3)
        
        # 提取表头
        current_columns = df.iloc[0, :].tolist()
        
        # 检查列数一致性
        if columns_ref is None:
            columns_ref = current_columns
            print(f"   参考列数：{len(columns_ref)}")
        else:
            if current_columns != columns_ref:
                print(f"⚠️  警告：{file_path} 列数不一致！")
        
        # 【核心修改2】提取数据
        min_row = df.iloc[1, :].values.astype(np.float64)  # 原文件第2行
        max_row = df.iloc[2, :].values.astype(np.float64)  # 原文件第3行
        
        all_min_values.append(min_row)
        all_max_values.append(max_row)
        
        print(f"✅ 成功读取：{file_path}")
        
    except Exception as e:
        print(f"❌ 读取失败 {file_path}：{e}")

if len(all_min_values) == 0:
    print("❌ 错误：没有成功读取任何文件！")
    exit()

# ==========================================
# 【计算全局极值】逐列对比
# ==========================================
# 把列表转成 numpy 矩阵 (文件数 x 列数)
min_matrix = np.array(all_min_values)
max_matrix = np.array(all_max_values)

# 【核心修改3】按列取最小值和最大值 (axis=0 表示按列操作)
global_min = min_matrix.min(axis=0)
global_max = max_matrix.max(axis=0)

# ==========================================
# 【打印结果】
# ==========================================
print("\n" + "="*80)
print("【计算完成】逐列对比结果")
print("="*80)

print("\n1. 全局最小行（所有文件第2行逐列取最小）：")
min_str = "  " + ", ".join([f"{x:.10f}" for x in global_min])
print(min_str)

print("\n2. 全局最大行（所有文件第3行逐列取最大）：")
max_str = "  " + ", ".join([f"{x:.10f}" for x in global_max])
print(max_str)

# ==========================================
# 【保存结果】格式和原CSV完全一致
# ==========================================
output_file = "global_min_max.csv"

# 构建结果：第1行是表头，第2行是全局MIN，第3行是全局MAX
result_data = [columns_ref, global_min.tolist(), global_max.tolist()]
result_df = pd.DataFrame(result_data)

# 保存：不带索引，不带表头（因为我们已经手动加了）
result_df.to_csv(output_file, index=False, header=False, float_format='%.10f')

print(f"\n✅ 结果已保存至：{output_file}")
print(f"   格式：第1行=表头 | 第2行=全局MIN | 第3行=全局MAX")