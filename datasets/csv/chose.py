import pandas as pd
import glob

# ==========================================
# 【配置区域】请修改这里
# ==========================================
# 1. 你的CSV文件路径（支持通配符，比如 "*.csv" 表示当前目录下所有csv）
csv_files = glob.glob("*.csv") 

# 2. 如果需要指定文件列表，用下面这行代替上面那行：
# csv_files = [
#     "05D_A0_046_102_train.csv",
#     "066D_A3_train.csv",
#     # 把你所有的文件名加在这里
# ]

# ==========================================
# 【核心处理逻辑】
# ==========================================
if not csv_files:
    print("❌ 错误：没有找到CSV文件！请检查文件路径。")
    exit()

print(f"✅ 找到 {len(csv_files)} 个CSV文件：")
for f in csv_files:
    print(f"   - {f}")

all_min_rows = []
all_max_rows = []
columns_ref = None

for file_path in csv_files:
    try:
        # 读取CSV，第1行作为表头
        df = pd.read_csv(file_path, header=0, nrows=3) # 只读取前3行（表头+min+max），速度快
        
        # 检查列数是否一致
        if columns_ref is None:
            columns_ref = df.columns.tolist()
        else:
            if df.columns.tolist() != columns_ref:
                print(f"⚠️  警告：文件 {file_path} 的列顺序与第一个文件不一致！")
                print(f"   参考列：{columns_ref}")
                print(f"   当前列：{df.columns.tolist()}")
        
        # 提取第2行（原文件第2行，DataFrame索引0）作为min行
        min_row = df.iloc[0].copy()
        # 提取第3行（原文件第3行，DataFrame索引1）作为max行
        max_row = df.iloc[1].copy()
        
        all_min_rows.append(min_row)
        all_max_rows.append(max_row)
        
        print(f"✅ 成功读取：{file_path}")
        
    except Exception as e:
        print(f"❌ 读取文件 {file_path} 失败：{e}")

if len(all_min_rows) == 0:
    print("❌ 错误：没有成功读取任何文件！")
    exit()

# ==========================================
# 【计算全局极值】
# ==========================================
# 将所有min行拼成一个DataFrame，然后按列取最小值
df_all_mins = pd.DataFrame(all_min_rows)
global_min_row = df_all_mins.min()

# 将所有max行拼成一个DataFrame，然后按列取最大值
df_all_maxs = pd.DataFrame(all_max_rows)
global_max_row = df_all_maxs.max()

# ==========================================
# 【输出结果】
# ==========================================
print("\n" + "="*80)
print("【最终结果】")
print("="*80)

print("\n1. 全局最小行（所有文件第2行的对应列最小值）：")
print(global_min_row.to_frame().T.to_string(index=False, float_format=lambda x: "%.10f" % x))

print("\n2. 全局最大行（所有文件第3行的对应列最大值）：")
print(global_max_row.to_frame().T.to_string(index=False, float_format=lambda x: "%.10f" % x))

# ==========================================
# 【可选：保存结果到新CSV】
# ==========================================
save_to_csv = True # 改成 False 就不保存
output_file = "global_min_max.csv"

if save_to_csv:
    # 构建结果DataFrame
    result_df = pd.DataFrame([global_min_row, global_max_row], index=["Global_Min", "Global_Max"])
    # 插入表头行
    result_df.columns = columns_ref
    # 保存
    result_df.to_csv(output_file, float_format='%.10f')
    print(f"\n✅ 结果已保存到：{output_file}")