import pandas as pd

def insert_min_max_to_csv(input_path, output_path):
    """
    读取CSV，为每列计算最值并插入到表头下第二、三行
    :param input_path: 输入CSV路径
    :param output_path: 输出CSV路径
    """
    # 1. 读取CSV文件（第一行作为表头）
    df = pd.read_csv(input_path)
    
    # 2. 提取纯数据（从第二行开始的所有数据，即原数据）
    data = df.copy()
    
    # 3. 计算每一列的 最小值 和 最大值
    min_vals = data.min()  # 每列最小值（一行数据）
    max_vals = data.max()  # 每列最大值（一行数据）
    
    # 4. 构建新的DataFrame：表头 + 最小值行 + 最大值行 + 原始数据
    # 先创建空的DataFrame，按顺序拼接
    new_df = pd.DataFrame(columns=df.columns)
    # 插入最小值（第二行）
    new_df = pd.concat([new_df, min_vals.to_frame().T], ignore_index=True)
    # 插入最大值（第三行）
    new_df = pd.concat([new_df, max_vals.to_frame().T], ignore_index=True)
    # 插入原始数据（从第四行开始）
    new_df = pd.concat([new_df, data], ignore_index=True)
    
    # 5. 保存文件（不保存索引，编码兼容）
    new_df.to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f"✅ 处理完成！")
    print(f"📊 每列最小值已插入第二行，最大值已插入第三行")
    print(f"💾 文件已保存至：{output_path}")

# ===================== 配置区 =====================
INPUT_CSV = "train.csv"    # 替换为你的原始文件路径
OUTPUT_CSV = "train_out.csv"       # 替换为输出文件路径
# ===================================================

if __name__ == '__main__':
    insert_min_max_to_csv(INPUT_CSV, OUTPUT_CSV)