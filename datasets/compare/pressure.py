import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# ==========================================
# 【参数设置区域】
# ==========================================
# --- 文件路径设置（请修改为你的实际文件路径）---
# 模型输出的无量纲压力曲线
model_upper_path = "up_cfd.csv"    # 模型上壁面
model_lower_path = "down_cfd.csv"    # 模型下壁面

# Fluent提取的无量纲压力真值点（空格分隔）
truth_upper_path = "up_true.csv"       # 真值上壁面
truth_lower_path = "down_true.csv"       # 真值下壁面

# --- 图形样式参数 ---
z_section = 0.0002              # 当前处理的Z截面坐标
x_inlet = 0.297183              # 入口X坐标
x_outlet = 0.515712             # 出口X坐标
P_ref = 61143                   # 压力无量纲化参考值
LINE_WIDTH = 2.5                # 模型曲线宽度
MARKER_SIZE = 8                 # 真值点方框大小
MARKER_EDGE_WIDTH = 1.5         # 真值点边框宽度
DPI = 150                       # 图片分辨率

# --- 颜色配置（与之前代码保持一致）---
COLOR_UPPER = '#d62728'         # 上壁面：红色
COLOR_LOWER = '#1f77b4'         # 下壁面：蓝色

# ==========================================
# 【全局字体设置（强制Times New Roman）】
# ==========================================
plt.rcParams.update({
    'font.family': 'Times New Roman',
    'font.size': 12,
    'axes.titlesize': 16,
    'axes.labelsize': 14,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 12,
    'mathtext.fontset': 'stix'  # 数学公式也使用Times New Roman风格
})

# ==========================================
# 【核心函数区域】
# ==========================================

def read_pressure_csv(file_path, name):
    """读取压力CSV文件，自动处理字符串类型数据并强制转换为数值"""
    if not os.path.exists(file_path):
        print(f"❌ 错误：文件不存在 {file_path}")
        return None, None
    
    try:
        # 自动检测任意分隔符，清理列名前后空格
        df = pd.read_csv(
            file_path, 
            sep=None, 
            engine='python', 
            skipinitialspace=True,
            dtype=str  # 先全部读为字符串，避免自动转换错误
        )
        
        # 清理列名前后空格
        df.columns = df.columns.str.strip()
        
        print(f"📄 {name} 文件列名：{df.columns.tolist()}")
        print(f"📄 {name} 原始行数：{len(df)}")
        
        # 兼容两种列名格式
        if "X(m)" in df.columns and "Pressure(nondim)" in df.columns:
            x_col = "X(m)"
            p_col = "Pressure(nondim)"
        elif "X(m)" in df.columns and "Pressure(Pa)" in df.columns:
            x_col = "X(m)"
            p_col = "Pressure(Pa)"
            is_dimensional = True
        else:
            # 支持无表头的两列数据
            if len(df.columns) == 2:
                print(f"⚠️  警告：{name}文件未检测到标准列名，默认第一列为X(m)，第二列为Pressure(nondim)")
                x_col = df.columns[0]
                p_col = df.columns[1]
                is_dimensional = False
            else:
                print(f"❌ 错误：{name}文件列数不正确，需要2列数据（X, Pressure）")
                return None, None
        
        # 强制转换为数值类型，无法转换的变为NaN
        df[x_col] = pd.to_numeric(df[x_col], errors='coerce')
        df[p_col] = pd.to_numeric(df[p_col], errors='coerce')
        
        # 提取数据
        x = df[x_col].values
        p = df[p_col].values
        
        # 有量纲转无量纲
        if 'is_dimensional' in locals() and is_dimensional:
            p = p / P_ref
            print(f"⚠️  警告：{name}文件为有量纲压力，已自动除以{P_ref}转换为无量纲")
        
        # 过滤所有NaN和Inf无效值
        valid_mask = ~np.isnan(x) & ~np.isnan(p) & ~np.isinf(x) & ~np.isinf(p)
        x_clean = x[valid_mask]
        p_clean = p[valid_mask]
        
        invalid_count = len(x) - len(x_clean)
        if invalid_count > 0:
            print(f"⚠️  警告：{name}文件中发现 {invalid_count} 个无效值（非数值/NaN/Inf），已自动过滤")
            # 打印前5个无效行方便调试
            invalid_rows = df[~valid_mask].head()
            if not invalid_rows.empty:
                print(f"📄 无效行示例：\n{invalid_rows}")
        
        if len(x_clean) == 0:
            print(f"❌ 错误：{name}文件过滤后无有效数据！")
            return None, None
        
        print(f"✅ 成功读取 {name}：{len(x_clean)} 个有效点")
        return x_clean, p_clean
    
    except Exception as e:
        print(f"❌ 读取 {name} 文件失败：{str(e)}")
        import traceback
        traceback.print_exc()  # 打印详细错误信息方便调试
        return None, None

def plot_pressure_comparison(
    model_upper_x, model_upper_p,
    model_lower_x, model_lower_p,
    truth_upper_x, truth_upper_p,
    truth_lower_x, truth_lower_p,
    z_section, x_inlet, x_outlet
):
    """绘制模型预测与真值的对比图（全英文+Times New Roman字体）"""
    print("\n正在绘制压力对比图...")
    
    fig, ax = plt.subplots(figsize=(12, 6), dpi=DPI)
    
    # ===================== 绘制模型曲线 =====================
    ax.plot(
        model_upper_x, model_upper_p,
        color=COLOR_UPPER,
        linewidth=LINE_WIDTH,
        linestyle='-',
        label='Upper Wall (Model Prediction)'
    )
    
    ax.plot(
        model_lower_x, model_lower_p,
        color=COLOR_LOWER,
        linewidth=LINE_WIDTH,
        linestyle='-',
        label='Lower Wall (Model Prediction)'
    )
    
    # ===================== 绘制真值点（方框标记） =====================
    ax.scatter(
        truth_upper_x, truth_upper_p,
        marker='s',
        s=MARKER_SIZE**2,
        facecolor=COLOR_UPPER,
        edgecolor='black',
        linewidth=MARKER_EDGE_WIDTH,
        zorder=5,
        label='Upper Wall (Fluent Ground Truth)'
    )
    
    ax.scatter(
        truth_lower_x, truth_lower_p,
        marker='s',
        s=MARKER_SIZE**2,
        facecolor=COLOR_LOWER,
        edgecolor='black',
        linewidth=MARKER_EDGE_WIDTH,
        zorder=5,
        label='Lower Wall (Fluent Ground Truth)'
    )
    
    # ===================== 图形设置（全英文） =====================
    ax.set_xlabel('X (m)', fontweight='normal')
    ax.set_ylabel(f'Static Pressure (P / {P_ref})', fontweight='normal')
    ax.set_title(
        f'Wall Pressure Comparison at Z={z_section:.4f}m\nModel Prediction vs Fluent Ground Truth',
        fontweight='bold',
        pad=20
    )
    
    ax.set_xlim(x_inlet, x_outlet)
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.legend(loc='upper right', framealpha=1.0)
    
    # 安全计算Y轴范围
    all_pressures = np.concatenate([
        model_upper_p, model_lower_p,
        truth_upper_p, truth_lower_p
    ])
    
    all_pressures_clean = all_pressures[~np.isnan(all_pressures) & ~np.isinf(all_pressures)]
    
    if len(all_pressures_clean) == 0:
        print("⚠️  警告：所有压力数据均为无效值，使用默认Y轴范围 [0.5, 1.5]")
        y_min, y_max = 0.5, 1.5
    else:
        y_min = np.min(all_pressures_clean) * 0.95
        y_max = np.max(all_pressures_clean) * 1.05
        print(f"✅ 自动计算Y轴范围：{y_min:.6f} ~ {y_max:.6f}")
    
    ax.set_ylim(y_min, y_max)
    
    # 保存图片（带Z坐标）
    z_str = f"{z_section:.4f}".replace('.', '_').replace('-', 'n')
    output_file = f"pressure_comparison_z={z_str}.png"
    plt.savefig(output_file, bbox_inches='tight', dpi=DPI)
    plt.show()
    
    print(f"✅ 压力对比图已保存为：{output_file}")
    return output_file

# ==========================================
# 【主程序】
# ==========================================
def main():
    print("="*70)
    print("壁面压力对比：模型预测 vs Fluent真值")
    print("="*70)
    
    # 1. 读取所有数据文件
    print("\n📥 正在读取数据文件...")
    model_upper_x, model_upper_p = read_pressure_csv(model_upper_path, "模型上壁面")
    model_lower_x, model_lower_p = read_pressure_csv(model_lower_path, "模型下壁面")
    truth_upper_x, truth_upper_p = read_pressure_csv(truth_upper_path, "真值上壁面")
    truth_lower_x, truth_lower_p = read_pressure_csv(truth_lower_path, "真值下壁面")
    
    # 检查数据是否完整
    data_list = [model_upper_x, model_lower_x, truth_upper_x, truth_lower_x]
    if any(data is None for data in data_list):
        print("\n❌ 数据读取失败，程序终止")
        return
    
    # 2. 绘制对比图
    output_file = plot_pressure_comparison(
        model_upper_x, model_upper_p,
        model_lower_x, model_lower_p,
        truth_upper_x, truth_upper_p,
        truth_lower_x, truth_lower_p,
        z_section, x_inlet, x_outlet
    )
    
    # 3. 打印统计信息
    print("\n" + "="*70)
    print("📊 对比统计信息")
    print("="*70)
    print(f"上壁面：模型 {len(model_upper_x)} 个点 | 真值 {len(truth_upper_x)} 个点")
    print(f"下壁面：模型 {len(model_lower_x)} 个点 | 真值 {len(truth_lower_x)} 个点")
    print(f"✅ 对比图已保存：{output_file}")
    print("="*70)

if __name__ == "__main__":
    main()