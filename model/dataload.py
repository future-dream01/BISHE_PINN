import torch 
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import os, sys

# 获取项目根目录的绝对路径
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)

class Train_Dataset(Dataset):
    def __init__(self, csv_path):
        self.csv_path = os.path.abspath(csv_path)
        
        # 1. 读取CSV前3行（第1行列名，第2行raw_min，第3行raw_max）
        header_df = pd.read_csv(self.csv_path, nrows=3, header=None)
        
        # 🔥 【核心1】提取CSV第2、3行的原始值（raw_min和raw_max）
        raw_min_csv = header_df.iloc[1, :13].values.astype(np.float64)  # 用float64防止精度损失
        raw_max_csv = header_df.iloc[2, :13].values.astype(np.float64)
        
        # 初始化input_min和input_max
        self.input_min = np.zeros(13, dtype=np.float32)
        self.input_max = np.zeros(13, dtype=np.float32)
        
        # 🔥 【核心2】处理前11列（x,y,z,d,Ma,Pr,U,V,W,P,T）：直接用CSV原始值
        print(f"\n" + "="*150)
        print(f"🔥 开始处理CSV极值：")
        for i in range(11):
            self.input_min[i] = raw_min_csv[i]
            self.input_max[i] = raw_max_csv[i]
            print(f"   列{i}：直接使用CSV原始值 | min={self.input_min[i]:.10f}, max={self.input_max[i]:.10f}")
        
        # 🔥 【核心3】处理第12列（K，索引11）：CSV读取原始值 → 代码取对数
        raw_k_min = raw_min_csv[11]
        raw_k_max = raw_max_csv[11]
        self.input_min[11] = np.log(raw_k_min + 1e-12)  # 代码里取对数
        self.input_max[11] = np.log(raw_k_max + 1e-12)
        print(f"   列11(K)：CSV原始值取对数 | raw_min={raw_k_min:.10f} → ln_min={self.input_min[11]:.10f}")
        print(f"                           | raw_max={raw_k_max:.10f} → ln_max={self.input_max[11]:.10f}")
        
        # 🔥 【核心4】处理第13列（Omega，索引12）：CSV读取原始值 → 代码取对数
        raw_omega_min = raw_min_csv[12]
        raw_omega_max = raw_max_csv[12]
        self.input_min[12] = np.log(raw_omega_min + 1e-12)  # 代码里取对数
        self.input_max[12] = np.log(raw_omega_max + 1e-12)
        print(f"   列12(O)：CSV原始值取对数 | raw_min={raw_omega_min:.10f} → ln_min={self.input_min[12]:.10f}")
        print(f"                           | raw_max={raw_omega_max:.10f} → ln_max={self.input_max[12]:.10f}")
        
        # 2. 读取实际采样数据
        df = pd.read_csv(self.csv_path, skiprows=[1, 2])
        self.inputs = df.iloc[:, :6].values.astype(np.float32)  # 输入：前6列
        self.outputs = df.iloc[:, 6:].values.astype(np.float32) # 输出：后7列
        
        # ===================== 第一步：处理输入归一化（前6列，直接用CSV极值） =====================
        in_min_6 = self.input_min[:6]
        in_max_6 = self.input_max[:6]
        self.inputs_normalized = (self.inputs - in_min_6) / (in_max_6 - in_min_6 + 1e-8)

        # ===================== 第二步：处理输出归一化（后7列） =====================
        outputs_norm = np.zeros_like(self.outputs)
        
        # --- 2.1 输出前5列：U, V, W, P, T（索引0-4）→ 直接用CSV极值 ---
        for i in range(5):
            col_data = self.outputs[:, i]
            c_min = self.input_min[6 + i]
            c_max = self.input_max[6 + i]
            outputs_norm[:, i] = (col_data - c_min) / (c_max - c_min + 1e-8)
            
        # --- 2.2 输出第6列：K（索引5）→ 先取对数，再用CSV的对数极值 ---
        k_raw = self.outputs[:, 5]
        ln_k = np.log(k_raw + 1e-12)  # 先取对数
        ln_k_min = self.input_min[11]    # 用之前算好的对数极值
        ln_k_max = self.input_max[11]
        outputs_norm[:, 5] = (ln_k - ln_k_min) / (ln_k_max - ln_k_min + 1e-8)
        
        # --- 2.3 输出第7列：Omega（索引6）→ 先取对数，再用CSV的对数极值 ---
        omega_raw = self.outputs[:, 6]
        ln_omega = np.log(omega_raw + 1e-12)  # 先取对数
        ln_omega_min = self.input_min[12]       # 用之前算好的对数极值
        ln_omega_max = self.input_max[12]
        outputs_norm[:, 6] = (ln_omega - ln_omega_min) / (ln_omega_max - ln_omega_min + 1e-8)
        
        self.outputs_normalized = outputs_norm

        # ===================== 最终确认打印 =====================
        print(f"\n🔥 最终 input_min（用于训练）：")
        print(np.array2string(self.input_min, separator=', ', max_line_width=200))
        print(f"\n🔥 最终 input_max（用于训练）：")
        print(np.array2string(self.input_max, separator=', ', max_line_width=200))
        print(f"="*150 + "\n")

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        input_tensor = torch.from_numpy(self.inputs_normalized[idx])
        output_tensor = torch.from_numpy(self.outputs_normalized[idx])
        return input_tensor, output_tensor


# Val_Dataset：100%复用训练集的input_min/input_max
class Val_Dataset(Dataset):
    def __init__(self, csv_path, input_min=None, input_max=None):
        self.csv_path = os.path.abspath(csv_path)
        self.input_min = input_min
        self.input_max = input_max
        
        # 读取数据
        df = pd.read_csv(self.csv_path, skiprows=[1, 2])
        self.inputs = df.iloc[:, :6].values.astype(np.float32)
        self.outputs = df.iloc[:, 6:].values.astype(np.float32)
        
        # 输入归一化
        in_min_6 = self.input_min[:6]
        in_max_6 = self.input_max[:6]
        self.inputs_normalized = (self.inputs - in_min_6) / (in_max_6 - in_min_6 + 1e-8)
        
        # 输出归一化
        outputs_norm = np.zeros_like(self.outputs)
        
        # U,V,W,P,T
        for i in range(5):
            c_min = self.input_min[6 + i]
            c_max = self.input_max[6 + i]
            outputs_norm[:, i] = (self.outputs[:, i] - c_min) / (c_max - c_min + 1e-8)
            
        # K
        k_raw = self.outputs[:, 5]
        ln_k = np.log(k_raw + 1e-12)
        ln_k_min = self.input_min[11]
        ln_k_max = self.input_max[11]
        outputs_norm[:, 5] = (ln_k - ln_k_min) / (ln_k_max - ln_k_min + 1e-8)
        
        # Omega
        omega_raw = self.outputs[:, 6]
        ln_omega = np.log(omega_raw + 1e-12)
        ln_omega_min = self.input_min[12]
        ln_omega_max = self.input_max[12]
        outputs_norm[:, 6] = (ln_omega - ln_omega_min) / (ln_omega_max - ln_omega_min + 1e-8)
        
        self.outputs_normalized = outputs_norm

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        input_tensor = torch.from_numpy(self.inputs_normalized[idx])
        output_tensor = torch.from_numpy(self.outputs_normalized[idx])
        return input_tensor, output_tensor


def data_prepare(batchsize, train_csv_path=None, val_csv_path=None):
    if train_csv_path is None:
        train_csv_path = os.path.join(project_root, "datasets/csv/train.csv")
    if val_csv_path is None:
        val_csv_path = os.path.join(project_root, "datasets/csv/val2.csv")
    
    train_set = Train_Dataset(train_csv_path)
    val_set = Val_Dataset(val_csv_path, train_set.input_min, train_set.input_max)

    train_dataloader = DataLoader(train_set, batch_size=batchsize, shuffle=True)
    val_dataloader = DataLoader(val_set, batch_size=batchsize, shuffle=False)
    
    return train_dataloader, val_dataloader, train_set.input_min, train_set.input_max