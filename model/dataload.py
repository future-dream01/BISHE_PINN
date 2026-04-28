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
        
        # 1. 读取CSV前3行（先读出来，我们会重新计算并覆盖，保持接口兼容）
        header_df = pd.read_csv(self.csv_path, nrows=3, header=None)
        # 【修改】现在是13列，初始化长度为13
        self.input_min = header_df.iloc[1, :].values.astype(np.float32)  # [13]
        self.input_max = header_df.iloc[2, :].values.astype(np.float32)  # [13]
        
        # 2. 读取实际采样数据
        df = pd.read_csv(self.csv_path, skiprows=[1, 2])
        # 【修改】输入：前6列 (X,Y,Z,d,Ma,Pr)
        self.inputs = df.iloc[:, :6].values.astype(np.float32)   
        # 【修改】输出：后7列 (U,V,W,P,T,K,Omega)
        self.outputs = df.iloc[:, 6:].values.astype(np.float32) 
        
        # ===================== 第一步：处理输入归一化 (X, Y, Z, d, Ma, Pr) =====================
        # 【修改】处理前6列输入
        in_min_6 = self.inputs.min(axis=0)
        in_max_6 = self.inputs.max(axis=0)
        self.inputs_normalized = (self.inputs - in_min_6) / (in_max_6 - in_min_6 + 1e-8)
        
        # 【修改】更新 input_min/max 的前6位
        self.input_min[:6] = in_min_6
        self.input_max[:6] = in_max_6

        # ===================== 第二步：处理输出归一化 (U, V, W, P, T, K, Omega) =====================
        # 创建一个临时数组存归一化后的 Label
        outputs_norm = np.zeros_like(self.outputs)
        
        # --- 2.1 U, V, W, P, T (输出索引 0,1,2,3,4)：普通 Min-Max ---
        # 【修改】对应总索引 6,7,8,9,10
        for i in range(5):
            col_data = self.outputs[:, i]
            c_min = col_data.min()
            c_max = col_data.max()
            
            outputs_norm[:, i] = (col_data - c_min) / (c_max - c_min + 1e-8)
            
            # 【修改】更新 input_min/max 的对应位置 (总索引 6,7,8,9,10)
            self.input_min[6 + i] = c_min
            self.input_max[6 + i] = c_max
            
        # --- 2.2 K (输出索引 5，总索引 11)：对数归一化 ---
        k_raw = self.outputs[:, 5]
        ln_k = np.log(k_raw + 1e-12) # 加极小值防止 log(0)
        ln_k_min = ln_k.min()
        ln_k_max = ln_k.max()
        
        outputs_norm[:, 5] = (ln_k - ln_k_min) / (ln_k_max - ln_k_min + 1e-8)
        
        # 【修改】更新 input_min/max 的第 11 位
        self.input_min[11] = ln_k_min
        self.input_max[11] = ln_k_max
        
        # --- 2.3 Omega (输出索引 6，总索引 12)：对数归一化 ---
        omega_raw = self.outputs[:, 6]
        ln_omega = np.log(omega_raw + 1e-12)
        ln_omega_min = ln_omega.min()
        ln_omega_max = ln_omega.max()
        
        outputs_norm[:, 6] = (ln_omega - ln_omega_min) / (ln_omega_max - ln_omega_min + 1e-8)
        
        # 【修改】更新 input_min/max 的第 12 位
        self.input_min[12] = ln_omega_min
        self.input_max[12] = ln_omega_max
        
        # 保存归一化后的输出
        self.outputs_normalized = outputs_norm

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        input_tensor = torch.from_numpy(self.inputs_normalized[idx])
        output_tensor = torch.from_numpy(self.outputs_normalized[idx])
        return input_tensor, output_tensor


class Val_Dataset(Dataset):
    def __init__(self, csv_path, input_min=None, input_max=None):
        self.csv_path = os.path.abspath(csv_path)
        # 直接使用训练集的 input_min/max（已包含对数后的 K 和 Omega 极值）
        self.input_min = input_min
        self.input_max = input_max
        
        # 读取数据
        df = pd.read_csv(self.csv_path, skiprows=[1, 2])
        # 【修改】输入：前6列 (X,Y,Z,d,Ma,Pr)
        self.inputs = df.iloc[:, :6].values.astype(np.float32)
        # 【修改】输出：后7列 (U,V,W,P,T,K,Omega)
        self.outputs = df.iloc[:, 6:].values.astype(np.float32)
        
        # ===================== 验证集：复用训练集参数 =====================
        
        # 1. 输入归一化 (前6列)
        # 【修改】处理前6列
        in_min_6 = self.input_min[:6]
        in_max_6 = self.input_max[:6]
        self.inputs_normalized = (self.inputs - in_min_6) / (in_max_6 - in_min_6 + 1e-8)
        
        # 2. 输出归一化
        outputs_norm = np.zeros_like(self.outputs)
        
        # U, V, W, P, T (输出索引 0-4，总索引 6-10)归一化
        # 【修改】索引调整
        for i in range(5):
            c_min = self.input_min[6 + i]
            c_max = self.input_max[6 + i]
            outputs_norm[:, i] = (self.outputs[:, i] - c_min) / (c_max - c_min + 1e-8)
            
        # K 归一化 (输出索引 5，总索引 11)
        k_raw = self.outputs[:, 5]
        ln_k = np.log(k_raw + 1e-12)
        ln_k_min = self.input_min[11]
        ln_k_max = self.input_max[11]
        outputs_norm[:, 5] = (ln_k - ln_k_min) / (ln_k_max - ln_k_min + 1e-8)
        
        # Omega归一化 (输出索引 6，总索引 12)
        omega_raw = self.outputs[:, 6]
        ln_omega = np.log(omega_raw + 1e-12)
        ln_omega_min = self.input_min[12]
        ln_omega_max = self.input_max[12]
        outputs_norm[:, 6] = (ln_omega - ln_omega_min) / (ln_omega_max - ln_omega_min + 1e-8)
        
        self.outputs_normalized = outputs_norm

        print(f"\n✅ 验证集加载完成")
        print(f"   实际采样数据点数: {len(self.inputs)}")

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        input_tensor = torch.from_numpy(self.inputs_normalized[idx])
        output_tensor = torch.from_numpy(self.outputs_normalized[idx])
        return input_tensor, output_tensor


def data_prepare(batchsize, train_csv_path=None, val_csv_path=None):
    if train_csv_path is None:
        train_csv_path = os.path.join(project_root, "datasets/066D_A3_train.csv")
    if val_csv_path is None:
        val_csv_path = os.path.join(project_root, "datasets/066D_A3_val.csv")
    
    train_set = Train_Dataset(train_csv_path)
    val_set = Val_Dataset(val_csv_path, train_set.input_min, train_set.input_max)

    train_dataloader = DataLoader(train_set, batch_size=batchsize, shuffle=True)
    val_dataloader = DataLoader(val_set, batch_size=batchsize, shuffle=False)
    
    # 完全保持原有返回值，无需修改任何传参！
    return train_dataloader, val_dataloader, train_set.input_min, train_set.input_max