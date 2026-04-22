# 训练数据集准备
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
        
        # 1. 读取CSV前3行：原始全局MIN/MAX (11列)
        header_df = pd.read_csv(self.csv_path, nrows=3, header=None)
        self.input_min = header_df.iloc[1, :].values.astype(np.float32)  # [11]
        self.input_max = header_df.iloc[2, :].values.astype(np.float32)  # [11]
        
        # 2. 读取实际采样数据
        df = pd.read_csv(self.csv_path, skiprows=[1, 2])
        self.inputs = df.iloc[:, :4].values.astype(np.float32)   # 输入：X,Y,Z,壁面距离
        self.outputs = df.iloc[:, 4:].values.astype(np.float32) # 输出：U,V,W,P,T,K,Omega
        
        # ===================== 核心：Omega 对数归一化 =====================
        OMEGA_COL_IDX = 10  # 总列11列中，Omega在索引10
        # 1. 提取原始无量纲Omega
        omega_raw = self.outputs[:, 6]  # 输出第7列 = Omega
        # 2. 对数变换（压缩量级）
        ln_omega = np.log(omega_raw)
        # 3. 计算对数后的极值
        ln_omega_min = np.min(ln_omega).astype(np.float32)
        ln_omega_max = np.max(ln_omega).astype(np.float32)
        # 4. ✅ 直接覆盖原input_min/max中Omega的位置
        self.input_min[OMEGA_COL_IDX] = ln_omega_min
        self.input_max[OMEGA_COL_IDX] = ln_omega_max
        # 5. 0-1归一化
        omega_norm = (ln_omega - ln_omega_min) / (ln_omega_max - ln_omega_min + 1e-8)
        # 6. 替换输出中的Omega
        self.outputs[:, 6] = omega_norm

        # 输入归一化（不变）
        in_min_4 = self.input_min[:4]
        in_max_4 = self.input_max[:4]
        self.inputs_normalized = (self.inputs - in_min_4) / (in_max_4 - in_min_4 + 1e-8)

        print(f"\n✅ 训练集加载完成")
        print(f"   已覆盖Omega极值：ln_min={self.input_min[10]:.4f}, ln_max={self.input_max[10]:.4f}")

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        input_tensor = torch.from_numpy(self.inputs_normalized[idx])
        output_tensor = torch.from_numpy(self.outputs[idx])
        return input_tensor, output_tensor


class Val_Dataset(Dataset):
    def __init__(self, csv_path, input_min=None, input_max=None):
        self.csv_path = os.path.abspath(csv_path)
        # 直接使用训练集的input_min/max（已包含对数Omega极值）
        self.input_min = input_min
        self.input_max = input_max
        
        # 读取数据
        df = pd.read_csv(self.csv_path, skiprows=[1, 2])
        self.inputs = df.iloc[:, :4].values.astype(np.float32)
        self.outputs = df.iloc[:, 4:].values.astype(np.float32)
        
        # ===================== 验证集：复用训练集对数参数 =====================
        OMEGA_COL_IDX = 10
        ln_omega_min = self.input_min[OMEGA_COL_IDX]
        ln_omega_max = self.input_max[OMEGA_COL_IDX]
        
        omega_raw = self.outputs[:, 6]
        ln_omega = np.log(omega_raw)
        omega_norm = (ln_omega - ln_omega_min) / (ln_omega_max - ln_omega_min + 1e-8)
        self.outputs[:, 6] = omega_norm

        # 输入归一化
        in_min_4 = self.input_min[:4]
        in_max_4 = self.input_max[:4]
        self.inputs_normalized = (self.inputs - in_min_4) / (in_max_4 - in_min_4 + 1e-8)

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        input_tensor = torch.from_numpy(self.inputs_normalized[idx])
        output_tensor = torch.from_numpy(self.outputs[idx])
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