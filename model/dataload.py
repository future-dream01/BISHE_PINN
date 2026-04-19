# 训练数据集准备
import torch 
import numpy as np
import pandas as pd  # 新增：用于读取CSV
from torch.utils.data import Dataset, DataLoader
import os, sys

# 获取项目根目录的绝对路径
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)

# 自定义数据集创建器（适配CSV）
class Train_Dataset(Dataset):
    def __init__(self, csv_path):
        self.csv_path = os.path.abspath(csv_path)
        
        # ==========================================
        # 【关键修改1】读取CSV的前3行获取【全部11列】全局极值
        # ==========================================
        # 1. 先读取前3行：第0行=表头，第1行=全局MIN，第2行=全局MAX
        header_df = pd.read_csv(self.csv_path, nrows=3, header=None)
        
        # 2. 提取【全部11列】的MIN和MAX（不再只取前4列）
        # 第1行（索引1）是MIN，取全部列
        self.input_min = header_df.iloc[1, :].values.astype(np.float32)
        # 第2行（索引2）是MAX，取全部列
        self.input_max = header_df.iloc[2, :].values.astype(np.float32)
        
        # ==========================================
        # 【关键修改2】从第4行（索引3）开始读取实际采样数据
        # ==========================================
        df = pd.read_csv(self.csv_path, skiprows=[1, 2]) # 跳过MIN/MAX行，保留表头
        
        # 分离输入和输出：前4列是输入，之后7列是输出
        self.inputs = df.iloc[:, :4].values.astype(np.float32)  # 输入：前4列
        self.outputs = df.iloc[:, 4:].values.astype(np.float32) # 输出：后7列
        
        # ==========================================
        # 【关键修改3】归一化依然只对【前4列输入】做
        # ==========================================
        # 切片取前4列的极值用于输入归一化
        in_min_4 = self.input_min[:4]
        in_max_4 = self.input_max[:4]
        self.inputs_normalized = (self.inputs - in_min_4) / (in_max_4 - in_min_4 + 1e-8)

        # 打印日志确认
        print(f"\n✅ 训练集加载完成：{self.csv_path}")
        print(f"   读取到的【全部11列】MIN: {self.input_min}")
        print(f"   读取到的【全部11列】MAX: {self.input_max}")
        print(f"   实际采样数据点数: {len(self.inputs)}")

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        # 直接返回归一化后的输入和原始输出
        input_tensor = torch.from_numpy(self.inputs_normalized[idx])
        output_tensor = torch.from_numpy(self.outputs[idx])
        return input_tensor, output_tensor


class Val_Dataset(Dataset):
    def __init__(self, csv_path, input_min=None, input_max=None):
        """
        csv_path: 验证集CSV路径
        input_min: (可选) 训练集的【全部11列】最小值
        input_max: (可选) 训练集的【全部11列】最大值
        """
        self.csv_path = os.path.abspath(csv_path)
        
        # ==========================================
        # 【关键修改】验证集也读取【全部11列】全局极值
        # ==========================================
        # 1. 先读取前3行
        header_df = pd.read_csv(self.csv_path, nrows=3, header=None)
        
        # 2. 如果外部传入了min/max（来自训练集的11列），优先用外部的；否则从自己CSV读
        if input_min is not None:
            self.input_min = input_min
            self.input_max = input_max
            print(f"\n⚠️  验证集：使用传入的训练集【全部11列】归一化参数")
        else:
            self.input_min = header_df.iloc[1, :].values.astype(np.float32)
            self.input_max = header_df.iloc[2, :].values.astype(np.float32)
            print(f"\n✅ 验证集：从自身CSV读取【全部11列】归一化参数")
        
        # 3. 从第4行开始读取实际数据
        df = pd.read_csv(self.csv_path, skiprows=[1, 2])
        
        # 分离输入和输出
        self.inputs = df.iloc[:, :4].values.astype(np.float32)
        self.outputs = df.iloc[:, 4:].values.astype(np.float32)
        
        # ==========================================
        # 【关键修改】归一化依然只对【前4列输入】做
        # ==========================================
        in_min_4 = self.input_min[:4]
        in_max_4 = self.input_max[:4]
        self.inputs_normalized = (self.inputs - in_min_4) / (in_max_4 - in_min_4 + 1e-8)

        print(f"   验证集加载完成：{self.csv_path}")
        print(f"   使用的【全部11列】MIN: {self.input_min}")
        print(f"   使用的【全部11列】MAX: {self.input_max}")
        print(f"   实际采样数据点数: {len(self.inputs)}")

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        input_tensor = torch.from_numpy(self.inputs_normalized[idx])
        output_tensor = torch.from_numpy(self.outputs[idx])
        return input_tensor, output_tensor


def data_prepare(batchsize, train_csv_path=None, val_csv_path=None):
    # 使用默认路径（你也可以在调用时传入自定义路径）
    if train_csv_path is None:
        train_csv_path = os.path.join(project_root, "datasets/066D_A3_train.csv") # 请修改为你的训练集CSV路径
    if val_csv_path is None:
        val_csv_path = os.path.join(project_root, "datasets/066D_A3_val.csv")   # 请修改为你的验证集CSV路径
    
    # 初始化训练集（自动从CSV第2/3行读取【全部11列】归一化参数）
    train_set = Train_Dataset(train_csv_path)
    
    # 初始化验证集（传入训练集的【全部11列】参数）
    val_set = Val_Dataset(val_csv_path, train_set.input_min, train_set.input_max)

    # 创建DataLoader
    train_dataloader = DataLoader(train_set, batch_size=batchsize, shuffle=True)
    val_dataloader = DataLoader(val_set, batch_size=batchsize, shuffle=False) # 验证集通常不shuffle
    
    # 返回dataloader以及【全部11列】的归一化参数（方便后续模型反归一化输出7个变量）
    return train_dataloader, val_dataloader, train_set.input_min, train_set.input_max