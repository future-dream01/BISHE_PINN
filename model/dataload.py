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
        
        # 读取CSV文件（跳过表头，第一行是列名）
        df = pd.read_csv(self.csv_path)
        
        # 分离输入和输出：前4列是输入，之后7列是输出
        self.inputs = df.iloc[:, :4].values.astype(np.float32)  # 输入：前4列
        self.outputs = df.iloc[:, 4:].values.astype(np.float32) # 输出：后7列
        
        # 【关键】计算训练集输入的归一化参数（min和max）
        self.input_min = self.inputs.min(axis=0)
        self.input_max = self.inputs.max(axis=0)
        
        # 0~1 归一化公式
        self.inputs_normalized = (self.inputs - self.input_min) / (self.input_max - self.input_min + 1e-8)

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        # 直接返回归一化后的输入和原始输出（输出不需要归一化，或根据你需求调整）
        input_tensor = torch.from_numpy(self.inputs_normalized[idx])
        output_tensor = torch.from_numpy(self.outputs[idx])
        return input_tensor, output_tensor


class Val_Dataset(Dataset):
    def __init__(self, csv_path, input_min, input_max):
        """
        csv_path: 验证集CSV路径
        input_min: 训练集计算得到的输入最小值
        input_max: 训练集计算得到的输入最大值
        """
        self.csv_path = os.path.abspath(csv_path)
        self.input_min = input_min
        self.input_max = input_max
        
        # 读取CSV文件
        df = pd.read_csv(self.csv_path)
        
        # 分离输入和输出
        self.inputs = df.iloc[:, :4].values.astype(np.float32)
        self.outputs = df.iloc[:, 4:].values.astype(np.float32)
        
        # 0~1 归一化公式
        self.inputs_normalized = (self.inputs - self.input_min) / (self.input_max - self.input_min + 1e-8)

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        input_tensor = torch.from_numpy(self.inputs_normalized[idx])
        output_tensor = torch.from_numpy(self.outputs[idx])
        return input_tensor, output_tensor


def data_prepare(batchsize, train_csv_path=None, val_csv_path=None):
    # 使用默认路径（你也可以在调用时传入自定义路径）
    if train_csv_path is None:
        train_csv_path = os.path.join(project_root, "datasets/066D_A3_hermites08_banmo_042_101_train.csv") # 请修改为你的训练集CSV路径
    if val_csv_path is None:
        val_csv_path = os.path.join(project_root, "datasets/066D_A3_hermites08_banmo_042_101_train.csv")   # 请修改为你的验证集CSV路径
    
    # 初始化训练集（自动计算归一化参数）
    train_set = Train_Dataset(train_csv_path)
    # 初始化验证集（传入训练集的归一化参数）
    val_set = Val_Dataset(val_csv_path, train_set.input_min, train_set.input_max)

    # 创建DataLoader
    train_dataloader = DataLoader(train_set, batch_size=batchsize, shuffle=True)
    val_dataloader = DataLoader(val_set, batch_size=batchsize, shuffle=True)
    
    # 返回dataloader以及训练集的归一化参数（方便后续推理时使用）
    return train_dataloader, val_dataloader, train_set.input_min, train_set.input_max