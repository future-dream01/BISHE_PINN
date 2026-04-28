import os, sys
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), './..'))
sys.path.append(project_root)
from model import PINN,data_prepare, train_loss_TOTAL,val_loss_TOTAL,psnr,ssim,allgraph,hard_consrain,denormalize_for_pde
import torch.optim as optim
from loguru import logger
from datetime import datetime
import torch
from torch.cuda.amp import autocast, GradScaler
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import random

# 无量纲参数设定
L=0.095    # 特征长度
T0=249.15  # 来流静温
P0=47181   # 来流静压

# 训练超参数设定
EPOCHES = 2000    # 轮次数
BATCHSIZE = 1024    # 批次数
PDEloss_start_epoch=500  # 开始加入PDE残差损失的轮次
train_nan_loss=val_nan_loss=0   # 一轮中出现异常损失值的批次数量
LOAD_CP=False    # 是否需要加载之前的检查点
CP_PATH= f'{project_root}/outputs/weights/04-24_21-04/416weights.pth'    # 检查点权重文件绝对路径
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")   # 计算设备
current_datetime = datetime.now().strftime("%m-%d_%H-%M")               # 当前时间
log_file_path=f'{project_root}/outputs/训练与性能情况/{current_datetime}/损失日志.log'  # 训练日志文件的绝对路径
os.makedirs(f'{project_root}/outputs/训练与性能情况/{current_datetime}', exist_ok=True)    # 创建训练与性能情况文件夹
os.makedirs(f'{project_root}/outputs/weights/{current_datetime}', exist_ok=True)         # 创建权重文件夹
logger.add(log_file_path,rotation="50000 MB",level="INFO")               #创建日志文件

def train():
    # 训练前备工作
    global train_nan_loss,val_nan_loss      
    train_dataloader, val_dataloader ,data_min ,data_max= data_prepare(BATCHSIZE)        # 创建数据加载器对象
    data_min = torch.tensor(data_min, dtype=torch.float32).to(device)  # numpy张量转torch张量
    data_max = torch.tensor(data_max, dtype=torch.float32).to(device)
    M = PINN()              # 创建模型对象
    M.to(device)                                # 将模型转移到计算设备上
    optimizer_M = optim.Adam(M.parameters(), lr=0.001)    # 创建梯度优化器
    start_epoch=1                               # 开始训练的轮次数，默认是1，如果从断点开始会更新为断点的轮次数
    train_losses = []                           # 训练集损失
    res_cont_epoches=[]                         # 训练集连续方程残差
    res_mx_epoches=[]                           # 训练集x方向动量方程残差
    res_my_epoches=[]                           # 训练集y方向动量方程残差
    res_mz_epoches=[]                           # 训练集z方向动量方程残差
    res_energy_epoches=[]                       # 训练集能量方程残差
    res_k_epoches=[]                            # 训练集湍流动能输运方程残差
    res_omega_epoches=[]                        # 训练集湍流比耗散率输方程残差
    val_losses=[]                               # 验证集损失

    # 是否加载先前的检查点文件
    if LOAD_CP:                                
        logger.info(f"正在加载模型文件")
        M,optimizer_M,start_epoch,train_losses,res_cont_epoches,res_mx_epoches,res_my_epoches,res_mz_epoches,res_energy_epoches,res_k_epoches,res_omega_epoches,val_losses=load_checkpoint(M,optimizer_M,CP_PATH)
        min_val_loss=val_losses[start_epoch-1]
        d_epoch_num=start_epoch
        start_epoch+=1
        logger.info(f"模型文件加载完成,最低的验证集损失:{min_val_loss}")

    # 开启混合精度
    #scaler=GradScaler(device)
    logger.info("开始训练")

    # 开始训练
    for epoch in range(int(start_epoch), EPOCHES + 1):

        ###################### 训练集训练 #########################
        M.train()                       # 将模型转换为训练模式
        M.requires_grad_(True)          # 开启梯度，这是为了损失函数计算能用自动微分
        train_loss_epoch=0
        res_cont_epoch=0
        res_mx_epoch=0
        res_my_epoch=0
        res_mz_epoch=0
        res_energy_epoch=0
        res_k_epoch=0
        res_omega_epoch=0
        train_batches = 0               # 批次计数
        logger.info(f"第{epoch}轮训练开始，训练集开始训练")
        for input, label in train_dataloader:
            # 每次迭代生成新的输入和输出
            input, label = input.to(device).requires_grad_(True), label.to(device)
            logger.info(f"输入数据检查:")
            logger.info(f"   输入的归一化x坐标范围: {input[:, 0].min().item():.6f} ~ {input[:, 0].max().item():.6f}")
            logger.info(f"   输入的归一化y坐标范围: {input[:, 1].min().item():.6f} ~ {input[:, 1].max().item():.6f}")
            logger.info(f"   输入的归一化z坐标范围: {input[:, 2].min().item():.6f} ~ {input[:, 2].max().item():.6f}")
            logger.info(f"   输入的归一化壁面距离范围: {input[:, 3].min().item():.6f} ~ {input[:, 3].max().item():.6f}")
            logger.info(f"   输入的归一化马赫数: {input[:, 4].min().item():.6f} ~ {input[:, 3].max().item():.6f}")
            logger.info(f"   输入的归一化压比: {input[:, 5].min().item():.6f} ~ {input[:, 3].max().item():.6f}")
            optimizer_M.zero_grad()  # 梯度归零
            # 前向传播
            #with autocast():
            output_raw = M(input)
            # input_sym=input.clone()                              # 构造对称输入
            # input_sym[:,2:3]=-input_sym[:,2:3]
            # input_sym=input_sym.detach()
            # output_raw_sym=M(input_sym)    
            # output_final=hard_consrain(input[:,3:4],output_raw,output_raw_sym) # 硬约束
            loss = train_loss_TOTAL(epoch,PDEloss_start_epoch,device, L,T0,P0,input,output_raw,label,data_min,data_max)  # 计算损失

            #loss = train_loss_TOTAL(epoch,PDEloss_start_epoch,device, L,M0,T0,P0,input,output_raw,label,data_min,data_max) 

            # 数据接收
            train_loss_batch=loss[0]
            res_cont_batch=loss[1]
            res_mx_batch=loss[2]
            res_my_batch=loss[3]
            res_mz_batch=loss[4]
            res_energy_batch=loss[5]
            res_k_batch=loss[6]
            res_omega_batch=loss[7]

            train_batches += 1  # 本轮次已迭代的批次总数更新
            # 每批次训练参数打印
            logger.info(f"epoch:{epoch},batch:{train_batches}")
            logger.info(f"loss:{train_loss_batch.item()}")
            logger.info(f"Res_cont:{res_cont_batch.item()}")
            logger.info(f"Res_mx:{res_mx_batch.item()}")
            logger.info(f"Res_my:{res_my_batch.item()}")
            logger.info(f"Res_mz:{res_mz_batch.item()}")
            logger.info(f"Res_energy:{res_energy_batch.item()}")
            logger.info(f"Res_k:{res_k_batch.item()}")
            logger.info(f"Res_omega:{res_omega_batch.item()}")
                
            if not torch.isnan(train_loss_batch):
                train_loss_batch.backward()      # 反向传播
                optimizer_M.step()                       # 梯度下降
                

                # 累加每轮当中的训练参数
                train_loss_epoch += train_loss_batch.item()  
                res_cont_epoch+=res_cont_batch.item()  
                res_mx_epoch+=res_mx_batch.item()  
                res_my_epoch+=res_my_batch.item()  
                res_mz_epoch+=res_mz_batch.item()  
                res_energy_epoch+=res_energy_batch.item()  
                res_k_epoch+=res_k_batch.item()  
                res_omega_epoch+=res_omega_batch.item()  
            else:
                train_nan_loss +=1
                logger.info("此批次损失计算出现NAN,已舍弃此损失值,不对此批次反向传播")
                continue

        # 计算每轮平均训练参数
        train_loss_epoch = train_loss_epoch / (train_batches-train_nan_loss)  # 本epoch平均损失
        res_cont_epoch=res_cont_epoch/ (train_batches-train_nan_loss)
        res_mx_epoch=res_mx_epoch/(train_batches-train_nan_loss)
        res_my_epoch=res_my_epoch/(train_batches-train_nan_loss)
        res_mz_epoch=res_mz_epoch/(train_batches-train_nan_loss)
        res_energy_epoch=res_energy_epoch/(train_batches-train_nan_loss)
        res_k_epoch=res_k_epoch/(train_batches-train_nan_loss)
        res_omega_epoch=res_omega_epoch/(train_batches-train_nan_loss)

        # 每轮训练参数添加到列表
        train_losses.append(train_loss_epoch)
        res_cont_epoches.append(res_cont_epoch)
        res_mx_epoches.append(res_mx_epoch)
        res_my_epoches.append(res_my_epoch)
        res_mz_epoches.append(res_mz_epoch)
        res_energy_epoches.append(res_energy_epoch)
        res_k_epoches.append(res_k_epoch)
        res_omega_epoches.append(res_omega_epoch)

        # 训练参数打印
        logger.info(f"epoch:{epoch}")
        logger.info(f"train_loss_average:{train_loss_epoch}")
        logger.info(f"Res_cont_average:{res_cont_epoch}")
        logger.info(f"Res_mx_average:{res_mx_epoch}")
        logger.info(f"Res_my_average:{res_my_epoch}")
        logger.info(f"Res_mz_average:{res_mz_epoch}")
        logger.info(f"Res_energy_average:{res_energy_epoch}")
        logger.info(f"Res_k_average:{res_k_epoch}")
        logger.info(f"Res_omega_average:{res_omega_epoch}")

        # 每轮参数归零
        train_nan_loss=train_loss_epoch = res_cont_epoch=res_mx_epoch=res_my_epoch=res_mz_epoch=\
            res_energy_epoch=res_k_epoch=res_omega_epoch=0
        
        ########################## 验证集评估 #################################
        logger.info(f"第{epoch}轮训练集训练完成,开始验证集校验工作")
        M.eval()
        val_batches=val_loss_epoch = 0
        with torch.no_grad():
            for input, label in val_dataloader:
                input, label = input.to(device), label.to(device)
                optimizer_M.zero_grad()  # 梯度归零
                output_raw = M(input)
                # input_sym=input.clone()                              # 构造对称输入
                # input_sym[:,2:3]=-input_sym[:,2:3]
                # input_sym=input_sym.detach()
                # output_raw_sym=M(input_sym)
                # output_final=hard_consrain(input[:,3:4],output_raw,output_raw_sym) # 硬约束
                val_loss_batch = val_loss_TOTAL(device, output_raw, label)  # 计算验证集损失
                val_batches += 1  # 本轮验证集已经迭代的批次数
                # 打印验证集参数
                logger.info(f"epoch:{epoch},batch:{val_batches},\n loss:{val_loss_batch.item()} ")
                if not torch.isnan(val_loss_batch):
                    val_loss_epoch += val_loss_batch.item()  # 累加损失
                else:
                    logger.info("损失值出现nan,已舍弃此批次验证，继续验证")
                    val_nan_loss+=1
        # 计算验证集平均损失
        val_loss_epoch = val_loss_epoch / (val_batches-val_nan_loss)  # 本epoch平均损失

        # 将验证集平均损失添加到列表中
        val_losses.append(val_loss_epoch)

        logger.info(f"epoch:{epoch},\n val_loss_average:{val_loss_epoch}")
        val_nan_loss=0

        ##################### 决定是否保存当前轮次 ###############################
        logger.info(f"第{epoch}轮验证集评估完成，开始判断是否保存本轮权重")
        if (epoch==start_epoch)and LOAD_CP==False:
            save_checkpoint(M,optimizer_M,epoch,train_losses,res_cont_epoches,res_mx_epoches,res_my_epoches,res_mz_epoches,res_energy_epoches,res_k_epoches,res_omega_epoches,val_losses, f'{project_root}/outputs/weights/{current_datetime}/{epoch}weights.pth')   # 保存当前模型权重的信息
            logger.info(f"第一轮权重已保存")
            min_val_loss= val_loss_epoch            # 初始化最小的验证集损失
            d_epoch_num = start_epoch             # 初始化效果最好的轮次数
        if (epoch==EPOCHES):
            save_checkpoint(M,optimizer_M,epoch,train_losses,res_cont_epoches,res_mx_epoches,res_my_epoches,res_mz_epoches,res_energy_epoches,res_k_epoches,res_omega_epoches,val_losses, f'{project_root}/outputs/weights/{current_datetime}/{epoch}weights.pth')   # 保存当前模型权重的信息
            logger.info(f"最后一轮权重已保存")
        else:
            if val_loss_epoch < min_val_loss:
                delpath=f'{project_root}/outputs/weights/{current_datetime}/{d_epoch_num}weights.pth' # 删除对应的权重
                if os.path.exists(delpath):
                    os.remove(delpath)
                    logger.info(f"删除了先前的第{d_epoch_num}轮权重")
                d_epoch_num=epoch                 # 更新效果最好的轮次数
                min_val_loss= val_loss_epoch        # 更新最小的验证集
                
                save_checkpoint(M,optimizer_M,epoch,train_losses,res_cont_epoches,res_mx_epoches,res_my_epoches,res_mz_epoches,res_energy_epoches,res_k_epoches,res_omega_epoches,val_losses,  f'{project_root}/outputs/weights/{current_datetime}/{epoch}weights.pth')   # 保存当前模型权重的信息
                logger.info(f"保存了当前的第{d_epoch_num}轮权重,最好效果为{min_val_loss}")
            else :
                logger.info("不保存此轮权重")

        ################### 绘图 ############################ 
        logger.info("开始绘图")
        epochs = range(1, len(val_losses) + 1)
        allgraph(current_datetime, epochs,train_losses,res_cont_epoches,res_mx_epoches,res_my_epoches,res_mz_epoches,res_energy_epoches,res_k_epoches,res_omega_epoches,val_losses)
        logger.info("绘图完成")
        logger.info(f"第{epoch}轮训练全部完成")
    logger.info("训练全部完成")

# 保存模型权重
def save_checkpoint(model,optimizer,epoch,train_losses,res_cont_epoches,res_mx_epoches,res_my_epoches,res_mz_epoches,res_energy_epoches,res_k_epoches,res_omega_epoches,val_losses,path):
    state={
        "epoch":epoch,
        "model_state_dict":model.state_dict(),
        "optimizer_state_dict":optimizer.state_dict(),
        "train_loss":train_losses,
        "res_cont_epoch":res_cont_epoches,
        "res_mx_epoch":res_mx_epoches,
        "res_my_epoch":res_my_epoches,
        "res_mz_epoch":res_mz_epoches,
        "res_energy_epoch":res_energy_epoches,
        "res_k_epoch":res_k_epoches,
        "res_omega_epoch":res_omega_epoches,
        "val_loss":val_losses,
    }
    torch.save(state,path)

# 获取断点的轮次数、损失值
def load_checkpoint(model,optimizer,path):
    if os.path.isfile(path):    # 判断是否存在该检查点权重文件
        checkpoint=torch.load(path) # 加载该权重文件
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch=checkpoint['epoch']
        train_losses=checkpoint['train_loss']
        res_cont_epoches=checkpoint['res_cont_epoch']
        res_mx_epoches=checkpoint['res_mx_epoch']
        res_my_epoches=checkpoint['res_my_epoch']
        res_mz_epoches=checkpoint['res_mz_epoch']
        res_energy_epoches=checkpoint['res_energy_epoch']
        res_k_epoches=checkpoint['res_k_epoch']
        res_omega_epoches=checkpoint['res_omega_epoch']
        val_losses=checkpoint['val_loss']

        logger.info(f"已成功加载检查点权重,上次结束轮次为：{start_epoch},训练集总损失值为:{train_losses[start_epoch-1]}")
        return model,optimizer,start_epoch,train_losses,res_cont_epoches,res_mx_epoches,\
            res_my_epoches,res_mz_epoches,res_energy_epoches,res_k_epoches,res_omega_epoches,val_losses
    else:
        logger.info("在所给的路径下未找到对应的权重文件，请检查后重试")

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

if __name__ == '__main__':
    set_seed(42)  # 设置随机种子
    train()