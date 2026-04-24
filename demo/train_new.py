import os, sys
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), './..'))
sys.path.append(project_root)
from model import PINN,data_prepare, train_loss_TOTAL,val_loss_TOTAL,psnr,ssim,allgraph,hard_consrain
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
M0=0.42    # 来流马赫数
T0=249.15  # 来流静温
P0=47181   # 来流静压
GAMMA = 1.4      # 比热比
R_gas = 287      # 气体常数

# 训练超参数设定
EPOCHES = 2000    # 轮次数
BATCHSIZE = 1024    # 批次数
PDEloss_start_epoch=100  # 开始加入PDE残差损失的轮次
train_nan_loss=val_nan_loss=0   # 一轮中出现异常损失值的批次数量
LOAD_CP=True    # 是否需要加载之前的检查点
CP_PATH= f'{project_root}/outputs/weights/04-23_20-05/84weights.pth'    # 检查点权重文件绝对路径
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
    M = PINN(device,data_min ,data_max)              # 创建模型对象
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
            optimizer_M.zero_grad()  # 梯度归零
            # 前向传播
            #with autocast():
            output_raw = M(input)
            #input_sym=input.clone()                              # 构造对称输入
            #input_sym[:,2:3]=-input_sym[:,2:3]
            #input_sym=input_sym.detach()
            #output_raw_sym=M(input_sym)    
            #output_final=hard_consrain(input[:,3:4],output_raw,output_raw_sym) # 硬约束
            #loss = train_loss_TOTAL(epoch,PDEloss_start_epoch,device, L,M0,T0,P0,input,output_raw,output_final,label,data_min,data_max)  # 计算损失

            loss = train_loss_TOTAL(epoch,PDEloss_start_epoch,device, L,M0,T0,P0,input,output_raw,label,data_min,data_max) 

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
            logger.info(f"epoch:{epoch},batch:{train_batches},\n loss:{train_loss_batch.item()} \n Res_cont:{res_cont_batch.item()}\
                 \n Res_mx:{res_mx_batch.item()} \n Res_my:{res_my_batch.item()}\
                \n Res_mz:{res_mz_batch.item()} \n Res_energy:{res_energy_batch.item()} \
                \n Res_k:{res_k_batch.item()} \n Res_omega:{res_omega_batch.item()}")
                
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
        logger.info(f"epoch:{epoch},\n train_loss_average:{train_loss_epoch} \n \
            Res_cont_average:{res_cont_epoch} \n Res_mx_average:{res_mx_epoch} \n Res_my_average:{res_my_epoch}\
            \n Res_mz_average:{res_mz_epoch} \n Res_energy_average:{res_energy_epoch} \
            \n Res_k_average:{res_k_epoch} \n Res_omega_average:{res_omega_epoch}")

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
                
                # 🔧 修正了这里的打印：原来是 [0:1] 切片，现在是取整个 batch
                if val_batches == 0:
                    logger.info(f"   验证集 Batch 0 数据范围:")
                    logger.info(f"   U范围: {output_raw[:, 0].min().item():.6f} ~ {output_raw[:, 0].max().item():.6f}")
                    logger.info(f"   V范围: {output_raw[:, 1].min().item():.6f} ~ {output_raw[:, 1].max().item():.6f}")
                    logger.info(f"   P范围: {output_raw[:, 3].min().item():.6f} ~ {output_raw[:, 3].max().item():.6f}")

                #input_sym=input.clone()                              # 构造对称输入
                #input_sym[:,2:3]=-input_sym[:,2:3]
                #input_sym=input_sym.detach()
                #output_raw_sym=M(input_sym)
                #output_final=hard_consrain(input[:,3:4],output_raw) # 硬约束
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

        # ===================== 【核心新增】训练完成后，直接用当前模型推理绘图 =====================
        if epoch == EPOCHES: 
            logger.info("\n" + "="*60)
            logger.info("🎨 训练完成，启动内嵌推理绘图 (无权重加载/无数据重新加载)...")
            logger.info("="*60)
            
            M.eval()
            
            # 全局存储
            infer_X, infer_Y, infer_Z, infer_WALL_D = [], [], [], []
            infer_U_pred, infer_V_pred, infer_W_pred, infer_P_pred, infer_T_pred = [], [], [], [], []
            infer_U_true, infer_V_true, infer_W_true, infer_P_true, infer_T_true = [], [], [], [], []
            
            # 直接用当前的 data_min/data_max
            in_min_4 = data_min[:4].cpu().numpy()
            in_max_4 = data_max[:4].cpu().numpy()
            in_range_4 = in_max_4 - in_min_4 + 1e-8
            
            U0_val = M0 * (GAMMA * R_gas * T0) ** 0.5
            q0_val = GAMMA * P0 * M0**2
            
            with torch.no_grad():
                for input_batch, target_batch in val_dataloader:
                    input_batch = input_batch.to(device)
                    target_batch = target_batch.to(device)
                    
                    # 【核心】和训练完全一致的前向传播
                    output_final = M(input_batch)
                    
                    # 提取数据
                    input_np = input_batch.cpu().numpy()
                    pred_np = output_final.cpu().numpy()
                    true_np = target_batch.cpu().numpy()
                    
                    # 反归一化坐标
                    input_denorm = input_np * in_range_4 + in_min_4
                    X, Y, Z, WALL_D = input_denorm[:, 0], input_denorm[:, 1], input_denorm[:, 2], input_denorm[:, 3]
                    
                    # 保存数据
                    infer_X.extend(X); infer_Y.extend(Y); infer_Z.extend(Z); infer_WALL_D.extend(WALL_D)
                    infer_U_pred.extend(pred_np[:, 0]); infer_V_pred.extend(pred_np[:, 1]); infer_W_pred.extend(pred_np[:, 2])
                    infer_P_pred.extend(pred_np[:, 3]); infer_T_pred.extend(pred_np[:, 4])
                    infer_U_true.extend(true_np[:, 0]); infer_V_true.extend(true_np[:, 1]); infer_W_true.extend(true_np[:, 2])
                    infer_P_true.extend(true_np[:, 3]); infer_T_true.extend(true_np[:, 4])
            
            # 转为数组
            infer_X = np.array(infer_X); infer_Y = np.array(infer_Y); infer_Z = np.array(infer_Z); infer_WALL_D = np.array(infer_WALL_D)
            infer_U_pred = np.array(infer_U_pred); infer_V_pred = np.array(infer_V_pred); infer_W_pred = np.array(infer_W_pred)
            infer_P_pred = np.array(infer_P_pred); infer_T_pred = np.array(infer_T_pred)
            infer_U_true = np.array(infer_U_true); infer_V_true = np.array(infer_V_true); infer_W_true = np.array(infer_W_true)
            infer_P_true = np.array(infer_P_true); infer_T_true = np.array(infer_T_true)
            
            # 打印确认
            logger.info("\n📊 【内嵌推理】数据范围检查 (无量纲):")
            logger.info(f"   U_pred: {infer_U_pred.min():.6f} ~ {infer_U_pred.max():.6f}")
            logger.info(f"   V_pred: {infer_V_pred.min():.6f} ~ {infer_V_pred.max():.6f}")
            logger.info(f"   P_pred: {infer_P_pred.min():.6f} ~ {infer_P_pred.max():.6f}")
            if infer_U_pred.min() != infer_U_pred.max():
                logger.info("✅ 确认：网络输出不是常数，有空间分布！")
            
            # 有量纲化
            Vel_pred = np.sqrt(infer_U_pred**2 + infer_V_pred**2 + infer_W_pred**2) * U0_val
            P_pred = infer_P_pred * q0_val + P0
            T_pred = infer_T_pred * T0
            c_pred = np.sqrt(GAMMA * R_gas * T_pred)
            Ma_pred = Vel_pred / (c_pred + 1e-12)
            
            Vel_true = np.sqrt(infer_U_true**2 + infer_V_true**2 + infer_W_true**2) * U0_val
            P_true = infer_P_true * q0_val + P0
            T_true = infer_T_true * T0
            
            X_dim = infer_X * L
            
            # 创建专门的推理结果文件夹
            infer_save_root = f'{project_root}/outputs/推理结果/{current_datetime}_内嵌推理'
            os.makedirs(infer_save_root, exist_ok=True)
            
            # 绘图参数
            plot_vars = [("Velocity", Vel_pred, Vel_true, "m/s"), 
                         ("Static_P", P_pred, P_true, "Pa"), 
                         ("Mach", Ma_pred, Ma_true, "-")]
            
            # 聚类切片
            from scipy.cluster.hierarchy import fclusterdata
            import matplotlib.tri as tri
            from matplotlib import cm
            
            X_TOL = 1e-4
            WALL_DIST_MAX_INFER = 0.6
            
            clusters = fclusterdata(X_dim.reshape(-1,1), t=X_TOL, criterion='distance')
            cluster_ids = np.unique(clusters)
            
            logger.info(f"开始绘图，共 {len(cluster_ids)} 个截面...")
            
            for cid in cluster_ids:
                mask = clusters == cid
                X_sec = X_dim[mask]
                Y_sec = infer_Y[mask]
                Z_sec = infer_Z[mask]
                WALL_D_sec = infer_WALL_D[mask]
                
                if len(X_sec) < 50: continue
                
                # 流体域过滤
                fluid_mask = (WALL_D_sec <= WALL_DIST_MAX_INFER) & (WALL_D_sec >= 0) & (~np.isnan(WALL_D_sec))
                Y_fluid = Y_sec[fluid_mask]
                Z_fluid = Z_sec[fluid_mask]
                WALL_D_fluid = WALL_D_sec[fluid_mask]
                
                if len(Y_fluid) < 50: continue
                
                for name, val_pred, val_true, unit in plot_vars:
                    v_pred_fluid = val_pred[mask][fluid_mask]
                    v_true_fluid = val_true[mask][fluid_mask]
                    
                    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6), dpi=150)
                    triang = tri.Triangulation(Y_fluid, Z_fluid)
                    
                    # 简单的掩码剔除坏三角形
                    triangles = triang.triangles
                    pts = np.column_stack([Y_fluid, Z_fluid])
                    y_tri = pts[triangles, 0]
                    z_tri = pts[triangles, 1]
                    a = np.sqrt((y_tri[:, 1] - y_tri[:, 0])**2 + (z_tri[:, 1] - z_tri[:, 0])**2)
                    b = np.sqrt((y_tri[:, 2] - y_tri[:, 1])**2 + (z_tri[:, 2] - z_tri[:, 1])**2)
                    c = np.sqrt((y_tri[:, 0] - y_tri[:, 2])**2 + (z_tri[:, 0] - z_tri[:, 2])**2)
                    s = (a + b + c) / 2.0
                    area = np.sqrt(np.clip(s * (s - a) * (s - b) * (s - c), 0, None))
                    area[area < 1e-12] = 1e-12
                    scale = np.max([np.max(Y_fluid) - np.min(Y_fluid), np.max(Z_fluid) - np.min(Z_fluid)])
                    circum_r = (a * b * c) / (4.0 * area)
                    mask_alpha = circum_r > (scale / 1.0)
                    triang.set_mask(mask_alpha)
                    
                    ax1.tricontourf(triang, v_pred_fluid, 80, cmap=cm.jet)
                    ax1.set_title(f"Prediction | {name}")
                    ax2.tricontourf(triang, v_true_fluid, 80, cmap=cm.jet)
                    ax2.set_title(f"Ground Truth | {name}")
                    ax1.set_aspect('equal')
                    ax2.set_aspect('equal')
                    plt.savefig(f"{infer_save_root}/X_{np.mean(X_sec):.4f}m_{name}_Compare.png")
                    plt.close()
            
            logger.info(f"✅ 内嵌推理绘图全部完成！保存在: {infer_save_root}")

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