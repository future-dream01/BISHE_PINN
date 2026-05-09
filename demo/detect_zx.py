import os, sys
import torch
import numpy as np
from loguru import logger
from datetime import datetime
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), './..'))
sys.path.append(project_root)
from model import PINN, data_prepare, denormalize_for_pde
from model.utils_zx import (
    set_seed, L,
    compute_physical_quantities,
    plot_slice_comparison,
    X_TOLERANCE, MA_TOLERANCE, WALL_DIST_MAX
)

# 路径

# ===================== 配置 =====================
BATCHSIZE = 1024
weight_name = "mapr_best_1233weights"    # 目前最好的是697
weight_PATH = f'{project_root}/outputs/weights/mapr_best/{weight_name}.pth'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 输出路径
current_datetime = datetime.now().strftime("%m-%d_%H-%M")
save_root = f'{project_root}/outputs/推理结果/{current_datetime}'
os.makedirs(save_root, exist_ok=True)
log_file_path = f'{save_root}/inference.log'
logger.add(log_file_path, rotation="500 MB", level="INFO")

# ===================== 主推理函数 =====================
def main():
    set_seed(42)
    logger.info(f"开始推理")
    import pyvista as pv
    pv.set_plot_theme("document")
    os.environ["PYVISTA_OFF_SCREEN"] = "true"

    logger.info("加载数据集")
    train_dataloader, val_dataloader, data_min_np, data_max_np = data_prepare(BATCHSIZE)
    data_min = torch.tensor(data_min_np, dtype=torch.float32).to(device)
    data_max = torch.tensor(data_max_np, dtype=torch.float32).to(device)
    
    # 反归一化参数
    in_min_4 = data_min[:4].cpu().numpy()  # x,y,z,d
    in_max_4 = data_max[:4].cpu().numpy()
    in_range_4 = in_max_4 - in_min_4 + 1e-8
    
    # 马赫数反归一化参数
    in_min_Ma = data_min[4].cpu().numpy()   # Ma
    in_max_Ma = data_max[4].cpu().numpy()
    in_range_Ma = in_max_Ma - in_min_Ma + 1e-8
    
    # 压比 Pr 的反归一化参数
    in_min_Pr = data_min[5].cpu().numpy() if len(data_min) > 5 else 1.0
    in_max_Pr = data_max[5].cpu().numpy() if len(data_max) > 5 else 1.0
    in_range_Pr = in_max_Pr - in_min_Pr + 1e-8

    # 2. 加载模型
    model = PINN().to(device)
    if os.path.isfile(weight_PATH):
        checkpoint = torch.load(weight_PATH, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'], strict=True)
        logger.info(f"权重加载成功 (Epoch {checkpoint['epoch']})")
    else:
        logger.error(f"权重文件未找到: {weight_PATH}")
        return
    model.eval()

    logger.info("正在进行推理...")
    buffer = {
        'X': [], 'Y': [], 'Z': [], 'WD': [], 
        'Ma': [], 'Pr': [], # 保存每个点的工况参数
        # 预测值 (无量纲)
        'Up': [], 'Vp': [], 'Wp': [], 'Pp': [], 'Tp': [], 'Kp': [], 'Op': [],
        # 真值 (无量纲)
        'Ut': [], 'Vt': [], 'Wt': [], 'Pt': [], 'Tt': [], 'Kt': [], 'Ot': []
    }

    with torch.no_grad():
        for batch_idx, (input_batch, target_batch) in enumerate(val_dataloader):
            input_batch = input_batch.to(device)
            target_batch = target_batch.to(device)

            # 推理
            output_norm = model(input_batch)
            pred_phys = denormalize_for_pde(device, output_norm.clone(), data_min.clone(), data_max.clone())
            true_phys = denormalize_for_pde(device, target_batch.clone(), data_min.clone(), data_max.clone())

            # 转 Numpy
            in_np = input_batch.cpu().numpy()
            p_np = pred_phys.cpu().numpy()
            t_np = true_phys.cpu().numpy()

            # 坐标、d反归一化
            in_denorm_4 = in_np[:, :4] * in_range_4 + in_min_4

            # 马赫数反归一化
            Ma_denorm = in_np[:, 4] * in_range_Ma + in_min_Ma
            
            # 压比 Pr 反归一化
            if in_np.shape[1] > 5:
                Pr_denorm = in_np[:, 5] * in_range_Pr + in_min_Pr
            else:
                Pr_denorm = np.ones_like(Ma_denorm)

            # 存入 Buffer
            buffer['X'].extend(in_denorm_4[:, 0])
            buffer['Y'].extend(in_denorm_4[:, 1])
            buffer['Z'].extend(in_denorm_4[:, 2])
            buffer['WD'].extend(in_denorm_4[:, 3])
            buffer['Ma'].extend(Ma_denorm)
            buffer['Pr'].extend(Pr_denorm)
            
            # 预测值
            buffer['Up'].extend(p_np[:, 0]); buffer['Vp'].extend(p_np[:, 1]); buffer['Wp'].extend(p_np[:, 2])
            buffer['Pp'].extend(p_np[:, 3]); buffer['Tp'].extend(p_np[:, 4]); buffer['Kp'].extend(p_np[:, 5]); buffer['Op'].extend(p_np[:, 6])
            
            # 真值
            buffer['Ut'].extend(t_np[:, 0]); buffer['Vt'].extend(t_np[:, 1]); buffer['Wt'].extend(t_np[:, 2])
            buffer['Pt'].extend(t_np[:, 3]); buffer['Tt'].extend(t_np[:, 4]); buffer['Kt'].extend(t_np[:, 5]); buffer['Ot'].extend(t_np[:, 6])

    # 转为 Numpy 数组
    for k in buffer: buffer[k] = np.array(buffer[k])
    X_dim = buffer['X'] * L

    # ==========================================
    # 分组 (按 X + Ma)
    # ==========================================
    logger.info("正在分组...")
    X_rounded = np.round(X_dim / X_TOLERANCE) * X_TOLERANCE
    Ma_rounded = np.round(buffer['Ma'] / MA_TOLERANCE) * MA_TOLERANCE
    x_ma_pairs = np.column_stack([X_rounded, Ma_rounded])
    unique_pairs, inverse_indices = np.unique(x_ma_pairs, axis=0, return_inverse=True)
    logger.info(f"✅ 共找到 {len(unique_pairs)} 组切片")

    # ==========================================
    # 主循环：遍历切片 -> 调用 utils 计算 -> 调用 utils 绘图
    # ==========================================
    logger.info("开始处理切片..")

    num_cont=0
    
    for i, (x_sec_rounded, ma_sec_rounded) in enumerate(unique_pairs):
        mask = inverse_indices == i

        # 提取该切片的所有数据
        x_loc = X_dim[mask]
        y_loc = buffer['Y'][mask]
        z_loc = buffer['Z'][mask]
        wd_loc = buffer['WD'][mask]
        
        # 提取该切片所有点的 Ma 和 Pr (完整数组，不是平均)
        ma_slice = buffer['Ma'][mask]
        pr_slice = buffer['Pr'][mask]
        
        # 计算平均工况 (仅用于标题显示)
        x_sec_mean = np.mean(x_loc)
        ma_sec_mean = np.mean(ma_slice)
        pr_sec_mean = np.mean(pr_slice)
        
        if len(x_loc) < 50: continue

        # 流体域过滤
        fluid_mask = (wd_loc <= WALL_DIST_MAX + 0.2) & (wd_loc >= -0.1) & (~np.isnan(wd_loc))
        
        # 提取最终数据
        y_f = y_loc
        z_f = z_loc
        wd_f = wd_loc
        
        # 提取无量纲物理量
        Up_f = buffer['Up'][mask]
        Vp_f = buffer['Vp'][mask]
        Wp_f = buffer['Wp'][mask]
        Pp_f = buffer['Pp'][mask]
        Tp_f = buffer['Tp'][mask]
        
        Ut_f = buffer['Ut'][mask]
        Vt_f = buffer['Vt'][mask]
        Wt_f = buffer['Wt'][mask]
        Pt_f = buffer['Pt'][mask]
        Tt_f = buffer['Tt'][mask]
        
        # 提取对应点的 Ma 和 Pr (应用流体域过滤)
        ma_f = ma_slice
        pr_f = pr_slice

        if len(y_f) < 50: continue

        # 有量纲化
        pred_dict, true_dict = compute_physical_quantities(
            Up_f, Vp_f, Wp_f, Pp_f, Tp_f,
            Ut_f, Vt_f, Wt_f, Pt_f, Tt_f,
            ma_f, pr_f
        )

        num_cont+=1
        logger.info(f"111")
        if num_cont==9:
            logger.info(f"找到目标切片,开始写入网络预测的csv文件")
            Pressure_pred=pred_dict["Static_P"]
            Total_Pressure_pred=pred_dict["Total_P"]
            Mach_pred=pred_dict["Mach"]
            data=np.column_stack([x_loc,y_f,z_f,Pressure_pred,Total_Pressure_pred,Mach_pred])
            np.savetxt("0515_042_pred.csv",data,delimiter=" ")
            logger.info(f"网络预测文件写入完成")

            logger.info(f"找到目标切片,开始写入真值的csv文件")
            Pressure_true=true_dict["Static_P"]
            Total_Pressure_true=true_dict["Total_P"]
            Mach_true=true_dict["Mach"]
            data=np.column_stack([x_loc,y_f,z_f,Pressure_true,Total_Pressure_true,Mach_true])
            np.savetxt("0515_042_true.csv",data,delimiter=" ")
            logger.info(f"网络预测文件写入完成")

        if num_cont==10:
            logger.info(f"找到目标切片,开始写入网络预测的csv文件")
            Pressure_pred=pred_dict["Static_P"]
            Total_Pressure_pred=pred_dict["Total_P"]
            Mach_pred=pred_dict["Mach"]
            data=np.column_stack([x_loc,y_f,z_f,Pressure_pred,Total_Pressure_pred,Mach_pred])
            np.savetxt("0515_090_pred.csv",data,delimiter=" ")
            logger.info(f"网络预测文件写入完成")

            logger.info(f"找到目标切片,开始写入真值的csv文件")
            Pressure_true=true_dict["Static_P"]
            Total_Pressure_true=true_dict["Total_P"]
            Mach_true=true_dict["Mach"]
            data=np.column_stack([x_loc,y_f,z_f,Pressure_true,Total_Pressure_true,Mach_true])
            np.savetxt("0515_090_true.csv",data,delimiter=" ")
            logger.info(f"网络预测文件写入完成")

        plot_slice_comparison(
            pred_dict, true_dict, y_f, z_f, wd_f,
            x_sec_mean, ma_sec_mean, pr_sec_mean,
            save_root
        )

    logger.info(f"完成 保存在: {save_root}")

if __name__ == "__main__":
    main()