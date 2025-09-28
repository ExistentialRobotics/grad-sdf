import os
import sys
import torch
import numpy as np
import yaml
import trimesh
import pysdf
from scipy.spatial import cKDTree
import open3d as o3d
from tqdm import tqdm
import random
import pandas as pd

# Add paths to import project modules
sys.path.insert(0, ".")
sys.path.insert(0, os.path.abspath('src'))

# 添加项目根目录到Python路径，确保可以导入demo模块
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

# 现在可以正确导入demo模块
from demo.parser import get_parser

from src.utils.import_util import get_dataset, get_decoder
from src.frame import DepthFrame
from src.loggers import BasicLogger
from src.mapping import Mapping
from src.functions.render_helpers import find_voxel_idx, get_features


def load_checkpoint(ckpt_path, args=None):
    """
    Load trained checkpoint file
    
    Args:
        ckpt_path (str): checkpoint file path, e.g.:
                        "mapping/logs/replica/room0/2025-08-06-19-44-27/ckpt/final_ckpt.pth"
        args: training parameters, if None then need to load from config file
    
    Returns:
        mapper: Mapping object with loaded state
        decoder: decoder with loaded state
    """
    
    # Load sparse octree library
    torch.classes.load_library(
        "third_party/sparse_octree/build/lib.linux-x86_64-cpython-310/svo.cpython-310-x86_64-linux-gnu.so")
    
    # 1. Load checkpoint file
    print(f"Loading checkpoint: {ckpt_path}")
    training_result = torch.load(ckpt_path)
    
    # Check checkpoint content
    print("Checkpoint keys:", list(training_result.keys()))
    
    # 3. Create data stream (for initialization)
    data_stream = get_dataset(args)
    # data_in = data_stream[0]
    # first_frame = DepthFrame(*data_in[:-1], offset=args.mapper_specs['offset'], 
    #                        ref_pose=data_in[-1]).cuda()
    # W, H = first_frame.rgb.shape[1], first_frame.rgb.shape[0]
    
    # 4. Create logger and mapper
    logger = BasicLogger(args, for_eva=True)
    mapper = Mapping(args, logger, data_stream=data_stream)
    
    # 5. Restore state from checkpoint
    print("Restoring model state...")
    
    # Restore decoder state
    mapper.decoder.load_state_dict(training_result['decoder_state'])
    
    # Restore map state
    mapper.map_states = training_result['map_state']
    
    # Set to evaluation mode
    mapper.decoder = mapper.decoder.cuda()
    mapper.decoder.eval()
    
    print("Checkpoint loading completed!")
    print(f"Decoder parameters: {sum(p.numel() for p in mapper.decoder.parameters())}")
    print(f"Map state keys: {list(mapper.map_states.keys())}")
    
    return mapper


def inference(mapper, points, batch_size=100000):
    points = points + mapper.offset
    points = torch.tensor(points, dtype=torch.float32, device='cuda', requires_grad=False)
    sdf_pred = []
    sdf_priors = []
    hash_features = []
    # print(points.min(axis=0), points.max(axis=0))
    with torch.no_grad():
        for i in range(0, points.shape[0], batch_size):
            batch_points = points[i:i+batch_size]
            batch_points_voxel_idx = find_voxel_idx(batch_points, mapper.map_states)
            batch_sdf_priors = get_features(batch_points, batch_points_voxel_idx, mapper.map_states, mapper.voxel_size)
            batch_hash_features = mapper.decoder(batch_points)
            sdf_priors_features = batch_sdf_priors['sdf_priors'].squeeze(1)
            batch_sdf_pred = sdf_priors_features + batch_hash_features['sdf']
            sdf_pred.append(batch_sdf_pred.cpu())
            sdf_priors.append(sdf_priors_features.cpu())
            hash_features.append(batch_hash_features['sdf'].cpu())
            # print(batch_hash_features['sdf'].min(), batch_hash_features['sdf'].max())
            del batch_sdf_priors, batch_hash_features, batch_sdf_pred
            torch.cuda.empty_cache()
        sdf_pred = torch.cat(sdf_pred, dim=0)
        sdf_priors = torch.cat(sdf_priors, dim=0)
        hash_features = torch.cat(hash_features, dim=0)
    return sdf_pred, sdf_priors, hash_features


def get_points(bound, res_per_meter, offset=0):
    """
    :param bound: 体素边界，形如[[xmin, xmax], [ymin, ymax], [zmin, zmax]]
    :param res_per_meter: 每米采样多少个点（分辨率）
    :param offset: 偏移量
    :return: points, (x_res, y_res, z_res)
    """
    bound = np.array(bound) - offset
    x_min, x_max = bound[0]
    y_min, y_max = bound[1]
    z_min, z_max = bound[2]

    x_range = x_max - x_min
    y_range = y_max - y_min
    z_range = z_max - z_min

    x_res = int(np.round(x_range * res_per_meter)) + 1
    y_res = int(np.round(y_range * res_per_meter)) + 1
    z_res = int(np.round(z_range * res_per_meter)) + 1

    x = np.linspace(x_min, x_max, x_res)
    y = np.linspace(y_min, y_max, y_res)
    z = np.linspace(z_min, z_max, z_res)
    x, y, z = np.meshgrid(x, y, z, indexing='ij')
    points = np.stack([x.flatten(), y.flatten(), z.flatten()], axis=-1)

    return points, (x_res, y_res, z_res)


def calculate_gt_sdf(gt_mesh, points):
    vertices = gt_mesh.vertices.astype(np.float32)
    faces = gt_mesh.faces.astype(np.int32)
    f = pysdf.SDF(vertices, faces)
    gt_sdf_values = f(points)
    return gt_sdf_values

def check_sdf_info(args):
    base = os.path.join(args.data_specs['data_path'], "sdf_info")
    files = [
        "points_for_eval.npy", "gt_sdf_values_for_eval.npy",
        "points_near_surface.npy", "gt_sdf_values_near_surface.npy",
        "points_far_surface.npy", "gt_sdf_values_far_surface.npy"
    ]
    all_exist = True
    for fname in files:
        fpath = os.path.join(base, fname)
        if not os.path.exists(fpath):
            print(f"文件不存在: {fpath}")
            all_exist = False
    print("所有SDF相关文件都存在。" if all_exist else "有缺失的SDF相关文件，重新生成")
    return all_exist

        

def get_sdf_loss(args, mapper, gt_mesh):
    if not check_sdf_info(args):
        points_for_eval, points_near_surface, points_far_surface, gt_sdf_values_for_eval, gt_sdf_values_near_surface, gt_sdf_values_far_surface = generate_sdf_info(args, gt_mesh)
    else:
        points_for_eval = np.load(os.path.join(args.data_specs['data_path'], "sdf_info", "points_for_eval.npy"))
        points_near_surface = np.load(os.path.join(args.data_specs['data_path'], "sdf_info", "points_near_surface.npy"))
        points_far_surface = np.load(os.path.join(args.data_specs['data_path'], "sdf_info", "points_far_surface.npy"))
        gt_sdf_values_for_eval = np.load(os.path.join(args.data_specs['data_path'], "sdf_info", "gt_sdf_values_for_eval.npy"))
        gt_sdf_values_near_surface = np.load(os.path.join(args.data_specs['data_path'], "sdf_info", "gt_sdf_values_near_surface.npy"))
        gt_sdf_values_far_surface = np.load(os.path.join(args.data_specs['data_path'], "sdf_info", "gt_sdf_values_far_surface.npy"))

    sdf_pred, sdf_priors, _ = inference(mapper, points_for_eval)
    sdf_near_surface, sdf_near_surface_priors, _ = inference(mapper, points_near_surface)
    sdf_far_surface, sdf_far_surface_priors, _ = inference(mapper, points_far_surface)

    # 计算MAE（平均绝对误差）
    sdf_mae = np.mean(np.abs(sdf_pred.numpy() - gt_sdf_values_for_eval))
    prior_sdf_mae = np.mean(np.abs(sdf_priors.numpy() - gt_sdf_values_for_eval))
    near_surface_sdf_mae = np.mean(np.abs(sdf_near_surface.numpy() - gt_sdf_values_near_surface))
    near_surface_prior_sdf_mae = np.mean(np.abs(sdf_near_surface_priors.numpy() - gt_sdf_values_near_surface))
    far_surface_sdf_mae = np.mean(np.abs(sdf_far_surface.numpy() - gt_sdf_values_far_surface))
    far_surface_prior_sdf_mae = np.mean(np.abs(sdf_far_surface_priors.numpy() - gt_sdf_values_far_surface))

    # 计算MSE（均方误差）
    sdf_mse = np.mean((sdf_pred.numpy() - gt_sdf_values_for_eval) ** 2)
    prior_sdf_mse = np.mean((sdf_priors.numpy() - gt_sdf_values_for_eval) ** 2)
    near_surface_sdf_mse = np.mean((sdf_near_surface.numpy() - gt_sdf_values_near_surface) ** 2)
    near_surface_prior_sdf_mse = np.mean((sdf_near_surface_priors.numpy() - gt_sdf_values_near_surface) ** 2)
    far_surface_sdf_mse = np.mean((sdf_far_surface.numpy() - gt_sdf_values_far_surface) ** 2)
    far_surface_prior_sdf_mse = np.mean((sdf_far_surface_priors.numpy() - gt_sdf_values_far_surface) ** 2)

    # 构建包含MAE和MSE指标的字典
    loss_dict = {
        'sdf_mae': sdf_mae,
        'prior_sdf_mae': prior_sdf_mae,
        'near_surface_sdf_mae': near_surface_sdf_mae,
        'near_surface_prior_sdf_mae': near_surface_prior_sdf_mae,
        'far_surface_sdf_mae': far_surface_sdf_mae,
        'far_surface_prior_sdf_mae': far_surface_prior_sdf_mae,
        'sdf_mse': sdf_mse,
        'prior_sdf_mse': prior_sdf_mse,
        'near_surface_sdf_mse': near_surface_sdf_mse,
        'near_surface_prior_sdf_mse': near_surface_prior_sdf_mse,
        'far_surface_sdf_mse': far_surface_sdf_mse,
        'far_surface_prior_sdf_mse': far_surface_prior_sdf_mse
    }

    return loss_dict, points_for_eval, points_near_surface, points_far_surface

def generate_sdf_info(args, gt_mesh):
    bound = args.decoder_specs['bound']
    offset = args.mapper_specs['offset']
    
    points, (x_res, y_res, z_res) = get_points(bound, 80, offset)
    
    gt_sdf_values = calculate_gt_sdf(gt_mesh, points)
    gt_sdf_values = gt_sdf_values.reshape(x_res, y_res, z_res)
    points = points.reshape(x_res, y_res, z_res, 3)
    # clear invalid values
    threshold = 2.0
    outliers_mask = np.abs(gt_sdf_values) > threshold
    valid_mask = ~outliers_mask

    os.makedirs(os.path.join(args.data_specs['data_path'], "sdf_info"), exist_ok=True)
    # near surface points
    near_surface_path = os.path.join(args.data_specs['data_path'], "sdf_info", "points_near_surface.npy")
    near_surface_mask = (gt_sdf_values >= -0.1) & (gt_sdf_values <= 0.2) & valid_mask
    points_near_surface = points[near_surface_mask]
    gt_sdf_values_near_surface = gt_sdf_values[near_surface_mask]
    num_true = np.count_nonzero(near_surface_mask)
    total = near_surface_mask.size
    print(f"Number of near surface points: {num_true} / {total} ({num_true / total * 100:.2f}%)")
    np.save(near_surface_path, points_near_surface)
    np.save(near_surface_path.replace("points", "gt_sdf_values"), gt_sdf_values[near_surface_mask])

    # far surface points
    far_surface_path = os.path.join(args.data_specs['data_path'], "sdf_info", "points_far_surface.npy")
    far_surface_mask = (gt_sdf_values > 0.2) & valid_mask
    points_far_surface = points[far_surface_mask]
    gt_sdf_values_far_surface = gt_sdf_values[far_surface_mask]
    num_true = np.count_nonzero(far_surface_mask)
    total = far_surface_mask.size
    print(f"Number of far surface points: {num_true} / {total} ({num_true / total * 100:.2f}%)")
    np.save(far_surface_path, points_far_surface)
    np.save(far_surface_path.replace("points", "gt_sdf_values"), gt_sdf_values_far_surface)

    # points for sdf and sdf grad
    points_for_eval_path = os.path.join(args.data_specs['data_path'], "sdf_info", "points_for_eval.npy")
    points_for_eval_mask = (gt_sdf_values >= -0.1) & valid_mask
    points_for_eval = points[points_for_eval_mask]
    gt_sdf_values_for_eval = gt_sdf_values[points_for_eval_mask]
    num_true = np.count_nonzero(points_for_eval_mask)
    total = points_for_eval_mask.size
    print(f"Number of points for eval: {num_true} / {total} ({num_true / total * 100:.2f}%)")
    np.save(points_for_eval_path, points_for_eval)
    np.save(points_for_eval_path.replace("points", "gt_sdf_values"), gt_sdf_values_for_eval)
    return points_for_eval, points_near_surface, points_far_surface, gt_sdf_values_for_eval, gt_sdf_values_near_surface, gt_sdf_values_far_surface


def calculate_gt_gradients(points, gt_mesh, h1, args=None, save_path=None):
    # points = points + args.mapper_specs['offset']
    grad = np.zeros_like(points, dtype=np.float32) 
    vertices = gt_mesh.vertices.astype(np.float32) 
    faces = gt_mesh.faces.astype(np.int32)
    f = pysdf.SDF(vertices, faces)
    
    valid_mask = np.ones(points.shape[0], dtype=bool)
    
    # 首先计算所有方向的梯度
    for i in range(3):
        offset = np.zeros_like(points, dtype=np.float32)
        offset[:, i] = h1
        a = f(points+offset)
        b = f(points-offset)
        
        # 检查当前方向上a和b的绝对值是否都小于2
        direction_valid = (np.abs(a) < 2.0) & (np.abs(b) < 2.0)
        
        # 更新总的有效性掩码（所有方向都必须有效）
        valid_mask = valid_mask & direction_valid
        
        # 计算所有点的梯度（暂时忽略有效性）
        grad[:, i] = (a - b) / (2*h1)
    
    # 对于任何方向invalid的点，将所有三个方向的梯度都设为0
    grad[~valid_mask] = 0.0
    grad_normalized = grad / (np.linalg.norm(grad, axis=1, keepdims=True)+1e-8)
    
    print(f"Valid gradient points: {valid_mask.sum()} / {len(valid_mask)} ({valid_mask.sum()/len(valid_mask)*100:.2f}%)")
    if save_path is not None:
        np.save(save_path, grad_normalized)
    return grad_normalized


def get_sdf_gradient_metrics(args, mapper, gt_mesh, points_for_grad, save_name=None):
    # 临时启用梯度计算并设置为训练模式
    with torch.enable_grad():
        mapper.decoder.train()  # 临时设置为训练模式以启用梯度
        
        # 使用torch.autograd求sdf_pred关于points_for_grad的梯度
        points_eval_grad = torch.tensor(points_for_grad, dtype=torch.float32, device='cuda', requires_grad=True) + args.mapper_specs['offset']
        batch_size = 500000
        grad_pred = []
        for i in range(0, points_eval_grad.shape[0], batch_size):
            batch_points = points_eval_grad[i:i+batch_size]
            batch_points.requires_grad_(True)
            batch_points_voxel_idx = find_voxel_idx(batch_points, mapper.map_states)
            batch_sdf_priors = get_features(batch_points, batch_points_voxel_idx.detach(), mapper.map_states, mapper.voxel_size)
            batch_hash_features = mapper.decoder(batch_points)
            sdf_priors_features = batch_sdf_priors['sdf_priors'].squeeze(1)
            batch_sdf_pred = sdf_priors_features + batch_hash_features['sdf']
            grad_outputs = torch.ones_like(batch_sdf_pred)
            batch_grad = torch.autograd.grad(
                outputs=batch_sdf_pred,
                inputs=batch_points,
                grad_outputs=grad_outputs,
                create_graph=False,
                retain_graph=False,
                only_inputs=True
            )[0]
            grad_pred.append(batch_grad.detach().cpu().numpy())
            del batch_points_voxel_idx, batch_sdf_priors, batch_hash_features, batch_sdf_pred, batch_grad
            torch.cuda.empty_cache()
        
        # 恢复eval模式
        mapper.decoder.eval()
        
    grad_pred = np.concatenate(grad_pred, axis=0)
    gt_grad_path = os.path.join(args.data_specs['data_path'], "sdf_info", f"grad_gt_{save_name}.npy")
    if not os.path.exists(gt_grad_path):
        grad_gt_normalized = calculate_gt_gradients(points_for_grad, gt_mesh, h1=0.01, args=args, save_path=gt_grad_path)
    else:
        grad_gt_normalized = np.load(gt_grad_path)
    grad_pred_normalized = grad_pred / (np.linalg.norm(grad_pred, axis=1, keepdims=True)+1e-8)
    cosine = np.sum(grad_pred_normalized * grad_gt_normalized, axis=1)
    cosine = np.clip(cosine, -1, 1)
    angle = np.arccos(cosine)
    grad_angle_diff = np.abs(angle).mean()
    return grad_angle_diff


def evaluate_sdf(args):
    ckpt_path = os.path.join(args.log_dir, args.exp_name, "ckpt", "final_ckpt.pth")
    gt_mesh_path = args.data_specs['data_path']+'_mesh.ply'  # './Datasets/Replica/room0'
    scene_name = os.path.basename(args.data_specs['data_path'])
    gt_mesh = trimesh.load(gt_mesh_path)
    mapper = load_checkpoint(ckpt_path, args)

    # 创建保存metrics的字典
    all_metrics = {}

    # Calculate SDF loss
    loss_dict, points_for_eval, points_near_surface, points_far_surface = get_sdf_loss(args, mapper, gt_mesh)
    print("SDF loss:")
    for key, value in loss_dict.items():
        print(f"  {key}: {value:.6f}")
        all_metrics[key] = value

    grad_angle_diff = get_sdf_gradient_metrics(args, mapper, gt_mesh, points_for_eval, save_name="all")
    grad_angle_diff_near_surface = get_sdf_gradient_metrics(args, mapper, gt_mesh, points_near_surface, save_name="near_surface")
    grad_angle_diff_far_surface = get_sdf_gradient_metrics(args, mapper, gt_mesh, points_far_surface, save_name="far_surface")
    print(f"SDF gradient angle diff: {grad_angle_diff}", f"SDF gradient angle diff near surface: {grad_angle_diff_near_surface}", f"SDF gradient angle diff far surface: {grad_angle_diff_far_surface}")
    all_metrics['sdf_gradient_angle_diff'] = grad_angle_diff
    all_metrics['sdf_gradient_angle_diff_near_surface'] = grad_angle_diff_near_surface
    all_metrics['sdf_gradient_angle_diff_far_surface'] = grad_angle_diff_far_surface
    
    # 保存所有metrics到txt文件
    results_dir = os.path.join(args.log_dir, args.exp_name, "results")
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    
    metrics_file_path = os.path.join(results_dir, f"evaluation_sdf_metrics_{scene_name}.txt")

    metrics_list = list(all_metrics.keys())
    df = pd.DataFrame(columns=metrics_list)
    df.loc[scene_name] = [all_metrics[key] for key in metrics_list]
    print(df)
    df.to_csv(os.path.join(results_dir, f"evaluation_sdf_metrics_{scene_name}.csv"), index=False)
    
    with open(metrics_file_path, 'w', encoding='utf-8') as f:
        f.write("=== Evaluation Results ===\n")
        f.write(f"Experiment Name: {args.exp_name}\n")
        f.write(f"Data Path: {args.data_specs['data_path']}\n")
        f.write(f"Checkpoint Path: {ckpt_path}\n")
        f.write("=" * 50 + "\n\n")
        
        f.write("=== SDF Loss Metrics ===\n")
        f.write(f"SDF MAE: {all_metrics['sdf_mae']:.4f}\n")
        f.write(f"Prior SDF MAE: {all_metrics['prior_sdf_mae']:.4f}\n")
        f.write(f"Near Surface SDF MAE: {all_metrics['near_surface_sdf_mae']:.4f}\n")
        f.write(f"Near Surface Prior SDF MAE: {all_metrics['near_surface_prior_sdf_mae']:.4f}\n")
        f.write(f"Far Surface SDF MAE: {all_metrics['far_surface_sdf_mae']:.4f}\n")
        f.write(f"Far Surface Prior SDF MAE: {all_metrics['far_surface_prior_sdf_mae']:.4f}\n")

        f.write(f"SDF Gradient Angle Diff: {all_metrics['sdf_gradient_angle_diff']:.3f}\n")
        f.write(f"SDF Gradient Angle Diff Near Surface: {all_metrics['sdf_gradient_angle_diff_near_surface']:.3f}\n")
        f.write(f"SDF Gradient Angle Diff Far Surface: {all_metrics['sdf_gradient_angle_diff_far_surface']:.3f}\n")
        
        f.write(f"SDF MSE: {all_metrics['sdf_mse']:.4f}\n")
        f.write(f"Prior SDF MSE: {all_metrics['prior_sdf_mse']:.4f}\n")
        f.write(f"Near Surface SDF MSE: {all_metrics['near_surface_sdf_mse']:.4f}\n")
        f.write(f"Near Surface Prior SDF MSE: {all_metrics['near_surface_prior_sdf_mse']:.4f}\n")
        f.write("\n")

        f.write("=== Raw Values (for further processing) ===\n")
        for key, value in all_metrics.items():
            if isinstance(value, (int, float)):
                f.write(f"{key}: {value}\n")
    
    print(f"\n评估结果已保存到: {metrics_file_path}")
    return all_metrics


# Usage examples
if __name__ == "__main__":
    args = get_parser().parse_args()
    print(args)
    
    evaluate_sdf(args)