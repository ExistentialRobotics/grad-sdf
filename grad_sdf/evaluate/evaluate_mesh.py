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

torch.set_grad_enabled(False)


def trimesh_to_open3d(trimesh_mesh):
    """将trimesh.Trimesh转换为open3d.geometry.TriangleMesh"""
    o3d_mesh = o3d.geometry.TriangleMesh()
    o3d_mesh.vertices = o3d.utility.Vector3dVector(trimesh_mesh.vertices)
    o3d_mesh.triangles = o3d.utility.Vector3iVector(trimesh_mesh.faces)
    o3d_mesh.compute_vertex_normals()
    return o3d_mesh


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
                        #    ref_pose=data_in[-1]).cuda()
    # W, H = first_frame.depth.shape[1], first_frame.depth.shape[0]
    
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


def load_and_extract_mesh(ckpt_path, args, mesh_res):
    """
    Load checkpoint and extract mesh
    
    Args:
        ckpt_path (str): checkpoint file path
        args: configuration parameters
        mesh_res (int): mesh resolution, default 256
        output_dir (str): output directory, default to mesh directory at same level as checkpoint
    
    Returns:
        mesh: extracted mesh object
        output_path: mesh save path
    """
    
    # 1. Load checkpoint
    print("=" * 50)
    print("Start loading checkpoint and reconstructing mesh")
    print("=" * 50)
    
    mapper = load_checkpoint(ckpt_path, args)
    
    # 4. Extract mesh
    print(f"\nStart extracting mesh, resolution: {mesh_res}")
    print("This may take several minutes...")
    
    mesh, mesh_priors, sdf_u, prior_u, decoder_u = mapper.extract_mesh(
        res=mesh_res, 
        map_states=mapper.map_states
    )

    print(f"\n✅ Mesh reconstruction completed!")
    print(f"📊 Vertex count: {len(mesh.vertices)}")
    print(f"📊 Face count: {len(mesh.faces)}")
    
    return mesh, mesh_priors

def rgba_visualize(points, values, alpha=1.0, save_path=None):
    """
    向量化版：values 为 ndarray，返回 shape=(N,4) 的 RGBA
    """
    v_min, v_max = np.min(values), np.max(values)
    v = ((values - v_min) / (v_max - v_min)).astype(np.float32)
    h = v * 5.0 + 1.0
    i = np.floor(h).astype(np.int32)
    f = h - i
    f = np.where((i % 2) == 0, 1.0 - f, f)  # 偶数翻转
    n = 1.0 - f

    # 分段赋值（与上面标量逻辑一致）
    r = np.where(i <= 1, n,
        np.where(i == 2, 0.0,
        np.where(i == 3, 0.0,
        np.where(i == 4, n, 1.0))))
    g = np.where(i <= 1, 0.0,
        np.where(i == 2, n,
        np.where(i == 3, 1.0,
        np.where(i == 4, 1.0, n))))
    b = np.where(i <= 1, 1.0,
        np.where(i == 2, 1.0,
        np.where(i == 3, n, 0.0)))

    a = np.full_like(v, float(alpha))

    rgba = np.stack([r, g, b, a], axis=-1)
    
    # 展平颜色数组 (只取RGB，忽略alpha通道)
    colors = rgba.reshape(-1, 4)[:, :3]  # 只要RGB通道
    points = points.reshape(-1, 3)
    
    # 可选：过滤掉值为0的点（如果不想显示空白区域）
    # non_zero_mask = values_flat > 1e-6
    # points = points[non_zero_mask]
    # colors = colors[non_zero_mask]
    
    # 创建Open3D点云对象
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    
    # 保存点云
    success = o3d.io.write_point_cloud(save_path, pcd)
    
    return pcd


def crop_mesh_to_aabb(mesh: trimesh.Trimesh, aabb_min, aabb_max):
    """Keep faces with all three vertices inside AABB and remove unreferenced vertices"""
    V = mesh.vertices
    F = mesh.faces
    vmask = np.all((V >= aabb_min) & (V <= aabb_max), axis=1)  # Whether vertices are inside the box
    fmask = vmask[F].all(axis=1)                               # All three vertices of faces are inside the box
    cropped = trimesh.Trimesh(vertices=V, faces=F[fmask], process=False)
    cropped.remove_unreferenced_vertices()
    return cropped


def sample_points_on_mesh(mesh: trimesh.Trimesh, n_points=200_000):
    """Area-weighted uniform sampling of surface points; n_points can be adjusted up/down based on scene"""
    if mesh.is_empty or len(mesh.faces) == 0:
        return np.empty((0, 3), dtype=np.float32)
    pts, _ = trimesh.sample.sample_surface(mesh, n_points)
    return pts.astype(np.float32)


def mesh_metrics(pts_a: np.ndarray, pts_b: np.ndarray, threshold: float, args=None):
    """
    基于点云的 Chamfer 距离和 F1 score
    
    Args:
        pts_a: GT 点云，shape (N, 3)
        pts_b: 预测点云，shape (M, 3)  
        threshold: 匹配的最大距离阈值（例如 0.05 表示 5 cm）
        
    Returns:
        dict: 包含所有评估指标的字典
    """
    tree_a = cKDTree(pts_a) #gt
    tree_b = cKDTree(pts_b) #pred

    d_b2a, _ = tree_a.query(pts_b, k=1, workers=-1)
    d_a2b, _ = tree_b.query(pts_a, k=1, workers=-1)
    comp_ratio = np.mean(d_a2b < threshold).astype(np.float32)
    comp_ratio_std = np.std(d_a2b < threshold).astype(np.float32)
    comp = np.mean(d_a2b)
    comp_std = np.std(d_a2b)
    acc = np.mean(d_b2a)
    acc_std = np.std(d_b2a)

    # Chamfer（非平方版）
    chamfer = float(d_a2b.mean() + d_b2a.mean())
    if args is not None:
        rgba_visualize(pts_a, d_a2b, save_path=os.path.join(args.log_dir, args.exp_name, "misc", "d_a2b.ply"))
        rgba_visualize(pts_b, d_b2a, save_path=os.path.join(args.log_dir, args.exp_name, "misc", "d_b2a.ply"))

    # F1 计算
    tp = np.sum(d_b2a <= threshold)      # Predicted points correctly matched to GT
    fp = np.sum(d_b2a > threshold)       # Predicted points too far from GT
    fn = np.sum(d_a2b > threshold)       # GT points not matched by predicted

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    return {
        'chamfer_distance': float(chamfer),
        'f1_score': float(f1),
        'precision': float(precision),
        'recall': float(recall),
        'complete_ratio': float(comp_ratio),  # GT点被覆盖的比例
        'complete_ratio_std': float(comp_ratio_std),
        'complete': float(comp),  # 平均完整性距离
        'complete_std': float(comp_std),
        'accuracy': float(acc),  # 平均准确性距离  
        'accuracy_std': float(acc_std),
        'threshold': float(threshold),
        'num_gt_points': int(pts_a.shape[0]),
        'num_pred_points': int(pts_b.shape[0])
    }


def get_distance_metrics(gt_mesh: trimesh.Trimesh, reconstructed_mesh: trimesh.Trimesh,
                         n_samples=200_000, threshold=0.2, args=None):
    """
    Returns (chamfer_distance, hausdorff_distance)
    - By default, crop reconstructed mesh using GT's AABB; can also read custom bound from args.
    - n_samples: number of sampling points per mesh (larger = more stable, slower)
    """
    # 1) Calculate bounds
    aabb_min = np.min(gt_mesh.vertices, axis=0)-0.01
    aabb_max = np.max(gt_mesh.vertices, axis=0)+0.01

    # 2) Crop reconstructed mesh
    recon_cropped = crop_mesh_to_aabb(reconstructed_mesh, aabb_min, aabb_max)
    # 3) Sample points
    pts_gt = sample_points_on_mesh(gt_mesh, n_points=n_samples)
    pts_rec = sample_points_on_mesh(recon_cropped, n_points=n_samples)

    # 4) Calculate metrics
    metrics = mesh_metrics(pts_gt, pts_rec, threshold=threshold, args=args)
    return metrics

def calculate_depth_L1(gt_mesh, rec_mesh, n_imgs=1000):
    def get_cam_position(gt_mesh):
        to_origin, extents = trimesh.bounds.oriented_bounds(gt_mesh)
        extents[2] *= 0.7
        extents[1] *= 0.7
        extents[0] *= 0.3
        transform = np.linalg.inv(to_origin)
        transform[2, 3] += 0.4
        return extents, transform


    def normalize(x):
        return x / np.linalg.norm(x)


    def viewmatrix(z, up, pos):
        vec2 = normalize(z)
        vec1_avg = up
        vec0 = normalize(np.cross(vec1_avg, vec2))
        vec1 = normalize(np.cross(vec2, vec0))
        m = np.stack([vec0, vec1, vec2, pos], 1)
        return m
    H = 500
    W = 500
    focal = 300
    fx = focal
    fy = focal
    cx = H / 2.0 - 0.5
    cy = W / 2.0 - 0.5

    # get vacant area inside the room
    extents, transform = get_cam_position(gt_mesh)

    # 转换trimesh为open3d格式
    gt_mesh_o3d = trimesh_to_open3d(gt_mesh)
    rec_mesh_o3d = trimesh_to_open3d(rec_mesh)

    vis = o3d.visualization.Visualizer()
    vis.create_window(width=W, height=H)
    vis.get_render_option().mesh_show_back_face = True
    errors = []
    for i in tqdm(range(n_imgs)):
        while True:
            up = [0, 0, -1]
            origin = trimesh.sample.volume_rectangular(
                extents, 1, transform=transform)
            origin = origin.reshape(-1)
            tx = round(random.uniform(-10000, +10000), 2)
            ty = round(random.uniform(-10000, +10000), 2)
            tz = round(random.uniform(-10000, +10000), 2)
            target = [tx, ty, tz]
            target = np.array(target) - np.array(origin)
            c2w = viewmatrix(target, up, origin)
            tmp = np.eye(4)
            tmp[:3, :] = c2w
            c2w = tmp

            param = o3d.camera.PinholeCameraParameters()
            param.extrinsic = np.linalg.inv(c2w)  # 4x4 numpy array

            param.intrinsic = o3d.camera.PinholeCameraIntrinsic(
                W, H, fx, fy, cx, cy)

            ctr = vis.get_view_control()
            ctr.set_constant_z_far(20)
            ctr.convert_from_pinhole_camera_parameters(param)

            vis.add_geometry(gt_mesh_o3d, reset_bounding_box=True, )
            ctr.convert_from_pinhole_camera_parameters(param)
            vis.poll_events()
            vis.update_renderer()
            gt_depth = vis.capture_depth_float_buffer(True)
            gt_depth = np.asarray(gt_depth)
            vis.remove_geometry(gt_mesh_o3d, reset_bounding_box=True, )
            if (gt_depth != 0).any():
                break

        vis.add_geometry(rec_mesh_o3d, reset_bounding_box=True, )
        ctr.convert_from_pinhole_camera_parameters(param)
        vis.poll_events()
        vis.update_renderer()
        ours_depth = vis.capture_depth_float_buffer(True)
        ours_depth = np.asarray(ours_depth)
        vis.remove_geometry(rec_mesh_o3d, reset_bounding_box=True, )

        mask = (gt_depth != 0) * (ours_depth != 0)
        errors += [np.abs(gt_depth[mask] - ours_depth[mask]).mean()]
    errors = np.array(errors)

    return errors

def evaluate_mesh(args):
    ckpt_path = os.path.join(args.log_dir, args.exp_name, "ckpt", "final_ckpt.pth")
    recon_mesh_path = os.path.join(args.log_dir, args.exp_name, "mesh", f"mesh_{args.mapper_specs['mesh_res']}.obj")
    recon_prior_mesh_path = os.path.join(args.log_dir, args.exp_name, "mesh", f"mesh_{args.mapper_specs['mesh_res']}_priors.obj")
    gt_mesh_path = args.data_specs['data_path']+'_mesh.ply'  # './Datasets/Replica/room0'
    gt_mesh = trimesh.load(gt_mesh_path)
    scene_name = os.path.basename(args.data_specs['data_path'])
    
    # 创建保存metrics的字典
    recon_mesh_metrics = {}
    recon_mesh_priors_metrics = {}
    
    # Load and extract mesh
    if not os.path.exists(recon_mesh_path):
        reconstructed_mesh, reconstructed_mesh_priors = load_and_extract_mesh(ckpt_path, args, mesh_res=args.mapper_specs['mesh_res'])
    else:
        reconstructed_mesh = trimesh.load(recon_mesh_path)
        reconstructed_mesh_priors = trimesh.load(recon_prior_mesh_path)


    # Calculate mesh metrics
    mesh_metrics_dict = get_distance_metrics(gt_mesh, reconstructed_mesh, threshold=0.05, args=args)
    mesh_metrics_dict_priors = get_distance_metrics(gt_mesh, reconstructed_mesh_priors, threshold=0.05, args=args)
    # print in cm and percentage
    print("--------------------------------")
    print("Reconstructed Mesh Metrics")
    print("--------------------------------")
    print(f"Chamfer distance: {mesh_metrics_dict['chamfer_distance']*100:.2f} cm, F1 score: {mesh_metrics_dict['f1_score']*100:.2f}%")
    print(f"Precision: {mesh_metrics_dict['precision']*100:.2f}%, Recall: {mesh_metrics_dict['recall']*100:.2f}%")
    print(f"Complete ratio: {mesh_metrics_dict['complete_ratio']*100:.2f} ± {mesh_metrics_dict['complete_ratio_std']*100:.2f}")
    print(f"Complete: {mesh_metrics_dict['complete']*100:.2f} ± {mesh_metrics_dict['complete_std']*100:.2f}")
    print(f"Accuracy: {mesh_metrics_dict['accuracy']*100:.2f} ± {mesh_metrics_dict['accuracy_std']*100:.2f}")
    print(f"GT points: {mesh_metrics_dict['num_gt_points']}, Pred points: {mesh_metrics_dict['num_pred_points']}")
    print("--------------------------------")
    print("Reconstructed Mesh Priors Metrics")
    print("--------------------------------")
    print(f"Chamfer distance priors: {mesh_metrics_dict_priors['chamfer_distance']*100:.2f} cm, F1 score priors: {mesh_metrics_dict_priors['f1_score']*100:.2f}%")
    print(f"Precision priors: {mesh_metrics_dict_priors['precision']*100:.2f}%, Recall priors: {mesh_metrics_dict_priors['recall']*100:.2f}%")
    print(f"Complete ratio priors: {mesh_metrics_dict_priors['complete_ratio']*100:.2f} ± {mesh_metrics_dict_priors['complete_ratio_std']*100:.2f}")
    print(f"Complete priors: {mesh_metrics_dict_priors['complete']*100:.2f} ± {mesh_metrics_dict_priors['complete_std']*100:.2f}")
    print(f"Accuracy priors: {mesh_metrics_dict_priors['accuracy']*100:.2f} ± {mesh_metrics_dict_priors['accuracy_std']*100:.2f}")
    print(f"GT points priors: {mesh_metrics_dict_priors['num_gt_points']}, Pred points priors: {mesh_metrics_dict_priors['num_pred_points']}")

    # 添加mesh metrics到总的metrics字典
    recon_mesh_metrics.update(mesh_metrics_dict)
    recon_mesh_priors_metrics.update(mesh_metrics_dict_priors)

    if args.eval_depth:
        depth_errors = calculate_depth_L1(gt_mesh, reconstructed_mesh, n_imgs=1000)
        print(f"Depth L1: {depth_errors.mean()*100:.2f} cm, Depth L1 std: {depth_errors.std()*100:.2f}")
        depth_errors_priors = calculate_depth_L1(gt_mesh, reconstructed_mesh_priors, n_imgs=1000)
        print(f"Depth L1 priors: {depth_errors_priors.mean()*100:.2f} cm, Depth L1 std priors: {depth_errors_priors.std()*100:.2f}")
        
        # 添加depth metrics
        recon_mesh_metrics['depth_l1_mean'] = depth_errors.mean()
        recon_mesh_metrics['depth_l1_std'] = depth_errors.std()
        recon_mesh_priors_metrics['depth_l1_mean_priors'] = depth_errors_priors.mean()
        recon_mesh_priors_metrics['depth_l1_std_priors'] = depth_errors_priors.std()
    
    # 保存所有metrics到txt文件
    results_dir = os.path.join(args.log_dir, args.exp_name, "results")
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    
    metrics_file_path = os.path.join(results_dir, f"mesh_metrics_{scene_name}.txt")

    print(recon_mesh_metrics)
    metrics_list = list(recon_mesh_metrics.keys())
    df = pd.DataFrame(columns=metrics_list)
    df.loc[scene_name] = [recon_mesh_metrics[key] for key in metrics_list]
    print(df)
    df.to_csv(os.path.join(results_dir, f"mesh_metrics_{scene_name}.csv"), index=False)

    df_priors = pd.DataFrame(columns=metrics_list)
    df_priors.loc[scene_name] = [recon_mesh_priors_metrics[key] for key in metrics_list]
    print(df_priors)
    df_priors.to_csv(os.path.join(results_dir, f"mesh_metrics_priors_{scene_name}.csv"), index=False)
    
    with open(metrics_file_path, 'w', encoding='utf-8') as f:
        f.write("=== Evaluation Results ===\n")
        f.write(f"Experiment Name: {args.exp_name}\n")
        f.write(f"Data Path: {args.data_specs['data_path']}\n")
        f.write(f"Checkpoint Path: {ckpt_path}\n")
        f.write("=" * 50 + "\n\n")
        
        f.write("=== Mesh Metrics ===\n")
        f.write(f"Chamfer Distance: {recon_mesh_metrics['chamfer_distance']*100:.2f} cm\n")
        f.write(f"F1 Score: {recon_mesh_metrics['f1_score']*100:.2f}%\n")
        f.write(f"Precision: {recon_mesh_metrics['precision']*100:.2f}%\n")
        f.write(f"Recall: {recon_mesh_metrics['recall']*100:.2f}%\n")
        f.write(f"Complete Ratio: {recon_mesh_metrics['complete_ratio']*100:.2f} ± {recon_mesh_metrics['complete_ratio_std']*100:.2f}%\n")
        f.write(f"Complete: {recon_mesh_metrics['complete']*100:.2f} ± {recon_mesh_metrics['complete_std']*100:.2f} cm\n")
        f.write(f"Accuracy: {recon_mesh_metrics['accuracy']*100:.2f} ± {recon_mesh_metrics['accuracy_std']*100:.2f} cm\n")
        f.write(f"GT Points: {recon_mesh_metrics['num_gt_points']}\n")
        f.write(f"Predicted Points: {recon_mesh_metrics['num_pred_points']}\n")
        f.write(f"Threshold: {recon_mesh_metrics['threshold']:.3f} m\n")
        f.write("\n")

        f.write("=== Mesh Priors Metrics ===\n")
        f.write(f"Chamfer Distance Priors: {recon_mesh_priors_metrics['chamfer_distance']*100:.2f} cm\n")
        f.write(f"F1 Score Priors: {recon_mesh_priors_metrics['f1_score']*100:.2f}%\n")
        f.write(f"Precision Priors: {recon_mesh_priors_metrics['precision']*100:.2f}%\n")
        f.write(f"Recall Priors: {recon_mesh_priors_metrics['recall']*100:.2f}%\n")
        f.write(f"Complete Ratio Priors: {recon_mesh_priors_metrics['complete_ratio']*100:.2f} ± {recon_mesh_priors_metrics['complete_ratio_std']*100:.2f}%\n")
        f.write(f"Complete Priors: {recon_mesh_priors_metrics['complete']*100:.2f} ± {recon_mesh_priors_metrics['complete_std']*100:.2f} cm\n")
        f.write(f"Accuracy Priors: {recon_mesh_priors_metrics['accuracy']*100:.2f} ± {recon_mesh_priors_metrics['accuracy_std']*100:.2f} cm\n")
        f.write(f"GT Points Priors: {recon_mesh_priors_metrics['num_gt_points']}\n")
        f.write(f"Predicted Points Priors: {recon_mesh_priors_metrics['num_pred_points']}\n")
        f.write(f"Threshold Priors: {recon_mesh_priors_metrics['threshold']:.3f} m\n")
        f.write("\n")

        if args.eval_depth:
            f.write("=== Depth Metrics ===\n")
            f.write(f"Depth L1 Mean: {recon_mesh_metrics['depth_l1_mean']*100:.2f} cm\n")
            f.write(f"Depth L1 Std: {recon_mesh_metrics['depth_l1_std']*100:.2f} cm\n")
            f.write("\n")

            f.write("=== Depth Metrics Priors ===\n")
            f.write(f"Depth L1 Mean Priors: {recon_mesh_priors_metrics['depth_l1_mean']*100:.2f} cm\n")
            f.write(f"Depth L1 Std Priors: {recon_mesh_priors_metrics['depth_l1_std']*100:.2f} cm\n")
            f.write("\n")
        
        f.write("=== Raw Values (for further processing) ===\n")
        for key, value in recon_mesh_metrics.items():
            if isinstance(value, (int, float)):
                f.write(f"{key}: {value}\n")
        for key, value in recon_mesh_priors_metrics.items():
            if isinstance(value, (int, float)):
                f.write(f"{key}: {value}\n")

    
    print(f"\n评估结果已保存到: {metrics_file_path}")


# Usage examples
if __name__ == "__main__":
    args = get_parser().parse_args()
    print(args)
    
    evaluate_mesh(args)