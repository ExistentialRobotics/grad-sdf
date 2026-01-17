import os
from pathlib import Path

import numpy as np
import open3d as o3d
from tqdm import tqdm

# 设置路径
data_dir = Path("/home/qihao/workplace/grad-sdf/data/newercollege-lidar-rotated2")
ply_dir = data_dir / "ply"
poses_file = data_dir / "poses.txt"
gt_pointcloud_file = data_dir / "gt-pointcloud.ply"
output_dir = data_dir / "ply-filtered-0.1"

# 创建输出目录
output_dir.mkdir(exist_ok=True, parents=True)

print("正在读取ground truth点云...")
gt_pcd = o3d.io.read_point_cloud(str(gt_pointcloud_file))
gt_points = np.asarray(gt_pcd.points)
print(f"Ground truth点云有 {len(gt_points)} 个点")

# 构建KD树用于快速最近邻搜索
print("正在构建KD树...")
gt_pcd_tree = o3d.geometry.KDTreeFlann(gt_pcd)

# 读取所有poses
print("正在读取poses...")
with open(poses_file, "r") as f:
    lines = f.readlines()

poses = []
for line in lines:
    if not line.strip():
        continue
    values = list(map(float, line.strip().split()))
    if len(values) == 16:
        matrix = np.array(values).reshape(4, 4)
        poses.append(matrix)

print(f"读取了 {len(poses)} 个poses")

# 距离阈值
distance_threshold = 0.1

# 统计信息
total_points_before = 0
total_points_after = 0

# 处理每个ply文件
ply_files = sorted(ply_dir.glob("*.ply"))
print(f"找到 {len(ply_files)} 个ply文件")

if len(ply_files) != len(poses):
    print(f"警告: ply文件数量 ({len(ply_files)}) 与poses数量 ({len(poses)}) 不匹配！")

for i, ply_file in enumerate(tqdm(ply_files, desc="处理ply文件")):
    # 读取局部点云
    local_pcd = o3d.io.read_point_cloud(str(ply_file))
    local_points = np.asarray(local_pcd.points)

    if len(local_points) == 0:
        # 如果点云为空，创建空文件
        output_file = output_dir / ply_file.name
        o3d.io.write_point_cloud(str(output_file), local_pcd)
        continue

    total_points_before += len(local_points)

    # 获取对应的pose
    pose = poses[i]

    # 转换到世界坐标系
    # 将局部点云转换为齐次坐标
    local_points_homo = np.hstack([local_points, np.ones((len(local_points), 1))])
    # 应用变换矩阵
    world_points_homo = (pose @ local_points_homo.T).T
    world_points = world_points_homo[:, :3]

    # 计算与ground truth点云的最近距离
    valid_mask = np.zeros(len(world_points), dtype=bool)

    for j, point in enumerate(world_points):
        # 搜索最近邻
        [k, idx, dist] = gt_pcd_tree.search_knn_vector_3d(point, 1)
        # dist是平方距离，需要开方
        distance = np.sqrt(dist[0])

        if distance <= distance_threshold:
            valid_mask[j] = True

    # 过滤点云
    filtered_points = local_points[valid_mask]
    total_points_after += len(filtered_points)

    # 创建过滤后的点云
    filtered_pcd = o3d.geometry.PointCloud()
    filtered_pcd.points = o3d.utility.Vector3dVector(filtered_points)

    # 如果原始点云有颜色，也复制颜色
    if local_pcd.has_colors():
        local_colors = np.asarray(local_pcd.colors)
        filtered_colors = local_colors[valid_mask]
        filtered_pcd.colors = o3d.utility.Vector3dVector(filtered_colors)

    # 保存过滤后的点云
    output_file = output_dir / ply_file.name
    o3d.io.write_point_cloud(str(output_file), filtered_pcd)

print(f"\n处理完成！")
print(f"总点数（过滤前）: {total_points_before}")
print(f"总点数（过滤后）: {total_points_after}")
print(f"保留比例: {total_points_after / total_points_before * 100:.2f}%")
print(f"过滤后的点云已保存到: {output_dir}")
