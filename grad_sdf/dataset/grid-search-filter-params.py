import os
from pathlib import Path
import json

import numpy as np
import open3d as o3d
import torch
from pytorch3d.ops import knn_points
from scipy.spatial import cKDTree
from tqdm import tqdm

# 设置路径
data_dir = Path("/home/qihao/workplace/grad-sdf/data/newercollege-lidar")
ply_dir = data_dir / "ply"
poses_file = data_dir / "poses.txt"
gt_pointcloud_file = data_dir / "gt-pointcloud.ply"

# 检测GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")

# 方法1的固定参数
distance_threshold_method1 = 0.1

# 方法2的参数搜索空间
param_grid = {
    "distance_threshold": [0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.10, 0.12, 0.15],
    "n_neighbors": [2, 3, 4, 5, 6, 8, 10],
}

print("正在读取ground truth点云...")
gt_pcd = o3d.io.read_point_cloud(str(gt_pointcloud_file))
gt_points = np.asarray(gt_pcd.points)
print(f"Ground truth点云有 {len(gt_points)} 个点")

# 使用scipy的cKDTree（对大点云更快）
print(f"正在构建GT点云的cKDTree...")
gt_kdtree = cKDTree(gt_points)
print(f"✓ GT cKDTree已准备好")

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

# 选择测试文件
ply_files = sorted(ply_dir.glob("*.ply"))
test_indices = [500, 600, 700, 800, 900, 1000, 1100, 1200, 1300, 1400]
test_files = [ply_files[i] for i in test_indices]
print(f"将使用 {len(test_files)} 个文件进行参数搜索（索引: {test_indices}）")

# 预计算方法1的结果（所有参数组合都一样）
print("\n预计算方法1的过滤结果...")
method1_results = []

for i, ply_file in enumerate(tqdm(test_files, desc="计算方法1")):
    local_pcd = o3d.io.read_point_cloud(str(ply_file))
    local_points = np.asarray(local_pcd.points)

    if len(local_points) == 0:
        method1_results.append(None)
        continue

    # 获取对应的pose
    actual_index = test_indices[i]
    pose = poses[actual_index]

    # 转换到世界坐标系
    local_points_homo = np.hstack([local_points, np.ones((len(local_points), 1))])
    world_points_homo = (pose @ local_points_homo.T).T
    world_points = world_points_homo[:, :3]

    # 使用scipy的cKDTree查询最近距离（对大点云更快）
    distances, _ = gt_kdtree.query(world_points, k=1)

    method1_valid_mask = distances <= distance_threshold_method1

    method1_results.append({"local_points": local_points, "valid_mask": method1_valid_mask})

# 网格搜索
print("\n开始网格搜索...")
best_params = None
best_score = -1
all_results = []

total_combinations = len(param_grid["distance_threshold"]) * len(param_grid["n_neighbors"])
pbar = tqdm(total=total_combinations, desc="参数搜索")

for dist_thresh in param_grid["distance_threshold"]:
    for n_neighbors in param_grid["n_neighbors"]:
        # 对每个参数组合计算方法2的结果
        agreement_rates = []
        total_points = 0
        both_filtered = 0
        only_method1 = 0
        only_method2 = 0
        method1_filtered_total = 0
        method2_filtered_total = 0

        for i, result in enumerate(method1_results):
            if result is None:
                continue

            local_points = result["local_points"]
            method1_valid_mask = result["valid_mask"]

            # 方法2: frame内部点云比较 (使用PyTorch3D加速)
            local_points_torch = torch.from_numpy(local_points).float().unsqueeze(0).to(device)  # [1, N, 3]

            # 搜索k个最近邻（包括自己）
            knn_result = knn_points(local_points_torch, local_points_torch, K=n_neighbors)
            # dists: [1, N, K], 第一个是自己（距离为0），后面是最近邻
            dists = knn_result.dists[0].cpu().numpy()  # [N, K]

            # 计算平均距离（排除自己，即从索引1开始）
            avg_distances = np.mean(np.sqrt(dists[:, 1:]), axis=1)  # [N]

            # 计算每个点的深度（在相机坐标系下，就是到原点的距离）
            depths = np.linalg.norm(local_points, axis=1)  # [N]

            # 深度自适应阈值：远处的点使用更大的阈值
            adaptive_thresholds = depths / 10.0 * dist_thresh  # [N]
            method2_valid_mask = avg_distances <= adaptive_thresholds

            # 计算一致性
            method1_filtered_mask = ~method1_valid_mask
            method2_filtered_mask = ~method2_valid_mask

            agreement = np.sum(method1_valid_mask == method2_valid_mask) / len(local_points)
            agreement_rates.append(agreement)

            total_points += len(local_points)
            both_filtered += np.sum(method1_filtered_mask & method2_filtered_mask)
            only_method1 += np.sum(method1_filtered_mask & ~method2_filtered_mask)
            only_method2 += np.sum(~method1_filtered_mask & method2_filtered_mask)
            method1_filtered_total += np.sum(method1_filtered_mask)
            method2_filtered_total += np.sum(method2_filtered_mask)

        # 计算评分指标
        avg_agreement = np.mean(agreement_rates)
        std_agreement = np.std(agreement_rates)

        # 计算过滤率
        method1_filter_rate = method1_filtered_total / total_points
        method2_filter_rate = method2_filtered_total / total_points
        filter_rate_diff = abs(method1_filter_rate - method2_filter_rate)

        # 关键指标：
        # Recall (召回率): 方法二找到了多少方法一过滤的点
        recall = both_filtered / method1_filtered_total if method1_filtered_total > 0 else 0

        # Precision (精确率): 方法二过滤的点中有多少是正确的（即方法一也过滤的）
        precision = both_filtered / method2_filtered_total if method2_filtered_total > 0 else 0

        # False Positive Rate: 方法二误过滤的点的比例
        false_positive_rate = only_method2 / total_points

        # F1-score: 召回率和精确率的调和平均
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

        # 综合评分：主要目标是召回率高，次要目标是误报率低
        # 使用加权组合：优先考虑召回率，然后惩罚误报
        score = recall - 1.0 * false_positive_rate

        all_results.append(
            {
                "distance_threshold": dist_thresh,
                "n_neighbors": n_neighbors,
                "avg_agreement": avg_agreement,
                "std_agreement": std_agreement,
                "score": score,
                "recall": recall,
                "precision": precision,
                "f1_score": f1_score,
                "false_positive_rate": false_positive_rate,
                "method1_filter_rate": method1_filter_rate,
                "method2_filter_rate": method2_filter_rate,
                "filter_rate_diff": filter_rate_diff,
                "both_filtered": both_filtered,
                "only_method1": only_method1,
                "only_method2": only_method2,
                "method1_filtered_total": method1_filtered_total,
                "method2_filtered_total": method2_filtered_total,
                "total_points": total_points,
            }
        )

        if score > best_score:
            best_score = score
            best_params = {"distance_threshold": dist_thresh, "n_neighbors": n_neighbors}

        pbar.update(1)

pbar.close()

# 按评分排序
all_results.sort(key=lambda x: x["score"], reverse=True)

# 保存完整的JSON结果（方便后续重新评分）
json_output_file = data_dir / "param_search_raw_results.json"
print(f"\n正在保存完整结果到 {json_output_file}...")


# 转换numpy类型为Python原生类型以便JSON序列化
def convert_to_serializable(obj):
    """递归转换numpy类型为Python原生类型"""
    if isinstance(obj, dict):
        return {key: convert_to_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_serializable(item) for item in obj]
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    else:
        return obj


serializable_results = convert_to_serializable(all_results)

with open(json_output_file, "w", encoding="utf-8") as f:
    json.dump(serializable_results, f, indent=2, ensure_ascii=False)
print(f"✓ 完整结果已保存（{len(all_results)} 组参数）")

# 输出结果
print("\n" + "=" * 110)
print("参数搜索结果 - Top 15")
print("=" * 110)
print(
    f"{'排名':<6} {'距离阈值':<12} {'邻居数':<8} {'评分':<12} "
    f"{'召回率':<12} {'精确率':<12} {'F1':<12} {'误报率':<12}"
)
print("-" * 110)

for rank, result in enumerate(all_results[:15], 1):
    print(
        f"{rank:<6} {result['distance_threshold']:<12.2f} {result['n_neighbors']:<8} "
        f"{result['score']:<12.4f} "
        f"{result['recall']*100:>10.2f}%  {result['precision']*100:>10.2f}%  "
        f"{result['f1_score']*100:>10.2f}%  {result['false_positive_rate']*100:>10.2f}%"
    )

print("\n" + "=" * 110)
print("🎯 最佳参数配置:")
print("=" * 110)
best_result = all_results[0]
print(f"distance_threshold_method2 = {best_result['distance_threshold']:.2f}")
print(f"n_neighbors = {best_result['n_neighbors']}")

print(f"\n📊 核心性能指标:")
print(f"  综合评分: {best_result['score']:.4f}")
print(f"  召回率 (Recall): {best_result['recall']*100:.2f}%  <- 方法二找到了方法一过滤点的比例")
print(f"  精确率 (Precision): {best_result['precision']*100:.2f}%  <- 方法二过滤的点中正确的比例")
print(f"  F1-Score: {best_result['f1_score']*100:.2f}%")
print(f"  误报率: {best_result['false_positive_rate']*100:.2f}%  <- 方法二误过滤的点的比例")
print(f"  一致性: {best_result['avg_agreement']*100:.2f}% (±{best_result['std_agreement']*100:.2f}%)")

print(f"\n📈 详细统计:")
print(f"  总点数: {best_result['total_points']}")
print(f"  方法1过滤点数: {best_result['method1_filtered_total']} ({best_result['method1_filter_rate']*100:.2f}%)")
print(f"  方法2过滤点数: {best_result['method2_filtered_total']} ({best_result['method2_filter_rate']*100:.2f}%)")
print(
    f"\n  ✅ 两种方法都过滤 (正确): {best_result['both_filtered']} ({best_result['both_filtered']/best_result['total_points']*100:.2f}%)"
)
print(
    f"  ⚠️  只被方法1过滤 (漏检): {best_result['only_method1']} ({best_result['only_method1']/best_result['total_points']*100:.2f}%)"
)
print(
    f"  ❌ 只被方法2过滤 (误报): {best_result['only_method2']} ({best_result['only_method2']/best_result['total_points']*100:.2f}%)"
)

print(f"\n💡 解读:")
if best_result["recall"] > 0.95:
    print(f"  ✓ 召回率很高 ({best_result['recall']*100:.1f}%)，方法二能找到几乎所有方法一过滤的点")
elif best_result["recall"] > 0.85:
    print(f"  ✓ 召回率较高 ({best_result['recall']*100:.1f}%)，方法二能找到大部分方法一过滤的点")
else:
    print(f"  ✗ 召回率偏低 ({best_result['recall']*100:.1f}%)，方法二漏检了较多点")

if best_result["false_positive_rate"] < 0.01:
    print(f"  ✓ 误报率很低 ({best_result['false_positive_rate']*100:.1f}%)，方法二很少误过滤")
elif best_result["false_positive_rate"] < 0.05:
    print(f"  ✓ 误报率较低 ({best_result['false_positive_rate']*100:.1f}%)，方法二误过滤较少")
else:
    print(f"  ✗ 误报率偏高 ({best_result['false_positive_rate']*100:.1f}%)，方法二过滤了较多不该过滤的点")

# 保存完整结果到文件
output_file = data_dir / "param_search_results.txt"
with open(output_file, "w") as f:
    f.write("完整参数搜索结果\n")
    f.write("=" * 110 + "\n")
    f.write(
        f"{'排名':<6} {'距离阈值':<12} {'邻居数':<8} {'评分':<12} "
        f"{'召回率':<12} {'精确率':<12} {'F1':<12} {'误报率':<12}\n"
    )
    f.write("-" * 110 + "\n")
    for rank, result in enumerate(all_results, 1):
        f.write(
            f"{rank:<6} {result['distance_threshold']:<12.2f} {result['n_neighbors']:<8} "
            f"{result['score']:<12.4f} "
            f"{result['recall']*100:>10.2f}%  {result['precision']*100:>10.2f}%  "
            f"{result['f1_score']*100:>10.2f}%  {result['false_positive_rate']*100:>10.2f}%\n"
        )

    f.write("\n" + "=" * 110 + "\n")
    f.write("最佳参数详情\n")
    f.write("=" * 110 + "\n")
    best = all_results[0]
    f.write(f"distance_threshold_method2 = {best['distance_threshold']:.2f}\n")
    f.write(f"n_neighbors = {best['n_neighbors']}\n")
    f.write(f"\n性能指标:\n")
    f.write(f"  召回率: {best['recall']*100:.2f}%\n")
    f.write(f"  精确率: {best['precision']*100:.2f}%\n")
    f.write(f"  F1-Score: {best['f1_score']*100:.2f}%\n")
    f.write(f"  误报率: {best['false_positive_rate']*100:.2f}%\n")
    f.write(f"  综合评分: {best['score']:.4f}\n")

print(f"\n完整结果已保存到: {output_file}")

print("\n" + "=" * 110)
print("💡 提示:")
print("=" * 110)
print("如果你想使用不同的评分公式重新评估参数，可以运行:")
print("  python grad_sdf/dataset/rescore-params.py")
print("\n在 rescore-params.py 中修改 custom_score() 函数来定义你的评分公式")
print(f"原始数据已保存在: {json_output_file}")
