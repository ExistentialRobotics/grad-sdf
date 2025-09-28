import os

import numpy as np
from tqdm import tqdm

from grad_sdf import MeshSdf, o3d
import argparse


def compute_sdf_ground_truth(
    gt_mesh: o3d.geometry.TriangleMesh, query_points: np.ndarray, eps: float
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute the ground truth SDF and its gradient at the query points.

    Args:
        gt_mesh: o3d.geometry.TriangleMesh, the ground truth mesh
        query_points: (N, 3) array of query points
        eps: float, small value for numerical gradient computation

    Returns:

    """
    # compute SDF
    tqdm.write("Computing ground truth SDF")
    f = MeshSdf(np.asarray(gt_mesh.vertices), np.asarray(gt_mesh.triangles))
    sdf = f(query_points.T)

    # compute gradient
    tqdm.write("Computing ground truth SDF gradient")
    grad = np.zeros_like(query_points)
    for i in range(3):
        offset = np.zeros((3,))
        offset[i] = eps
        a = f((query_points + offset).T)
        b = f((query_points - offset).T)
        grad[:, i] = (a - b) / (2 * eps)

    # normalize gradient
    grad_norm = np.linalg.norm(grad, axis=1, keepdims=True)
    grad_norm[grad_norm == 0] = 1
    grad = grad / grad_norm

    return sdf, grad


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mesh-path", type=str, required=True, help="Path to the ground truth mesh file (e.g., .ply)")
    parser.add_argument("--bound-min", type=float, nargs=3)
    parser.add_argument("--bound-max", type=float, nargs=3)
    parser.add_argument("--offset", type=float, nargs=3, help="Offset to move the bounds")
    parser.add_argument("--grid-resolution", type=float, default=0.0125, help="Resolution of the grid to generate")
    parser.add_argument("--eps", type=float, default=0.001, help="Small value for numerical gradient computation")
    parser.add_argument(
        "--near-surface-sdf-range",
        type=float,
        nargs=2,
        default=(-0.1, 0.2),
        help="SDF range to consider as near surface",
    )
    parser.add_argument("--output-dir", type=str, required=True, help="Output directory")

    args = parser.parse_args()

    mesh_path: str = args.mesh_path
    bound_min: list[float] = args.bound_min
    bound_max: list[float] = args.bound_max
    offset: list[float] = args.offset
    grid_resolution: float = args.grid_resolution
    eps: float = args.eps
    near_surface_sdf_range: list[float] = args.near_surface_sdf_range
    output_dir: str = args.output_dir

    mesh = o3d.io.read_triangle_mesh(mesh_path)
    if bound_min is None:
        bound_min = mesh.get_min_bound()
        bound_min -= args.near_surface_sdf_range[0] * 2
    if bound_max is None:
        bound_max = mesh.get_max_bound()
        bound_max += args.near_surface_sdf_range[0] * 2
    if offset is None:
        offset = np.zeros((3,))
    bound_min -= offset
    bound_max -= offset

    os.makedirs(output_dir, exist_ok=True)

    x = np.arange(bound_min[0], bound_max[0], grid_resolution)
    y = np.arange(bound_min[1], bound_max[1], grid_resolution)
    z = np.arange(bound_min[2], bound_max[2], grid_resolution)
    x_size = x.shape[0]
    y_size = y.shape[0]
    z_size = z.shape[0]
    grid_points = np.stack(np.meshgrid(x, y, z, indexing="ij"), axis=-1)

    gt_sdf_values, gt_sdf_grad = compute_sdf_ground_truth(mesh, grid_points.reshape(-1, 3), eps)
    gt_sdf_values = gt_sdf_values.reshape(x_size, y_size, z_size)
    gt_sdf_grad = gt_sdf_grad.reshape(x_size, y_size, z_size, 3)

    np.save(os.path.join(output_dir, "grid_points.npy"), grid_points)
    np.save(os.path.join(output_dir, "gt_sdf_values.npy"), gt_sdf_values)
    np.save(os.path.join(output_dir, "gt_sdf_grad.npy"), gt_sdf_grad)

    # remove outliers
    threshold = max(bound_max[0] - bound_min[0], bound_max[1] - bound_min[1], bound_max[2] - bound_min[2]) * 0.5
    valid_mask = np.abs(gt_sdf_values) <= threshold

    # near surface points
    near_surface_mask = (gt_sdf_values >= near_surface_sdf_range[0]) & (gt_sdf_values <= near_surface_sdf_range[1])
    near_surface_mask &= valid_mask
    print(f"Number of near surface points: {np.sum(near_surface_mask)}/{gt_sdf_values.size}")
    np.save(os.path.join(output_dir, "mask_near_surface.npy"), near_surface_mask)

    # far away points
    far_away_mask = (gt_sdf_values > near_surface_sdf_range[1]) & valid_mask
    print(f"Number of far surface points: {np.sum(far_away_mask)}/{gt_sdf_values.size}")
    np.save(os.path.join(output_dir, "mask_far_surface.npy"), far_away_mask)


if __name__ == "__main__":
    main()
