import trimesh
import numpy as np
import random
import open3d as o3d
from tqdm import tqdm

def trimesh_to_open3d(trimesh_mesh):
    """将trimesh.Trimesh转换为open3d.geometry.TriangleMesh"""
    o3d_mesh = o3d.geometry.TriangleMesh()
    o3d_mesh.vertices = o3d.utility.Vector3dVector(trimesh_mesh.vertices)
    o3d_mesh.triangles = o3d.utility.Vector3iVector(trimesh_mesh.faces)
    o3d_mesh.compute_vertex_normals()
    return o3d_mesh

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


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="计算重建mesh与GT mesh的深度L1误差")
    parser.add_argument("--gt_mesh_path", type=str, required=True, help="GT mesh的路径")
    parser.add_argument("--rec_mesh_path", type=str, required=True, help="重建mesh的路径")
    parser.add_argument("--n_imgs", type=int, default=1000, help="采样视角数量，默认1000")
    args = parser.parse_args()

    gt_mesh = trimesh.load(args.gt_mesh_path)
    rec_mesh = trimesh.load(args.rec_mesh_path)
    errors = calculate_depth_L1(gt_mesh, rec_mesh, n_imgs=args.n_imgs)
    print(f"平均深度L1误差: {errors.mean()}")