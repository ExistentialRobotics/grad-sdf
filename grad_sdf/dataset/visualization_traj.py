import argparse
import numpy as np
import open3d as o3d
import os

def tum_row_to_mat(row):
    # row: [t, tx, ty, tz, qx, qy, qz, qw] (w-last)
    tx, ty, tz, qx, qy, qz, qw = row[1:]
    q = np.array([qw, qx, qy, qz], dtype=float)  # (w, x, y, z)
    # quaternion -> rotation matrix
    w, x, y, z = q
    R = np.array([
        [1-2*(y*y+z*z),   2*(x*y - z*w),   2*(x*z + y*w)],
        [  2*(x*y + z*w), 1-2*(x*x+z*z),   2*(y*z - x*w)],
        [  2*(x*z - y*w),   2*(y*z + x*w), 1-2*(x*x+y*y)]
    ], dtype=float)
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3]  = np.array([tx, ty, tz], dtype=float)
    return T

def load_poses(traj_path):
    # 兼容三种：每行8/12/16，或整个文件是连续浮点数
    floats = []
    rows = []
    with open(traj_path, 'r') as f:
        for line in f:
            if not line.strip():
                continue
            parts = line.strip().split()
            try:
                vals = [float(x) for x in parts]
            except:
                continue
            rows.append(vals)
            floats.extend(vals)

    poses = []
    # 优先按“逐行列数稳定”来判定
    if len(rows) > 0:
        lens = [len(r) for r in rows]
        uniq = set(lens)
        if len(uniq) == 1:
            L = lens[0]
            if L == 8:
                for r in rows:
                    poses.append(tum_row_to_mat(r))
            elif L == 12:
                for r in rows:
                    T = np.eye(4)
                    T[:3, :4] = np.array(r, dtype=float).reshape(3, 4)
                    poses.append(T)
            elif L == 16:
                for r in rows:
                    poses.append(np.array(r, dtype=float).reshape(4, 4))
            else:
                # 回退到原始连续数组解析
                poses = parse_by_flat(floats)
        else:
            # 行长不一致，回退
            poses = parse_by_flat(floats)
    else:
        # 空文件？
        poses = []

    if len(poses) == 0:
        raise ValueError("无法从轨迹文件解析出位姿。请检查格式。")
    return np.stack(poses, axis=0)

def parse_by_flat(floats):
    arr = np.array(floats, dtype=float)
    poses = []
    if arr.size % 16 == 0:
        poses = arr.reshape(-1, 4, 4)
    elif arr.size % 12 == 0:
        poses = arr.reshape(-1, 3, 4)
        poses = np.concatenate([poses, np.tile(np.array([0,0,0,1.0]).reshape(1,1,4), (poses.shape[0],1,1))], axis=1)
    else:
        raise ValueError("无法按 16 或 12 浮点数一帧来整除。")
    return poses

def make_trajectory_lineset(positions, color=(0.95, 0.2, 0.2)):
    pts = o3d.utility.Vector3dVector(positions)
    lines = [[i, i+1] for i in range(len(positions)-1)]
    ls = o3d.geometry.LineSet(points=pts, lines=o3d.utility.Vector2iVector(lines))
    colors = o3d.utility.Vector3dVector([color for _ in lines])
    ls.colors = colors
    return ls

def main(args):
    # 1) 载入mesh
    mesh = o3d.io.read_triangle_mesh(args.mesh)
    if mesh is None or len(mesh.triangles) == 0:
        raise ValueError(f"读取网格失败或为空：{args.mesh}")
    if not mesh.has_vertex_normals():
        mesh.compute_vertex_normals()

    # 2) 载入轨迹 -> 相机中心（t）
    poses = load_poses(args.traj)
    positions = poses[:, :3, 3]  # camera-to-world 的平移
    # 也可按相机前向画箭头，这里只画折线

    # 3) 几何体
    traj = make_trajectory_lineset(positions, color=(0.96, 0.2, 0.2))
    # 小相机点
    cam_points = o3d.geometry.PointCloud()
    cam_points.points = o3d.utility.Vector3dVector(positions)
    import numpy as np
    cam_points.colors = o3d.utility.Vector3dVector(np.tile([0.1, 0.1, 0.1], (len(positions), 1)))

    # 4) 渲染（离屏）
    W, H = args.width, args.height
    renderer = o3d.visualization.rendering.OffscreenRenderer(W, H)
    scene = renderer.scene
    scene.set_background([1, 1, 1, 1])  # 白底

    # Mesh 材质（线框模式，方便看到内部轨迹）
    mat_mesh = o3d.visualization.rendering.MaterialRecord()
    mat_mesh.shader = "unlitLine"  # 使用线条shader
    mat_mesh.base_color = [0.6, 0.6, 0.6, 1.0]  # 灰色线框
    mat_mesh.line_width = 1.0
    
    # 转换mesh为线框
    mesh_wireframe = o3d.geometry.LineSet.create_from_triangle_mesh(mesh)

    # Traj 材质（线）
    mat_line = o3d.visualization.rendering.MaterialRecord()
    mat_line.shader = "unlitLine"

    # 相机点材质（不发光的点）
    mat_pts = o3d.visualization.rendering.MaterialRecord()
    mat_pts.shader = "defaultUnlit"
    mat_pts.point_size = 2.0

    scene.add_geometry("mesh", mesh_wireframe, mat_mesh)
    scene.add_geometry("traj", traj, mat_line)
    scene.add_geometry("cams", cam_points, mat_pts)

    # 5) 视角自动居中
    # 计算联合包围盒
    bbox_mesh = mesh.get_axis_aligned_bounding_box()
    pts = np.vstack([np.asarray(mesh.vertices), positions])  # 网格顶点 + 相机轨迹点
    bbox_all = o3d.geometry.AxisAlignedBoundingBox.create_from_points(
        o3d.utility.Vector3dVector(pts)
    )
    center  = bbox_all.get_center()
    extents = bbox_all.get_extent()
    diag = np.linalg.norm(extents)
    print(f"bbox_all: {bbox_all}")
    print(f"center: {center}")
    print(f"extents: {extents}")
    print(f"diag: {diag}")

    # 设置相机：从对角方向看过去
    eye = center + np.array([+0.9*extents[0], -1.2*extents[1], +0.7*extents[2]])
    up  = np.array([0.0, 0.0, 1.0])
    scene.camera.look_at(center, eye, up)
    near = max(1e-3, diag * 1e-3)
    far  = diag * 5.0 + 1.0
    try:
        FovType = o3d.visualization.rendering.Camera.FovType  # 新版 API
        scene.camera.set_projection(60.0, W / H, near, far, FovType.Vertical)
    except Exception:
        # 老版本可能只接受整数枚举（0=Vertical, 1=Horizontal），再退一层
        try:
            scene.camera.set_projection(60.0, W / H, near, far, 0)
        except Exception:
            # 实在不行就不改投影，保持默认的透视参数（一般也够用）
            pass

    # 6) 轴系（可选）
    if args.axes:
        axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=diag*0.05, origin=center)
        scene.add_geometry("axes", axis, mat_mesh)

    # 7) 保存图片（稳健版）
    img = renderer.render_to_image()

    # 确保目录存在
    out_dir = os.path.dirname(args.out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    # 先试 Open3D 自带写出
    ok = False
    try:
        ok = o3d.io.write_image(args.out, img)  # 不要传 quality；PNG 会忽略
    except Exception:
        ok = False

    # 失败就用 PIL 兜底（最稳）
    if not ok:
        try:
            import numpy as np
            np_img = np.asarray(img)  # HxWxC，通常 uint8
            # 某些版本可能给 float [0,1]，转成 uint8
            if np_img.dtype != np.uint8:
                np_img = (np.clip(np_img, 0, 1) * 255).astype(np.uint8)

            # 用 PIL 保存（支持 RGBA/PNG）
            from PIL import Image
            Image.fromarray(np_img).save(args.out)
            ok = True
        except Exception as e:
            print("PIL 写图失败：", repr(e))
            ok = False

    # 再失败，用 imageio 兜底
    if not ok:
        try:
            import imageio.v3 as iio
            np_img = np.asarray(img)
            if np_img.dtype != np.uint8:
                np_img = (np.clip(np_img, 0, 1) * 255).astype(np.uint8)
            iio.imwrite(args.out, np_img)
            ok = True
        except Exception as e:
            print("imageio 写图失败：", repr(e))
            ok = False

    if not ok:
        raise RuntimeError(f"写出图片失败：{args.out}")
    else:
        print(f"Done. Saved to {args.out}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--traj", required=True, help="路径：Replica 轨迹文件 traj.txt")
    parser.add_argument("--mesh", required=True, help="路径：room0_mesh.ply")
    parser.add_argument("--out", default="trajectory.png", help="输出 PNG 文件名")
    parser.add_argument("--width", type=int, default=1600)
    parser.add_argument("--height", type=int, default=1000)
    parser.add_argument("--line_width", type=float, default=2.5)
    parser.add_argument("--axes", action="store_true", help="叠加坐标轴")
    args = parser.parse_args()
    main(args)
