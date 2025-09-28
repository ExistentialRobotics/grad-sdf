import numpy as np
import open3d as o3d


class MeshSdf:
    def __init__(self, vertices: np.ndarray, triangles: np.ndarray):
        self.scene = o3d.t.geometry.RaycastingScene()

        mesh = o3d.geometry.TriangleMesh()
        mesh.vertices = o3d.utility.Vector3dVector(vertices.astype(np.float64))
        mesh.triangles = o3d.utility.Vector3iVector(triangles.astype(np.int32))
        mesh.compute_triangle_normals()
        mesh.compute_vertex_normals()

        self.scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))

        # Open3D determines the sign by checking if the point is inside the closed surface.
        # If inside the closed surface, the returned sign is -1.
        # However, the mesh may have flipped normals. We need to check this.
        self.flip_sign = False
        pcd = mesh.sample_points_uniformly(1000, True)
        points = np.asarray(pcd.points) + 0.01 * np.asarray(pcd.normals)
        sdf = self(points)
        self.flip_sign = np.mean(sdf) < 0

    def __call__(self, points: np.ndarray, *args, **kwargs):
        positions = o3d.core.Tensor(points.astype(np.float32), o3d.core.Dtype.Float32)
        sdf = self.scene.compute_signed_distance(positions, nsamples=3)
        sdf = sdf.numpy()
        if self.flip_sign:
            sdf = -sdf
        return sdf
