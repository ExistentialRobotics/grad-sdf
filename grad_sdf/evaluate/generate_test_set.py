import open3d as o3d
import numpy as np


def generate_test_set(gt_mesh: o3d.geometry.TriangleMesh, bound: list, resolution: float, offset: float):

    # generate grid
    bound = np.array(bound) - offset
    x_min, x_max = bound[0]
    y_min, y_max = bound[1]
    z_min, z_max = bound[2]

    x_range = x_max - x_min
    y_range = y_max - y_min
    z_range = z_max - z_min

    x_size = int(np.ceil(x_range / resolution))
    y_size = int(np.ceil(y_range / resolution))
    z_size = int(np.ceil(z_range / resolution))

    x = np.linspace(x_min, x_max, x_size)
    y = np.linspace(y_min, y_max, y_size)
    z = np.linspace(z_min, z_max, z_size)
    xx, yy, zz = np.meshgrid(x, y, z, indexing="xy")

    grid_points = np.stack([xx, yy, zz], axis=-1)

    # calculate sdf
