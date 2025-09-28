try:
    import numpy as np
    from erl_geometry import MarchingCubes, o3d, torch

    erl_geometry_loaded = True
except ImportError as e:
    print(f"Failed to import erl_geometry. Please ensure it is installed correctly: {e}")
    print("Will use fallback implementations.")
    print("However, some features may be missing or have reduced performance.")

    erl_geometry_loaded = False

    import numpy as np
    import open3d as o3d
    import torch

    # from pysdf import SDF

    from .utils.mcubes_wrapper import MarchingCubesWrapper as MarchingCubes

    # class MeshSdf(SDF):
    #     def __init__(self, vertices, faces):
    #         vertices = vertices.astype(np.float32)
    #         faces = faces.astype(np.int32)
    #         super().__init__(vertices, faces)

    #     def __call__(self, points, *args, **kwargs):
    #         points = points.astype(np.float32).T
    #         return super().__call__(points).astype(np.float64)

from open3d.visualization import gui as o3d_gui
from open3d.visualization import rendering as o3d_rendering

__all__ = [
    "erl_geometry_loaded",
    "MarchingCubes",
    "MeshSdf",
    "torch",
    "o3d",
    "np",
    "o3d_gui",
    "o3d_rendering",
]
