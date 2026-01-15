from dataclasses import dataclass

from grad_sdf.utils.config_abc import ConfigABC


@dataclass
class OctreeConfig(ConfigABC):
    resolution: float = 0.5
    tree_depth: int = 8
    semi_sparse_depth: int = 2
    init_voxel_num: int = 200000  # reduce if your memory explodes
    insertion_threshold: int = 3  # Minimum number of points to insert a voxel
    # If True, skip insertion if voxel of size 1 already exists. This can speed up insertion if the search
    # implementation is efficient enough.
    skip_insertion_if_exists: bool = True
    gradient_augmentation: bool = True
