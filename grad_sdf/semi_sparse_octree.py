import os

from grad_sdf import erl_geometry_loaded

use_octree_v2 = os.getenv("USE_OCTREE_V2", "1")  # default to use octree_v2 if erl_geometry is available

if erl_geometry_loaded and use_octree_v2:
    from .semi_sparse_octree_v2 import SemiSparseOctree
else:
    from .semi_sparse_octree_v1 import SemiSparseOctree

__all__ = ["SemiSparseOctree"]
