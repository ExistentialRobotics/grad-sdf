import glob
import os

import torch

pkg_dir = os.path.dirname(__file__)
svo_lib = glob.glob("svo.*.so", root_dir=pkg_dir)[0]
torch.classes.load_library(os.path.join(pkg_dir, svo_lib))

Octree = torch.classes.svo.Octree

__all__ = ["Octree"]
