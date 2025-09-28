from dataclasses import dataclass

import tinycudann as tcnn
import torch
import torch.nn as nn

from grad_sdf.utils.config_abc import ConfigABC


@dataclass
class ResidualNetConfig(ConfigABC):
    bound_min: list = None  # (3, ) minimum corner of the bounding box of the scene
    bound_max: list = None  # (3, ) maximum corner of the bounding box of the scene
    resolution: float = 0.05  # the size of the smallest voxel
    num_levels: int = 4  # number of levels in the hash encoding
    feature_dims: int = 2  # number of features per level
    log2_T: int = 19  # log2 of the hash table size
    per_level_scale: float = 2.0  # per level scale factor
    mlp_activation: str = "LeakyReLU"  # activation function for the MLP
    hidden_dims: int = 64  # number of hidden dimensions
    n_hidden_layers: int = 5  # number of hidden layers
    scale_with_bound: bool = True  # whether to scale the input with the bounding box
    output_sdf_scale: float = 0.1  # scale the output SDF


class ResidualNet(nn.Module):
    def __init__(self, cfg: ResidualNetConfig):
        """
        Args:
            cfg: configuration of the network
        """
        super().__init__()
        self.cfg = cfg

        if self.cfg.bound_min is not None and self.cfg.bound_max is not None:
            self.register_buffer("bound_min", torch.FloatTensor(cfg.bound_min))
            self.register_buffer("bound_max", torch.FloatTensor(cfg.bound_max))
            self.register_buffer("bound_dis", self.bound_max - self.bound_min)
        else:
            self.bound_min = None
            self.bound_max = None
            self.bound_dis = None

        self.max_dis = torch.ceil(torch.max(self.bound_dis))
        N_min = int(self.max_dis / cfg.resolution)
        self.hash_sdf_out = tcnn.NetworkWithInputEncoding(
            n_input_dims=3,
            n_output_dims=1,
            encoding_config={
                "otype": "Grid",
                "type": "Hash",
                "n_levels": cfg.num_levels,  # number of levels
                "n_features_per_level": cfg.feature_dims,  # number of features per level
                "log2_hashmap_size": cfg.log2_T,  # each level has a hash table of size 2^log2_T
                "base_resolution": N_min,  # coarsest grid size
                # each level's grid size = base_resolution * per_level_scale^level
                "per_level_scale": cfg.per_level_scale,
                "interpolation": "Linear",
            },
            network_config={
                "otype": "FullyFusedMLP",
                "activation": cfg.mlp_activation,
                "output_activation": "None",
                "n_neurons": cfg.hidden_dims,
                "n_hidden_layers": cfg.n_hidden_layers,
            },
        )

    def get_sdf(self, xyz: torch.Tensor):
        if self.cfg.scale_with_bound:
            assert self.bound_min is not None
            assert self.bound_dis is not None
            xyz = (xyz - self.bound_min.to(xyz.device)) / self.bound_dis.to(xyz.device)
        sdf = self.hash_sdf_out(xyz) * self.cfg.output_sdf_scale
        return sdf

    def forward(self, xyz: torch.Tensor):
        sdf = self.get_sdf(xyz.view(-1, 3)).view(xyz.shape[:-1])
        return sdf
