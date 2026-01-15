"""
Common methods for semi-sparse octree implementations. This class is agnostic to the underlying octree implementation.
The derived classes should implement the octree operations of inserting points and finding voxel indices. When the tree
is modified, the buffers should be updated.

The sdf_priors and grad_priors are learnable parameters that store the SDF and gradient priors for each vertex in the
octree. The buffers store the octree structure:
- voxels: (N, 4) [x, y, z, voxel_size]
- voxel_centers: (N, 3) in meter
- vertex_indices: (N, 8) index of vertices, -1 if not exists
- structure: (N, 8) [children(8)]
"""

from abc import ABC, abstractmethod

from grad_sdf import torch
from grad_sdf.ga_trilinear import ga_trilinear
from grad_sdf.octree_config import OctreeConfig


class SemiSparseOctreeBase(torch.nn.Module, ABC):

    def __init__(self, cfg: OctreeConfig):
        super(SemiSparseOctreeBase, self).__init__()
        self.cfg = cfg

        # Initialize learnable parameters for SDF and gradient priors of each vertex
        self.sdf_priors = torch.nn.Parameter(
            torch.zeros((self.cfg.init_voxel_num,), dtype=torch.float32),
            requires_grad=True,
        )
        self.grad_priors = torch.nn.Parameter(
            torch.zeros((self.cfg.init_voxel_num, 3), dtype=torch.float32),
            requires_grad=True,
        )

        self.ever_inserted = False

        n = self.cfg.init_voxel_num
        self.register_buffer("voxels", torch.zeros((n, 4), dtype=torch.float32))
        self.register_buffer("voxel_centers", torch.zeros((n, 3), dtype=torch.float32))
        self.register_buffer("vertex_indices", torch.zeros((n, 8), dtype=torch.int32))
        self.register_buffer("structure", torch.zeros((n, 8), dtype=torch.int32))

        self.voxels: torch.Tensor  # (N, 4) [x, y, z, voxel_size]
        self.voxel_centers: torch.Tensor  # (N, 3) in meter
        self.vertex_indices: torch.Tensor  # (N, 8) index of vertices, -1 if not exists
        self.structure: torch.Tensor  # (N, 8) [children(8)]

    @abstractmethod
    def points_to_voxels(self, points: torch.Tensor) -> torch.Tensor:
        """
        Converts points to voxel coordinates.
        Args:
            points: (..., 3) point cloud in world coordinates
        Returns:
            voxels: (..., 3) voxel coordinates
        """
        pass

    @abstractmethod
    def insert_voxels(self, voxels: torch.Tensor) -> torch.Tensor:
        """
        Inserts voxels into the octree, updates the buffers and returns the voxel indices.
        Args:
            voxels: (n_voxels, 3) voxel coordinates
        Returns:
            voxel_indices: (n_voxels,) index of the voxel for each voxel
        """
        pass

    @torch.no_grad()
    def insert_points(self, points: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Inserts points into the octree.
        Args:
            points: (n_points, 3) point cloud in world coordinates
        Returns:
            voxels_unique: (n_unique, 3) unique voxel coordinates inserted
            voxel_indices: (n_unique,) index of the voxel for each voxel, on CPU.
            points_valid: (n_valid_points, 3) points that fall into valid voxels (count > insertion_threshold)
            point_voxel_indices: (n_valid_points,) voxel index for each point in points_valid, on CPU.
        """
        voxels = self.points_to_voxels(points)  # (n_points, 3) voxel coordinates
        voxels_raw, inverse, counts = torch.unique(voxels, dim=0, return_inverse=True, return_counts=True)
        valid_raw_mask = counts > self.cfg.insertion_threshold  # (n_raw,)
        voxels_unique = voxels_raw[valid_raw_mask]  # (n_unique, 3) of grid coordinates

        # select points in valid voxels
        valid_point_mask = valid_raw_mask[inverse]  # (n_points,)
        points_valid = points[valid_point_mask]  # (n_valid_points, 3)
        inverse_valid = inverse[valid_point_mask]  # (n_valid_points,)

        if self.cfg.skip_insertion_if_exists and self.ever_inserted:
            device = self.sdf_priors.device
            voxel_indices = self.find_voxel_indices(voxels_unique.to(device), True)  # (n_unique,)
            voxel_sizes = self.get_voxel_discrete_size(voxel_indices)  # (n_unique,)
            mask = voxel_sizes != 1  # only insert voxels that do not exist at size 1
            voxel_indices[mask] = self.insert_voxels(voxels_unique[mask]).to(device)
        else:
            voxel_indices = self.insert_voxels(voxels_unique)

        # map each valid point to its voxel index (based on voxels_raw -> voxels_unique)
        remap = torch.full((voxels_raw.shape[0],), -1, dtype=torch.long, device=voxels_raw.device)  # (n_raw,)
        remap[valid_raw_mask] = torch.arange(voxels_unique.shape[0], device=voxels_raw.device, dtype=torch.long)
        point_voxel_unique_idx = remap[inverse_valid]  # (n_valid_points,)
        # NOTE: derived implementations may return voxel_indices on CPU (e.g. inserting via a CPU octree backend),
        # while point_voxel_unique_idx lives on the same device as points/voxels (often GPU). Align devices before indexing.
        if voxel_indices.device != point_voxel_unique_idx.device:
            voxel_indices_for_index = voxel_indices.to(point_voxel_unique_idx.device)
        else:
            voxel_indices_for_index = voxel_indices
        point_voxel_indices = voxel_indices_for_index[point_voxel_unique_idx]  # (n_valid_points,)

        return voxels_unique, voxel_indices.cpu(), points_valid, point_voxel_indices.cpu()

    @torch.no_grad()
    def get_voxel_discrete_size(self, voxel_indices: torch.Tensor) -> torch.Tensor:
        """
        Get the voxel sizes for the given voxel indices.
        Args:
            voxel_indices: (...) index of the voxels

        Returns:
            (..., ) voxel discrete sizes
        """
        assert self.voxels is not None, "Octree is empty. Please insert points first."
        assert voxel_indices.dtype == torch.long, "voxel_indices must be of type torch.long"

        voxel_sizes = self.voxels[voxel_indices.view(-1), -1]  # (..., ) discrete sizes
        voxel_sizes = voxel_sizes.view(voxel_indices.shape)
        voxel_sizes[voxel_indices < 0] = -1  # set invalid voxel sizes to -1
        return voxel_sizes

    @abstractmethod
    def find_voxel_indices(self, points: torch.Tensor, are_voxels: bool) -> torch.Tensor:
        """
        Finds the voxel indices for the given points.
        Args:
            points: (n_points, 3) point cloud in world coordinates
            are_voxels: bool, if True, points are treated as voxel coordinates
        Returns:
            voxel_indices: (n_points,) index of the voxel for each point, -1 if not exists
        """
        pass

    def forward(self, points: torch.Tensor, voxel_indices: torch.Tensor = None, batch_size: int = -1) -> torch.Tensor:
        """
        Forward pass of the octree.
        Args:
            points: (n_points, 3) point cloud in world coordinates
            voxel_indices: (n_points,) index of the voxel for each point, -1 if not exists
            batch_size: int, number of points to process in a batch. If -1, process all points at once.
        Returns:
            (n_points, ) of sdf predictions.
            (n_points,) of voxel indices for each point. If voxel_indices is provided, it will be returned as is.
        """
        if voxel_indices is None:
            voxel_indices = self.find_voxel_indices(points, False)

        if batch_size > 0:
            n_points = points.shape[0]
            sdf_preds = torch.zeros((n_points,), dtype=torch.float32, device=points.device)
            for start in range(0, n_points, batch_size):
                end = min(start + batch_size, n_points)
                sdf_preds[start:end] = self.forward(points[start:end], voxel_indices[start:end])
            return sdf_preds, voxel_indices

        # Implement the forward pass logic here
        assert voxel_indices.dtype == torch.long

        # find the voxel centers for each point
        voxel_centers = self.voxel_centers[voxel_indices]  # (n_points, 3)
        # find the vertex indices for each point
        vertex_indices = self.vertex_indices[voxel_indices]  # (n_points, 8)
        # find the voxel sizes for each point
        voxel_sizes = self.voxels[voxel_indices, -1:]  # (n_points, 1)
        # get the sdf priors and gradient priors for each vertex
        vertex_sdf_priors = self.sdf_priors[vertex_indices]  # (n_points, 8)
        vertex_grad_priors = self.grad_priors[vertex_indices]  # (n_points, 8, 3)

        sdf_preds = ga_trilinear(
            points=points,
            voxel_centers=voxel_centers,
            voxel_sizes=voxel_sizes,
            vertex_values=vertex_sdf_priors,
            vertex_grad=vertex_grad_priors,
            resolution=self.cfg.resolution,
            gradient_augmentation=self.cfg.gradient_augmentation,
            little_endian=self.little_endian_vertex_order,
        )
        return sdf_preds, voxel_indices

    @property
    @abstractmethod
    def little_endian_vertex_order(self) -> bool:
        """
        Returns:
            bool: True if the vertex order is little-endian, False if big-endian.
        """
        pass
