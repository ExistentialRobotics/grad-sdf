import torch


class Frame:
    def get_frame_index(self) -> int:
        raise NotImplementedError

    def get_ref_pose(self) -> torch.Tensor:
        raise NotImplementedError

    def get_points(self, to_world_frame: bool, device: str) -> torch.Tensor:
        raise NotImplementedError

    def get_rays_direction(self) -> torch.Tensor:
        raise NotImplementedError

    def get_depth(self) -> torch.Tensor:
        raise NotImplementedError

    def get_valid_mask(self) -> torch.Tensor:
        raise NotImplementedError

    def apply_bound(self, bound_min: torch.Tensor, bound_max: torch.Tensor):
        raise NotImplementedError

    def sample_points(
        self,
        num_points: int = -1,
        ratio: float = 0.25,
        to_world_frame: bool = True,
        device: str = None,
    ) -> torch.Tensor:
        raise NotImplementedError


class DepthFrame(Frame):
    def __init__(
        self,
        fid: int,
        depth: torch.Tensor,
        intrinsic: torch.Tensor,
        offset: torch.Tensor,
        ref_pose: torch.Tensor,
        project_to_boundry: bool = False,
    ) -> None:
        """
        Args:
            fid: int, frame idx
            depth: (H, W) in meter
            intrinsic: (3, 3) intrinsic matrix
            offset: (3, ) offset to be added to the translation of ref_pose
            ref_pose: (4, 4) reference pose in world coordinates
        """
        super().__init__()
        self.stamp = fid
        self.h, self.w = depth.shape
        if not isinstance(depth, torch.Tensor):
            depth = torch.FloatTensor(depth)  # / 2
        self.depth = depth
        self.project_to_boundry = project_to_boundry
        self.K = intrinsic

        if ref_pose.ndim != 2:
            ref_pose = ref_pose.reshape(4, 4)
        if not isinstance(ref_pose, torch.Tensor):  # from gt data
            self.ref_pose = torch.tensor(
                ref_pose, requires_grad=False, dtype=torch.float32
            )
        else:  # from tracked data
            self.ref_pose = ref_pose.clone().requires_grad_(False)
        self.ref_pose[:3, 3] += offset  # Offset ensures voxel coordinates > 0

        self.rays_d: torch.Tensor = self.get_rays(
            K=self.K
        )  # (H, W, 3) in camera coordinates

        self.points: torch.Tensor = (
            self.rays_d * self.depth[..., None]
        )  # (H, W, 3) in world coordinates
        self.valid_mask: torch.Tensor = self.depth > 0  # (H, W) depth > 0

    def get_frame_index(self):
        return self.stamp

    def get_ref_pose(self):
        return self.ref_pose

    def get_ref_translation(self):
        return self.ref_pose[:3, 3]

    def get_ref_rotation(self):
        return self.ref_pose[:3, :3]

    @torch.no_grad()
    def get_rays(self, w=None, h=None, K=None):
        w = self.w if w is None else w
        h = self.h if h is None else h
        if K is None:
            K = torch.eye(3)
            K[0, 0] = self.K[0, 0] * w / self.w
            K[1, 1] = self.K[1, 1] * h / self.h
            K[0, 2] = self.K[0, 2] * w / self.w
            K[1, 2] = self.K[1, 2] * h / self.h
        ix, iy = torch.meshgrid(torch.arange(w), torch.arange(h), indexing="xy")
        rays_d = torch.stack(
            [(ix - K[0, 2]) / K[0, 0], (iy - K[1, 2]) / K[1, 1], torch.ones_like(ix)],
            -1,
        ).float()  # camera coordinate
        return rays_d

    def get_points(self, to_world_frame: bool, device: str):
        points = self.points[self.valid_mask].reshape(-1, 3).to(device)  # [N,3]
        if to_world_frame:
            pose = self.get_ref_pose().to(device)
            points = points @ pose[:3, :3].T + pose[:3, 3]  # to world coordinates
        return points

    def get_rays_direction(self):
        return self.rays_d

    def get_depth(self):
        return self.depth

    def get_valid_mask(self):
        return self.valid_mask

    def _project_points_to_boundary(
        self,
        bound_min: torch.Tensor,
        bound_max: torch.Tensor,
        points_out_of_bound: torch.Tensor,
    ):
        origin = self.get_ref_translation().view(1, 3)  # (1, 3)

        ray_dir = points_out_of_bound - origin

        inv_dir = 1.0 / (ray_dir + 1e-8)
        t_min_planes = (bound_min.view(1, 3) - origin) * inv_dir
        t_max_planes = (bound_max.view(1, 3) - origin) * inv_dir

        t_max_each_dim = torch.max(t_min_planes, t_max_planes)
        t_exit_final = torch.min(t_max_each_dim, dim=-1)[0]

        projected_points_world = origin + t_exit_final.view(-1, 1) * ray_dir

        R_w2c = self.ref_pose[:3, :3].T
        t_w2c = -R_w2c @ self.ref_pose[:3, 3]

        projected_points_cam = projected_points_world @ R_w2c.T + t_w2c

        return projected_points_world, projected_points_cam[:, 2]

    def apply_bound(self, bound_min: torch.Tensor, bound_max: torch.Tensor):
        if not self.project_to_boundry:
            points = self.points @ self.ref_pose[:3, :3].T + self.ref_pose[:3, 3]
            mask = points >= bound_min.view(1, 1, 3)
            mask = mask & (points <= bound_max.view(1, 1, 3))
            mask = mask.all(dim=-1)
            self.valid_mask = self.valid_mask & mask
        else:
            points = self.points @ self.ref_pose[:3, :3].T + self.ref_pose[:3, 3]
            mask = (points >= bound_min.view(1, 1, 3)) & (
                points <= bound_max.view(1, 1, 3)
            )
            mask = mask.all(dim=-1)

            points_out_of_bound = points[~mask]
            if points_out_of_bound.shape[0] > 0:
                new_points_world, new_points_depth = self._project_points_to_boundary(
                    bound_min, bound_max, points_out_of_bound
                )

                self.depth[~mask] = new_points_depth
                self.points[~mask] = new_points_world

    def sample_points(
        self,
        num_points: int = -1,
        ratio: float = 0.25,
        to_world_frame: bool = True,
        device: str = None,
    ) -> torch.Tensor:
        if num_points <= 0:
            num_points = int(self.h * self.w * ratio)
        indices = torch.argwhere(self.valid_mask)
        if len(indices) <= num_points:
            sampled_indices = indices
        else:
            perm = torch.randperm(len(indices))[:num_points]
            sampled_indices = indices[perm]
        points = self.points[sampled_indices[:, 0], sampled_indices[:, 1]]
        if device is not None:
            points = points.to(device)
        if to_world_frame:
            pose = self.get_ref_pose().to(points.device)
            points = points @ pose[:3, :3].T + pose[:3, 3]  # to world coordinates
        return points
