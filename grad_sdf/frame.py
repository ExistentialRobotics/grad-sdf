import torch
import numpy as np
import cv2


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
        self.K = intrinsic

        if ref_pose.ndim != 2:
            ref_pose = ref_pose.reshape(4, 4)
        if not isinstance(ref_pose, torch.Tensor):  # from gt data
            self.ref_pose = torch.tensor(ref_pose, requires_grad=False, dtype=torch.float32)
        else:  # from tracked data
            self.ref_pose = ref_pose.clone().requires_grad_(False)
        self.ref_pose[:3, 3] += offset  # Offset ensures voxel coordinates > 0

        self.rays_d: torch.Tensor = self.get_rays(K=self.K)  # (H, W, 3) in camera coordinates
        self.points: torch.Tensor = self.rays_d * self.depth[..., None]  # (H, W, 3) in world coordinates
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
            [(ix - K[0, 2]) / K[0, 0], (iy - K[1, 2]) / K[1, 1], torch.ones_like(ix)], -1
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

    def apply_bound(self, bound_min: torch.Tensor, bound_max: torch.Tensor):
        points = self.points @ self.ref_pose[:3, :3].T + self.ref_pose[:3, 3]
        mask = points >= bound_min.view(1, 1, 3)
        mask = mask & (points <= bound_max.view(1, 1, 3))
        mask = mask.all(dim=-1)
        self.valid_mask = self.valid_mask & mask

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


class LiDARFrame:
    def __init__(
        self,
        fid: int,
        pointcloud: torch.Tensor,
        offset: torch.Tensor,
        ref_pose: torch.Tensor,
    ) -> None:
        self.stamp = fid
        self.points = pointcloud
        self.offset = offset

        if ref_pose.ndim != 2:
            ref_pose = ref_pose.reshape(4, 4)
        if not isinstance(ref_pose, torch.Tensor):  # from gt data
            self.ref_pose = torch.tensor(ref_pose, requires_grad=False, dtype=torch.float32)
        else:  # from tracked data
            self.ref_pose = ref_pose.clone().requires_grad_(False)
        self.ref_pose[:3, 3] += offset  # Offset ensures voxel coordinates > 0
        self.rays_d: torch.Tensor = self.get_rays()  # (N, 3) in world coordinates

        self.valid_mask: torch.Tensor = torch.ones(pointcloud.shape[0], dtype=torch.bool)

    def get_frame_index(self):
        return self.stamp

    def get_ref_pose(self):
        return self.ref_pose

    def get_ref_translation(self):
        return self.ref_pose[:3, 3]

    def get_ref_rotation(self):
        return self.ref_pose[:3, :3]

    def get_points(self, to_world_frame: bool, device: str):
        points = self.points[self.valid_mask].reshape(-1, 3).to(device)
        if to_world_frame:
            pose = self.get_ref_pose().to(device)
            points = points @ pose[:3, :3].T + pose[:3, 3]  # to world coordinates
        return points

    def get_depth(self):
        return torch.norm(self.points, dim=-1)  # (N,)

    @torch.no_grad()
    def get_rays(self):
        # 返回局部坐标系中的归一化射线方向，与 DepthFrame 保持一致
        # 后续在 key_frame_set.sample_rays 中会通过旋转矩阵转换到世界坐标系
        rays_d = torch.nn.functional.normalize(self.points, p=2, dim=-1)
        return rays_d

    def get_rays_direction(self):
        return self.rays_d

    def get_valid_mask(self):
        return self.valid_mask

    def apply_bound(self, bound_min: torch.Tensor, bound_max: torch.Tensor):
        points = self.points @ self.ref_pose[:3, :3].T + self.ref_pose[:3, 3]
        mask = points >= bound_min.view(1, 3)
        mask = mask & (points <= bound_max.view(1, 3))
        mask = mask.all(dim=-1)
        self.valid_mask = mask & self.valid_mask

    def pcd_to_image(self, width=1024, height=128):
        """
        将点云转换为深度图像
        Args:
            width: 图像宽度，对应水平分辨率 (360度)
            height: 图像高度，对应垂直分辨率 (90度)
        Returns:
            depth_image: (height, width) 深度图像
            mask: (height, width) 有效点的mask
            point_to_pixel: (N, 2) 点云到像素的对应关系 [u, v]
            pixel_to_point: (height, width) 像素到点云的对应关系，存储点索引，-1表示无效
        """
        points = self.points[self.valid_mask]  # (N, 3)
        depth = self.get_depth()[self.valid_mask]  # (N,)

        # 计算方位角 (Azimuth angle): atan2(y, x)
        # 范围 [-π, π]，对应水平360度
        azimuth = torch.atan2(points[:, 1], points[:, 0])  # (N,)

        # 计算仰角 (Elevation angle): atan2(z, sqrt(x^2 + y^2))
        # 范围 [-π/4, π/4]，对应垂直90度 (±45度)
        xy_dist = torch.sqrt(points[:, 0] ** 2 + points[:, 1] ** 2)
        elevation = torch.atan2(points[:, 2], xy_dist)  # (N,)

        # 将角度映射到像素坐标
        # 水平: azimuth ∈ [-π, π] -> u ∈ [0, width)
        # 反转方向以匹配LiDAR的扫描方向
        u = ((azimuth + torch.pi) / (2 * torch.pi) * width).long()
        u = torch.clamp(u, 0, width - 1)
        u = width - 1 - u  # 反转水平方向

        # 垂直: elevation ∈ [-π/4, π/4] -> v ∈ [0, height)
        # 注意：图像坐标v从上到下增加，所以需要翻转
        v = ((torch.pi / 4 - elevation) / (torch.pi / 2) * height).long()
        v = torch.clamp(v, 0, height - 1)

        # 创建深度图像和像素到点云的映射
        depth_image = torch.zeros(height, width, dtype=torch.float32)
        mask = torch.zeros(height, width, dtype=torch.bool)
        pixel_to_point = torch.full((height, width), -1, dtype=torch.long)

        # 将深度值填充到对应的像素位置
        # 如果多个点映射到同一像素，保留最近的深度值
        for i in range(len(points)):
            pixel_v, pixel_u = v[i].item(), u[i].item()
            if not mask[pixel_v, pixel_u] or depth[i] < depth_image[pixel_v, pixel_u]:
                depth_image[pixel_v, pixel_u] = depth[i]
                mask[pixel_v, pixel_u] = True
                pixel_to_point[pixel_v, pixel_u] = i

        # 构建点云到像素的对应关系 (N, 2)，格式为 [u, v]
        point_to_pixel = torch.stack([u, v], dim=1)

        return depth_image, mask, point_to_pixel, pixel_to_point

    def apply_noise_filter(self, noise_filter_threshold: float = 0.5, min_blob_size: int = 30):
        """
        应用噪声滤波：深度跳变过滤 + 连通域孤立点过滤
        Args:
            noise_filter_threshold: 深度跳变阈值 (米)，建议 0.5 - 1.5
            min_blob_size: 最小连通域像素数，小于此值的点簇将被视为噪声剔除
        """
        # 1. 获取投影后的图像和映射关系
        depth_image, mask, point_to_pixel, pixel_to_point = self.pcd_to_image()
        height, width = depth_image.shape
        device = depth_image.device

        # --- 步骤 A: 深度跳变过滤 (Jump Distance Filter) ---
        # 利用 torch.roll 实现水平 360 度环绕对比 (同一 Ring 邻居)
        left_depth = torch.roll(depth_image, shifts=1, dims=1)
        right_depth = torch.roll(depth_image, shifts=-1, dims=1)

        # 计算水平跳变：如果当前点比左右邻居都远/近超过阈值
        # 注意：我们只对比 mask 为 True 的点，避免背景 0 干扰
        h_jump = (torch.abs(depth_image - left_depth) > noise_filter_threshold) & (
            torch.abs(depth_image - right_depth) > noise_filter_threshold
        )

        # 垂直方向跳变 (相邻 Beam 之间)
        v_jump = torch.zeros_like(h_jump)
        # 计算上方和下方的深度差 (不循环 roll)
        v_jump[1:-1, :] = (torch.abs(depth_image[1:-1, :] - depth_image[:-2, :]) > noise_filter_threshold) & (
            torch.abs(depth_image[1:-1, :] - depth_image[2:, :]) > noise_filter_threshold
        )

        # 如果在水平或垂直方向都是孤立的跳变点，标记为疑似噪声
        jump_mask = h_jump & v_jump

        # --- 步骤 B: 连通域离群点剔除 (Connected Component Filter) ---
        # 这一步能非常有效地去除玻璃反射产生的浮空点簇
        # 先得到初步过滤后的有效 mask
        pre_mask = mask & (~jump_mask)

        # 转换为 numpy 以使用 OpenCV 的连通域分析
        pre_mask_np = pre_mask.cpu().numpy().astype(np.uint8)

        # 使用 8 连通域分析
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(pre_mask_np, connectivity=8)

        # 过滤掉像素数过少的“孤岛”
        final_mask_np = np.zeros_like(pre_mask_np)
        for i in range(1, num_labels):  # label 0 是背景
            if stats[i, cv2.CC_STAT_AREA] >= min_blob_size:
                final_mask_np[labels == i] = 1

        final_mask_torch = torch.from_numpy(final_mask_np).bool().to(device)

        # --- 步骤 C: 将过滤结果映射回原始点云 ---
        # 1. 找出被保留下来的像素坐标对应的点索引
        # 只有在 final_mask_torch 为 True 且 pixel_to_point 有效的地方才保留
        keep_indices_in_subset = pixel_to_point[final_mask_torch]
        # 排除掉可能存在的 -1 值（虽然逻辑上在 mask 里的点索引应该都 >= 0）
        keep_indices_in_subset = keep_indices_in_subset[keep_indices_in_subset >= 0]

        # 2. 更新 self.valid_mask
        # 此时 keep_indices_in_subset 是相对于 points = self.points[self.valid_mask] 的索引
        # 我们需要将其映射回原始点云的全量索引
        original_valid_indices = torch.where(self.valid_mask)[0]
        final_valid_indices = original_valid_indices[keep_indices_in_subset]

        # 创建新的全局 valid_mask
        new_valid_mask = torch.zeros_like(self.valid_mask, dtype=torch.bool)
        new_valid_mask[final_valid_indices] = True

        # 3. 打印过滤结果
        num_before = self.valid_mask.sum().item()
        num_after = final_valid_indices.shape[0]

        print(f"noise filter 保留比例: {num_after / num_before * 100:.2f}%")

        self.valid_mask = self.valid_mask & new_valid_mask

    def sample_points(
        self,
        num_points: int = -1,
        ratio: float = 0.25,
        to_world_frame: bool = True,
        device: str = None,
    ) -> torch.Tensor:
        if num_points <= 0:
            num_points = int(self.points.shape[0] * ratio)
        indices = torch.argwhere(self.valid_mask).flatten()
        if len(indices) <= num_points:
            sampled_indices = indices
        else:
            perm = torch.randperm(len(indices))[:num_points]
            sampled_indices = indices[perm]
        points = self.points[sampled_indices]
        if device is not None:
            points = points.to(device)
        if to_world_frame:
            pose = self.get_ref_pose().to(points.device)
            points = points @ pose[:3, :3].T + pose[:3, 3]  # to world coordinates
        return points
