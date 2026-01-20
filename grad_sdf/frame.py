import torch
import torch.nn.functional as F
import numpy as np
import cv2
from pytorch3d.ops import knn_points


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
        self.device = pointcloud.device

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
        bound_mask = points >= bound_min.view(1, 3)
        bound_mask = bound_mask & (points <= bound_max.view(1, 3))
        bound_mask = bound_mask.all(dim=-1)
        self.valid_mask = bound_mask & self.valid_mask

    @torch.no_grad()
    def pcd_to_image(self, width=1024, height=128):
        # 仅处理当前 valid_mask 为 True 的点，或者直接处理全量点
        # 建议处理全量点，通过返回值中的 mask 来区分
        points = self.points
        depth = torch.norm(points, dim=-1)

        # 1. 计算角度
        azimuth = torch.atan2(points[:, 1], points[:, 0])  # [-pi, pi]
        xy_dist = torch.sqrt(points[:, 0] ** 2 + points[:, 1] ** 2)
        elevation = torch.atan2(points[:, 2], xy_dist)  # [-pi/4, pi/4]

        # 2. 水平映射 (360度)
        u = ((azimuth + torch.pi) / (2 * torch.pi) * width).long()
        u = torch.clamp(width - 1 - u, 0, width - 1)

        # 3. 垂直映射 (针对你的 90度 FOV)
        # 公式: v = (max_angle - current_angle) / total_range * height
        # max_angle = pi/4 (+45°), total_range = pi/2 (90°)
        v = ((torch.pi / 4 - elevation) / (torch.pi / 2) * height).long()
        v = torch.clamp(v, 0, height - 1)

        # 3. 创建结果容器
        # 用 -1 初始化，表示该像素没有点
        pixel_to_point = torch.full((height, width), -1, dtype=torch.long, device=self.device)
        depth_image = torch.zeros((height, width), dtype=torch.float32, device=self.device)

        # 4. 核心：处理点到像素的竞争 (保留最近点)
        # 只有当前有效且深度 > 0 的点参与投影
        valid_idx = torch.where(self.valid_mask & (depth > 0))[0]
        if valid_idx.shape[0] > 0:
            # 按深度降序排序：远点在前，近点在后
            sort_idx = valid_idx[torch.argsort(depth[valid_idx], descending=True)]

            # 利用 Tensor 索引的特性：后面的赋值会覆盖前面的
            # 这样近距离的点会最终留在像素中
            pixel_to_point[v[sort_idx], u[sort_idx]] = sort_idx
            depth_image[v[sort_idx], u[sort_idx]] = depth[sort_idx]

        mask = pixel_to_point != -1
        return depth_image, mask, pixel_to_point

    def apply_noise_filter(self, noise_filter_threshold: float = 0.5, min_blob_size: int = 30):
        # 1. 获取投影（此时 pixel_to_point 存储的是 self.points 的原始索引）
        depth_image, mask, pixel_to_point = self.pcd_to_image()

        # --- 步骤 A: 深度跳变过滤 ---
        # 在稀疏图上，roll 操作依然有效，但要注意空像素（depth=0）的干扰
        # 我们只对 mask 为 True 的像素计算跳变
        l_diff = torch.abs(depth_image - torch.roll(depth_image, 1, 1))
        r_diff = torch.abs(depth_image - torch.roll(depth_image, -1, 1))

        # 如果左右邻居都有点，且当前点比它们都远/近超过阈值，视为噪声
        # 注意：如果邻居是 0 (空像素)，这种过滤会失效，这是基于图像过滤的固有局限
        is_jump = (mask) & (l_diff > noise_filter_threshold) & (r_diff > noise_filter_threshold)

        # --- 步骤 B: 连通域过滤 (移除孤立小点簇) ---
        # 将跳变点从 mask 中移除
        clean_mask = mask & (~is_jump)

        # OpenCV 处理 (CPU)
        clean_mask_np = clean_mask.cpu().numpy().astype(np.uint8)
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(clean_mask_np, connectivity=8)

        final_mask_np = np.zeros_like(clean_mask_np)
        for i in range(1, num_labels):
            if stats[i, cv2.CC_STAT_AREA] >= min_blob_size:
                final_mask_np[labels == i] = 1

        final_mask_torch = torch.from_numpy(final_mask_np).to(self.device).bool()

        # --- 步骤 C: 精确映射回原始点云 ---
        # 1. 找出被判定为“噪声”的像素
        # 那些本来有点（mask 为 True），但最后没在 final_mask_torch 里的点
        noise_pixels = mask & (~final_mask_torch)

        # 2. 获取这些像素对应的原始点云索引
        noise_indices = pixel_to_point[noise_pixels]
        noise_indices = noise_indices[noise_indices >= 0]  # 过滤掉无效索引

        # 3. 在全局 mask 中剔除这些点
        noise_filtered_mask = self.valid_mask.clone()
        noise_filtered_mask[noise_indices] = False

        print(f"Noise filtered: {self.valid_mask.sum() - noise_filtered_mask.sum()} points removed.")
        self.valid_mask = noise_filtered_mask

    def apply_noise_filter_gpu(self, noise_filter_threshold: float = 0.5, min_neighbor_count: int = 5):
        """
        完全在 GPU 上运行的滤波逻辑
        :param noise_filter_threshold: 深度跳变阈值
        :param min_neighbor_count: 在 3x3 窗口内，最少需要多少个邻居点才不被判定为孤立噪声
        """
        # 1. 获取投影映射
        depth_image, mask, pixel_to_point = self.pcd_to_image()

        # --- 步骤 A: GPU 跳变过滤 (Jump Filter) ---
        # 利用 roll 实现快速位移
        mask_l = torch.roll(mask, shifts=1, dims=1)
        mask_r = torch.roll(mask, shifts=-1, dims=1)
        diff_l = torch.abs(depth_image - torch.roll(depth_image, 1, 1))
        diff_r = torch.abs(depth_image - torch.roll(depth_image, -1, 1))

        # 逻辑：只有当邻居存在且深度差异大时，才标记为跳变候选
        # 这里的逻辑是：如果该点与它“存在的邻居”差异都很大，则它是噪声
        jump_l = mask & mask_l & (diff_l > noise_filter_threshold)
        jump_r = mask & mask_r & (diff_r > noise_filter_threshold)

        is_jump = jump_l & jump_r

        # --- 步骤 B: GPU 孤立点过滤 (替代 Connected Components) ---
        # 先剔除跳变点，得到初步干净的 mask
        clean_mask = mask & (~is_jump)

        # 使用 2D 卷积统计 3x3 或 5x5 邻域内有效点的数量
        # 卷积核全为 1，计算结果就是该像素周围有效点的总数
        kernel_size = 3
        kernel = torch.ones((1, 1, kernel_size, kernel_size), device=self.device)

        # 将 mask 转为 float 进行卷积
        mask_float = clean_mask.float().unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
        # padding=1 保证尺寸不变
        neighbor_count_map = F.conv2d(mask_float, kernel, padding=kernel_size // 2).squeeze()

        # 最终保留的 mask：原本有点，且周围邻居数量达标
        final_mask_gpu = clean_mask & (neighbor_count_map >= min_neighbor_count)

        # --- 步骤 C: 逻辑运算更新有效索引 ---
        # 找出被滤掉的点：原本在 mask 中，但不在 final_mask_gpu 中
        removed_pixel_mask = mask & (~final_mask_gpu)

        # 获取这些像素对应的原始点云索引
        noise_indices = pixel_to_point[removed_pixel_mask]
        noise_indices = noise_indices[noise_indices >= 0]

        # 更新全局 valid_mask
        new_valid_mask = self.valid_mask.clone()
        new_valid_mask[noise_indices] = False

        print(f"GPU Noise Filter: {self.valid_mask.sum() - new_valid_mask.sum()} points removed.")
        self.valid_mask = new_valid_mask

    def apply_knn_noise_filter(self, knn_distance_threshold: float = 0.12, knn_neighbor_count: int = 10):
        points = self.points.to("cuda")
        # 搜索K个最近邻（包括自己）
        knn_result = knn_points(points.unsqueeze(0), points.unsqueeze(0), K=knn_neighbor_count)

        # dists: [1, N, K]，距离的平方
        dists = knn_result.dists[0]  # [N, K]

        # 排除第一个邻居（自己，距离为0），计算后面K-1个邻居的平均距离
        avg_distances = torch.sqrt(dists[:, 1:]).mean(dim=1)  # [N]

        depth = self.get_depth()
        valid_threshold = depth / 20.0 * knn_distance_threshold

        # 保留平均距离小于等于阈值的点
        knn_valid_mask = (avg_distances <= valid_threshold).to(self.valid_mask.device)

        kept = knn_valid_mask.sum().item()
        total = len(knn_valid_mask)
        print(f"KNN Noise Filter: {kept}/{total} points kept ({100 * kept / total:.2f}%)")

        self.valid_mask = self.valid_mask & knn_valid_mask

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
