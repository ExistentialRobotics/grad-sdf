import torch
from dataclasses import dataclass
from typing import Optional
from grad_sdf.utils.config_abc import ConfigABC
from grad_sdf.frame import Frame


@dataclass
class SampleTableConfig(ConfigABC):
    pts_per_voxel: int = 100
    surface_voxel_num: int = 1000000
    device: str = "cuda"
    frames_num: int = 1988


class SampleTable:
    def __init__(self, cfg: SampleTableConfig):
        self.pts_per_voxel = cfg.pts_per_voxel
        self.surface_voxel_num = cfg.surface_voxel_num
        self.device = cfg.device
        self.frames_num = cfg.frames_num
        # table存储: [rays_d_world(3), depth(1), frame_idx(1)] = 5个值
        # self.table = torch.full(
        #     (self.surface_voxel_num, self.pts_per_voxel, 5), -1, dtype=torch.float32, device=cfg.device
        # )
        self.table_rays_d = torch.full(
            (self.surface_voxel_num, self.pts_per_voxel, 3), -1, dtype=torch.float16, device=cfg.device
        )
        self.table_depth = torch.full(
            (self.surface_voxel_num, self.pts_per_voxel), -1, dtype=torch.float32, device=cfg.device
        )
        self.table_frame_idx = torch.full(
            (self.surface_voxel_num, self.pts_per_voxel), -1, dtype=torch.int32, device=cfg.device
        )
        self.table_ray_origin = torch.full((self.frames_num, 3), -1, dtype=torch.float16, device=cfg.device)
        # 记录每个体素当前填充到的位置
        self.voxel_counters = torch.zeros(self.surface_voxel_num, dtype=torch.long, device=cfg.device)
        # 记录每个体素已填充的有效样本数（最多 pts_per_voxel）；voxel_counters 是“写指针”（会取模），不能当有效数量用
        self.voxel_filled_counts = torch.zeros(self.surface_voxel_num, dtype=torch.long, device=cfg.device)

    def add_observation(self, frame: Frame, voxel_indices: torch.Tensor):
        """
        使用并行化方式将射线方向、深度和帧索引添加到采样表中
        超出长度后循环填充（从上次填充的位置继续）

        Args:
            frame: Frame对象
            voxel_indices: 每个点对应的体素索引 (N,)，N是有效点的数量
        """

        frame_idx = frame.get_frame_index()

        # 获取相机坐标系下的射线方向 (H, W, 3)
        rays_d_local = frame.get_rays_direction()
        # 获取深度 (H, W)
        depth = frame.get_depth()
        # 获取有效点mask (H, W)
        valid_mask = frame.get_valid_mask()

        # 只取有效点
        rays_d_local_valid = rays_d_local[valid_mask].to(self.device)  # (N, 3)
        depth_valid = depth[valid_mask].to(self.device)  # (N,)

        # 将射线方向转换到世界坐标系（仅旋转，方向向量不需要平移）
        rotation = frame.get_ref_rotation().to(self.device)  # (3, 3)
        rays_d_world = rays_d_local_valid @ rotation.T  # (N, 3)
        # 归一化射线方向
        rays_d_world = rays_d_world / (torch.norm(rays_d_world, dim=-1, keepdim=True) + 1e-8)

        # 检查体素索引是否超出范围
        assert (
            voxel_indices.max() < self.surface_voxel_num
        ), f"Voxel index {voxel_indices.max().item()} exceeds table size {self.surface_voxel_num}"

        # 获取唯一的体素索引和每个体素的点数
        unique_voxels, inverse_indices, counts = torch.unique(voxel_indices, return_inverse=True, return_counts=True)

        # 获取每个唯一体素当前的计数器值（即上次填充到的位置）
        current_counters = self.voxel_counters[unique_voxels]  # shape: (num_unique_voxels,)

        # 计算每个体素内的起始位置（基于当前批次）
        batch_starts = counts.cumsum(0) - counts  # [0, count[0], count[0]+count[1], ...]

        # 使用并行技巧计算每个点在当前批次中的局部索引
        global_indices = torch.arange(rays_d_world.shape[0], device=self.device)
        batch_local_indices = global_indices - torch.repeat_interleave(batch_starts, counts)

        # 将批次局部索引映射到对应的唯一体素，然后加上该体素当前的计数器值
        voxel_base_positions = current_counters[inverse_indices]  # 每个点对应体素的当前计数
        local_indices = (voxel_base_positions + batch_local_indices) % self.pts_per_voxel

        # 并行更新表格：存储 [rays_d_world(3), depth(1), frame_idx(1)]
        # table shape: (surface_voxel_num, pts_per_voxel, 5)
        self.table_rays_d[voxel_indices, local_indices, :] = rays_d_world.to(self.table_rays_d.dtype)  # 射线方向
        self.table_depth[voxel_indices, local_indices] = depth_valid.to(self.table_depth.dtype)  # 深度
        self.table_frame_idx[voxel_indices, local_indices] = torch.tensor(
            frame_idx, dtype=self.table_frame_idx.dtype, device=self.device
        )  # 帧索引
        self.table_ray_origin[frame_idx, :] = frame.get_ref_translation().to(
            device=self.device, dtype=self.table_ray_origin.dtype
        )  # 帧原点

        # 更新每个唯一体素的计数器
        self.voxel_counters[unique_voxels] = (current_counters + counts) % self.pts_per_voxel
        # 更新每个唯一体素已填充的有效样本数（封顶到 pts_per_voxel）
        self.voxel_filled_counts[unique_voxels] = torch.minimum(
            self.voxel_filled_counts[unique_voxels] + counts,  # type: ignore[arg-type]
            torch.tensor(self.pts_per_voxel, device=self.device, dtype=self.voxel_filled_counts.dtype),
        )

    def sample_rays(
        self,
        num_samples: int,
        main_voxel_indices: torch.Tensor,
        main_ratio: float,
    ):
        """
        从采样表中采样射线，并返回对应的 rays 和 depth。

        采样策略（“尽量均匀”）：
        - 先在 voxel 维度做尽量均匀的分配：当 S < |V| 时无放回选 S 个 voxel 各取 1 个；当 S >= |V| 时对所有 voxel 均分，
          余数随机分给部分 voxel，避免固定偏置。
        - 再在每个 voxel 内尽量均匀取样：当需要样本数 <= 已填充数时优先无放回；不足时按随机置换循环重复（允许重复采样）。

        Args:
            num_samples: 需要采样的总点数
            main_voxel_indices: 主 voxel 索引集合（用于按 main_ratio 分配一部分样本）
            main_ratio: 主 voxel 样本占比
        Returns:
            rays_o: ray起点 (num_samples, 3)
            rays_d: ray方向（归一化） (num_samples, 3)
            depth_samples: 深度值 (num_samples,)
        """

        device = self.device

        main_voxel_indices = main_voxel_indices.to(device=device).long().view(-1)

        valid_voxel_mask = self.voxel_filled_counts > 0  # (surface_voxel_num,)
        valid_voxel_indices = torch.nonzero(valid_voxel_mask, as_tuple=False).squeeze(-1).long()

        main_sample_num = int(num_samples * main_ratio)

        extra_valid_indices = valid_voxel_indices[~torch.isin(valid_voxel_indices, main_voxel_indices)]
        if extra_valid_indices.numel() == 0:
            print("no extra valid indices, use main voxel only")
            extra_sample_num = 0
            main_sample_num = num_samples
        else:
            extra_sample_num = num_samples - main_sample_num

        def allocate_counts(voxels: torch.Tensor, total: int) -> tuple[torch.Tensor, torch.Tensor]:
            """
            返回 (selected_voxels, counts)，其中 counts 与 selected_voxels 对齐。
            """
            m = int(voxels.numel())
            if m == 0:
                return torch.empty((0,), dtype=torch.long, device=device), torch.empty(
                    (0,), dtype=torch.long, device=device
                )
            base = total // m
            rem = total - base * m
            selected = voxels
            counts = torch.full((m,), base, dtype=torch.long, device=device)
            if rem > 0:
                perm = torch.randperm(m, device=device)[:rem]
                counts[perm] += 1
            return selected, counts

        def fill_samples(voxels, counts):
            device = voxels.device
            total = counts.sum()
            if total == 0:
                return (
                    torch.empty((0, 3), dtype=torch.float16, device=device),
                    torch.empty((0,), dtype=torch.float32, device=device),
                    torch.empty((0,), dtype=torch.int32, device=device),
                )

            voxel_ids = torch.repeat_interleave(voxels, counts)

            # 生成 group-wise arange
            # 思路：对 [0..total-1] 减去该 voxel 的起始 index
            starts = torch.cumsum(counts, dim=0) - counts
            starts = torch.repeat_interleave(starts, counts)

            offsets = torch.arange(total, device=device) - starts

            filled = self.voxel_filled_counts[voxel_ids]
            slot_ids = offsets % filled

            return (
                self.table_rays_d[voxel_ids, slot_ids, :],
                self.table_depth[voxel_ids, slot_ids],
                self.table_frame_idx[voxel_ids, slot_ids],
            )

        # 1) 主 voxel
        main_sel, main_cnt = allocate_counts(main_voxel_indices, main_sample_num)
        main_rays_d, main_depth, main_frame_idx = fill_samples(main_sel, main_cnt)

        # 2) 其它 voxel
        if extra_sample_num > 0:
            extra_sel, extra_cnt = allocate_counts(extra_valid_indices, extra_sample_num)
            extra_rays_d, extra_depth, extra_frame_idx = fill_samples(extra_sel, extra_cnt)

            # 拼接 + 截断
            rays_d = torch.cat([main_rays_d, extra_rays_d], dim=0)
            depth = torch.cat([main_depth, extra_depth], dim=0)
            frame_idx = torch.cat([main_frame_idx, extra_frame_idx], dim=0)

        else:
            rays_d = main_rays_d
            depth = main_depth
            frame_idx = main_frame_idx
        # 预先计算所有需要的相机位置（ray origin）
        rays_o = self.table_ray_origin[frame_idx, :]

        return rays_o, rays_d, depth

    def clear(self):
        """清空采样表和计数器"""
        self.table.fill_(-1)
        self.voxel_counters.fill_(0)
        self.voxel_filled_counts.fill_(0)
