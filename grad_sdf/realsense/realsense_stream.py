import os.path as osp
from glob import glob

import cv2
import numpy as np
import open3d as o3d
import pyrealsense2 as rs
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

from grad_sdf.frame import DepthFrame


class RealsenseStream:
    def __init__(
        self,
        data_path: str,
        min_depth: float = 0.0,
        max_depth: float = -1.0,
        offset: torch.Tensor | None = None,
        bound_min: torch.Tensor | None = None,
        bound_max: torch.Tensor | None = None,
    ):
        self.min_depth = min_depth
        self.max_depth = max_depth
        self.bound_min = bound_min
        self.bound_max = bound_max
        self.offset = torch.zeros(3)
        if offset is not None:
            self.offset = torch.tensor(offset).float()
        if self.bound_min is not None:
            assert self.bound_max is not None
            self.bound_min = torch.tensor(self.bound_min).float()
        if self.bound_max is not None:
            assert self.bound_min is not None
            self.bound_max = torch.tensor(self.bound_max).float()

        self.initialize_realsense()

    def initialize_realsense(self):
        self.pipeline = rs.pipeline()
        # TODO: add config for resolution, frame rate, depth type etc
        self.rs_config = rs.config()
        self.rs_config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        self.profile = self.pipeline.start(self.rs_config)
        # Wait for a valid frame so profiles are initialized
        frames = self.pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()

        depth_profile = depth_frame.profile.as_video_stream_profile()
        intr = depth_profile.get_intrinsics()
        K = torch.eye(3)
        K[0, 0] = intr.fx
        K[1, 1] = intr.fy
        K[0, 2] = intr.ppx
        K[1, 2] = intr.ppy
        self.K = K
        depth_sensor = self.profile.get_device().first_depth_sensor()
        self.depth_scale = depth_sensor.get_depth_scale()

    def get_next_frame(self) -> DepthFrame:
        # TODO: get pose from somewhere, seemingly best way is to query into the EKF of erl_agilib at the timestamp of the frame

        frames = self.pipeline.wait_for_frames()
        rs_depth = frames.get_depth_frame()

        # rs_timestamp = rs_depth.get_timestamp()
        pose = torch.eye(4)
        depth = np.array(rs_depth.get_data(), dtype=np.float32)
        depth *= self.depth_scale
        if self.min_depth > 0:
            depth[depth < self.min_depth] = 0
        if self.max_depth > 0:
            depth[depth > self.max_depth] = 0
        depth = torch.from_numpy(depth).float()
        index = rs_depth.get_frame_number()
        depth_frame = DepthFrame(index, depth, self.K, self.offset, pose)
        return depth_frame
