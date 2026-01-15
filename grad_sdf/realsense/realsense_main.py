import copy
import os
import random
import threading
import time
from typing import Callable, Optional

import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
import torch
from tqdm import tqdm

from grad_sdf import torch
from grad_sdf.criterion import Criterion
from grad_sdf.evaluator_grad_sdf import GradSdfEvaluator
from grad_sdf.frame import DepthFrame, Frame
from grad_sdf.key_frame_set import KeyFrameSet
from grad_sdf.loggers import BasicLogger
from grad_sdf.model import SdfNetwork
from grad_sdf.realsense.realsense_stream import RealsenseStream
from grad_sdf.realsense.realsense_trainer import Trainer
from grad_sdf.trainer_config import TrainerConfig
from grad_sdf.utils.profiling import GpuTimer
from grad_sdf.utils.sampling import SampleResults, generate_sdf_samples


class ModelSnapshotter:
    def __init__(self, src_model, model_ctor, ctor_args, device, interval_sec=0.2):
        self._src_model = src_model
        self._interval = interval_sec
        self._device = device

        # create snapshot model once
        self._snapshot = model_ctor(*ctor_args).to(device)
        self._snapshot.load_state_dict(src_model.state_dict())
        self._snapshot.eval()

        self._lock = threading.Lock()
        self._stop_event = threading.Event()

        self._thread = threading.Thread(target=self._loop)
        self._thread.start()

    def _loop(self):
        while not self._stop_event.is_set():
            with self._lock:
                self._snapshot.load_state_dict(
                    self._src_model.state_dict(), strict=True
                )
            self._stop_event.wait(self._interval)  # allows immediate stop

    def get_model(self):
        with self._lock:
            return self._snapshot

    def stop(self):
        self._stop_event.set()
        self._thread.join()


def save_mesh(mesh, name):
    out_path = os.path.join(os.getcwd(), name)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    o3d.io.write_triangle_mesh(out_path, mesh)


def main():
    parser = TrainerConfig.get_argparser()
    cfg: TrainerConfig = parser.parse_args()
    trainer = Trainer(cfg)
    train_thread = threading.Thread(target=trainer.train)
    train_thread.start()
    snapshotter = ModelSnapshotter(
        src_model=trainer.model,
        model_ctor=SdfNetwork,
        ctor_args=(cfg.model,),
        device=cfg.device,
        interval_sec=0.2,
    )
    cnt = 0
    time.sleep(5)  # wait a bit before first extraction
    try:
        while train_thread.is_alive() and cnt < 10:
            print("Extracting mesh...")
            logger = BasicLogger(cfg.log_dir, cfg.exp_name, cfg.as_dict())
            evaluator = GradSdfEvaluator(
                batch_size=cfg.batch_size,
                clean_mesh=cfg.clean_mesh,
                model_cfg=cfg.model,
                model=snapshotter.get_model(),
                model_input_offset=None,
                device=cfg.device,
            )
            bound_min = cfg.model.residual_net_cfg.bound_min
            bound_max = cfg.model.residual_net_cfg.bound_max
            mesh_prior, mesh = evaluator.extract_mesh(
                bound_min=bound_min,
                bound_max=bound_max,
                grid_resolution=cfg.mesh_resolution,
                fields=["sdf_prior", "sdf"],
                iso_value=cfg.mesh_iso_value,
            )

            save_mesh(mesh_prior, f"realsense_data/mesh_prior{cnt}.ply")
            save_mesh(mesh, f"realsense_data/mesh{cnt}.ply")
            cnt += 1
            time.sleep(1)
    finally:
        snapshotter.stop()
        train_thread.join()


if __name__ == "__main__":
    main()
