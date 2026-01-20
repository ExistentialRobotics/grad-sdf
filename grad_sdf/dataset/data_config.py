from dataclasses import dataclass, field

from grad_sdf.utils.config_abc import ConfigABC


@dataclass
class DataConfig(ConfigABC):
    dataset_name: str = "newercolleage"
    dataset_args: dict = field(
        default_factory=lambda: {
            "data_path": "data/newercollege-lidar",
            "max_depth": -1.0,
        }
    )
    start_frame: int = 0
    end_frame: int = -1
    offset: list = field(default_factory=lambda: [0.0, 0.0, 0.0])
    noise_filter_threshold: float = 0.5
    min_blob_size: int = 30
    knn_distance_threshold: float = 0.12
    knn_neighbor_count: int = 10
