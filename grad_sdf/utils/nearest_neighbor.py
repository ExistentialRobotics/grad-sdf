import torch
from pytorch3d.ops import knn_points


def nearest_neighbor(src: torch.Tensor, dst: torch.Tensor) -> torch.Tensor:
    """
    Find the nearest neighbor in dst for each point in src.
    Args:
        src: (n_points, 3) source points
        dst: (m_points, 3) destination points
    Returns:
        dists: (n_points,) distances to the nearest neighbor in dst for each point in src
        idx: (n_points,) indices of the nearest neighbor in dst for each point in src
    """
    assert src.ndim == 2 and src.shape[1] == 3
    assert dst.ndim == 2 and dst.shape[1] == 3
    dists, idx, _ = knn_points(src.unsqueeze(0), dst.unsqueeze(0), K=1)
    dists = dists[0, :, 0].sqrt()  # (n_points,)
    idx = idx[0, :, 0]  # (n_points,)
    return dists, idx
