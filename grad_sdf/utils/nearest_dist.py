import time

import torch
from pytorch3d.ops import knn_points


def print_memory(prefix=""):
    allocated = torch.cuda.memory_allocated() / 1024**2
    reserved = torch.cuda.memory_reserved() / 1024**2
    max_alloc = torch.cuda.max_memory_allocated() / 1024**2
    print(f"{prefix} Allocated: {allocated:.2f} MB | Reserved: {reserved:.2f} MB | Max: {max_alloc:.2f} MB")


n1 = 20480
n2 = n1
samp_1 = torch.rand(n1, 3, dtype=torch.float32) * 2
samp_2 = torch.rand(n2, 3, dtype=torch.float32) * 2

samp_1 = samp_1.cuda()
samp_2 = samp_2.cuda()

K = 1
n_repeats = 200

torch.cuda.reset_peak_memory_stats()
print_memory("Before computation")

torch.cuda.synchronize()
t0 = time.time()
for _ in range(n_repeats):
    nn_dist_p3d, _, _ = knn_points(samp_1.unsqueeze(0), samp_2.unsqueeze(0), K=K)
    torch.cuda.synchronize()
t1 = time.time()
dt_p3d = (t1 - t0) / n_repeats
print_memory("After pytorch3d knn computation")

torch.cuda.reset_peak_memory_stats()

torch.cuda.synchronize()
t0 = time.time()
for _ in range(n_repeats):
    nn_dist_th = torch.cdist(samp_1, samp_2).topk(k=K, dim=1, largest=False).values.mean(dim=1)
    torch.cuda.synchronize()
t1 = time.time()
dt_th = (t1 - t0) / n_repeats
print_memory("After torch cdist knn computation")

print(f"p3d: {dt_p3d*1000:.2f} ms, th: {dt_th*1000:.2f} ms")

diff = nn_dist_p3d[0].sqrt().mean(1) - nn_dist_th
print(-diff.min(), diff.max())
