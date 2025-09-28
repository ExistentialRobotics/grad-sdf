import time

import torch
from tqdm import tqdm


class CpuTimer:

    def __init__(self, message, warmup: int = 0, enable: bool = True, verbose: bool = True):
        self.message = message
        self.warmup = warmup
        self.enable = enable
        self.verbose = verbose
        self.cnt = 0
        self.t = 0
        self.average_t = 0
        self._total_t = 0
        self.total_t = 0

    def __enter__(self):
        if not self.enable:
            return self
        self.start = time.perf_counter()
        return self

    def __exit__(self, *args):
        self.end = time.perf_counter()
        if self.cnt < self.warmup:
            self.cnt += 1
            return
        self.cnt += 1
        self.t = self.end - self.start
        self._total_t += self.t
        self.average_t = self._total_t / (self.cnt - self.warmup)
        self.total_t = self.average_t * self.cnt
        if self.verbose:
            tqdm.write(f"{self.message}: {self.t:.6f}(cur)/{self.average_t:.6f}(avg)/{self.total_t:.6f}(total) seconds")


def cpu_timer(message, warmup=0, enable=True, verbose=True):
    def decorator(func):
        def wrapper(*args, **kwargs):
            with CpuTimer(message, warmup=warmup, enable=enable, verbose=verbose):
                return func(*args, **kwargs)

        return wrapper

    return decorator


class GpuTimer:

    def __init__(self, message, warmup: int = 0, enable: bool = True, verbose: bool = True):
        self.message = message
        self.warmup = warmup
        self.enable = enable
        self.verbose = verbose
        self.cnt = 0
        self.t = 0
        self.average_t = 0
        self._total_t = 0
        self.total_t = 0

    def __enter__(self):
        if not self.enable:
            return self
        self.start = torch.cuda.Event(enable_timing=True)
        self.end = torch.cuda.Event(enable_timing=True)
        self.start.record()
        return self

    def __exit__(self, *args):
        if not self.enable:
            return
        self.end.record()
        torch.cuda.synchronize()
        self.t = self.start.elapsed_time(self.end) / 1e3
        if self.cnt < self.warmup:
            self.cnt += 1
            return
        self.cnt += 1
        self._total_t += self.t
        self.average_t = self._total_t / (self.cnt - self.warmup)
        self.total_t = self.average_t * self.cnt
        if self.verbose:
            tqdm.write(f"{self.message}: {self.t:.6f}(cur)/{self.average_t:.6f}(avg)/{self.total_t:.6f}(total) seconds")


def gpu_timer(message, warmup=0, enable=True, verbose=True):
    def decorator(func):
        def wrapper(*args, **kwargs):
            with GpuTimer(message, warmup=warmup, enable=enable, verbose=verbose):
                return func(*args, **kwargs)

        return wrapper

    return decorator
