"""CUDA and CPU-RSS memory samples for the SAM tracker logs."""

from dataclasses import dataclass

import torch

MIB = 2 ** 20


def cpu_rss_bytes() -> int:
    with open("/proc/self/status") as status:
        for line in status:
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) * 1024
    raise RuntimeError("VmRSS missing from /proc/self/status")


@dataclass(frozen=True)
class MemorySample:
    cuda_allocated: int
    cuda_reserved: int
    cuda_peak_allocated: int
    cpu_rss: int

    def line(self) -> str:
        return (f"cuda alloc {self.cuda_allocated / MIB:.0f} reserved {self.cuda_reserved / MIB:.0f} "
                f"peak {self.cuda_peak_allocated / MIB:.0f} MiB | rss {self.cpu_rss / MIB:.0f} MiB")


def sample_memory() -> MemorySample:
    return MemorySample(torch.cuda.memory_allocated(), torch.cuda.memory_reserved(),
                        torch.cuda.max_memory_allocated(), cpu_rss_bytes())


def reset_peak_memory() -> None:
    torch.cuda.reset_peak_memory_stats()
