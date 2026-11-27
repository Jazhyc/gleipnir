"""Diagnostic-only worker: initialize CUPTI before model loading and graphs."""

import torch
from vllm.v1.worker.gpu_worker import Worker


def probe_cupti(device: torch.device) -> None:
    """Fail closed unless a bounded pre-model probe records CUDA activity."""
    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ]
    ) as profiler:
        probe = torch.ones(1, device=device)
        probe.add_(1)
        torch.cuda.synchronize()
    kernels = sum(
        e.device_type == torch.autograd.DeviceType.CUDA for e in profiler.events()
    )
    print(f"Early CUPTI probe: {kernels} CUDA events", flush=True)
    if kernels == 0:
        raise RuntimeError("Early CUPTI initialization did not capture CUDA events")


class EarlyCuptiWorker(Worker):
    def init_device(self) -> None:
        super().init_device()
        probe_cupti(self.device)
