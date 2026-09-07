"""Distributed-safe dispatch for selected-token auxiliary projections."""

import os
import subprocess
from collections.abc import Callable
from functools import wraps
from pathlib import Path
from typing import Any

import torch


def prepare_rank_caches(rank: int) -> None:
    """Seed immutable Triton binaries/tuning entries; isolate future writes."""
    if rank < 0:
        raise ValueError("rank must be nonnegative")
    for key in (
        "TORCHINDUCTOR_CACHE_DIR",
        "TRITON_CACHE_DIR",
        "TILELANG_CACHE_DIR",
        "TVM_CACHE_DIR",
    ):
        if key not in os.environ:
            continue
        source = Path(os.environ[key]).resolve()
        if source == Path("/") or source == Path.home():
            raise ValueError("cache must be a dedicated directory")
        destination = source / f"rank-{rank}"
        if key == "TRITON_CACHE_DIR" and source.is_dir():
            destination.mkdir(parents=True, exist_ok=True)
            subprocess.run(
                [
                    "rsync",
                    "-a",
                    "--ignore-existing",
                    "--exclude=rank-*",
                    str(source) + "/",
                    str(destination) + "/",
                ],
                check=True,
            )
        os.environ[key] = str(destination)


def install_mil_forward(model: Any, projection: Callable) -> None:
    """Keep PEFT identity/save format while routing MIL through DDP.forward."""
    original = model.forward

    @wraps(original)
    def forward(*args: Any, **kwargs: Any) -> Any:
        positions = kwargs.pop("gleipnir_mil_positions", None)
        if positions is None:
            return original(*args, **kwargs)
        binary_ids = kwargs.pop("gleipnir_binary_ids")
        input_ids = kwargs.pop("input_ids")
        # Accelerate wraps model.forward in BF16 autocast. The legacy selected
        # projection autocasts only the decoder, leaving its FP32 head outside.
        # Restore that boundary; projection owns the decoder's autocast context.
        with torch.autocast(device_type=input_ids.device.type, enabled=False):
            final, mil, _ = projection(
                model, input_ids, kwargs.pop("attention_mask"), positions, binary_ids
            )
        if args or kwargs:
            raise ValueError("unsupported selected MIL forward arguments")
        return {"logits": final, "mil_logits": mil}

    model.forward = forward


def verify_distributed_parameters(model: Any) -> dict[str, Any]:
    """Fail closed on replica divergence; compare every FP32 trainable value."""
    if not torch.distributed.is_initialized():
        return {"world_size": 1, "max_replica_difference": 0.0}
    maximum = torch.zeros((), device=next(model.parameters()).device)
    for parameter in model.parameters():
        if not parameter.requires_grad:
            continue
        reference = parameter.detach().clone()
        torch.distributed.broadcast(reference, src=0)
        maximum = torch.maximum(maximum, (parameter.detach() - reference).abs().max())
    torch.distributed.all_reduce(maximum, op=torch.distributed.ReduceOp.MAX)
    difference = float(maximum)
    if not torch.isfinite(maximum) or difference != 0.0:
        raise RuntimeError(f"DDP trainable replicas diverged: {difference}")
    return {
        "world_size": torch.distributed.get_world_size(),
        "max_replica_difference": difference,
    }
