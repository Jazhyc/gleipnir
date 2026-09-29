"""Checksum-bound native FP4 artifacts, separate from original model weights."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import torch


def tensor_sha256(tensor: torch.Tensor) -> str:
    """Hash original contiguous tensor bytes without converting BF16 precision."""
    data = tensor.detach().cpu().contiguous().view(torch.uint8).numpy()
    return hashlib.sha256(data.tobytes()).hexdigest()


class NvFp4Artifact:
    """Require complete coverage, frozen provenance, and safe tensor-only loading."""

    def __init__(self, root: Path, manifest_sha256: str, merge_sha256: str) -> None:
        self.root = root
        content = (root / "manifest.json").read_bytes()
        if hashlib.sha256(content).hexdigest() != manifest_sha256:
            raise ValueError("Changed prepared NVFP4 manifest")
        self.manifest = json.loads(content)
        required = {f"{i}_{name}" for i in range(32) for name in ("gate_up", "down")}
        if (
            self.manifest["state"] != "complete"
            or set(self.manifest["layers"]) != required
        ):
            raise ValueError("Incomplete prepared NVFP4 projection coverage")
        if self.manifest["merge_manifest_sha256"] != merge_sha256:
            raise ValueError("Prepared NVFP4 merge provenance mismatch")

    def load(
        self, prefix: str, source: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Validate identity/layout before transferring packed serving tensors."""
        index = re.search(r"(?:^|\.)layers\.(\d+)\.mlp\.(gate_up|down)_proj$", prefix)
        if index is None:
            raise ValueError("Prepared NVFP4 artifact supports decoder MLP projections")
        key = f"{index.group(1)}_{index.group(2)}"
        record = self.manifest["layers"][key]
        filename = record["file"]
        if Path(filename).name != filename or Path(filename).suffix != ".pt":
            raise ValueError("Invalid prepared artifact filename")
        if (
            list(source.shape) != record["shape_nk"]
            or tensor_sha256(source) != record["original_weight_sha256"]
        ):
            raise ValueError("Prepared NVFP4 source weight mismatch")
        content = (self.root / filename).read_bytes()
        if hashlib.sha256(content).hexdigest() != record["sha256"]:
            raise ValueError("Changed prepared NVFP4 packed tensor")
        payload = torch.load(
            self.root / filename, map_location="cpu", weights_only=True
        )
        packed, blocks, scale = (
            payload[k] for k in ("packed", "blocks", "global_scale")
        )
        rows, width = source.shape
        if packed.dtype != torch.uint8 or packed.shape != (rows, width // 2):
            raise ValueError("Invalid prepared NVFP4 packed shape/dtype")
        if blocks.dtype != torch.float8_e4m3fn or blocks.shape != (rows, width // 16):
            raise ValueError("Invalid prepared NVFP4 scale shape/dtype")
        if scale.dtype != torch.float32 or scale.shape != (1,):
            raise ValueError("Invalid prepared NVFP4 global scale")
        if (
            not torch.isfinite(blocks.float()).all().item()
            or (blocks.float() < 0).any().item()
        ):
            raise ValueError("Invalid prepared NVFP4 block scale values")
        if not torch.isfinite(scale).all().item() or scale.item() <= 0:
            raise ValueError("Invalid prepared NVFP4 global scale value")
        return tuple(t.to(source.device) for t in (packed, blocks, scale))
