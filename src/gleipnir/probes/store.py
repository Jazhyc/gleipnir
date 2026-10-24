"""Append-only per-token activation stores with a row index.

A store directory holds flat token-major files and ``index.jsonl``:

- ``bf16.bin``: raw BF16 bits, ``[tokens, hidden]`` (optional);
- ``fp8.bin`` / ``fp8_scales.bin``: blockwise E4M3 codes and FP32 scales;
- ``index.jsonl``: one row per sequence with ``offset`` and ``length`` in tokens.

Rows are read back as memory maps, so a store larger than RAM streams from disk.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from . import fp8

FORMATS = ("bf16", "fp8")


class StoreWriter:
    def __init__(self, root: Path, hidden: int, formats: tuple[str, ...]):
        if not formats or set(formats) - set(FORMATS):
            raise ValueError(f"formats must be a nonempty subset of {FORMATS}")
        root.mkdir(parents=True, exist_ok=False)
        self.root, self.hidden, self.formats = root, hidden, formats
        self.offset = 0
        self.files = {
            name: (root / name).open("wb")
            for fmt in formats
            for name in (
                ("bf16.bin",) if fmt == "bf16" else ("fp8.bin", "fp8_scales.bin")
            )
        }
        self.index = (root / "index.jsonl").open("w")

    def append(self, values: torch.Tensor, record: dict[str, Any]) -> None:
        """Write one sequence's ``[tokens, hidden]`` activations."""
        if values.ndim != 2 or values.shape[1] != self.hidden or not len(values):
            raise ValueError(f"expected nonempty [tokens, {self.hidden}] activations")
        if not bool(torch.isfinite(values).all()):
            raise ValueError(f"nonfinite activations in row {record.get('id')}")
        if "bf16" in self.formats:
            bits = values.to(torch.bfloat16).view(torch.int16).cpu().numpy()
            self.files["bf16.bin"].write(bits.tobytes())
        if "fp8" in self.formats:
            codes, scales = fp8.quantize(values)
            self.files["fp8.bin"].write(codes.cpu().numpy().tobytes())
            self.files["fp8_scales.bin"].write(scales.cpu().numpy().tobytes())
        entry = {**record, "offset": self.offset, "length": int(values.shape[0])}
        self.index.write(json.dumps(entry, sort_keys=True) + "\n")
        self.offset += int(values.shape[0])

    def close(self, meta: dict[str, Any]) -> None:
        for handle in self.files.values():
            handle.close()
        self.index.close()
        meta = {**meta, "hidden": self.hidden, "formats": list(self.formats)}
        meta["tokens"] = self.offset
        (self.root / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")


class Store:
    """Read-only view of one store in one format."""

    def __init__(self, root: Path, fmt: str):
        self.root, self.fmt = root, fmt
        self.meta = json.loads((root / "meta.json").read_text())
        if fmt not in self.meta["formats"]:
            raise ValueError(f"{root} has no {fmt} copy")
        self.rows = [json.loads(line) for line in (root / "index.jsonl").open()]
        hidden, tokens = self.meta["hidden"], self.meta["tokens"]
        if fmt == "bf16":
            self.values = np.memmap(
                root / "bf16.bin", np.int16, "r", shape=(tokens, hidden)
            )
        else:
            self.values = np.memmap(
                root / "fp8.bin", np.uint8, "r", shape=(tokens, hidden)
            )
            self.scales = np.memmap(
                root / "fp8_scales.bin",
                np.float32,
                "r",
                shape=(tokens, hidden // fp8.BLOCK),
            )

    def __len__(self) -> int:
        return len(self.rows)

    def load(self, index: int, device: torch.device | str = "cpu") -> torch.Tensor:
        """Return row ``index`` as BF16 ``[tokens, hidden]`` on ``device``."""
        row = self.rows[index]
        span = slice(row["offset"], row["offset"] + row["length"])
        values = torch.from_numpy(np.array(self.values[span])).to(device)
        if self.fmt == "bf16":
            return values.view(torch.bfloat16)
        scales = torch.from_numpy(np.array(self.scales[span])).to(device)
        return fp8.dequantize(values, scales)
