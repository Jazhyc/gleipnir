"""Append-only per-token activation stores with a row index.

A store directory holds flat token-major files and ``index.jsonl``:

- ``bf16.bin``: raw BF16 bits, ``[tokens, hidden]`` (optional);
- ``fp8.bin`` / ``fp8_scales.bin``: blockwise E4M3 codes and FP32 scales;
- ``index.jsonl``: one row per sequence with ``offset`` and ``length`` in tokens.

Writes run on one background thread (in order, with a bounded backlog), so
file I/O overlaps the next forward pass. Rows are read back with one positional
read each, so a store larger than RAM streams from disk.
"""

from __future__ import annotations

import json
import os
from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
import torch

from . import fp8

FORMATS = ("bf16", "fp8")
MAX_PENDING_WRITES = 8


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
        self.executor = ThreadPoolExecutor(max_workers=1)
        self.pending: deque[Future] = deque()

    def _write(self, chunks: list[tuple[str, np.ndarray]], entry: str) -> None:
        for name, array in chunks:
            self.files[name].write(memoryview(array).cast("B"))
        self.index.write(entry)

    def append(self, values: torch.Tensor, record: dict[str, Any]) -> None:
        """Queue one sequence's ``[tokens, hidden]`` activations for writing."""
        if values.ndim != 2 or values.shape[1] != self.hidden or not len(values):
            raise ValueError(f"expected nonempty [tokens, {self.hidden}] activations")
        if not bool(torch.isfinite(values).all()):
            raise ValueError(f"nonfinite activations in row {record.get('id')}")
        chunks = []
        if "bf16" in self.formats:
            bits = values.to(torch.bfloat16).view(torch.int16)
            chunks.append(("bf16.bin", bits.cpu().numpy()))
        if "fp8" in self.formats:
            codes, scales = fp8.quantize(values)
            chunks.append(("fp8.bin", codes.cpu().numpy()))
            chunks.append(("fp8_scales.bin", scales.cpu().numpy()))
        entry = {**record, "offset": self.offset, "length": int(values.shape[0])}
        self.offset += int(values.shape[0])
        while len(self.pending) >= MAX_PENDING_WRITES:
            self.pending.popleft().result()
        line = json.dumps(entry, sort_keys=True) + "\n"
        self.pending.append(self.executor.submit(self._write, chunks, line))

    def close(self, meta: dict[str, Any]) -> None:
        while self.pending:
            self.pending.popleft().result()
        self.executor.shutdown()
        for handle in self.files.values():
            handle.close()
        self.index.close()
        meta = {**meta, "hidden": self.hidden, "formats": list(self.formats)}
        meta["tokens"] = self.offset
        (self.root / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")


class Store:
    """Read-only view of one store in one format."""

    def __init__(self, root: Path, fmt: str, *, skip: int = 0):
        """``skip`` drops each row's first tokens (e.g. the NTP attention sink)."""
        self.root, self.fmt = root, fmt
        self.meta = json.loads((root / "meta.json").read_text())
        if fmt not in self.meta["formats"]:
            raise ValueError(f"{root} has no {fmt} copy")
        rows = [json.loads(line) for line in (root / "index.jsonl").open()]
        if any(r["length"] <= skip for r in rows):
            raise ValueError(f"a row in {root} has no tokens left after skip={skip}")
        self.rows = [
            {**r, "offset": r["offset"] + skip, "length": r["length"] - skip}
            for r in rows
        ]
        self.hidden = self.meta["hidden"]
        # One large positional read per row: memory-mapped page faults over
        # NFS were orders of magnitude slower than sequential reads.
        names = ("bf16.bin",) if fmt == "bf16" else ("fp8.bin", "fp8_scales.bin")
        self.fds = {name: os.open(root / name, os.O_RDONLY) for name in names}

    def _read(self, name: str, row: dict, width: int, dtype) -> np.ndarray:
        """Read ``row``'s ``[length, width]`` slice of one flat token-major file.

        Reads straight into a fresh buffer (no intermediate bytes copy); the
        syscall releases the GIL, so prefetch threads overlap with compute. The
        buffer is a ``bytearray``, not ``np.empty``: NumPy requests transparent
        huge pages for large arrays, and under memory pressure the kernel then
        stalled reads for seconds compacting memory.
        """
        itemsize = np.dtype(dtype).itemsize
        buffer = bytearray(row["length"] * width * itemsize)
        view = memoryview(buffer)
        offset = row["offset"] * width * itemsize
        done = 0
        while done < len(view):
            count = os.preadv(self.fds[name], [view[done:]], offset + done)
            if count == 0:
                raise OSError(f"short read from {self.root / name}")
            done += count
        return np.frombuffer(buffer, dtype).reshape(row["length"], width)

    def __len__(self) -> int:
        return len(self.rows)

    def read_raw(self, index: int) -> tuple[np.ndarray, ...]:
        """Row ``index`` in its stored format (thread-safe; no device work)."""
        row = self.rows[index]
        if self.fmt == "bf16":
            return (self._read("bf16.bin", row, self.hidden, np.int16),)
        return (
            self._read("fp8.bin", row, self.hidden, np.uint8),
            self._read("fp8_scales.bin", row, self.hidden // fp8.BLOCK, np.float32),
        )

    def from_raw(
        self, raw: tuple[np.ndarray, ...], device: torch.device | str = "cpu"
    ) -> torch.Tensor:
        """Turn :meth:`read_raw` output into BF16 ``[tokens, hidden]``."""
        if self.fmt == "bf16":
            return torch.from_numpy(raw[0]).to(device).view(torch.bfloat16)
        codes, scales = (torch.from_numpy(a).to(device) for a in raw)
        return fp8.dequantize(codes, scales)

    def load(self, index: int, device: torch.device | str = "cpu") -> torch.Tensor:
        """Return row ``index`` as BF16 ``[tokens, hidden]`` on ``device``."""
        return self.from_raw(self.read_raw(index), device)


class ResidentStore:
    """A store's rows held in memory (GPU, or pinned CPU) in their stored format.

    Rows are copied once from disk; each :meth:`load` then slices the resident
    buffer, so training epochs after the first do no disk or page-cache I/O.
    FP8 rows stay as codes plus scales and are dequantized per load.
    """

    def __init__(self, store: Store, device: torch.device | str):
        self.fmt, self.meta, self.hidden = store.fmt, store.meta, store.hidden
        device = torch.device(device)
        pin = device.type == "cpu" and torch.cuda.is_available()
        total = sum(r["length"] for r in store.rows)
        dtype = torch.int16 if self.fmt == "bf16" else torch.uint8
        self.values = torch.empty((total, self.hidden), dtype=dtype, pin_memory=pin)
        self.scales = (
            torch.empty((total, self.hidden // fp8.BLOCK), pin_memory=pin)
            if self.fmt == "fp8"
            else None
        )
        self.rows, offset = [], 0
        for row in store.rows:
            span = slice(offset, offset + row["length"])
            if self.fmt == "bf16":
                bits = store._read("bf16.bin", row, self.hidden, np.int16)
                self.values[span] = torch.from_numpy(bits)
            else:
                codes = store._read("fp8.bin", row, self.hidden, np.uint8)
                scales = store._read(
                    "fp8_scales.bin", row, self.hidden // fp8.BLOCK, np.float32
                )
                self.values[span] = torch.from_numpy(codes)
                self.scales[span] = torch.from_numpy(scales)
            self.rows.append({**row, "offset": offset})
            offset += row["length"]
        if device.type != "cpu":
            self.values = self.values.to(device)
            self.scales = self.scales.to(device) if self.scales is not None else None

    def __len__(self) -> int:
        return len(self.rows)

    def load(self, index: int, device: torch.device | str = "cpu") -> torch.Tensor:
        row = self.rows[index]
        span = slice(row["offset"], row["offset"] + row["length"])
        values = self.values[span].to(device, non_blocking=True)
        if self.fmt == "bf16":
            return values.view(torch.bfloat16)
        scales = self.scales[span].to(device, non_blocking=True)
        return fp8.dequantize(values, scales)
