"""Capture block-15 activations from the optimized Lens server and score probes.

Each row's bare trajectory (NTP rendering) is sent to the running Lens server,
which returns the post-block-15 residual stream at every position. The client
checks the prompt length against the Transformers tokenizer, drops the
attention-sink token, applies the probe study's blockwise FP8 round trip, and
scores the full01 NTP probe bank at each requested checkpoint. Only logits
are kept, written in resumable parts; a few diagnostic rows also keep their
raw BF16 activations for the comparison with the BF16 extraction.
"""

from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path

import numpy as np
import torch

from gleipnir.probes import fp8
from gleipnir.probes.training import make_sweep, score

ROOT = Path(__file__).resolve().parents[2]
LAYER = 15
PART_ROWS = 256


class Chunk:
    """A few rows of in-memory activations, shaped like a probe store."""

    def __init__(self, values: list[torch.Tensor]):
        self.values = values
        self.rows = [{"length": len(v)} for v in values]

    def __len__(self) -> int:
        return len(self.values)

    def load(self, index: int, device: str = "cpu") -> torch.Tensor:
        return self.values[index].to(device)


def load_sweeps(run: Path, epochs: list[int]) -> tuple[dict, list[str]]:
    config = json.loads((run / "config.json").read_text())
    sweeps = {}
    for epoch in epochs:
        sweep = make_sweep(
            config["architectures"],
            config["objectives"],
            config["seeds"],
            2560,
            lr=config["lr"],
            weight_decay=config["weight_decay"],
            device="cuda",
        )
        state = torch.load(run / f"probes_e{epoch}.pt", map_location="cuda")
        if [tuple(s) for s in state["specs"]] != [
            (s.name, s.architecture) for s in sweep.specs
        ]:
            raise ValueError(f"probe order differs in probes_e{epoch}.pt")
        sweep.bank.load_state_dict(state["bank"])
        sweeps[epoch] = sweep
    return sweeps, [s.name for s in next(iter(sweeps.values())).specs]


def score_chunk(sweeps: dict, values: list[torch.Tensor]) -> dict[str, np.ndarray]:
    """``{variant: [rows, epochs, probes]}`` for raw and FP8-round-tripped input."""
    out = {}
    for variant in ("fp8", "raw"):
        chunk = Chunk(
            [fp8.dequantize(*fp8.quantize(v)) for v in values]
            if variant == "fp8"
            else values
        )
        per_epoch = []
        for sweep in sweeps.values():
            logits = score(sweep, chunk, 16, "cuda", 65536)
            per_epoch.append(np.array([logits[s.name] for s in sweep.specs]).T)
        out[variant] = np.stack(per_epoch, axis=1).astype(np.float32)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", required=True, help="jsonl with id, trajectory")
    parser.add_argument("--out", required=True)
    parser.add_argument("--snapshot", required=True, help="base model snapshot dir")
    parser.add_argument(
        "--probe-run", default="results/probe_generalization/full01/ntp-fp8"
    )
    parser.add_argument("--epochs", nargs="+", type=int, default=[2, 4, 8, 16])
    parser.add_argument("--url", default="http://127.0.0.1:8010")
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--limit", type=int)
    parser.add_argument(
        "--keep-ids", help="file of row ids whose raw activations are saved"
    )
    args = parser.parse_args()

    import threading

    import zstandard
    from transformers import AutoTokenizer
    from vllm_lens._helpers import _serialize

    from gleipnir.serving.lens import MonitorLensClient

    class PerThreadDecompressor:
        """Lens shares one module-level zstd decompressor, which is not
        thread-safe: concurrent responses segfault in ``decompress``."""

        local = threading.local()

        def decompress(self, raw: bytes) -> bytes:
            if not hasattr(self.local, "codec"):
                self.local.codec = zstandard.ZstdDecompressor()
            return self.local.codec.decompress(raw)

    _serialize._ZSTD_DECOMPRESSOR = PerThreadDecompressor()

    out = Path(args.out)
    (out / "parts").mkdir(parents=True, exist_ok=True)
    rows = [json.loads(line) for line in open(args.rows)]
    rows = rows[: args.limit] if args.limit else rows
    done = set()
    for part in sorted((out / "parts").glob("*.npz")):
        done.update(np.load(part)["ids"].tolist())
    todo = [r for r in rows if r["id"] not in done]
    keep = set(Path(args.keep_ids).read_text().split()) if args.keep_ids else set()
    (out / "activations").mkdir(exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(args.snapshot)
    sweeps, names = load_sweeps(ROOT / args.probe_run, args.epochs)
    (out / "probes.json").write_text(
        json.dumps({"epochs": args.epochs, "probes": names}) + "\n"
    )
    client = MonitorLensClient(args.url, timeout=1800)

    def capture(row: dict) -> tuple[dict, torch.Tensor, int]:
        result = client.score(
            row["trajectory"], capture_layers=[LAYER], capture_positions="all"
        )
        values = result["activations"]["residual_stream"][0]
        return row, values, result["prompt_tokens"]

    started, tokens, pending, buffer = time.time(), 0, set(), []
    queue = iter(todo)
    with ThreadPoolExecutor(args.workers) as pool:
        for _ in range(args.workers * 2):
            row = next(queue, None)
            if row is not None:
                pending.add(pool.submit(capture, row))
        while pending:
            finished, pending = wait(pending, return_when=FIRST_COMPLETED)
            for future in finished:
                row, values, prompt_tokens = future.result()
                expected = len(
                    tokenizer(row["trajectory"], add_special_tokens=False)["input_ids"]
                )
                if prompt_tokens != expected or len(values) != expected:
                    raise ValueError(
                        f"token mismatch in {row['id']}: server {prompt_tokens}, "
                        f"captured {len(values)}, tokenizer {expected}"
                    )
                if not bool(torch.isfinite(values).all()):
                    raise ValueError(f"nonfinite activations in row {row['id']}")
                if row["id"] in keep:
                    torch.save(
                        values.to(torch.bfloat16).cpu(),
                        out / "activations" / f"{row['id']}.pt",
                    )
                buffer.append((row["id"], expected, values[1:].to("cuda")))
                tokens += expected
                nxt = next(queue, None)
                if nxt is not None:
                    pending.add(pool.submit(capture, nxt))
            if len(buffer) >= PART_ROWS or (not pending and buffer):
                scored = score_chunk(sweeps, [v for _, _, v in buffer])
                index = len(list((out / "parts").glob("*.npz")))
                np.savez(
                    out / "parts" / f"{index:05d}.npz",
                    ids=np.array([i for i, _, _ in buffer]),
                    tokens=np.array([t for _, t, _ in buffer]),
                    **scored,
                )
                done.update(i for i, _, _ in buffer)
                buffer = []
                rate = tokens / (time.time() - started)
                print(f"rows {len(done)}/{len(rows)} {rate:.0f} tok/s", flush=True)
    (out / "complete.json").write_text(
        json.dumps({"rows": len(done), "seconds": time.time() - started}) + "\n"
    )


if __name__ == "__main__":
    main()
