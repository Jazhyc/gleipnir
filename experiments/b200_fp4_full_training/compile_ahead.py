"""Populate native plan caches in spare CPUs without replaying the model."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import multiprocessing
import os
import time
from pathlib import Path

from experiments.b200_fp4_full_training.campaign import (
    OUTPUT,
    ROOT,
    configuration,
    file_hash,
    verify_sources,
    write_json,
)


def future_shapes(records: list[dict], minimum_update: int) -> list[int]:
    """Preserve first-use order while compiling each future token count once."""
    selected = dict.fromkeys(
        int(row["tokens"]) for row in records if row["update"] >= minimum_update
    )
    if any(m <= 0 for m in selected):
        raise ValueError("invalid native plan token count")
    return list(selected)


def initialize() -> None:
    import torch

    from gleipnir.native_fp4_training import verify_native_fp4_runtime

    torch.set_num_threads(1)
    verify_native_fp4_runtime()


def compile_shape(m: int) -> dict:
    from cudnn.frost import compiled_cache

    from gleipnir.cudnn_fp4_epilogue import Nvfp4ScaledGemm

    started = time.monotonic()
    before = compiled_cache.stats()
    for k, n in ((2560, 18432), (9216, 2560), (18432, 2560), (2560, 9216)):
        plan = Nvfp4ScaledGemm(m, k, n)
        # The backend uses a one-byte marker even when the served JIT needs none.
        if plan.workspace_bytes or plan.reference.plan.workspace_bytes > 1:
            raise ValueError("compile-only helper refuses nontrivial workspaces")
        del plan
    after = compiled_cache.stats()
    return {
        "pid": os.getpid(),
        "m": m,
        "seconds": time.monotonic() - started,
        "cache_delta": {
            key: after.get(key, 0) - before.get(key, 0)
            for key in set(before) | set(after)
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=16, choices=range(1, 17))
    parser.add_argument("--minimum-update", type=int, required=True)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()
    config = configuration()
    verify_sources(config)
    metadata = json.loads((ROOT / config["control_metadata"]).read_text())
    shapes = future_shapes(
        metadata["adaptive_microbatching"]["records"], args.minimum_update
    )
    if args.limit:
        shapes = shapes[: args.limit]
    import torch

    free, _ = torch.cuda.mem_get_info()
    if free < (8 + args.workers) * 1024**3:
        raise ValueError("insufficient spare GPU memory for compiler contexts")
    path = OUTPUT / f"compile_ahead_{args.minimum_update}"
    path.mkdir(exist_ok=False)
    (path / "executed_compile_ahead.py").write_bytes(Path(__file__).read_bytes())
    contract = {
        "workers": args.workers,
        "minimum_update": args.minimum_update,
        "shapes": shapes,
        "source_sha256": file_hash(Path(__file__)),
        "control_metadata_sha256": config["control_metadata_sha256"],
        "model_loaded": False,
        "model_replayed": False,
        "training_source_changed": False,
        "cache_paths": {k: v for k, v in os.environ.items() if "CACHE" in k},
    }
    write_json(path / "contract.json", contract)
    started = time.time()
    completed = []
    with concurrent.futures.ProcessPoolExecutor(
        max_workers=args.workers,
        mp_context=multiprocessing.get_context("spawn"),
        initializer=initialize,
    ) as pool:
        futures = [pool.submit(compile_shape, m) for m in shapes]
        for future in concurrent.futures.as_completed(futures):
            worker = json.loads((OUTPUT / "worker.json").read_text())
            if worker["status"] == "failed":
                for pending in futures:
                    pending.cancel()
                raise RuntimeError("training failed; stopping cache population")
            try:
                result = future.result()
            except BaseException:
                for pending in futures:
                    pending.cancel()
                raise
            completed.append(result)
            print(json.dumps(result), flush=True)
            write_json(
                path / "progress.json",
                {
                    "completed": len(completed),
                    "total": len(shapes),
                    "elapsed_seconds": time.time() - started,
                },
            )
    write_json(path / "complete.json", {"plans": completed, **contract})
    print("native_compile_ahead_complete", flush=True)


if __name__ == "__main__":
    main()
