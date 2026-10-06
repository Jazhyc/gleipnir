"""Persistent native tile sweep: identical operands, graph replay and live receipts."""

import argparse
import gc
import hashlib
import json
import shutil
import statistics
import time
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    import torch

    from gleipnir.cudnn_fp4_epilogue import Nvfp4ScaledGemm
    from gleipnir.cudnn_fp4_gemm import pack_operand
    from gleipnir.serving_fp4_tuning import (
        BASE_TILE,
        CANDIDATES,
        SHAPES,
        retile,
        row_band,
    )

    if args.output.exists():
        raise FileExistsError(args.output)
    sources = [
        Path(__file__).relative_to(Path.cwd()),
        Path("src/gleipnir/serving_fp4_tuning.py"),
        Path("src/gleipnir/cudnn_fp4_epilogue.py"),
        Path("src/gleipnir/cudnn_fp4_gemm.py"),
        Path(".cache/kernels/nvidia_mxfp8/frontend/cudnn/gemm/frost/tile_config.py"),
    ]
    report = {
        "passed": False,
        "state": "starting",
        "started_at_unix": time.time(),
        "gpu": torch.cuda.get_device_name(),
        "baseline_tile": BASE_TILE,
        "candidates": list(CANDIDATES),
        "relative_l2_limit": 0.01,
        "selection_minimum_gain": 0.02,
        "sources": {
            str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources
        },
        "results": [],
        "selected": {},
        "timing_scope": "graph-replayed row descale plus GEMM, excludes packing",
    }

    def save():
        args.output.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.output.with_suffix(".tmp")
        temporary.write_text(json.dumps(report, indent=2) + "\n")
        temporary.replace(args.output)

    save()
    for p in sources:
        dest = args.output.with_suffix("").with_name(args.output.stem + "_sources") / p
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(p, dest)
    torch.manual_seed(173)
    torch.backends.cuda.matmul.allow_tf32 = False

    def pack(x):
        return pack_operand(x, row_amax=True, chunked_rows=True, hardware_packing=True)

    def timing(fn):
        fn()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            output = fn()
        for _ in range(5):
            graph.replay()
        samples = []
        for _ in range(5):
            start, end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
            start.record()
            for _ in range(32):
                graph.replay()
            end.record()
            end.synchronize()
            samples.append(start.elapsed_time(end) / 32)
        return statistics.median(samples), samples, graph, output

    def compare(out, ref):
        finite = bool(torch.isfinite(out).all())
        relative = float(
            (out.float() - ref.float()).norm() / ref.float().norm().clamp_min(1e-12)
        )
        return {
            "finite": finite,
            "relative_l2": relative,
            "passed": finite and relative <= 0.01,
        }

    try:
        for name, (k, n) in SHAPES.items():
            print("shape_start", name, k, n, flush=True)
            plans = {BASE_TILE: Nvfp4ScaledGemm(16384, k, n, runtime_m=True)}
            weight = pack_operand(
                torch.randn(n, k, device="cuda", dtype=torch.bfloat16), weight=True
            )
            for tile in CANDIDATES[1:]:
                print("compile", name, tile, flush=True)
                try:
                    plans[tile] = retile(
                        Nvfp4ScaledGemm(16384, k, n, runtime_m=True), tile
                    )
                except Exception as exc:
                    report["results"].append(
                        {"shape": name, "tile": tile, "compile_failed": repr(exc)}
                    )
                    save()
            shape_records = []
            for m in (1, 17, 129, 1536, 2304, 4096, 29184, 32768):
                x = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
                x[0].zero_()
                if m > 1:
                    x[1].mul_(1000)
                a = pack(x)
                ref = plans[BASE_TILE](a, weight)
                order = list(plans)
                if m % 2:
                    order.reverse()
                for tile in order:
                    record = {
                        "shape": name,
                        "k": k,
                        "n": n,
                        "m": m,
                        "band": row_band(m),
                        "tile": tile,
                    }
                    print("measure", name, m, tile, flush=True)
                    try:
                        plan = plans[tile]
                        out = plan(a, weight)
                        record.update(compare(out, ref))
                        record["zero_row_exact"] = bool((out[0] == 0).all())
                        record["passed"] &= record["zero_row_exact"]
                        ms, samples, graph, captured = timing(
                            lambda p=plan, aa=a, bb=weight: p(aa, bb)
                        )
                        record.update(milliseconds=ms, samples_ms=samples)
                        # Copy freshly packed changed inputs into captured addresses.
                        original = captured.clone()
                        changed_x = x.clone()
                        changed_x[0].fill_(0.5)
                        changed_a = pack(changed_x)
                        a.codes.view(torch.uint8).copy_(
                            changed_a.codes.view(torch.uint8)
                        )
                        a.scales.copy_(changed_a.scales)
                        a.inverse.copy_(changed_a.inverse)
                        changed_ref = plans[BASE_TILE](a, weight)
                        graph.replay()
                        torch.cuda.synchronize()
                        changed = compare(captured, changed_ref)
                        record["changed_replay"] = changed
                        record["unchanged_rows_exact"] = bool(
                            torch.equal(original[1:], captured[1:])
                        )
                        record["passed"] &= (
                            changed["passed"] and record["unchanged_rows_exact"]
                        )
                        del (
                            graph,
                            captured,
                            original,
                            changed_ref,
                            out,
                            changed_x,
                            changed_a,
                        )
                        # Restore zero-row inputs for identical candidates.
                        a = pack(x)
                    except Exception as exc:
                        record.update(passed=False, execution_failed=repr(exc))
                    report["results"].append(record)
                    shape_records.append(record)
                    save()
                del a, ref, x
                gc.collect()
                torch.cuda.empty_cache()
            selected = {}
            for band in ("small", "large"):
                scores = {}
                for tile in plans:
                    checks = [r for r in shape_records if r["tile"] == tile]
                    if len(checks) != 8 or not all(r["passed"] for r in checks):
                        continue
                    timings = [r for r in checks if r["band"] == band and r["m"] >= 129]
                    ratios = []
                    for r in timings:
                        base = next(
                            b
                            for b in shape_records
                            if b["tile"] == BASE_TILE and b["m"] == r["m"]
                        )
                        ratios.append(base["milliseconds"] / r["milliseconds"])
                    scores[tile] = statistics.geometric_mean(ratios)
                if BASE_TILE not in scores:
                    raise ValueError(f"baseline validation failed for {name}")
                best = max(scores, key=scores.get)
                selected[band] = best if scores[best] >= 1.02 else BASE_TILE
                report.setdefault("selection_scores", {}).setdefault(name, {})[band] = (
                    scores
                )
            report["selected"][name] = selected
            report["state"] = "shape_completed"
            print("shape_done", name, selected, flush=True)
            save()
            del plans, weight
            gc.collect()
            torch.cuda.empty_cache()
        report.update(passed=True, state="completed", completed_at_unix=time.time())
        save()
    except BaseException as exc:
        report.update(state="failed", error=repr(exc))
        save()
        raise


if __name__ == "__main__":
    main()
