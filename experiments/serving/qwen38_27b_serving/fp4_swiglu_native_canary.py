"""Validate and select config-derived native FP4 SwiGLU output.

The 4B comparison (`experiments/b200_attention_gdn_serving/
fp4_swiglu_native_output_compare.py`) at the served geometry from ``--model``,
for each tile in ``fp4_geometry_swiglu.TILES``, on the same nine row counts:
weight permutation exactness, packing against the independent E4M3/E2M1
reference, decoded down-projection arithmetic, zero rows, changed-input graph
replay and row isolation (all at the 1% limit), plus the precision change
against the fused-producer baseline (reported, not gated). It also requires
the generated source at the 4B width to equal the 4B module's byte for byte.

Selection: the passing tile with the best geometric-mean full-MLP speed, and
its native row band, the longest contiguous run of tested row counts where it
is at least 2% faster (none: native is not used).
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import statistics
import time
import traceback
from pathlib import Path

ROWS = (1, 17, 129, 1536, 1537, 2304, 4096, 29184, 32768)
GAIN = 1.02


def select(results: list[dict]) -> dict:
    """Pick a tile and its native-output row band from full-MLP timings.

    The band is the longest contiguous run of tested row counts where the
    tile is at least ``GAIN`` faster; other rows keep the fused producer.
    """
    scores = {}
    for tile in {r["tile"] for r in results}:
        rows = [r for r in results if r["tile"] == tile]
        if {r["m"] for r in rows if r.get("passed")} != set(ROWS):
            continue
        scores[tile] = statistics.geometric_mean(
            r["full_mlp_speed_ratio"] for r in rows
        )
    if not scores:
        return {"tile": None, "minimum_rows": None, "maximum_rows": None}
    tile = max(scores, key=scores.get)
    ratios = {r["m"]: r["full_mlp_speed_ratio"] for r in results if r["tile"] == tile}
    best, run = [], []
    for m in sorted(ratios):
        run = run + [m] if ratios[m] >= GAIN else []
        if len(run) > len(best):
            best = run
    return {
        "tile": tile,
        "minimum_rows": best[0] if best else None,
        "maximum_rows": best[-1] if best else None,
        "scores": scores,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    import torch

    from gleipnir.kernels.fp4.cudnn_fp4_epilogue import Nvfp4ScaledGemm
    from gleipnir.kernels.fp4.cudnn_fp4_gemm import decode_operand, pack_operand
    from gleipnir.serving.fp4.swiglu import UPSTREAM, interleaved_rows, prepare_weight
    from gleipnir.serving.fp4.swiglu_block_reference import reference_decode
    from gleipnir.serving.fp4.swiglu_native_output import native_output_source
    from gleipnir.serving.fp4_geometry_producers import silu_pack
    from gleipnir.serving.fp4_geometry_swiglu import (
        TILES,
        NativeSwiGlu,
        load_kernel,
        native_source,
    )

    torch.backends.cuda.matmul.allow_tf32 = False
    config = json.loads((args.model / "config.json").read_text())
    text = config.get("text_config", config)
    hidden, mlp = int(text["hidden_size"]), int(text["intermediate_size"])
    sources = [
        Path("src/gleipnir/__init__.py"),
        Path("src/gleipnir/_compat.py"),
        Path(__file__),
        Path("src/gleipnir/serving/fp4_geometry_swiglu.py"),
        Path("src/gleipnir/serving/fp4_geometry_producers.py"),
        Path("src/gleipnir/serving/fp4/swiglu.py"),
        Path("src/gleipnir/serving/fp4/swiglu_overhead.py"),
        Path("src/gleipnir/serving/fp4/swiglu_native_output.py"),
        Path("src/gleipnir/serving/fp4/swiglu_block_reference.py"),
        Path("src/gleipnir/serving/fp4/fusion.py"),
        Path("src/gleipnir/kernels/fp4/cudnn_fp4_epilogue.py"),
        Path("src/gleipnir/kernels/fp4/cudnn_fp4_gemm.py"),
        UPSTREAM,
    ]
    original = UPSTREAM.read_text()
    archive = args.output.with_name(args.output.stem + "_kernel")
    kernel, hashes = load_kernel(Path.cwd(), archive, mlp)
    report = {
        "state": "starting",
        "passed": False,
        "started_at_unix": time.time(),
        "gpu": torch.cuda.get_device_name(),
        "geometry": {"hidden": hidden, "mlp": mlp},
        "kernel": hashes,
        "base_identity": native_source(original, 9216)
        == native_output_source(original),
        "scaling": "unit_global_inverse_local_16_element_e4m3",
        "relative_l2_limit": 0.01,
        "selection_minimum_gain": GAIN,
        "rows": list(ROWS),
        "tiles": [list(t) for t in TILES],
        "sources": {
            str(p.relative_to(Path.cwd()) if p.is_absolute() else p): hashlib.sha256(
                p.read_bytes()
            ).hexdigest()
            for p in sources
        },
        "results": [],
    }

    def save():
        args.output.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.output.with_suffix(".tmp")
        temporary.write_text(json.dumps(report, indent=2) + "\n")
        temporary.replace(args.output)

    def compare(out, ref):
        out, ref = out.float(), ref.float()
        finite = bool(torch.isfinite(out).all())
        relative = float((out - ref).norm() / ref.norm().clamp_min(1e-12))
        return {
            "finite": finite,
            "relative_l2": relative,
            "passed": finite and relative <= 0.01,
        }

    def timing(fn):
        fn()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            captured = fn()
        for _ in range(5):
            graph.replay()
        samples = []
        for _ in range(5):
            begin, end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
            begin.record()
            for _ in range(32):
                graph.replay()
            end.record()
            end.synchronize()
            samples.append(begin.elapsed_time(end) / 32)
        return statistics.median(samples), graph, captured

    def pack(x):
        return pack_operand(x, row_amax=True, chunked_rows=True, hardware_packing=True)

    save()
    try:
        assert report["base_identity"], "generated source differs from the 4B module"
        gen = torch.Generator(device="cuda").manual_seed(173)
        ref = Nvfp4ScaledGemm(16384, hidden, 2 * mlp, runtime_m=True)
        down = Nvfp4ScaledGemm(16384, mlp, hidden, runtime_m=True)
        weight = pack_operand(
            torch.randn(
                2 * mlp, hidden, device="cuda", dtype=torch.bfloat16, generator=gen
            )
            * 0.02,
            weight=True,
        )
        down_weight = pack_operand(
            torch.randn(hidden, mlp, device="cuda", dtype=torch.bfloat16, generator=gen)
            * 0.02,
            weight=True,
        )
        interleaved = prepare_weight(weight)
        report["weight_permutation_exact"] = bool(
            torch.equal(
                decode_operand(interleaved),
                decode_operand(weight)[interleaved_rows(2 * mlp)],
            )
        )
        down16 = decode_operand(down_weight)[:16].float().T
        plans = {}
        for tile in TILES:
            name = f"{tile[0]}x{tile[1]}"
            try:
                plans[name] = NativeSwiGlu(kernel, hidden, mlp, tile=tile)
                plans[name].compile()
            except Exception as exc:
                report["results"].append(
                    {"tile": name, "compile_failed": repr(exc), "passed": False}
                )
                plans.pop(name, None)
                save()
        report["compiled_tiles"] = list(plans)

        def baseline(aa):
            return silu_pack(ref(aa, weight), mlp, warps=8)

        for m in ROWS:
            print("row_start", m, flush=True)
            x = torch.randn(
                m, hidden, device="cuda", dtype=torch.bfloat16, generator=gen
            )
            x[0].zero_()
            if m > 1:
                x[1].mul_(20)
            a = pack(x)
            gateup = ref(a, weight)
            gate, up = gateup.float().chunk(2, dim=1)
            activated = (gate / (1 + torch.exp(-gate)) * up).bfloat16()
            output_ref = down(baseline(a), down_weight)
            baseline_ms, g, c = timing(lambda aa=a: baseline(aa))
            full_ms, g2, c2 = timing(lambda aa=a: down(baseline(aa), down_weight))
            del g, c, g2, c2
            for name, plan in plans.items():
                record = {"m": m, "tile": name, "baseline_ms": baseline_ms}
                try:
                    candidate = plan(a, interleaved)
                    decoded = decode_operand(candidate)
                    record["packing_reference"] = compare(
                        decoded, reference_decode(activated)
                    )
                    out = down(candidate, down_weight)
                    record["arithmetic"] = compare(
                        out[:, :16], decoded.float() @ down16
                    )
                    record["baseline_precision"] = compare(out, output_ref)
                    record["activation_precision"] = compare(decoded, activated)
                    record["zero_row_exact"] = bool((out[0] == 0).all())
                    record["finite"] = bool(torch.isfinite(out).all())
                    if not record["finite"]:
                        raise FloatingPointError("nonfinite native FP4 output")
                    ms, graph, captured = timing(
                        lambda p=plan, aa=a: p(aa, interleaved)
                    )
                    cm, cg, cc = timing(
                        lambda p=plan, aa=a: down(p(aa, interleaved), down_weight)
                    )
                    del cg, cc
                    record.update(
                        milliseconds=ms,
                        speed_ratio=baseline_ms / ms,
                        baseline_full_mlp_ms=full_ms,
                        full_mlp_ms=cm,
                        full_mlp_speed_ratio=full_ms / cm,
                    )
                    first = down(captured, down_weight).clone()
                    changed = x.clone()
                    changed[0].fill_(0.5)
                    fresh = pack(changed)
                    a.codes.view(torch.uint8).copy_(fresh.codes.view(torch.uint8))
                    a.scales.copy_(fresh.scales)
                    a.inverse.copy_(fresh.inverse)
                    graph.replay()
                    torch.cuda.synchronize()
                    replayed = decode_operand(captured)
                    changed_out = down(captured, down_weight)
                    record["changed_replay"] = compare(
                        changed_out[:, :16], replayed.float() @ down16
                    )
                    fresh_gate, fresh_up = ref(a, weight).float().chunk(2, dim=1)
                    record["packing_replay"] = compare(
                        replayed,
                        reference_decode(
                            (
                                fresh_gate / (1 + torch.exp(-fresh_gate)) * fresh_up
                            ).bfloat16()
                        ),
                    )
                    record["unchanged_rows_exact"] = bool(
                        torch.equal(first[1:], changed_out[1:])
                    )
                    record["passed"] = all(
                        record[k]["passed"]
                        for k in (
                            "packing_reference",
                            "arithmetic",
                            "changed_replay",
                            "packing_replay",
                        )
                    ) and (record["zero_row_exact"] and record["unchanged_rows_exact"])
                    del graph, captured, first, changed_out, replayed, candidate, out
                    a = pack(x)
                except Exception as exc:
                    record.update(
                        passed=False,
                        execution_failed=repr(exc),
                        traceback=traceback.format_exc(),
                    )
                    print(record["traceback"], flush=True)
                    if isinstance(exc, (torch.AcceleratorError, FloatingPointError)):
                        report["results"].append(record)
                        save()
                        raise
                report["results"].append(record)
                print(
                    "measure",
                    json.dumps(
                        {
                            k: v
                            for k, v in record.items()
                            if not isinstance(v, (dict, str))
                        }
                    ),
                    flush=True,
                )
                save()
            del x, a, gateup, gate, up, activated, output_ref
            gc.collect()
            torch.cuda.empty_cache()
        timed = [r for r in report["results"] if "full_mlp_speed_ratio" in r]
        report["selection"] = select(timed)
        report["compile_count"] = {n: p.compile_count for n, p in plans.items()}
        report["passed"] = bool(
            report["weight_permutation_exact"] and report["selection"]["tile"]
        )
        report["strict_baseline_precision_passed"] = all(
            r.get("baseline_precision", {}).get("passed") for r in timed
        )
        report["state"] = "completed"
    except Exception as exc:
        report.update(state="failed", error=repr(exc), traceback=traceback.format_exc())
        print(report["traceback"], flush=True)
    finally:
        report["elapsed_seconds"] = time.time() - report["started_at_unix"]
        save()


if __name__ == "__main__":
    main()
