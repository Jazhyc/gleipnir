"""Complete native FP4 attention projection cost and quantized-reference checks."""

import argparse
import gc
import hashlib
import json
import shutil
import statistics
import time
import traceback
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    import torch

    from gleipnir.cudnn_fp4_epilogue import Nvfp4ScaledGemm
    from gleipnir.cudnn_fp4_gemm import PackedNvfp4, decode_operand, pack_operand
    from gleipnir.serving_attention_fp4 import ROWS, SHAPES
    from gleipnir.serving_fp4_prepare import vendor_pack
    from gleipnir.serving_fp4_tuning import BASE_TILE, retile, row_band

    sources = [
        Path("src/gleipnir/__init__.py"),
        Path("src/gleipnir/_compat.py"),
        Path(__file__).relative_to(Path.cwd()),
        Path("src/gleipnir/serving/fp4/attention.py"),
        Path("src/gleipnir/serving/vllm/frost_attention_fp4.py"),
        Path("src/gleipnir/serving/vllm/frost_fp4.py"),
        Path("src/gleipnir/kernels/fp4/cudnn_fp4_gemm.py"),
        Path("src/gleipnir/kernels/fp4/cudnn_fp4_epilogue.py"),
        Path("src/gleipnir/serving/fp4/prepare.py"),
        Path("src/gleipnir/serving/fp4/tuning.py"),
    ]
    report = {
        "state": "starting",
        "passed": False,
        "started_at_unix": time.time(),
        "gpu": torch.cuda.get_device_name(),
        "sources": {
            str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources
        },
        "checks": [],
        "selected": {},
        "relative_l2_limit": 0.01,
        "runtime": {"torch": torch.__version__},
        "timing_scope": "packing + GEMM/descale vs BF16; CUDA graph replay",
    }
    archive = args.output.with_suffix("").with_name(args.output.stem + "_sources")
    for p in sources:
        dest = archive / p
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(p, dest)

    def save():
        args.output.parent.mkdir(parents=True, exist_ok=True)
        tmp = args.output.with_suffix(".tmp")
        tmp.write_text(json.dumps(report, indent=2) + "\n")
        tmp.replace(args.output)

    def timing(fn):
        fn()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            out = fn()
        for _ in range(5):
            graph.replay()
        samples = []
        for _ in range(5):
            start, end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
            start.record()
            for _ in range(16):
                graph.replay()
            end.record()
            end.synchronize()
            samples.append(start.elapsed_time(end) / 16)
        return statistics.median(samples), samples, graph, out

    def relative(out, ref):
        return float(
            (out.float() - ref.float()).norm() / ref.float().norm().clamp_min(1e-12)
        )

    torch.backends.cuda.matmul.allow_tf32 = False
    gen = torch.Generator(device="cuda").manual_seed(173)
    selection = json.loads(
        Path("results/b200_attention_gdn_serving/fp4_gemm_tune01.json").read_text()
    )["selected"]
    save()
    try:
        for projection, (k, n) in SHAPES.items():
            chosen = (
                {b: BASE_TILE for b in ("small", "large")}
                if projection == "qkv_proj"
                else selection["gdn_output"]
            )
            plans = {
                tile: retile(Nvfp4ScaledGemm(16384, k, n, runtime_m=True), tile)
                for tile in set(chosen.values())
            }
            report["selected"][projection] = chosen
            weight = (
                torch.randn(n, k, device="cuda", dtype=torch.bfloat16, generator=gen)
                * 0.02
            )
            b = pack_operand(weight, weight=True)
            bn = (
                decode_operand(PackedNvfp4(b.codes[:16], b.scales, b.inverse))
                * b.inverse
            )

            def project(a, ps=plans, s=chosen, bb=b):
                return ps[s[row_band(a.codes.shape[0])]](a, bb)

            def reference(a, ww=bn, bb=b):
                normalized = decode_operand(a) * a.inverse[:, None]
                raw = (normalized @ ww.T).bfloat16().float()
                return (raw / (a.inverse[:, None] * bb.inverse)).bfloat16()

            for m in sorted(ROWS):
                print("measure", projection, m, flush=True)
                x = torch.randn(
                    m, k, device="cuda", dtype=torch.bfloat16, generator=gen
                )
                x[0].zero_()
                if m > 1:
                    x[1].mul_(20)
                a = vendor_pack(x)
                out = project(a)
                ref = reference(a)
                record = {
                    "projection": projection,
                    "shape": [k, n],
                    "rows": m,
                    "tile": chosen[row_band(m)],
                    "finite": bool(torch.isfinite(out).all()),
                    "relative_l2": relative(out[:, :16], ref),
                    "zero_row_exact": bool((out[0] == 0).all()),
                }
                bf16 = torch.nn.functional.linear(x, weight)
                record["bf16_relative_l2"] = relative(out, bf16)
                bf_ms, bs, bg, bo = timing(
                    lambda xx=x, ww=weight: torch.nn.functional.linear(xx, ww)
                )
                ms, ss, pg, po = timing(lambda xx=x: project(vendor_pack(xx)))
                record.update(
                    milliseconds=ms,
                    samples_ms=ss,
                    bf16_ms=bf_ms,
                    bf16_samples_ms=bs,
                    speed_ratio=bf_ms / ms,
                )
                del bg, bo, pg, po, bf16
                # Replay the same full packing+GEMM graph with changed BF16 input.
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    captured = project(vendor_pack(x))
                graph.replay()
                torch.cuda.synchronize()
                original = captured.clone()
                x[0].fill_(0.5)
                graph.replay()
                torch.cuda.synchronize()
                record["replay_changed"] = not torch.equal(captured[0], original[0])
                record["unchanged_rows_exact"] = torch.equal(captured[1:], original[1:])
                record["replay_relative_l2"] = relative(
                    captured[:, :16], reference(vendor_pack(x))
                )
                record["passed"] = (
                    record["finite"]
                    and record["relative_l2"] <= 0.01
                    and record["replay_relative_l2"] <= 0.01
                    and all(
                        record[key]
                        for key in (
                            "zero_row_exact",
                            "replay_changed",
                            "unchanged_rows_exact",
                        )
                    )
                )
                report["checks"].append(record)
                save()
                if not record["passed"]:
                    raise ValueError("native attention projection admission failed")
                del graph, captured, original, x, a, out, ref
                gc.collect()
                torch.cuda.empty_cache()
            del plans, weight, b, bn
        report.update(state="completed", passed=True)
    except Exception as exc:
        report.update(state="failed", error=repr(exc), traceback=traceback.format_exc())
        print(report["traceback"], flush=True)
    finally:
        report["elapsed_seconds"] = time.time() - report["started_at_unix"]
        save()


if __name__ == "__main__":
    main()
