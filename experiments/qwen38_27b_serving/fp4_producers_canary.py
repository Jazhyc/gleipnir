"""Validate config-derived fused FP4 producers at a served model's widths.

The 4B fusion and vendor canaries (`experiments/b200_attention_gdn_serving/
fp4_fusion_canary.py`, `fp4_prepare_canary.py`) with K read from ``--model``:
fused SwiGLU (K = intermediate size), fused residual + RMSNorm (K = hidden)
and the vendor CUDA packer (GDN projection inputs). For rows 1, 17, 129 and
32768 and each warp count, each producer is checked against the compiled BF16
producer plus the FROST packer (decoded relative L2 and GEMM relative L2 at
most 0.01), with exact residuals, row isolation and changed-input graph
replay, and timed against that baseline. One receipt holds all three modes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

ROWS = (1, 17, 129, 32768)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    import torch

    from gleipnir.kernels.fp4.cudnn_fp4_epilogue import Nvfp4ScaledGemm
    from gleipnir.kernels.fp4.cudnn_fp4_gemm import decode_operand, pack_operand
    from gleipnir.serving.fp4_geometry_producers import (
        NORM_WARPS,
        SILU_WARPS,
        norm_pack,
        producer_k,
        silu_pack,
        vendor_k,
        vendor_pack,
    )

    config = json.loads((args.model / "config.json").read_text())
    text = SimpleNamespace(**config.get("text_config", config))
    widths, allowed = producer_k(text), sorted(vendor_k(text))
    torch.manual_seed(71)
    torch.backends.cuda.matmul.allow_tf32 = False
    sources = [
        Path("src/gleipnir/__init__.py"),
        Path("src/gleipnir/_compat.py"),
        Path(__file__),
        Path("src/gleipnir/serving/fp4_geometry_producers.py"),
        Path("src/gleipnir/serving/fp4/fusion.py"),
        Path("src/gleipnir/serving/fp4/prepare.py"),
        Path("src/gleipnir/serving/fp4/integration.py"),
        Path("src/gleipnir/kernels/fp4/cudnn_fp4_gemm.py"),
        Path("src/gleipnir/kernels/fp4/cudnn_fp4_epilogue.py"),
    ]
    report = {
        "passed": False,
        "relative_l2_limit": 0.01,
        "producer_k": widths,
        "vendor_k": allowed,
        "modes": {},
        "sources": {
            str(p.relative_to(Path.cwd()) if p.is_absolute() else p): hashlib.sha256(
                p.read_bytes()
            ).hexdigest()
            for p in sources
        },
        "gpu": torch.cuda.get_device_name(),
        "bf16_producer_boundary_preserved": True,
    }

    def save() -> None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")

    def pack(x):
        return pack_operand(x, row_amax=True, chunked_rows=True, hardware_packing=True)

    def activated(x):
        k = x.shape[1] // 2
        return torch.nn.functional.silu(x[:, :k]) * x[:, k:]

    def normalized(x, residual, weight):
        summed = x.float() + residual.float()
        norm = summed * torch.rsqrt(summed.square().mean(-1, keepdim=True) + 1e-6)
        return (norm * (weight.float() + 1)).to(torch.bfloat16), summed.to(
            torch.bfloat16
        )

    act_compiled = torch.compile(activated, dynamic=True, fullgraph=True)
    norm_compiled = torch.compile(normalized, dynamic=True, fullgraph=True)

    def bench(fn):
        fn()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            output = fn()
        for _ in range(5):
            graph.replay()
        samples = []
        for _ in range(3):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            for _ in range(32):
                graph.replay()
            end.record()
            end.synchronize()
            samples.append(start.elapsed_time(end) / 32)
        return sorted(samples)[1], graph, output

    plan = [("silu", widths["silu"], SILU_WARPS), ("norm", widths["norm"], NORM_WARPS)]
    plan += [("vendor", k, (None,)) for k in allowed]
    save()
    try:
        for mode, k, warps_values in plan:
            section = report["modes"].setdefault(mode, {"checks": []})
            weight = pack_operand(
                torch.randn(256, k, device="cuda", dtype=torch.bfloat16), weight=True
            )
            native = Nvfp4ScaledGemm(16384, k, 256, runtime_m=True)
            bn = decode_operand(weight).float() * weight.inverse
            norm_weight = torch.randn(k, device="cuda", dtype=torch.bfloat16) * 0.1
            for m in ROWS:
                width = 2 * k if mode == "silu" else k
                x = torch.randn(m, width, device="cuda", dtype=torch.bfloat16)
                residual = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
                if m > 1:
                    x[0].zero_()
                    residual[0].zero_()
                    x[1].mul_(1000)
                    residual[1].mul_(1000)
                expected_residual = None
                if mode == "silu":
                    expected = pack(act_compiled(x))

                    def baseline(inputs=x):
                        return pack(act_compiled(inputs)), None
                elif mode == "norm":
                    values, expected_residual = norm_compiled(x, residual, norm_weight)
                    expected = pack(values)

                    def baseline(inputs=x, r=residual, w=norm_weight):
                        values, summed = norm_compiled(inputs, r, w)
                        return pack(values), summed
                else:
                    expected = pack(x)

                    def baseline(inputs=x):
                        return pack(inputs), None

                reference = decode_operand(expected).float()
                old_ms, _, _ = bench(baseline)
                for warps in warps_values:
                    if mode == "silu":

                        def fn(inputs=x, threads=warps, width=k):
                            return silu_pack(inputs, width, warps=threads), None
                    elif mode == "norm":

                        def fn(inputs=x, r=residual, w=norm_weight, t=warps, width=k):
                            return norm_pack(inputs, r, w, 1e-6, width, warps=t)
                    else:

                        def fn(inputs=x):
                            return vendor_pack(inputs, set(allowed)), None

                    observed, summed = fn()
                    decoded = decode_operand(observed).float()
                    relative = (
                        (decoded - reference).norm() / reference.norm().clamp_min(1e-30)
                    ).item()
                    finite = bool(
                        torch.isfinite(decoded).all()
                        and torch.isfinite(observed.inverse).all()
                    )
                    if mode == "norm":
                        assert torch.equal(summed, expected_residual), "residual"
                    raw = (
                        (decoded * observed.inverse[:, None] @ bn.t())
                        .to(torch.bfloat16)
                        .float()
                    )
                    ref_gemm = (
                        (raw / (observed.inverse[:, None] * weight.inverse))
                        .to(torch.bfloat16)
                        .float()
                    )
                    got_gemm = native(observed, weight).float()
                    gemm_error = (
                        (got_gemm - ref_gemm).norm() / ref_gemm.norm().clamp_min(1e-30)
                    ).item()
                    new_ms, graph, graph_output = bench(fn)
                    saved_x, saved_r = x.clone(), residual.clone()
                    if m > 1:
                        x[0].fill_(5)
                        altered = decode_operand(fn()[0]).float()
                        assert torch.equal(decoded[1:], altered[1:]), "row coupling"
                    x.mul_(0.5)
                    residual.mul_(0.25)
                    graph.replay()
                    replay = decode_operand(graph_output[0]).float()
                    fresh = decode_operand(fn()[0]).float()
                    assert torch.equal(replay, fresh), "changed-input graph replay"
                    x.copy_(saved_x)
                    residual.copy_(saved_r)
                    row = {
                        "m": m,
                        "k": k,
                        "warps": warps,
                        "relative_l2": relative,
                        "gemm_relative_l2": gemm_error,
                        "finite": finite,
                        "codes_equal": torch.equal(
                            expected.codes.view(torch.uint8),
                            observed.codes.view(torch.uint8),
                        ),
                        "baseline_graph_ms": old_ms,
                        "candidate_graph_ms": new_ms,
                        "throughput_ratio": old_ms / new_ms,
                        "isolation_passed": True,
                        "updated_input_graph_passed": True,
                    }
                    section["checks"].append(row)
                    save()
                    print(mode, json.dumps(row), flush=True)
                    assert finite and relative <= 0.01 and gemm_error <= 0.01, row
            if mode in {"silu", "norm"}:
                largest = [r for r in section["checks"] if r["m"] == max(ROWS)]
                section["selected_warps"] = min(
                    largest, key=lambda r: r["candidate_graph_ms"]
                )["warps"]
            del weight, native, bn
            torch.cuda.empty_cache()
        report["passed"] = True
    except BaseException as error:
        report["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        save()


if __name__ == "__main__":
    main()
