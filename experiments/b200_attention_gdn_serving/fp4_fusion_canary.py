"""Validate both explicit row-scaled producer fusions on one offline worker."""

import argparse
import hashlib
import json
import shutil
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-prefix", type=Path, required=True)
    args = parser.parse_args()
    import torch

    from gleipnir.cudnn_fp4_epilogue import Nvfp4ScaledGemm
    from gleipnir.cudnn_fp4_gemm import decode_operand, pack_operand
    from gleipnir.serving_fp4_fusion import norm_pack, silu_pack

    torch.manual_seed(71)
    torch.backends.cuda.matmul.allow_tf32 = False

    def pack(x):
        return pack_operand(x, row_amax=True, chunked_rows=True, hardware_packing=True)

    def activated(x):
        return torch.nn.functional.silu(x[:, :9216]) * x[:, 9216:]

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

    for mode, k, warps_values in [("silu", 9216, (8, 16)), ("norm", 2560, (4, 8))]:
        output_path = args.output_prefix.with_name(
            args.output_prefix.name + "_" + mode + ".json"
        )
        sources = [
            Path("src/gleipnir/serving_fp4_fusion.py"),
            Path(__file__),
            Path("src/gleipnir/cudnn_fp4_gemm.py"),
            Path("src/gleipnir/cudnn_fp4_epilogue.py"),
        ]
        report = {
            "passed": False,
            "mode": mode,
            "checks": [],
            "relative_l2_limit": 0.01,
            "sources": {
                str(source): hashlib.sha256(source.read_bytes()).hexdigest()
                for source in sources
            },
            "gpu": torch.cuda.get_device_name(),
            "bf16_producer_boundary_preserved": True,
        }

        def save(path=output_path, receipt=report):
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(receipt, indent=2) + "\n")

        save()
        for source in sources:
            target = output_path.with_name(
                output_path.stem + "_sources"
            ) / source.resolve().relative_to(Path.cwd())
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
        try:
            weight = pack_operand(
                torch.randn(256, k, device="cuda", dtype=torch.bfloat16), weight=True
            )
            native = Nvfp4ScaledGemm(16384, k, 256, runtime_m=True)
            bn = decode_operand(weight).float() * weight.inverse
            norm_weight = torch.randn(k, device="cuda", dtype=torch.bfloat16) * 0.1
            for m in (1, 17, 129, 32768):
                x = torch.randn(
                    m,
                    k * (2 if mode == "silu" else 1),
                    device="cuda",
                    dtype=torch.bfloat16,
                )
                residual = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
                if m > 1:
                    x[0].zero_()
                    residual[0].zero_()
                    x[1].mul_(1000)
                    residual[1].mul_(1000)
                if mode == "silu":
                    plain = activated(x)
                    compiled = act_compiled(x)
                    expected = pack(compiled)

                    def baseline_fn(inputs=x):
                        return pack(act_compiled(inputs)), None
                else:
                    plain, _ = normalized(x, residual, norm_weight)
                    compiled, expected_residual = norm_compiled(
                        x, residual, norm_weight
                    )
                    expected = pack(compiled)

                    def baseline_fn(inputs=x, r=residual, w=norm_weight):
                        values, summed = norm_compiled(inputs, r, w)
                        return pack(values), summed

                reference = decode_operand(expected).float()
                old_ms, _, _ = bench(baseline_fn)
                for warps in warps_values:
                    if mode == "silu":

                        def fn(inputs=x, threads=warps):
                            return silu_pack(inputs, warps=threads), None
                    else:

                        def fn(inputs=x, r=residual, w=norm_weight, threads=warps):
                            return norm_pack(inputs, r, w, 1e-6, warps=threads)

                    observed, summed = fn()
                    decoded = decode_operand(observed).float()
                    relative = (
                        (decoded - reference).norm() / reference.norm().clamp_min(1e-30)
                    ).item()
                    finite = (
                        torch.isfinite(decoded).all().item()
                        and torch.isfinite(observed.inverse).all().item()
                    )
                    if mode == "norm":
                        assert torch.equal(summed, expected_residual), (
                            "residual rounding"
                        )
                    a_norm = decoded * observed.inverse[:, None]
                    raw = (a_norm @ bn.t()).to(torch.bfloat16).float()
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
                        assert torch.equal(decoded[1:], altered[1:]), (
                            "cross-row coupling"
                        )
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
                        "scales_equal": torch.equal(
                            expected.scales.view(torch.uint8),
                            observed.scales.view(torch.uint8),
                        ),
                        "eager_compiled_producer_equal": torch.equal(plain, compiled),
                        "baseline_graph_ms": old_ms,
                        "candidate_graph_ms": new_ms,
                        "throughput_ratio": old_ms / new_ms,
                        "isolation_passed": True,
                        "updated_input_graph_passed": True,
                    }
                    report["checks"].append(row)
                    save()
                    print(mode, json.dumps(row), flush=True)
                    assert finite and relative <= 0.01 and gemm_error <= 0.01, row
            candidates = [r for r in report["checks"] if r["m"] == 32768]
            report["selected_warps"] = min(
                candidates, key=lambda r: r["candidate_graph_ms"]
            )["warps"]
            report["passed"] = True
        except BaseException as error:
            report["error"] = f"{type(error).__name__}: {error}"
            print(mode, report["error"], flush=True)
        finally:
            save()


if __name__ == "__main__":
    main()
