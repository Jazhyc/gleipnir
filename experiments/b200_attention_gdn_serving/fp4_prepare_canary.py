"""Bounded producer arithmetic, isolation, graph and timing screen."""

import argparse
import hashlib
import json
import shutil
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    import torch

    from gleipnir.cudnn_fp4_epilogue import Nvfp4ScaledGemm
    from gleipnir.cudnn_fp4_gemm import decode_operand, pack_operand
    from gleipnir.serving_fp4_prepare import vendor_pack

    torch.manual_seed(37)
    torch.backends.cuda.matmul.allow_tf32 = False
    sources = [
        Path("src/gleipnir/__init__.py"),
        Path("src/gleipnir/_compat.py"),
        Path("src/gleipnir/serving/fp4/prepare.py"),
        Path(__file__),
        Path("src/gleipnir/kernels/fp4/cudnn_fp4_gemm.py"),
        Path("src/gleipnir/kernels/fp4/cudnn_fp4_epilogue.py"),
    ]
    report = {
        "passed": False,
        "arithmetic_passed": False,
        "mode": "vendor_cuda",
        "checks": [],
        "relative_l2_limit": 0.01,
        "sources": {
            str(source): hashlib.sha256(source.read_bytes()).hexdigest()
            for source in sources
        },
        "gpu": torch.cuda.get_device_name(),
        "bitwise_agreement_is_separate": True,
    }

    def save():
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")

    archive = args.output.with_name(args.output.stem + "_sources")
    for source in sources:
        target = archive / source.resolve().relative_to(Path.cwd())
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)

    def baseline(x):
        return pack_operand(x, row_amax=True, chunked_rows=True, hardware_packing=True)

    def time_graph(fn, x):
        fn(x)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            value = fn(x)
        for _ in range(5):
            graph.replay()
        samples = []
        for _ in range(3):
            start, end = (
                torch.cuda.Event(enable_timing=True),
                torch.cuda.Event(enable_timing=True),
            )
            start.record()
            for _ in range(32):
                graph.replay()
            end.record()
            end.synchronize()
            samples.append(start.elapsed_time(end) / 32)
        return sorted(samples)[1], graph, value

    save()
    try:
        for k in (2560, 4096, 9216):
            weight = pack_operand(
                torch.randn(256, k, device="cuda", dtype=torch.bfloat16), weight=True
            )
            native = Nvfp4ScaledGemm(16384, k, 256, runtime_m=True)
            bn = decode_operand(weight).float() * weight.inverse
            for m in (1, 17, 129, 32768):
                x = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
                if m > 1:
                    x[0].zero_()
                    x[1].mul_(1000)
                    if m > 2:
                        x[2].mul_(1e-6)
                expected, observed = baseline(x), vendor_pack(x)
                ref = decode_operand(expected).float()
                got = decode_operand(observed).float()
                relative = ((ref - got).norm() / ref.norm().clamp_min(1e-30)).item()
                finite = (
                    torch.isfinite(got).all().item()
                    and torch.isfinite(observed.inverse).all().item()
                )
                an = got * observed.inverse[:, None]
                raw = (an @ bn.t()).to(torch.bfloat16).float()
                gemm_reference = (
                    (raw / (observed.inverse[:, None] * weight.inverse))
                    .to(torch.bfloat16)
                    .float()
                )
                gemm_observed = native(observed, weight).float()
                gemm_error = (
                    (gemm_observed - gemm_reference).norm()
                    / gemm_reference.norm().clamp_min(1e-30)
                ).item()
                old_ms, _, _ = time_graph(baseline, x)
                new_ms, graph, output = time_graph(vendor_pack, x)
                saved = got[1:].clone()
                if m > 1:
                    x[0].fill_(7)
                    changed = decode_operand(vendor_pack(x)).float()
                    assert torch.equal(saved, changed[1:]), "cross-row coupling"
                x.mul_(0.25)
                graph.replay()
                replay = decode_operand(output).float()
                fresh = decode_operand(vendor_pack(x)).float()
                assert torch.equal(replay, fresh), "changed-input graph replay"
                row = {
                    "m": m,
                    "k": k,
                    "relative_l2": relative,
                    "finite": finite,
                    "gemm_relative_l2": gemm_error,
                    "precision_passed": relative <= report["relative_l2_limit"],
                    "codes_equal": torch.equal(
                        expected.codes.view(torch.uint8),
                        observed.codes.view(torch.uint8),
                    ),
                    "scales_equal": torch.equal(
                        expected.scales.view(torch.uint8),
                        observed.scales.view(torch.uint8),
                    ),
                    "inverse_equal": torch.equal(expected.inverse, observed.inverse),
                    "baseline_graph_ms": old_ms,
                    "candidate_graph_ms": new_ms,
                    "throughput_ratio": old_ms / new_ms,
                    "isolation_passed": True,
                    "updated_input_graph_passed": True,
                }
                report["checks"].append(row)
                save()
                print(json.dumps(row), flush=True)
                assert (
                    finite
                    and torch.isfinite(gemm_observed).all()
                    and gemm_error <= 0.01
                ), row
                del x, expected, observed, ref, got, graph, output
        report["arithmetic_passed"] = True
        report["passed"] = all(r["precision_passed"] for r in report["checks"])
        report["diagnostic_only"] = not report["passed"]
    except BaseException as error:
        report["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        save()


if __name__ == "__main__":
    main()
