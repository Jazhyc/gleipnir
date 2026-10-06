"""Persistent matched producer benchmark for native NVFP4 GEMM/SwiGLU fusion."""

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

    parser.add_argument(
        "--rows",
        nargs="+",
        type=int,
        default=[1536, 1537, 32768, 1, 17, 129, 2304, 4096, 29184],
    )
    args = parser.parse_args()
    if len(set(args.rows)) != len(args.rows) or any(
        not 1 <= m <= 32768 for m in args.rows
    ):
        parser.error("rows must be unique and within 1..32768")
    if args.output.exists():
        raise FileExistsError(args.output)
    import torch
    import triton

    torch.backends.cuda.matmul.allow_tf32 = False

    from gleipnir.cudnn_fp4_epilogue import Nvfp4ScaledGemm
    from gleipnir.cudnn_fp4_gemm import decode_operand, pack_operand
    from gleipnir.serving_fp4_fusion import silu_pack
    from gleipnir.serving_fp4_swiglu import prepare_weight
    from gleipnir.serving_fp4_swiglu_block_reference import reference_decode
    from gleipnir.serving_fp4_swiglu_native_output import (
        NativeOutputSwiGlu,
        load_kernel,
    )

    sources = [
        Path(__file__).relative_to(Path.cwd()),
        Path("src/gleipnir/serving_fp4_swiglu.py"),
        Path("src/gleipnir/serving_fp4_swiglu_overhead.py"),
        Path("src/gleipnir/serving_fp4_swiglu_native_output.py"),
        Path("src/gleipnir/serving_fp4_swiglu_block_reference.py"),
        Path("src/gleipnir/serving_fp4_swiglu_padding.py"),
        Path("src/gleipnir/serving_fp4_swiglu_pack.py"),
        Path("src/gleipnir/cudnn_fp4_epilogue.py"),
        Path("src/gleipnir/cudnn_fp4_gemm.py"),
        Path("src/gleipnir/serving_fp4_fusion.py"),
    ]
    archive = args.output.with_suffix("").with_name(args.output.stem + "_sources")
    for p in sources:
        dest = archive / p
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(p, dest)
    kernel, hashes = load_kernel(Path.cwd(), archive)
    report = dict(
        state="starting",
        passed=False,
        started_at_unix=time.time(),
        gpu=torch.cuda.get_device_name(),
        runtime={"torch": torch.__version__, "triton": triton.__version__},
        sources={str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
        kernel=hashes,
        relative_l2_limit=0.01,
        selection_minimum_gain=0.02,
        timing_scope=("gate/up, row descale, SiLU, NVFP4 packing; CUDA graph replay"),
        results=[],
    )

    def save():
        args.output.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.output.with_suffix(".tmp")
        temporary.write_text(json.dumps(report, indent=2) + "\n")
        temporary.replace(args.output)

    def compare(out, ref):
        out, ref = out.float(), ref.float()
        finite = bool(torch.isfinite(out).all())
        relative = float((out - ref).norm() / ref.norm().clamp_min(1e-12))
        return dict(
            finite=finite, relative_l2=relative, passed=finite and relative <= 0.01
        )

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
        return statistics.median(samples), samples, graph, captured

    save()
    try:
        gen = torch.Generator(device="cuda").manual_seed(173)
        ref = Nvfp4ScaledGemm(16384, 2560, 18432, runtime_m=True)
        down = Nvfp4ScaledGemm(16384, 9216, 2560, runtime_m=True)
        weight = pack_operand(
            torch.randn(18432, 2560, device="cuda", dtype=torch.bfloat16, generator=gen)
            * 0.02,
            weight=True,
        )
        down_weight = pack_operand(
            torch.randn(2560, 9216, device="cuda", dtype=torch.bfloat16, generator=gen)
            * 0.02,
            weight=True,
        )
        interleaved = prepare_weight(weight)
        decoded = decode_operand(interleaved)
        from gleipnir.serving_fp4_swiglu import interleaved_rows

        report["weight_permutation_exact"] = bool(
            torch.equal(decoded, decode_operand(weight)[interleaved_rows(18432)])
        )
        del decoded
        candidates = {"native_block_fp4_output": NativeOutputSwiGlu(kernel)}
        report["scaling"] = "unit_global_inverse_local_16_element_e4m3"
        report["n192_scale_fix"] = True
        report["candidates"] = list(candidates)
        for m in args.rows:
            print("row_start", m, flush=True)
            x = torch.randn(m, 2560, device="cuda", dtype=torch.bfloat16, generator=gen)
            x[0].zero_()
            if m > 1:
                x[1].mul_(20)
            a = pack_operand(x, row_amax=True, chunked_rows=True, hardware_packing=True)
            gateup = ref(a, weight)
            gate, up = gateup.float().chunk(2, dim=1)
            activated = (gate / (1 + torch.exp(-gate)) * up).bfloat16()
            packed_ref = silu_pack(gateup, warps=8)
            output_ref = down(packed_ref, down_weight)
            baseline_ms, samples, g, packed = timing(
                lambda aa=a: silu_pack(ref(aa, weight), warps=8)
            )
            del g, packed
            for name, plan in candidates.items():
                record = dict(
                    m=m, tile=name, baseline_ms=baseline_ms, baseline_samples_ms=samples
                )
                print("measure", m, name, flush=True)
                try:
                    candidate = plan(a, interleaved)
                    decoded = decode_operand(candidate)
                    packed_expected = reference_decode(activated)
                    record["packing_reference"] = compare(
                        decoded, packed_expected
                    )
                    out = down(candidate, down_weight)
                    expected = (
                        decoded.float() @ decode_operand(down_weight)[:16].float().T
                    )
                    record["arithmetic"] = compare(out[:, :16], expected)
                    record["baseline_precision"] = compare(out, output_ref)
                    record["activation_precision"] = compare(decoded, activated)
                    record["zero_row_exact"] = bool((out[0] == 0).all())
                    record.update(
                        passed=record["packing_reference"]["passed"]
                        and record["arithmetic"]["passed"]
                        and record["zero_row_exact"],
                        finite=bool(torch.isfinite(out).all()),
                    )
                    if not record["finite"]:
                        raise FloatingPointError("nonfinite native FP4 output")
                    ms, ss, graph, captured = timing(
                        lambda p=plan, aa=a: p(aa, interleaved)
                    )
                    record.update(
                        milliseconds=ms, samples_ms=ss, speed_ratio=baseline_ms / ms
                    )
                    bm, bss, bg, bc = timing(
                        lambda aa=a: down(
                            silu_pack(ref(aa, weight), warps=8), down_weight
                        )
                    )
                    cm, css, cg, cc = timing(
                        lambda p=plan, aa=a: down(p(aa, interleaved), down_weight)
                    )
                    record.update(
                        baseline_full_mlp_ms=bm,
                        baseline_full_mlp_samples_ms=bss,
                        full_mlp_ms=cm,
                        full_mlp_samples_ms=css,
                        full_mlp_speed_ratio=bm / cm,
                    )
                    del bg, bc, cg, cc
                    del decoded, expected, packed_expected
                    original = down(captured, down_weight).clone()
                    changed = x.clone()
                    changed[0].fill_(0.5)
                    changed_a = pack_operand(
                        changed, row_amax=True, chunked_rows=True, hardware_packing=True
                    )
                    a.codes.view(torch.uint8).copy_(changed_a.codes.view(torch.uint8))
                    a.scales.copy_(changed_a.scales)
                    a.inverse.copy_(changed_a.inverse)
                    graph.replay()
                    torch.cuda.synchronize()
                    changed_out = down(captured, down_weight)
                    decoded_replay = decode_operand(captured)
                    record["changed_replay"] = compare(
                        changed_out[:, :16],
                        decoded_replay.float()
                        @ decode_operand(down_weight)[:16].float().T,
                    )
                    fresh_gateup = ref(a, weight).float()
                    fresh_gate, fresh_up = fresh_gateup.chunk(2, dim=1)
                    fresh_activation = (
                        fresh_gate / (1 + torch.exp(-fresh_gate)) * fresh_up
                    ).bfloat16()
                    record["packing_replay"] = compare(
                        decoded_replay,
                        reference_decode(fresh_activation),
                    )
                    del (
                        decoded_replay,
                        fresh_gateup,
                        fresh_gate,
                        fresh_up,
                        fresh_activation,
                    )
                    record["unchanged_rows_exact"] = bool(
                        torch.equal(original[1:], changed_out[1:])
                    )
                    record["passed"] &= (
                        record["changed_replay"]["passed"]
                        and record["packing_replay"]["passed"]
                        and record["unchanged_rows_exact"]
                    )
                    del (
                        graph,
                        captured,
                        original,
                        changed_out,
                        changed,
                        changed_a,
                        candidate,
                        out,
                    )
                    a = pack_operand(
                        x, row_amax=True, chunked_rows=True, hardware_packing=True
                    )
                except Exception as exc:
                    record.update(
                        passed=False,
                        execution_failed=repr(exc),
                        traceback=traceback.format_exc(),
                    )
                    print(record["traceback"], flush=True)
                    if isinstance(
                        exc, (torch.AcceleratorError, FloatingPointError)
                    ) or "illegal instruction" in str(exc):
                        report["results"].append(record)
                        save()
                        raise
                report["results"].append(record)
                save()
            del x, a, gateup, gate, up, activated, packed_ref, output_ref
            gc.collect()
            torch.cuda.empty_cache()
            if not any(r["passed"] for r in report["results"]):
                print("stop_no_executable_candidate", flush=True)
                break
        report["validated_candidates"] = [
            name
            for name in candidates
            if {r["m"] for r in report["results"] if r["tile"] == name and r["passed"]}
            == set(args.rows)
        ]
        report["full_envelope"] = set(args.rows) == {
            1,
            17,
            129,
            1536,
            2304,
            4096,
            29184,
            32768,
            1537,
        }
        report["passed"] = bool(
            report["weight_permutation_exact"]
            and report["validated_candidates"]
            and report["full_envelope"]
        )
        report["compile_count"] = next(iter(candidates.values())).compile_count
        report["strict_baseline_precision_passed"] = bool(
            report["passed"]
            and all(
                r.get("baseline_precision", {}).get("passed") for r in report["results"]
            )
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
