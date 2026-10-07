"""Exact host-binding parity on unchanged FROST plans, outputs and streams."""

import argparse
import hashlib
import json
import statistics
import time
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    import torch
    from cudnn.gemm.frost import compiler

    from gleipnir.cudnn_fp4_epilogue import Nvfp4ScaledGemm
    from gleipnir.cudnn_fp4_gemm import pack_operand
    from gleipnir.serving_attention_fp4 import SHAPES as ATTENTION
    from gleipnir.serving_fp4_tuning import SHAPES, retile
    from gleipnir.serving_frost_wrappers import COMPILER_SHA, DirectBindings

    root = Path.cwd()
    report = {
        "passed": False,
        "checks": [],
        "gpu": torch.cuda.get_device_name(),
        "helper_sha256": hashlib.sha256(
            (root / "src/gleipnir/serving/frost_wrappers.py").read_bytes()
        ).hexdigest(),
        "compiler_sha256": hashlib.sha256(
            Path(compiler.__file__).read_bytes()
        ).hexdigest(),
        "arithmetic_changed": False,
        "timing_scope": "synchronized native calls, excludes operand packing",
    }
    assert report["compiler_sha256"] == COMPILER_SHA
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def save():
        args.output.write_text(json.dumps(report, indent=2) + "\n")

    save()
    selected = json.loads(
        Path("results/b200_attention_gdn_serving/fp4_gemm_tune01.json").read_text()
    )["selected"]
    attention = json.loads(
        Path(
            "results/b200_attention_gdn_serving/attention_fp4_canary02.json"
        ).read_text()
    )["selected"]
    specs = [(name, dims, selected[name]) for name, dims in SHAPES.items()]
    # Output uses the already tuned shared GDN-output geometry; only QKV is new.
    specs.append(("attention_qkv", ATTENTION["qkv_proj"], attention["qkv_proj"]))
    torch.manual_seed(41)

    def timing(fn):
        samples = []
        for _ in range(3):
            torch.cuda.synchronize()
            start = time.perf_counter()
            for _ in range(16):
                fn()
            torch.cuda.synchronize()
            samples.append((time.perf_counter() - start) * 1000 / 16)
        return statistics.median(samples)

    try:
        for name, (k, n), tiles in specs:
            weight = pack_operand(
                torch.randn(n, k, device="cuda", dtype=torch.bfloat16), weight=True
            )
            plans = {
                tile: retile(Nvfp4ScaledGemm(16384, k, n, runtime_m=True), tile)
                for tile in set(tiles.values())
            }
            wrappers = {tile: DirectBindings(plan) for tile, plan in plans.items()}
            for m in [1, 129, 4096, 32768]:
                print("wrapper_check", name, m, flush=True)
                tile = tiles["small" if m <= 4096 else "large"]
                plan, direct = plans[tile], wrappers[tile]
                x = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
                x[0].zero_()
                a = pack_operand(
                    x, row_amax=True, chunked_rows=True, hardware_packing=True
                )
                ref, out = plan(a, weight), direct(a, weight)
                exact = torch.equal(ref, out)
                finite = bool(torch.isfinite(out).all())
                saved = out.clone()
                changed = pack_operand(
                    x + 1, row_amax=True, chunked_rows=True, hardware_packing=True
                )
                changed_out = direct(changed, weight)
                independent = out.data_ptr() != changed_out.data_ptr() and torch.equal(
                    saved, out
                )
                changed_exact = torch.equal(changed_out, plan(changed, weight))
                old_ms, new_ms = (
                    (
                        timing(lambda plan=plan, a=a, weight=weight: plan(a, weight)),
                        timing(
                            lambda direct=direct, a=a, weight=weight: direct(a, weight)
                        ),
                    )
                    if m <= 129
                    else (None, None)
                )
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    captured = direct(a, weight)
                a.codes.copy_(changed.codes)
                a.scales.copy_(changed.scales)
                a.inverse.copy_(changed.inverse)
                graph.replay()
                replay_exact = torch.equal(captured, plan(a, weight))
                # Cached frozen views must observe writes to the same weight storage.
                weight.scales.copy_(
                    (weight.scales.float() * 0.5).to(weight.scales.dtype)
                )
                graph.replay()
                weight_replay_exact = torch.equal(captured, plan(a, weight))
                stream = torch.cuda.Stream()
                with torch.cuda.stream(stream):
                    alternate = direct(a, weight)
                torch.cuda.current_stream().wait_stream(stream)
                stream_exact = torch.equal(alternate, plan(a, weight))
                check = dict(
                    shape=name,
                    rows=m,
                    tile=tile,
                    bitwise_equal=exact,
                    finite=finite,
                    independent_outputs=independent,
                    changed_input_exact=changed_exact,
                    graph_replay_exact=replay_exact,
                    changed_weight_replay_exact=weight_replay_exact,
                    alternate_stream_exact=stream_exact,
                    original_ms=old_ms,
                    direct_ms=new_ms,
                )
                report["checks"].append(check)
                save()
                if not all(
                    [
                        exact,
                        finite,
                        independent,
                        changed_exact,
                        replay_exact,
                        weight_replay_exact,
                        stream_exact,
                    ]
                ):
                    raise ValueError("host wrapper native parity failed")
                del (
                    graph,
                    captured,
                    alternate,
                    out,
                    ref,
                    saved,
                    changed_out,
                    changed,
                    a,
                    x,
                )
            del wrappers, plans, weight
        report["passed"] = True
        save()
    except BaseException as error:
        report["error"] = f"{type(error).__name__}: {error}"
        save()
        raise


if __name__ == "__main__":
    main()
