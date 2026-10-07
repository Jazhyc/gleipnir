"""Check exact direct-output/state parity and changed-input graph replay."""

import argparse
import hashlib
import importlib.metadata
import json
import time
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    import torch
    from flashinfer.gdn_prefill import chunk_gated_delta_rule
    from vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn import (
        fi_chunk_gated_delta_rule,
        l2norm_fwd,
    )

    from gleipnir.serving_gdn_direct_output import make_forward, validate_native

    started = time.time()
    sources = [
        Path("src/gleipnir/__init__.py"),
        Path("src/gleipnir/_compat.py"),
        Path(__file__),
        Path("src/gleipnir/serving/gdn/direct_output.py"),
        Path(fi_chunk_gated_delta_rule.__code__.co_filename),
        Path(".venv/lib/python3.12/site-packages/flashinfer/gdn_prefill.py"),
        Path(
            ".venv/lib/python3.12/site-packages/flashinfer/gdn_kernels/blackwell/gdn_prefill.py"
        ),
        Path(
            ".venv/lib/python3.12/site-packages/flashinfer/gdn_kernels/blackwell/gated_delta_net_chunked.py"
        ),
    ]
    receipt = {
        "passed": False,
        "intervention": "flashinfer_gdn_direct_output",
        "gpu": torch.cuda.get_device_name(),
        "started_at_unix": started,
        "sources": {
            str(p.relative_to(Path.cwd()))
            if p.is_absolute()
            else str(p): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sources
        },
        "software": {
            name: importlib.metadata.version(name)
            for name in ["torch", "vllm", "flashinfer-python", "nvidia-cutlass-dsl"]
        },
        "checks": [],
        "graph_replay_passed": False,
        "isolation_passed": False,
        "zero_passed": False,
        "arithmetic_changed": False,
        "comparison": "bitwise output and FP32 final-state equality",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    archive = args.output.parent / f"{args.output.stem}_sources"
    for source in receipt["sources"]:
        target = archive / source
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(Path(source).read_bytes())

    def save():
        args.output.write_text(json.dumps(receipt, indent=2) + "\n")

    def same(actual, expected):
        return bool(torch.isfinite(actual).all()) and torch.equal(actual, expected)

    candidate = make_forward(chunk_gated_delta_rule, l2norm_fwd)
    generator = torch.Generator(device="cuda").manual_seed(0)
    save()
    try:
        for lengths in ([1], [17], [129], [1, 127, 513], [4096], [8192, 8192], [32768]):
            total = sum(lengths)
            print("case_start", lengths, flush=True)
            q, k = [
                torch.randn(
                    1,
                    total,
                    16,
                    128,
                    device="cuda",
                    dtype=torch.bfloat16,
                    generator=generator,
                )
                for _ in range(2)
            ]
            v = torch.randn(
                1,
                total,
                32,
                128,
                device="cuda",
                dtype=torch.bfloat16,
                generator=generator,
            )
            if total == 129:
                packed = torch.randn(
                    1,
                    total,
                    8192,
                    device="cuda",
                    dtype=torch.bfloat16,
                    generator=generator,
                )
                v = packed[..., 4096:].view(1, total, 32, 128)
                assert not v.is_contiguous()
            g = -torch.rand(1, total, 32, device="cuda", generator=generator) * 0.1
            beta = torch.rand(1, total, 32, device="cuda", generator=generator)
            state = (
                torch.randn(
                    len(lengths), 32, 128, 128, device="cuda", generator=generator
                )
                * 0.01
            )
            before = state.clone()
            cu = (
                torch.tensor([0, *lengths], device="cuda", dtype=torch.int32)
                .cumsum(0)
                .int()
            )
            inputs = q, k, v, g, beta, state, True, cu
            reference = fi_chunk_gated_delta_rule(*inputs)
            backing = torch.full(
                (total + 8, 32, 128), -123.0, device="cuda", dtype=torch.bfloat16
            )
            destination = backing[1:]
            actual = candidate(*inputs, core_attn_out=destination)
            check = {
                "lengths": lengths,
                "output_exact": same(actual[0], reference[0]),
                "state_exact": same(actual[1], reference[1]),
                "destination_alias": actual[0].data_ptr() == destination.data_ptr(),
                "prefix_tail_untouched": bool(
                    (backing[:1] == -123).all() and (backing[total + 1 :] == -123).all()
                ),
                "initial_state_unchanged": torch.equal(state, before),
            }
            if total == 17:
                for final in (False, True):
                    expected = fi_chunk_gated_delta_rule(
                        q, k, v, g, beta, state, final, cu, False
                    )
                    value = candidate(
                        q,
                        k,
                        v,
                        g,
                        beta,
                        state,
                        final,
                        cu,
                        use_qk_l2norm_in_kernel=False,
                        core_attn_out=destination,
                    )
                    check[f"normalization_off_final_{final}"] = same(
                        value[0], expected[0]
                    ) and (same(value[1], expected[1]) if final else value[1] is None)
                allocated = candidate(*inputs)
                check["optional_destination"] = same(
                    allocated[0], reference[0]
                ) and same(allocated[1], reference[1])
                for operand in (q, k, v, state):
                    operand.zero_()
                zeros = candidate(*inputs, core_attn_out=destination)
                check["zero"] = bool((zeros[0] == 0).all() and (zeros[1] == 0).all())
                receipt["zero_passed"] = check["zero"]
            if len(lengths) == 3:
                outputs, states = [], []
                start = 0
                for index, length in enumerate(lengths):
                    stop = start + length
                    value = candidate(
                        q[:, start:stop].contiguous(),
                        k[:, start:stop].contiguous(),
                        v[:, start:stop],
                        g[:, start:stop],
                        beta[:, start:stop],
                        state[index : index + 1],
                        True,
                        torch.tensor([0, length], device="cuda", dtype=torch.int32),
                        core_attn_out=torch.empty(
                            length, 32, 128, device="cuda", dtype=torch.bfloat16
                        ),
                    )
                    outputs.append(value[0])
                    states.append(value[1])
                    start = stop
                check["isolation"] = same(
                    torch.cat(outputs, dim=1), actual[0]
                ) and same(torch.cat(states), actual[1])
                receipt["isolation_passed"] = check["isolation"]
            if total in (129, 32768):
                for _ in range(3):
                    candidate(*inputs, core_attn_out=destination)
                torch.cuda.synchronize()
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    captured = candidate(*inputs, core_attn_out=destination)
                q.mul_(0.5)
                expected = fi_chunk_gated_delta_rule(*inputs)
                graph.replay()
                torch.cuda.synchronize()
                check["changed_input_graph"] = same(captured[0], expected[0]) and same(
                    captured[1], expected[1]
                )
                check["graph_tail_untouched"] = bool(
                    (backing[total + 1 :] == -123).all()
                )
                receipt["graph_replay_passed"] = (
                    check["changed_input_graph"]
                    and check["graph_tail_untouched"]
                    and (receipt["graph_replay_passed"] if total == 32768 else True)
                )
            check["passed"] = all(v is True for k, v in check.items() if k != "lengths")
            receipt["checks"].append(check)
            save()
            print("case_complete", json.dumps(check), flush=True)
            if not check["passed"]:
                raise ValueError("direct GDN output native check failed")
        receipt.update(passed=True, elapsed_seconds=time.time() - started)
        validate_native(receipt)
        save()
        print("direct_gdn_native_passed", flush=True)
    except BaseException as error:
        receipt.update(
            passed=False,
            error=f"{type(error).__name__}: {error}",
            elapsed_seconds=time.time() - started,
        )
        save()
        raise


if __name__ == "__main__":
    main()
