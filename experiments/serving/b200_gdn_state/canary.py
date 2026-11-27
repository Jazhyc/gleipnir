"""Bounded native BF16 state admission and independent gate-rounding probe."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import inspect
import json
import time
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    import torch
    from flashinfer.gdn_kernels.blackwell import gdn_prefill
    from flashinfer.gdn_kernels.blackwell.gated_delta_net_chunked import (
        GatedDeltaNetChunkedKernel,
    )
    from flashinfer.gdn_prefill import chunk_gated_delta_rule
    from vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn import (
        fi_chunk_gated_delta_rule,
        l2norm_fwd,
    )

    from gleipnir.serving.gdn.state import make_forward, validate_native

    if torch.cuda.get_device_capability() != (10, 0):
        raise ValueError("this screen requires the existing SM100 B200")
    paths = [
        Path(__file__).resolve(),
        Path(inspect.getfile(make_forward)),
        Path(inspect.getfile(fi_chunk_gated_delta_rule)),
        Path(inspect.getfile(chunk_gated_delta_rule)),
        Path(inspect.getfile(gdn_prefill)),
        Path(inspect.getfile(GatedDeltaNetChunkedKernel)),
    ]
    root = Path.cwd()
    sources = {
        str(p.relative_to(root)) if p.is_relative_to(root) else str(p): hashlib.sha256(
            p.read_bytes()
        ).hexdigest()
        for p in paths
    }
    receipt = {
        "passed": False,
        "intervention": "flashinfer_gdn_bf16_state",
        "state_dtype": "bfloat16",
        "gate_dtype": "float32",
        "accumulation_dtype": "float32",
        "relative_l2_limit": 0.03,
        "gpu": torch.cuda.get_device_name(),
        "device_capability": [10, 0],
        "sources": sources,
        "software": {
            name: importlib.metadata.version(name)
            for name in ("torch", "vllm", "flashinfer-python", "nvidia-cutlass-dsl")
        },
        "checks": [],
        "gate_rounding_checks": [],
        "oracle_passed": False,
        "continuation_passed": False,
        "isolation_passed": False,
        "graph_replay_passed": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    archive = args.output.parent / "native_sources"
    for index, path in enumerate(paths):
        target = archive / f"{index}_{path.name}"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(path.read_bytes())

    def save() -> None:
        args.output.write_text(json.dumps(receipt, indent=2) + "\n")

    def comparison(actual, reference) -> dict:
        a, b = actual.float(), reference.float()
        return {
            "finite": bool(torch.isfinite(a).all()),
            "relative_l2": float((a - b).norm() / b.norm().clamp_min(1e-12)),
            "max_absolute": float((a - b).abs().max()),
        }

    def require(actual, reference, name, check) -> None:
        value = comparison(actual, reference)
        check[name] = value
        if not value["finite"] or value["relative_l2"] > 0.03:
            raise ValueError(f"BF16 GDN native {name} failed: {value}")

    candidate = make_forward(chunk_gated_delta_rule, l2norm_fwd)
    gate_probe = make_forward(chunk_gated_delta_rule, l2norm_fwd, round_gates=True)
    generator = torch.Generator(device="cuda").manual_seed(0)
    started = time.perf_counter()
    save()
    try:
        with torch.inference_mode():
            for lengths in (
                [1],
                [17],
                [129],
                [1, 127, 513],
                [4096],
                [8192, 8192],
                [32768],
            ):
                print("native_case_start", lengths, flush=True)
                total = sum(lengths)
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
                    pq, pk, pv = packed.split([2048, 2048, 4096], dim=-1)
                    q, k, v = (
                        pq.reshape(1, total, 16, 128),
                        pk.reshape(1, total, 16, 128),
                        pv.reshape(1, total, 32, 128),
                    )
                g = -torch.rand(1, total, 32, device="cuda", generator=generator) * 0.1
                beta = torch.rand(1, total, 32, device="cuda", generator=generator)
                state = (
                    torch.randn(
                        len(lengths), 32, 128, 128, device="cuda", generator=generator
                    )
                    * 0.01
                )
                cu = (
                    torch.tensor([0, *lengths], device="cuda", dtype=torch.int32)
                    .cumsum(0)
                    .int()
                )
                original_state = state.clone()
                inputs = q, k, v, g, beta, state, True, cu
                reference = fi_chunk_gated_delta_rule(
                    q.contiguous(), k.contiguous(), v, g, beta, state, True, cu
                )
                actual = candidate(*inputs)
                check = {"lengths": lengths, "passed": False}
                receipt["checks"].append(check)
                require(actual[0], reference[0], "output", check)
                require(actual[1], reference[1], "state", check)
                if actual[1].dtype != torch.bfloat16 or not torch.equal(
                    state, original_state
                ):
                    raise ValueError("BF16 state dtype or input isolation failed")
                check["state_input_unchanged"] = True
                check["output_state_dtype"] = str(actual[1].dtype)
                rounded = gate_probe(*inputs)
                receipt["gate_rounding_checks"].append(
                    {
                        "lengths": lengths,
                        "output": comparison(rounded[0], reference[0]),
                        "state": comparison(rounded[1], reference[1]),
                        "relative_to_bf16_state_output": comparison(
                            rounded[0], actual[0]
                        ),
                        "numerical_probe_only": True,
                        "native_gate_boundary_dtype": "float32",
                    }
                )
                if len(lengths) == 3:
                    outputs, states = [], []
                    begin = 0
                    for i, length in enumerate(lengths):
                        end = begin + length
                        o, h = candidate(
                            q[:, begin:end],
                            k[:, begin:end],
                            v[:, begin:end],
                            g[:, begin:end],
                            beta[:, begin:end],
                            state[i : i + 1],
                            True,
                            torch.tensor([0, length], device="cuda", dtype=torch.int32),
                        )
                        outputs.append(o)
                        states.append(h)
                        begin = end
                    require(torch.cat(outputs, 1), actual[0], "isolation_output", check)
                    require(torch.cat(states), actual[1], "isolation_state", check)
                    receipt["isolation_passed"] = True
                if lengths == [17]:
                    nq, nk = [
                        l2norm_fwd(x.contiguous())
                        .float()
                        .cpu()
                        .repeat_interleave(2, dim=2)
                        for x in (q, k)
                    ]
                    nv, ng, nb = [x.float().cpu() for x in (v, g, beta)]
                    h = state.float().cpu()[0]
                    outputs = []
                    for t in range(17):
                        h = h * ng[0, t].exp()[:, None, None]
                        residual = nv[0, t] - (h * nk[0, t, :, None, :]).sum(-1)
                        h += (
                            residual[:, :, None]
                            * nk[0, t, :, None, :]
                            * nb[0, t, :, None, None]
                        )
                        outputs.append((h * nq[0, t, :, None, :]).sum(-1) / 128**0.5)
                    require(
                        actual[0].cpu(),
                        torch.stack(outputs).unsqueeze(0),
                        "oracle_output",
                        check,
                    )
                    require(actual[1].cpu(), h.unsqueeze(0), "oracle_state", check)
                    receipt["oracle_passed"] = True
                    first = candidate(
                        q[:, :7],
                        k[:, :7],
                        v[:, :7],
                        g[:, :7],
                        beta[:, :7],
                        state,
                        True,
                        torch.tensor([0, 7], device="cuda", dtype=torch.int32),
                    )
                    second = candidate(
                        q[:, 7:],
                        k[:, 7:],
                        v[:, 7:],
                        g[:, 7:],
                        beta[:, 7:],
                        first[1],
                        True,
                        torch.tensor([0, 10], device="cuda", dtype=torch.int32),
                    )
                    require(
                        torch.cat((first[0], second[0]), 1),
                        actual[0],
                        "continuation_output",
                        check,
                    )
                    require(second[1], actual[1], "continuation_state", check)
                    receipt["continuation_passed"] = True
                    without_final = candidate(q, k, v, g, beta, state, False, cu)
                    require(without_final[0], actual[0], "no_final_state_output", check)
                    if without_final[1] is not None:
                        raise ValueError("unexpected final state when disabled")
                    nq, nk = [l2norm_fwd(x) for x in (q, k)]
                    unnormalized = candidate(nq, nk, v, g, beta, state, True, cu, False)
                    require(unnormalized[0], actual[0], "normalization_disabled", check)
                if total in (129, 32768):
                    bfstate = state.bfloat16()
                    graph_inputs = q, k, v, g, beta, bfstate, True, cu
                    for _ in range(3):
                        candidate(*graph_inputs)
                    torch.cuda.synchronize()
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph):
                        captured = candidate(*graph_inputs)
                    q.mul_(0.5)
                    bfstate.mul_(0.5)
                    expected = candidate(*graph_inputs)
                    graph.replay()
                    torch.cuda.synchronize()
                    for a, b in zip(captured, expected, strict=True):
                        if not torch.equal(a, b):
                            raise ValueError(
                                "changed-input state graph replay mismatch"
                            )
                    check["changed_input_graph_exact"] = True
                    receipt["graph_replay_passed"] = True
                for name, kernel in (
                    ("fp32", fi_chunk_gated_delta_rule),
                    ("bf16", candidate),
                ):
                    timed = (
                        q.contiguous(),
                        k.contiguous(),
                        v,
                        g,
                        beta,
                        state if name == "fp32" else state.bfloat16(),
                        True,
                        cu,
                    )
                    for _ in range(3):
                        kernel(*timed)
                    torch.cuda.synchronize()
                    before = time.perf_counter()
                    for _ in range(10):
                        kernel(*timed)
                    torch.cuda.synchronize()
                    check[name + "_host_inclusive_seconds"] = (
                        time.perf_counter() - before
                    ) / 10
                check["passed"] = True
                save()
                print("native_case_complete", json.dumps(check), flush=True)
        receipt.update(passed=True, elapsed_seconds=time.perf_counter() - started)
        validate_native(receipt)
        save()
        print("bf16_state_native_passed", flush=True)
    except BaseException as error:
        receipt.update(passed=False, error=f"{type(error).__name__}: {error}")
        save()
        raise


if __name__ == "__main__":
    main()
