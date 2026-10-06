"""Matched native GDN forward/state checks before changing a serving backend."""

import argparse
import hashlib
import importlib.metadata
import json
import time
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=["cutedsl", "flashqla"], required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    import torch
    from vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn import (
        fi_chunk_gated_delta_rule,
        l2norm_fwd,
    )

    from gleipnir.flashqla_training import tensor_comparison

    def reference_kernel(q, k, v, g, beta, state, final, cu):
        # vLLM's normalizer flattens with view(); preserve its contract before
        # comparing low-level kernels on separately strided QKV views.
        return fi_chunk_gated_delta_rule(
            q.contiguous(), k.contiguous(), v, g, beta, state, final, cu
        )

    receipt = {
        "passed": False,
        "backend": args.backend,
        "output_relative_l2_limit": 0.03,
        "state_relative_l2_limit": 0.03,
        "checks": [],
        "sources": {
            str(path.resolve()): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in (
                Path(__file__),
                Path("src/gleipnir/serving_gdn_kernels.py"),
                Path(fi_chunk_gated_delta_rule.__code__.co_filename),
            )
        },
        "gpu": torch.cuda.get_device_name(),
        "device_capability": list(torch.cuda.get_device_capability()),
        "software": {
            name: importlib.metadata.version(name)
            for name in (
                "torch",
                "vllm",
                "flashinfer-python",
                "triton",
                "nvidia-cutlass-dsl",
            )
        },
    }

    def save() -> None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(receipt, indent=2) + "\n")
        archive = args.output.parent / (args.output.stem + "_sources")
        for source, expected in receipt["sources"].items():
            path = Path(source)
            content = path.read_bytes()
            if hashlib.sha256(content).hexdigest() != expected:
                raise ValueError(f"canary source changed during execution: {source}")
            target = archive / path.relative_to(Path.cwd())
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(content)

    save()
    try:
        if args.backend == "flashqla":
            from flash_qla.ops.gated_delta_rule.chunk import chunk_gated_delta_rule_fwd

            from gleipnir.flashqla_training import load_flashqla
            from gleipnir.serving_gdn_kernels import make_flashqla_prefill

            _, receipt["runtime"] = load_flashqla()
            candidate = make_flashqla_prefill(
                chunk_gated_delta_rule_fwd, l2norm_fwd, auto_cp=False
            )
        else:
            from cutlass._mlir.dialects import nvvm

            from gleipnir.serving_gdn_kernels import install_nvvm_compatibility

            receipt["compatibility"] = install_nvvm_compatibility(nvvm)
            from vllm.model_executor.layers.mamba.ops.gdn_chunk_cutedsl import (
                chunk_gated_delta_rule_cutedsl,
                prepare_metadata_cutedsl,
            )

            directory = Path(chunk_gated_delta_rule_cutedsl.__code__.co_filename).parent
            receipt["sources"].update(
                {
                    str(path.resolve()): hashlib.sha256(path.read_bytes()).hexdigest()
                    for path in directory.glob("*.py")
                }
            )

            def candidate(q, k, v, g, beta, state, final, cu):
                indices, offsets = prepare_metadata_cutedsl(cu, q.shape[1])
                return chunk_gated_delta_rule_cutedsl(
                    l2norm_fwd(q.contiguous()),
                    l2norm_fwd(k.contiguous()),
                    v,
                    g,
                    beta,
                    state,
                    cu,
                    indices,
                    offsets,
                )

        generator = torch.Generator(device="cuda").manual_seed(0)
        for lengths in ([1], [17], [129], [1, 127, 513], [4096], [8192, 8192]):
            print(json.dumps({"case_start_lengths": lengths}), flush=True)
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
            if lengths == [129]:
                packed = torch.randn(
                    1,
                    total,
                    8192,
                    device="cuda",
                    dtype=torch.bfloat16,
                    generator=generator,
                )
                pq, pk, pv = packed.split([2048, 2048, 4096], dim=-1)
                q, k = pq.reshape(1, total, 16, 128), pk.reshape(1, total, 16, 128)
                v = pv.reshape(1, total, 32, 128)
                assert not v.is_contiguous()
            g = -torch.rand(1, total, 32, device="cuda", generator=generator) * 0.1
            beta = torch.rand(1, total, 32, device="cuda", generator=generator)
            state = (
                torch.randn(
                    len(lengths),
                    32,
                    128,
                    128,
                    device="cuda",
                    generator=generator,
                )
                * 0.01
            )
            cu = torch.tensor([0, *lengths], device="cuda", dtype=torch.int32).cumsum(0)
            cu = cu.to(torch.int32)
            inputs = q, k, v, g, beta, state, True, cu
            original_state = state.clone()
            reference = reference_kernel(*inputs)
            observed = candidate(*inputs)
            check = {
                "lengths": lengths,
                "input_strides": {
                    "q": list(q.stride()),
                    "k": list(k.stride()),
                    "v": list(v.stride()),
                },
            }
            for name, actual, expected in zip(
                ("output", "state"), observed, reference, strict=True
            ):
                check[name] = tensor_comparison(actual, expected)
                if not check[name]["finite"] or check[name]["relative_l2"] > 0.03:
                    receipt["checks"].append(check)
                    save()
                    raise ValueError(f"GDN {name} mismatch: {check[name]}")
            if not torch.equal(state, original_state):
                raise ValueError("GDN kernel mutated its initial state")
            if len(lengths) == 3:
                separate_outputs, separate_states = [], []
                start = 0
                for index, length in enumerate(lengths):
                    stop = start + length
                    isolated = candidate(
                        q[:, start:stop],
                        k[:, start:stop],
                        v[:, start:stop],
                        g[:, start:stop],
                        beta[:, start:stop],
                        original_state[index : index + 1],
                        True,
                        torch.tensor([0, length], device="cuda", dtype=torch.int32),
                    )
                    separate_outputs.append(isolated[0])
                    separate_states.append(isolated[1])
                    start = stop
                isolated = (
                    torch.cat(separate_outputs, dim=1),
                    torch.cat(separate_states),
                )
                for name, actual, expected in zip(
                    ("output", "state"), isolated, observed, strict=True
                ):
                    metric = tensor_comparison(actual, expected)
                    check["ragged_isolation_" + name] = metric
                    if not metric["finite"] or metric["relative_l2"] > 0.03:
                        raise ValueError(f"ragged isolation {name} mismatch: {metric}")
            if lengths == [17]:
                # Independent sequential FP32 oracle, with the same normalized
                # BF16 operands; catches log/exp, GVA and V/K-state mistakes.
                nq, nk = [
                    l2norm_fwd(x).float().cpu().repeat_interleave(2, dim=2)
                    for x in (q, k)
                ]
                nv, ng, nb = [x.float().cpu() for x in (v, g, beta)]
                h = original_state.float().cpu()[0]
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
                oracle = torch.stack(outputs).unsqueeze(0), h.unsqueeze(0)
                for name, actual, expected in zip(
                    ("output", "state"), observed, oracle, strict=True
                ):
                    metric = tensor_comparison(actual, expected)
                    check["sequential_oracle_" + name] = metric
                    if not metric["finite"] or metric["relative_l2"] > 0.03:
                        raise ValueError(f"sequential oracle {name} mismatch: {metric}")
                # A nonempty continued state must reproduce a single prefill.
                first = candidate(
                    q[:, :7],
                    k[:, :7],
                    v[:, :7],
                    g[:, :7],
                    beta[:, :7],
                    original_state,
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
                continued = torch.cat((first[0], second[0]), dim=1), second[1]
                for name, actual, expected in zip(
                    ("output", "state"), continued, observed, strict=True
                ):
                    metric = tensor_comparison(actual, expected)
                    check["continuation_" + name] = metric
                    if not metric["finite"] or metric["relative_l2"] > 0.03:
                        raise ValueError(f"continued-state {name} mismatch: {metric}")
            for name, kernel in [
                ("flashinfer", reference_kernel),
                (args.backend, candidate),
            ]:
                for _ in range(2):
                    kernel(*inputs)
                torch.cuda.synchronize()
                start = time.perf_counter()
                for _ in range(5):
                    kernel(*inputs)
                torch.cuda.synchronize()
                check[name + "_seconds"] = (time.perf_counter() - start) / 5
            receipt["checks"].append(check)
            save()
            print(json.dumps(check), flush=True)
        receipt["passed"] = True
        save()
    except BaseException as error:
        receipt["error"] = f"{type(error).__name__}: {error}"
        save()
        raise


if __name__ == "__main__":
    main()
