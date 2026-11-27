"""Native hardware-exp2 MXFP8 backward versus the current FA4 surrogate."""

import json
import time
from itertools import accumulate
from pathlib import Path

import torch

from gleipnir.data.monitoring import file_hash, write_json
from gleipnir.training.bf16_lora import configure_bf16_reductions
from gleipnir.training.mxfp8_backward import native_attention
from gleipnir.training.quantization_recovery import attention_runtime


def main():
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

    from gleipnir.training.backends.native_fp4 import verify_native_fp4_runtime

    root = Path.cwd()
    output = root / "results/native-fp8-qat01/mxfp8_backward"
    output.mkdir(parents=True, exist_ok=True)
    if (output / "result.json").exists():
        raise ValueError("preserve previous native backward probe")
    configure_bf16_reductions(allow_reduced_precision=False, allow_split_k=False)
    runtime = verify_native_fp4_runtime()
    torch.manual_seed(31)

    def reference(q, k, v, cuts, maximum):
        with attention_runtime():
            fn = ALL_ATTENTION_FUNCTIONS["sdpa"]
            result, _ = fn(
                None,
                q.transpose(0, 1)[None],
                k.transpose(0, 1)[None],
                v.transpose(0, 1)[None],
                None,
                cu_seq_lens_q=cuts,
                cu_seq_lens_k=cuts,
                max_length_q=maximum,
                max_length_k=maximum,
                dropout=0,
            )
            return result[0]

    result = {
        "status": "running",
        "runtime": runtime,
        "cases": [],
        "sources": {
            "src/gleipnir/training/mxfp8_backward.py": file_hash(
                root / "src/gleipnir/training/mxfp8_backward.py"
            )
        },
    }
    try:
        for lengths in ((1, 3, 31, 33, 127, 129), (4096, 4096), (8192,)):
            cuts = torch.tensor(
                (0, *accumulate(lengths)), device="cuda", dtype=torch.int32
            )
            q = torch.randn(sum(lengths), 16, 256, device="cuda", dtype=torch.bfloat16)
            k, v = [
                torch.randn(sum(lengths), 4, 256, device="cuda", dtype=torch.bfloat16)
                for _ in range(2)
            ]
            grad = torch.randn_like(q)
            row = {"lengths": lengths}
            baseline_values = None
            for name, fn in (
                ("fa4_surrogate", reference),
                ("native_mxfp8", native_attention),
            ):
                operands = [x.detach().clone().requires_grad_(True) for x in (q, k, v)]

                def iteration(
                    operands=operands, fn=fn, cuts=cuts, lengths=lengths, grad=grad
                ):
                    for x in operands:
                        x.grad = None
                    out = fn(*operands, cuts, max(lengths))
                    out.backward(grad)
                    if not torch.isfinite(out).all() or any(
                        x.grad is None or not torch.isfinite(x.grad).all()
                        for x in operands
                    ):
                        raise FloatingPointError("nonfinite attention or gradients")
                    return out

                for _ in range(3):
                    out = iteration()
                torch.cuda.synchronize()
                begin = time.monotonic()
                for _ in range(5):
                    out = iteration()
                torch.cuda.synchronize()
                row[name + "_milliseconds"] = (time.monotonic() - begin) * 1000 / 5
                values = [out.detach(), *[x.grad.detach() for x in operands]]
                if baseline_values is None:
                    baseline_values = [x.clone() for x in values]
                else:
                    row["relative_l2"] = [
                        float(
                            (a.float() - b.float()).norm()
                            / b.float().norm().clamp_min(1e-12)
                        )
                        for a, b in zip(values, baseline_values, strict=True)
                    ]
                    if row["relative_l2"][0] > 0.01:
                        raise ValueError(
                            "saving native LSE changed serving forward beyond 1%"
                        )
                    for x in operands:
                        x.grad = None
                    out = fn(*operands, cuts, max(lengths))
                    out[: lengths[0]].sum().backward()
                    leakage = max(
                        float(x.grad[lengths[0] :].abs().max())
                        if len(lengths) > 1
                        else 0.0
                        for x in operands
                    )
                    row["gradient_leakage"] = leakage
                    if leakage != 0:
                        raise ValueError(
                            "native MXFP8 gradient crossed packed boundaries"
                        )
                    changed = [x.detach().clone() for x in operands]
                    if len(lengths) > 1:
                        changed[0][: lengths[0]] += 0.5
                        observed = fn(*changed, cuts, max(lengths))
                        cross = float(
                            (
                                observed[lengths[0] :].float()
                                - out.detach()[lengths[0] :].float()
                            )
                            .abs()
                            .max()
                        )
                    else:
                        cross = 0.0
                    row["cross_example_output_difference"] = cross
                    if cross != 0:
                        raise ValueError(
                            "native MXFP8 output crossed packed boundaries"
                        )
                del operands, values, out
            result["cases"].append(row)
            print("native_backward_case", json.dumps(row), flush=True)
            write_json(output / "progress.json", result)
            del q, k, v, grad, baseline_values
        result.update(status="complete", promotion=False)
        write_json(output / "result.json", result)
    except Exception as exc:
        result.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        write_json(output / "result.json", result)
        raise


if __name__ == "__main__":
    main()
