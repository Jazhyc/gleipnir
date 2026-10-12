"""Matched complete-update screens in one persistent merged-QAT process."""

from __future__ import annotations

import hashlib
import json
import math
import time
from contextlib import nullcontext
from pathlib import Path

import torch
import yaml

from experiments.quantization_recovery.train import binary_kl, finite_gradients
from gleipnir.data.monitoring import file_hash, read_rows, write_json
from gleipnir.evaluation.binary import binary_token_ids
from gleipnir.training.distillation_targets import load_binary_targets
from gleipnir.training.packing import forward_packed_monitoring_logits, packed_partition
from gleipnir.training.qat_optimization import native_fp8_linear, optimized_path
from gleipnir.training.quantization_recovery import _PLANS, attention_runtime, install


def main():
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from gleipnir.training.backends.flashqla import (
        load_flashqla,
        make_flashqla_kernel,
        make_precision_boundary,
    )
    from gleipnir.training.backends.native_fp4 import verify_native_fp4_runtime
    from gleipnir.training.bf16_lora import configure_bf16_reductions

    root = Path.cwd()
    config_path = root / "experiments/qat_training_optimization/config.yaml"
    config = yaml.safe_load(config_path.read_text())
    original = yaml.safe_load((root / config["reference_config"]).read_text())
    output = root / "results" / config["experiment"]
    status_path = output / "status.json"
    started = time.monotonic()

    def status(phase, **fields):
        write_json(status_path, {"phase": phase, **fields})
        print(phase, json.dumps(fields), flush=True)

    try:
        status("loading")
        configure_bf16_reductions(allow_reduced_precision=False, allow_split_k=False)
        runtime = verify_native_fp4_runtime()
        receipt_path = root / "results/quantization-recovery01/preflight.json"
        receipt = json.loads(receipt_path.read_text())
        if not receipt["passed"]:
            raise ValueError("reference startup preflight failed")
        for path, checksum in receipt["sources"].items():
            if file_hash(root / path) != checksum:
                raise ValueError(f"reference source drift: {path}")
        uuid = str(torch.cuda.get_device_properties(0).uuid)
        if uuid.removeprefix("GPU-") != original["gpu_uuid"].removeprefix("GPU-"):
            raise ValueError(f"GPU identity mismatch: {uuid}")
        if (
            file_hash(root / original["initial_adapter"] / "adapter_model.safetensors")
            != original["initial_adapter_sha256"]
        ):
            raise ValueError("initial master drift")
        sources = {
            str(path.relative_to(root)): file_hash(path)
            for path in (
                config_path,
                Path(__file__).resolve(),
                root / "src/gleipnir/training/qat_optimization.py",
                root / "src/gleipnir/training/distillation_targets.py",
            )
        }
        workload = read_rows(root / "data/quantization-recovery01/workload.jsonl")
        if len(workload) != original["training_rows"]:
            raise ValueError("workload population drift")
        targets, target_receipt = load_binary_targets(
            workload,
            root / config["teacher_cache"],
            expected_sha256=config["teacher_sha256"],
            teacher_model=config["teacher_model"],
            prompt_contract="same",
        )
        _, kimi_receipt = load_binary_targets(
            workload,
            root / config["kimi_cache"],
            expected_sha256=config["kimi_sha256"],
            teacher_model="moonshotai/kimi-k3",
            prompt_contract="privileged",
        )
        write_json(output / "target_identity.json", target_receipt)
        write_json(output / "kimi_target_identity.json", kimi_receipt)
        write_json(
            output / "contract.json",
            {
                "sources": sources,
                "config": config,
                "reference": original,
                "reference_preflight_sha256": file_hash(receipt_path),
                "reused_reference_startup": True,
                "runtime": runtime,
                "gpu_uuid": uuid,
                "workload_sha256": file_hash(
                    root / "data/quantization-recovery01/workload.jsonl"
                ),
            },
        )
        torch.manual_seed(original["seed"])
        tokenizer = AutoTokenizer.from_pretrained(root / original["base_model"])
        ids = binary_token_ids(tokenizer)
        base = AutoModelForCausalLM.from_pretrained(
            root / original["base_model"],
            dtype=torch.bfloat16,
            device_map={"": 0},
            attn_implementation="sdpa",
        )
        model = PeftModel.from_pretrained(
            base, root / original["initial_adapter"], is_trainable=True
        )
        named = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
        for name, parameter in named:
            if "lora_" not in name:
                raise ValueError("unexpected trainable parameter")
            parameter.data = parameter.data.float()
        masters = [p.detach().cpu().clone() for _, p in named]
        function, recurrence = load_flashqla()
        kernel = make_precision_boundary(
            make_flashqla_kernel(function, auto_cp=False),
            policy="bf16_fp32_gates_norm",
        )
        layers = [m for m in model.modules() if hasattr(m, "chunk_gated_delta_rule")]
        if len(layers) != 24:
            raise ValueError("GDN scope drift")
        for module in layers:
            module.chunk_gated_delta_rule = kernel
        installation = install(model, root)
        write_json(
            output / "installation.json", {**installation, "recurrence": recurrence}
        )
        model.train()
        parameters = [p for _, p in named]
        generator = torch.Generator().manual_seed(original["seed"])
        order = torch.randperm(len(workload), generator=generator).tolist()
        batches = [
            order[i : i + original["logical_batch"]]
            for i in range(
                0,
                config["updates"] * original["logical_batch"],
                original["logical_batch"],
            )
        ]
        write_json(
            output / "batch_identity.json",
            {
                "ids": [[workload[i]["id"] for i in batch] for batch in batches],
                "partitions": [
                    packed_partition(
                        [workload[i]["prompt_tokens"] for i in batch],
                        original["packing_budget"],
                    )
                    for batch in batches
                ],
            },
        )

        def digest():
            h = hashlib.sha256()
            for name, p in named:
                h.update(name.encode())
                h.update(p.detach().cpu().numpy().tobytes())
            return h.hexdigest()

        initial_digest = digest()

        def reset():
            model.zero_grad(set_to_none=True)
            with torch.no_grad():
                for (_, parameter), master in zip(named, masters, strict=True):
                    parameter.copy_(master)
            if digest() != initial_digest:
                raise ValueError("FP32 master reset failed")
            torch.manual_seed(original["seed"])
            torch.cuda.manual_seed_all(original["seed"])

        def logits(rows):
            tokens = torch.tensor(
                [[t for row in rows for t in row["input_ids"]]], device="cuda"
            )
            with torch.autocast("cuda", dtype=torch.bfloat16):
                full, _ = forward_packed_monitoring_logits(
                    model, tokens, tuple(r["prompt_tokens"] for r in rows)
                )
            return full[:, ids]

        def backward_batch(chosen):
            rows = [workload[i] for i in chosen]
            partitions = packed_partition(
                [r["prompt_tokens"] for r in rows], original["packing_budget"]
            )
            total, scores = 0.0, []
            for indices in partitions:
                physical = [rows[i] for i in indices]
                output_logits = logits(physical)
                teacher = torch.tensor(
                    [targets[chosen[i]] for i in indices], device="cuda"
                )
                loss = binary_kl(output_logits, teacher).sum() / len(rows)
                if not torch.isfinite(loss):
                    raise FloatingPointError("nonfinite loss")
                total += float(loss.detach())
                scores.extend(
                    output_logits.detach().float().softmax(-1)[:, 1].cpu().tolist()
                )
                loss.backward()
            return total, scores, len(partitions)

        status("operator_canaries")
        checks = []
        torch.manual_seed(8)
        for m in (137, 1536, 4097):
            for n, k in ((8192, 2560), (1024, 2560), (2560, 8192)):
                a = torch.randn(m, k, device="cuda").to(torch.float8_e4m3fn)
                b = torch.randn(n, k, device="cuda").to(torch.float8_e4m3fn)
                native = native_fp8_linear(a, b)
                with torch.autocast("cuda", enabled=False):
                    expected = a.float() @ b.float().t()
                relative = float((native - expected).norm() / expected.norm())
                checks.append({"m": m, "n": n, "k": k, "relative_l2": relative})
                if (
                    not torch.isfinite(native).all()
                    or relative > config["numerical"]["operator_relative_l2"]
                ):
                    raise ValueError("native FP8 operator parity failed")
                del a, b, native, expected
        write_json(
            output / "operator_canaries.json", {"passed": True, "checks": checks}
        )

        with attention_runtime():
            status("matched_gradient_check")
            comparisons = []
            gradient_reference = None
            for variant in ("baseline", "baseline_repeat", "optimized"):
                reset()
                context = (
                    optimized_path(model, **config["implementation"])
                    if variant == "optimized"
                    else nullcontext()
                )
                with context:
                    loss, scores, _ = backward_batch(batches[0])
                    gradients = [p.grad.detach().cpu().clone() for p in parameters]
                    norm = finite_gradients(model)
                if gradient_reference is None:
                    gradient_reference = gradients
                    relative = 0.0
                else:
                    numerator = sum(
                        float((a.double() - b.double()).square().sum())
                        for a, b in zip(gradients, gradient_reference, strict=True)
                    )
                    denominator = sum(
                        float(b.double().square().sum()) for b in gradient_reference
                    )
                    relative = math.sqrt(numerator / max(denominator, 1e-30))
                comparisons.append(
                    {
                        "variant": variant,
                        "loss": loss,
                        "scores": scores,
                        "gradient_norm": norm,
                        "gradient_relative_l2": relative,
                    }
                )
                del gradients
            del gradient_reference
            score_difference = torch.tensor(comparisons[0]["scores"]) - torch.tensor(
                comparisons[2]["scores"]
            )
            limits = config["numerical"]
            diagnostic = {
                "comparisons": comparisons,
                "score_mae": float(score_difference.abs().mean()),
                "score_max": float(score_difference.abs().max()),
                "loss_absolute": abs(comparisons[0]["loss"] - comparisons[2]["loss"]),
            }
            diagnostic["passed"] = (
                diagnostic["score_mae"] <= limits["score_mae"]
                and diagnostic["score_max"] <= limits["score_max"]
                and diagnostic["loss_absolute"] <= limits["loss_absolute"]
                and comparisons[2]["gradient_relative_l2"]
                <= limits["gradient_relative_l2"]
            )
            write_json(output / "gradient_check.json", diagnostic)
            if not diagnostic["passed"]:
                raise ValueError(
                    "matched numerical gate failed; candidate not selected"
                )
            reset()
            status("longest_backward")
            longest = max(workload, key=lambda r: r["prompt_tokens"])
            with optimized_path(model, **config["implementation"]):
                torch.cuda.reset_peak_memory_stats()
                loss = binary_kl(
                    logits([longest]), torch.tensor([0.5], device="cuda")
                ).mean()
                if not torch.isfinite(loss):
                    raise FloatingPointError("nonfinite longest loss")
                loss.backward()
                norm = finite_gradients(model)
                write_json(
                    output / "longest_backward.json",
                    {
                        "passed": True,
                        "tokens": longest["prompt_tokens"],
                        "loss": float(loss),
                        "gradient_norm": norm,
                        "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                    },
                )
            summaries = []
            for trial in ("baseline", "optimized", "optimized_repeat"):
                reset()
                optimizer = torch.optim.AdamW(
                    parameters,
                    lr=original["learning_rate"],
                    weight_decay=original["weight_decay"],
                )
                updates = []
                context = (
                    nullcontext()
                    if trial == "baseline"
                    else optimized_path(model, **config["implementation"])
                )
                status("training", trial=trial, step=0)
                with context, (output / f"{trial}_updates.jsonl").open("x") as stream:
                    for step, chosen in enumerate(batches, 1):
                        if time.monotonic() - started > 1800:
                            raise TimeoutError("active-work cap reached")
                        torch.cuda.synchronize()
                        before, plans_before = time.monotonic(), len(_PLANS)
                        warmup = math.ceil(
                            original["updates"] * original["warmup_ratio"]
                        )
                        lr = original["learning_rate"] * (
                            step / warmup
                            if step <= warmup
                            else (original["updates"] - step + 1)
                            / (original["updates"] - warmup)
                        )
                        for group in optimizer.param_groups:
                            group["lr"] = lr
                        loss, _, physical = backward_batch(chosen)
                        norm = finite_gradients(model)
                        optimizer.step()
                        optimizer.zero_grad(set_to_none=True)
                        torch.cuda.synchronize()
                        update = {
                            "step": step,
                            "loss_kl": loss,
                            "gradient_norm": norm,
                            "lr": lr,
                            "physical_rows": physical,
                            "tokens": sum(workload[i]["prompt_tokens"] for i in chosen),
                            "seconds": time.monotonic() - before,
                            "new_fp4_plans": len(_PLANS) - plans_before,
                        }
                        updates.append(update)
                        stream.write(json.dumps(update, allow_nan=False) + "\n")
                        stream.flush()
                        status("training", trial=trial, **update)
                measured = updates[config["measured_start"] - 1 :]
                summary = {
                    "trial": trial,
                    "mean_seconds": sum(u["seconds"] for u in measured) / len(measured),
                    "tokens_per_second": sum(u["tokens"] for u in measured)
                    / sum(u["seconds"] for u in measured),
                    "warmup_seconds": sum(
                        u["seconds"] for u in updates[: config["measured_start"] - 1]
                    ),
                    "measured_new_fp4_plans": sum(u["new_fp4_plans"] for u in measured),
                    "final_master_digest": digest(),
                    "initial_master_digest": initial_digest,
                }
                summaries.append(summary)
                model.save_pretrained(
                    root / "results/systems_training_scratch/causal_adapter",
                    safe_serialization=True,
                )
                write_json(output / f"{trial}_summary.json", summary)
                del optimizer
            control = summaries[0]["mean_seconds"]
            for summary in summaries[1:]:
                summary["time_reduction"] = 1 - summary["mean_seconds"] / control
            result = {
                "trials": summaries,
                "numerical_passed": diagnostic["passed"],
                "selected_for_systems": all(
                    s["time_reduction"] >= config["minimum_speedup"]
                    and s["measured_new_fp4_plans"] == 0
                    for s in summaries[1:]
                ),
                "exact_repeat_master": summaries[1]["final_master_digest"]
                == summaries[2]["final_master_digest"],
                "held_out_quality_measured": False,
                "promotion": False,
            }
            write_json(output / "summary.json", result)
        status("idle", summary="summary.json")
        # Keep the loaded model and kernel plans resident. Explicit requests can
        # retire this worker without terminating the authorized Pod.
        request = output / "stop_worker"
        while not request.exists():
            time.sleep(5)
        status("worker_stopped")
    except Exception as exc:
        status("failed", error=f"{type(exc).__name__}: {exc}")
        raise


if __name__ == "__main__":
    main()
