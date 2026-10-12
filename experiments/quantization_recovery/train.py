"""Fixed-epoch binary QAD with fresh alignment/gradient/memory gates."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from pathlib import Path

import torch
import torch.nn.functional as F
import yaml

from gleipnir.data.monitoring import file_hash, read_rows, write_json
from gleipnir.evaluation.binary import binary_token_ids
from gleipnir.training.packing import forward_packed_monitoring_logits, packed_partition
from gleipnir.training.quantization_recovery import attention_runtime, install


def binary_kl(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Per-row T=1 KL, retaining teacher entropy for interpretable receipts."""
    probabilities = torch.stack((1 - targets, targets), -1).float()
    return F.kl_div(
        F.log_softmax(logits.float(), -1), probabilities, reduction="none"
    ).sum(-1)


def finite_gradients(model):
    named = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    bad = [n for n, p in named if p.grad is None or not torch.isfinite(p.grad).all()]
    if bad:
        raise FloatingPointError(f"missing/nonfinite FP32 adapter gradients: {bad[:5]}")
    if any(p.dtype != torch.float32 or p.grad.dtype != torch.float32 for _, p in named):
        raise ValueError("QAD master/gradient dtype drift")
    return float(
        torch.nn.utils.clip_grad_norm_(
            [p for _, p in named], 1.0, error_if_nonfinite=True
        )
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from gleipnir.campaigns.monitoring.evaluation import agreement
    from gleipnir.training.backends.flashqla import (
        load_flashqla,
        make_flashqla_kernel,
        make_precision_boundary,
    )
    from gleipnir.training.backends.native_fp4 import verify_native_fp4_runtime
    from gleipnir.training.bf16_lora import configure_bf16_reductions

    root = Path.cwd()
    config_path = root / "experiments/quantization_recovery/config.yaml"
    config = yaml.safe_load(config_path.read_text())
    output = root / "results" / config["experiment"]
    directory = root / "data" / config["experiment"]
    output.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(config["seed"])
    configure_bf16_reductions(allow_reduced_precision=False, allow_split_k=False)
    runtime = verify_native_fp4_runtime()
    tokenizer = AutoTokenizer.from_pretrained(root / config["base_model"])
    ids = binary_token_ids(tokenizer)
    base = AutoModelForCausalLM.from_pretrained(
        root / config["base_model"],
        dtype=torch.bfloat16,
        device_map={"": 0},
        attn_implementation="sdpa",
    )
    model = PeftModel.from_pretrained(
        base, root / config["initial_adapter"], is_trainable=True
    )
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            if "lora_" not in name:
                raise ValueError("unexpected trainable base parameter")
            parameter.data = parameter.data.float()
    initial_sha = file_hash(
        root / config["initial_adapter"] / "adapter_model.safetensors"
    )
    if initial_sha != config["initial_adapter_sha256"]:
        raise ValueError("QAD initial adapter drift")
    # Recurrence is the unchanged selected BF16 FlashQLA path. Fresh QAD
    # forward/gradient/isolation/memory gates below cover the combined model.
    function, recurrence = load_flashqla()
    kernel = make_precision_boundary(
        make_flashqla_kernel(function, auto_cp=False), policy="bf16_fp32_gates_norm"
    )
    recurrence_layers = [
        m for m in model.modules() if hasattr(m, "chunk_gated_delta_rule")
    ]
    if len(recurrence_layers) != 24:
        raise ValueError("GDN scope drift")
    for module in recurrence_layers:
        module.chunk_gated_delta_rule = kernel
    installation = install(model, root)
    from experiments.quantization_recovery.kernel_canary import main as decoder_canary

    decoder_canary()
    workload = read_rows(directory / "workload.jsonl")
    if len(workload) != config["training_rows"]:
        raise ValueError("training population drift")
    reference = json.loads(
        (
            root
            / "results/filtered-fp4-fa4-baseline01/4b/monitor/parity_reference.json"
        ).read_text()
    )
    server = json.loads(
        (
            root / "results/filtered-fp4-fa4-optimized-id01/canary_repeat1.json"
        ).read_text()
    )
    by_id = {r["id"]: r for r in workload}
    if len(by_id) != len(workload) or len(server) != len(reference["ids"]):
        raise ValueError("training/server canary identity drift")
    canary = [by_id[i] for i in reference["ids"]]
    for row, prediction in zip(canary, server, strict=True):
        if row["prompt_sha256"] != prediction["prompt_sha256"]:
            raise ValueError("simulator/server prompt mismatch")

    def logits(rows):
        lengths = tuple(r["prompt_tokens"] for r in rows)
        tokens = torch.tensor(
            [[t for r in rows for t in r["input_ids"]]], device="cuda"
        )
        with torch.autocast("cuda", dtype=torch.bfloat16):
            full, _ = forward_packed_monitoring_logits(model, tokens, lengths)
        return full[:, ids]

    with attention_runtime():
        model.eval()
        before = time.monotonic()
        scores = []
        with torch.no_grad():
            for row in canary:
                scores.append(float(logits([row]).float().softmax(-1)[0, 1]))
                print("alignment_rows", len(scores), len(canary), flush=True)
        gate = agreement(
            scores, [p["score"] for p in server], reference["base"], config["alignment"]
        )
        write_json(
            output / "alignment.json",
            {
                "gate": gate,
                "scores": scores,
                "ids": reference["ids"],
                "server_predictions_sha256": file_hash(
                    root / "results/filtered-fp4-fa4-optimized-id01/canary_repeat1.json"
                ),
                "installation": installation,
                "fp4_operator_checks": [
                    group.operator_check
                    for group in model._gleipnir_qad_groups
                    if group.precision == "fp4"
                ],
                "recurrence": recurrence,
                "seconds": time.monotonic() - before,
                "config_sha256": file_hash(config_path),
                "master_sha256": initial_sha,
                "scope": "training_source_canary_simulator_vs_optimized_server_c4",
                "passed": gate["passed"],
            },
        )
        if not gate["passed"]:
            raise ValueError("QAD simulator/server score alignment failed")
        # Independent sequence boundaries must survive packing. Use short
        # examples below the SwiGLU row-policy switch to isolate boundaries.
        short = sorted(workload, key=lambda r: r["prompt_tokens"])[:2]
        with torch.no_grad():
            separate = torch.cat([logits([row]).float().softmax(-1) for row in short])
            packed = logits(short).float().softmax(-1)
        difference = float((packed - separate).abs().max())
        isolation = {
            "maximum_probability_difference": difference,
            "limit": 0.02,
            "passed": difference <= 0.02,
        }
        write_json(output / "isolation.json", isolation)
        if not isolation["passed"]:
            raise ValueError("QAD packed-sequence isolation failed")
        model.train()
        model.zero_grad(set_to_none=True)
        longest = max(workload, key=lambda r: r["prompt_tokens"])
        torch.cuda.reset_peak_memory_stats()
        loss = binary_kl(logits([longest]), torch.tensor([0.5], device="cuda")).mean()
        if not torch.isfinite(loss):
            raise FloatingPointError("nonfinite QAD preflight loss")
        loss.backward()
        norm = finite_gradients(model)
        write_json(
            output / "preflight.json",
            {
                "passed": True,
                "longest_tokens": longest["prompt_tokens"],
                "loss": float(loss.detach()),
                "unclipped_gradient_norm": norm,
                "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                "initial_master_sha256": initial_sha,
                "backward": "approximate_low_rank_STE_and_quantized_QKV_FA4",
                "runtime": {**runtime, "gpu": torch.cuda.get_device_name()},
                "sources": {
                    str(path.relative_to(root)): file_hash(path)
                    for path in [
                        Path(__file__).resolve(),
                        root / "src/gleipnir/training/quantization_recovery.py",
                        root / "src/gleipnir/kernels/fp4/decode.py",
                    ]
                },
            },
        )
        model.zero_grad(set_to_none=True)
        print("QAD_preflight_passed", flush=True)
        if args.preflight_only:
            return
        receipt = json.loads((directory / "teacher_complete.json").read_text())
        if receipt["teacher_sha256"] != file_hash(directory / "teacher.jsonl"):
            raise ValueError("BF16 teacher cache drift")
        teachers = read_rows(directory / "teacher.jsonl")
        if [r["index"] for r in teachers] != [r["id"] for r in workload]:
            raise ValueError("teacher row order mismatch")
        for row, target in zip(workload, teachers, strict=True):
            if target["rendered_prompt_sha256"] != row["prompt_sha256"]:
                raise ValueError("teacher prompt identity mismatch")
        params = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(
            params, lr=config["learning_rate"], weight_decay=config["weight_decay"]
        )
        generator = torch.Generator().manual_seed(config["seed"])
        order = torch.randperm(len(workload), generator=generator).tolist()
        write_json(
            output / "training_order.json",
            {
                "ids": [workload[i]["id"] for i in order],
                "sha256": hashlib.sha256(json.dumps(order).encode()).hexdigest(),
            },
        )
        updates = math.ceil(len(order) / config["logical_batch"])
        if updates != config["updates"]:
            raise ValueError("QAD update count drift")
        warmup = math.ceil(updates * config["warmup_ratio"])
        started = time.monotonic()
        with (output / "updates.jsonl").open("x") as log:
            for step, offset in enumerate(
                range(0, len(order), config["logical_batch"]), start=1
            ):
                before = time.monotonic()
                chosen = order[offset : offset + config["logical_batch"]]
                rows = [workload[i] for i in chosen]
                partitions = packed_partition(
                    [r["prompt_tokens"] for r in rows], config["packing_budget"]
                )
                lr = config["learning_rate"] * (
                    step / warmup
                    if step <= warmup
                    else (updates - step + 1) / (updates - warmup)
                )
                for group in optimizer.param_groups:
                    group["lr"] = lr
                total = 0.0
                for indices in partitions:
                    physical = [rows[i] for i in indices]
                    targets = torch.tensor(
                        [teachers[chosen[i]]["soft_target"] for i in indices],
                        device="cuda",
                    )
                    loss = binary_kl(logits(physical), targets).sum() / len(rows)
                    if not torch.isfinite(loss):
                        raise FloatingPointError(f"nonfinite QAD loss: {step}")
                    total += float(loss.detach())
                    loss.backward()
                norm = finite_gradients(model)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                torch.cuda.synchronize()
                update = {
                    "step": step,
                    "loss_kl": total,
                    "lr": lr,
                    "gradient_norm": norm,
                    "seconds": time.monotonic() - before,
                    "rows": len(rows),
                    "physical_rows": len(partitions),
                }
                log.write(json.dumps(update, allow_nan=False) + "\n")
                log.flush()
                print("QAD_update", json.dumps(update), flush=True)
        master = output / "causal_adapter"
        model.save_pretrained(master, safe_serialization=True)
        tokenizer.save_pretrained(master)
        import shutil

        from safetensors.torch import load_file, save_file

        export = output / "model"
        export.mkdir(exist_ok=True)
        shutil.copy2(master / "adapter_config.json", export / "adapter_config.json")
        tensors = load_file(master / "adapter_model.safetensors")
        if any(t.dtype != torch.float32 for t in tensors.values()):
            raise ValueError("saved QAD master is not FP32")
        save_file(
            {n: t.to(torch.bfloat16) for n, t in tensors.items()},
            export / "adapter_model.safetensors",
        )
        write_json(
            output / "training_complete.json",
            {
                "updates": updates,
                "rows": len(order),
                "seconds": time.monotonic() - started,
                "master_sha256": file_hash(master / "adapter_model.safetensors"),
                "export_sha256": file_hash(export / "adapter_model.safetensors"),
                "teacher_sha256": receipt["teacher_sha256"],
                "config_sha256": file_hash(config_path),
                "selection": "final_epoch_only",
                "promotion": False,
            },
        )


if __name__ == "__main__":
    main()
