"""Bounded adaptive optimizer trajectories for a frozen precision intervention."""

from __future__ import annotations

import gc
import importlib.metadata
import json
import math
import time
from collections.abc import Callable
from contextlib import contextmanager
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as functional

from gleipnir.adaptive_microbatching import MicrobatchPolicy
from gleipnir.fouroversix_training import FrozenFourOverSixLinear, native_call_counts
from gleipnir.training_execution_audit import tensor_digest, use_forwards


@contextmanager
def dense_mlp_evaluation(model: torch.nn.Module):
    """Evaluate native candidates using their original BF16 MLP master weights."""
    forwards = []
    for module in model.modules():
        if isinstance(module, FrozenFourOverSixLinear):

            def dense(inputs, layer=module):
                return functional.linear(inputs.to(layer.weight.dtype), layer.weight)

            forwards.append((module, dense))
    with use_forwards(forwards):
        yield


def run_precision_training_screen(
    *,
    model: torch.nn.Module,
    features: list[dict[str, Any]],
    collator: Callable,
    loss_forward: Callable,
    optimizer_factory: Callable,
    scheduler_factory: Callable,
    install_compile: Callable,
    output: Path,
    seed: int,
    policy: MicrobatchPolicy,
    steps: int,
    max_grad_norm: float,
    metadata: dict[str, Any],
    diagnostics_only: bool = False,
    capture_native_operands: bool = False,
) -> dict[str, Any]:
    """Run memory/compile canaries and ten updates without held-out selection."""
    if steps not in {1, 10} or len(features) != steps * 32:
        raise ValueError("precision screen requires one or ten complete batches of 32")
    named = [(name, p) for name, p in model.named_parameters() if p.requires_grad]
    parameters = [p for _, p in named]
    if not parameters or any(p.dtype != torch.float32 for p in parameters):
        raise ValueError("precision screen requires FP32 master adapters")
    device = parameters[0].device
    initial = [p.detach().cpu().clone() for p in parameters]
    initial_digest = tensor_digest(initial)
    order = torch.randperm(
        len(features), generator=torch.Generator().manual_seed(seed)
    ).tolist()
    original_forwards = [(m, m.forward) for m in model.modules()]
    probe = [
        dict(features[i])
        for i in sorted(
            range(len(features)), key=lambda i: -len(features[i]["direct_input_ids"])
        )[:8]
    ]
    for item, length in zip(
        probe, [2048, 2048, 2048, 2048, 1024, 512, 256, 128], strict=True
    ):
        item["direct_input_ids"] = item["direct_input_ids"][-length:]
    report: dict[str, Any] = {
        "status": "running",
        "metadata": metadata,
        "policy": asdict(policy),
        "initial_master_sha256": initial_digest,
        "trainable_names": [n for n, _ in named],
        "order": order,
        "steps": [],
        "logical_batch_size": 32,
        "gpu": torch.cuda.get_device_name(device),
        "gpu_total_bytes": torch.cuda.get_device_properties(device).total_memory,
        "software": {
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            **{
                name: importlib.metadata.version(name)
                for name in [
                    "transformers",
                    "peft",
                    "triton",
                    "fla-core",
                    "causal-conv1d",
                ]
            },
        },
        "preflight_selection": "longest_32_of_supplied_selection",
        "common_probe": (
            "eager_singletons; native candidate uses original BF16 MLP masters; "
            "controls retain their own NF4 or BF16 MLP bases"
        ),
        "probe_lengths": [len(p["direct_input_ids"]) for p in probe],
    }
    output.mkdir(parents=True, exist_ok=True)

    def publish():
        path = output / "screen.tmp"
        path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        path.replace(output / "screen.json")

    def restore():
        model.zero_grad(set_to_none=True)
        with torch.no_grad():
            for parameter, value in zip(parameters, initial, strict=True):
                parameter.copy_(value)
        if tensor_digest(parameters) != initial_digest:
            raise ValueError("adapter restoration hash mismatch")
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        model.train()

    def evaluate(*, dense: bool):
        model.eval()
        with use_forwards(original_forwards), torch.no_grad():
            if dense:
                with dense_mlp_evaluation(model):
                    losses = [
                        float(loss_forward(collator([item])).detach()) for item in probe
                    ]
            else:
                losses = [
                    float(loss_forward(collator([item])).detach()) for item in probe
                ]
        model.train()
        if not all(math.isfinite(x) for x in losses):
            raise FloatingPointError("nonfinite probe loss")
        return {"losses": losses, "mean_loss": sum(losses) / len(losses)}

    def backward(items):
        model.zero_grad(set_to_none=True)
        lengths = [len(item["direct_input_ids"]) for item in items]
        partition = policy.partition(lengths)
        total = 0.0
        for indices in partition:
            loss = loss_forward(collator([items[i] for i in indices]))
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError("nonfinite training loss")
            fraction = len(indices) / len(items)
            total += float(loss.detach()) * fraction
            (loss * fraction).backward()
        for parameter in parameters:
            if parameter.grad is None or not bool(torch.isfinite(parameter.grad).all()):
                raise FloatingPointError("missing or nonfinite adapter gradient")
        return {
            "mean_loss": total,
            "physical_sizes": [len(i) for i in partition],
            "actual_tokens": sum(lengths),
            "padded_tokens": sum(
                len(i) * max(lengths[j] for j in i) for i in partition
            ),
        }

    publish()
    try:
        restore()
        report["before_native_probe"] = evaluate(dense=False)
        report["before_common_probe"] = evaluate(dense=True)
        canary = [dict(probe[0]), dict(probe[-1])]
        batch = collator(canary)
        model.eval()
        with torch.no_grad():
            eager = float(loss_forward(batch).detach())
            eager_repeat = float(loss_forward(batch).detach())
        report["compiled_layers"] = install_compile()
        with torch.no_grad():
            compiled = float(loss_forward(batch).detach())
            compiled_repeat = float(loss_forward(batch).detach())
        passed = all(
            math.isfinite(value) and abs(eager - value) <= 0.01 + 0.01 * abs(eager)
            for value in [eager_repeat, compiled, compiled_repeat]
        )
        report["compile_canary"] = {
            "eager_loss": eager,
            "eager_repeat_loss": eager_repeat,
            "compiled_loss": compiled,
            "compiled_repeat_loss": compiled_repeat,
            "passed": passed,
        }
        publish()
        if diagnostics_only:
            if capture_native_operands:
                from gleipnir.fp4_compiler_diagnostic import compare_native_operands

                report["native_operand_comparison"] = compare_native_operands(
                    model=model,
                    batch=batch,
                    loss_forward=loss_forward,
                    original_forwards=original_forwards,
                    eager_loss=eager,
                    compiled_loss=compiled,
                )
                publish()
            base = model.get_base_model() if hasattr(model, "get_base_model") else model
            layers = base.model.layers
            original_by_id = {
                id(module): forward for module, forward in original_forwards
            }
            compiled_forwards = [layer.forward for layer in layers]
            prefix_losses = []
            for count in [0, 1, 4, 8, 16, 24, len(layers)]:
                forwards = [
                    (
                        layer,
                        compiled_forwards[i]
                        if i < count
                        else original_by_id[id(layer)],
                    )
                    for i, layer in enumerate(layers)
                ]
                with use_forwards(forwards), torch.no_grad():
                    value = float(loss_forward(batch).detach())
                prefix_losses.append({"compiled_prefix_layers": count, "loss": value})
            report["prefix_compile_losses"] = prefix_losses
            report["diagnostics_only"] = True
            report["native_calls"] = native_call_counts(model)
            report["master_unchanged"] = tensor_digest(parameters) == initial_digest
            report["status"] = "diagnosed"
            publish()
            return report
        if not passed:
            raise ValueError(
                f"same-weight compilation loss canary failed: {eager}, {compiled}"
            )
        model.train()
        longest = sorted(features, key=lambda item: -len(item["direct_input_ids"]))[:32]
        torch.cuda.reset_peak_memory_stats(device)
        preflight = backward(longest)
        norm = torch.nn.utils.clip_grad_norm_(
            parameters, max_grad_norm, error_if_nonfinite=True
        )
        if float(norm) <= 0:
            raise ValueError("preflight adapter effect is zero")
        preflight.update(
            gradient_norm=float(norm),
            peak_allocated_bytes=torch.cuda.max_memory_allocated(device),
            longest_tokens=max(len(x["direct_input_ids"]) for x in longest),
        )
        report["preflight"] = preflight
        restore()
        optimizer = optimizer_factory()
        scheduler = scheduler_factory(optimizer, steps)
        report["preflight_passed"] = True
        print(f"precision_preflight={preflight}", flush=True)
        publish()
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
        start = time.perf_counter()
        for step in range(steps):
            indices = order[step * 32 : (step + 1) * 32]
            torch.cuda.synchronize(device)
            step_start = time.perf_counter()
            measurement = backward([features[i] for i in indices])
            norm = torch.nn.utils.clip_grad_norm_(
                parameters, max_grad_norm, error_if_nonfinite=True
            )
            lr = optimizer.param_groups[0]["lr"]
            optimizer.step()
            scheduler.step()
            torch.cuda.synchronize(device)
            measurement.update(
                step=step + 1,
                dataset_indices=indices,
                gradient_norm=float(norm),
                learning_rate=lr,
                seconds=time.perf_counter() - step_start,
                peak_allocated_bytes=torch.cuda.max_memory_allocated(device),
                peak_reserved_bytes=torch.cuda.max_memory_reserved(device),
            )
            report["steps"].append(measurement)
            print(f"precision_step={json.dumps(measurement)}", flush=True)
            publish()
        report["training_seconds"] = time.perf_counter() - start
        report["after_native_probe"] = evaluate(dense=False)
        report["after_common_probe"] = evaluate(dense=True)
        final_digest = tensor_digest(parameters)
        if final_digest == initial_digest:
            raise ValueError("optimizer trajectory produced no adapter change")
        report["final_master_sha256"] = final_digest
        report["native_calls"] = native_call_counts(model)
        if (
            metadata["mlp"]["precision"] == "fouroversix"
            and report["native_calls"]["backward"] <= 0
        ):
            raise ValueError("frozen FP4-base input gradients were not exercised")
        report["dynamo_counters"] = {
            name: dict(values) for name, values in torch._dynamo.utils.counters.items()
        }
        torch.save(
            {name: p.detach().cpu() for name, p in named}, output / "fp32_master.pt"
        )
        report["status"] = "complete"
        publish()
        return report
    except Exception as error:
        report.update(
            status="failed",
            error=f"{type(error).__name__}: {error}",
            native_calls=native_call_counts(model),
        )
        publish()
        raise
    finally:
        model.zero_grad(set_to_none=True)
        gc.collect()
