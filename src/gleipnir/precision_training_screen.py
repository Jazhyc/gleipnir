"""Bounded adaptive optimizer trajectories for a frozen precision intervention."""

from __future__ import annotations

import gc
import importlib.metadata
import json
import math
import statistics
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


def timing_summary(passes: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate complete measured passes without dropping slower batches."""
    if not passes or any(len(item["steps"]) != 10 for item in passes):
        raise ValueError("timing requires complete ten-step measured passes")
    reference = passes[0]["steps"]
    for item in passes:
        for actual, expected in zip(item["steps"], reference, strict=True):
            for key in (
                "dataset_indices",
                "actual_tokens",
                "padded_tokens",
                "physical_sizes",
                "learning_rate",
            ):
                if actual[key] != expected[key]:
                    raise ValueError(f"timing replay mismatch: {key}")
    measurements = [step for item in passes for step in item["steps"]]
    seconds = [step["seconds"] for step in measurements]
    if any(not math.isfinite(value) or value <= 0 for value in seconds):
        raise ValueError("timing requires finite positive step durations")
    total = sum(seconds)
    return {
        "measured_steps": len(seconds),
        "mean_step_seconds": statistics.mean(seconds),
        "median_step_seconds": statistics.median(seconds),
        "min_step_seconds": min(seconds),
        "max_step_seconds": max(seconds),
        "total_step_seconds": total,
        "actual_tokens_per_second": sum(s["actual_tokens"] for s in measurements)
        / total,
        "examples_per_second": 32 * len(seconds) / total,
        "pass_mean_step_seconds": [
            statistics.mean(s["seconds"] for s in item["steps"]) for item in passes
        ],
        "per_batch_mean_seconds": [
            statistics.mean(item["steps"][i]["seconds"] for item in passes)
            for i in range(10)
        ],
        "new_dynamo_graphs": sum(item["new_dynamo_graphs"] for item in passes),
        "scope": (
            "synchronized forward/backward/clip/AdamW/scheduler and finite checks; "
            "excludes report I/O, resets, probes and export"
        ),
    }


@contextmanager
def dense_mlp_evaluation(model: torch.nn.Module):
    """Evaluate native candidates using their original BF16 MLP master weights."""
    forwards = []
    for module in model.modules():
        if isinstance(module, FrozenFourOverSixLinear):

            def dense(inputs, layer=module):
                weight = layer.weight.to(inputs.device)
                return functional.linear(inputs.to(weight.dtype), weight)

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
    expected_initial_master_sha256: str | None = None,
    timing_repeats: int = 0,
    profile_batch: int | None = None,
    gradient_validation: str = "per_tensor",
    reference_weights_on_cpu: bool = False,
    gated_delta_backend: str = "fla",
    flashqla_auto_cp: bool = False,
    gated_delta_bf16_boundary: bool = False,
    gated_delta_boundary_policy: str = "bf16",
    flashqla_layer_indices: list[int] | None = None,
    flashqla_layer_sweep: list[list[int]] | None = None,
    ten_step_learning_comparison: bool = False,
) -> dict[str, Any]:
    """Run memory/compile canaries and ten updates without held-out selection."""
    if steps not in {1, 10} or len(features) != steps * 32:
        raise ValueError("precision screen requires one or ten complete batches of 32")
    if timing_repeats not in {0, 3} or (
        timing_repeats and (steps != 10 or diagnostics_only)
    ):
        raise ValueError(
            "timing benchmark requires ten steps and three measured replays"
        )
    if profile_batch is not None and (
        steps != 10 or diagnostics_only or not 1 <= profile_batch <= 10
    ):
        raise ValueError("profiling requires one of ten warmed training batches")
    if gradient_validation not in {"per_tensor", "clip_norm"}:
        raise ValueError("unknown adapter gradient validation mode")
    if gated_delta_backend not in {"fla", "flashqla", "fla_bf16"}:
        raise ValueError("unknown gated-delta backend")
    if flashqla_auto_cp and gated_delta_backend != "flashqla":
        raise ValueError("automatic FlashQLA partitioning requires FlashQLA")
    if ten_step_learning_comparison and (
        steps != 10
        or timing_repeats
        or diagnostics_only
        or profile_batch is not None
        or gated_delta_backend not in {"fla", "flashqla"}
        or flashqla_layer_indices is not None
        or flashqla_layer_sweep is not None
        or metadata.get("fp32_projection")
        or metadata.get("mlp", {}).get("precision") != "nf4"
    ):
        raise ValueError(
            "ten-step comparison requires ten uniform updates with current head"
        )
    original_kernels = [
        (m, m.chunk_gated_delta_rule)
        for m in model.modules()
        if hasattr(m, "chunk_gated_delta_rule")
    ]
    named = [(name, p) for name, p in model.named_parameters() if p.requires_grad]
    parameters = [p for _, p in named]
    if not parameters or any(p.dtype != torch.float32 for p in parameters):
        raise ValueError("precision screen requires FP32 master adapters")
    device = parameters[0].device
    initial = [p.detach().cpu().clone() for p in parameters]
    initial_digest = tensor_digest(initial)
    if (
        expected_initial_master_sha256 is not None
        and initial_digest != expected_initial_master_sha256
    ):
        raise ValueError("initial adapter hash mismatch before precision preflight")
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
        "timing_repeats": timing_repeats,
        "ten_step_learning_comparison": ten_step_learning_comparison,
        "profile_batch": profile_batch,
        "gradient_validation": gradient_validation,
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

    @contextmanager
    def common_attention(dense):
        saved = []
        if dense and ten_step_learning_comparison:
            saved = [(m, m.chunk_gated_delta_rule) for m, _ in original_kernels]
            for module, kernel in original_kernels:
                module.chunk_gated_delta_rule = kernel
        try:
            yield
        finally:
            for module, kernel in saved:
                module.chunk_gated_delta_rule = kernel

    def evaluate(*, dense: bool):
        model.eval()
        with common_attention(dense), use_forwards(original_forwards), torch.no_grad():
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
            if parameter.grad is None or (
                gradient_validation == "per_tensor"
                and not bool(torch.isfinite(parameter.grad).all())
            ):
                raise FloatingPointError("missing or nonfinite adapter gradient")
        # clip_norm checks the aggregate norm with error_if_nonfinite=True
        # immediately afterward in every caller, before any optimizer update.
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
        if gated_delta_backend != "fla":
            from gleipnir.flashqla_training import install_with_model_canary

            if flashqla_layer_sweep and not diagnostics_only:
                raise ValueError("layer sweep permits diagnostics only")
            for indices in flashqla_layer_sweep or [flashqla_layer_indices]:
                report["attention_backend_canary"] = install_with_model_canary(
                    model,
                    [collator([item]) for item in probe]
                    + [collator([probe[0], probe[-1]])],
                    loss_forward,
                    auto_cp=flashqla_auto_cp,
                    backend=gated_delta_backend,
                    bf16_boundary=gated_delta_bf16_boundary,
                    boundary_policy=gated_delta_boundary_policy,
                    layer_indices=indices,
                    ten_step_learning_comparison=ten_step_learning_comparison,
                )
                report.setdefault("attention_backend_layer_sweep", []).append(
                    report["attention_backend_canary"]
                )
                publish()
                if report["attention_backend_canary"]["passed"]:
                    break
            if not (
                report["attention_backend_canary"]["passed"]
                or report["attention_backend_canary"].get(
                    "accepted_for_ten_step_learning_comparison", False
                )
            ):
                raise ValueError("FlashQLA model loss/gradient canary failed")
        report["before_native_probe"] = evaluate(dense=False)
        report["before_common_probe"] = evaluate(dense=True)
        if reference_weights_on_cpu:
            from gleipnir.fp4_memory import offload_reference_weights

            torch.cuda.synchronize(device)
            before_bytes = torch.cuda.memory_allocated(device)
            report["reference_offload"] = offload_reference_weights(model)
            report["reference_offload"]["allocated_bytes_before"] = before_bytes
            report["reference_offload"]["allocated_bytes_after"] = (
                torch.cuda.memory_allocated(device)
            )
            publish()
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
        report["preflight_passed"] = True
        print(f"precision_preflight={preflight}", flush=True)
        publish()
        if ten_step_learning_comparison:
            report["compile_warmup"] = []
            for step in range(steps):
                torch.cuda.synchronize(device)
                warm_start = time.perf_counter()
                indices = order[step * 32 : (step + 1) * 32]
                measurement = backward([features[i] for i in indices])
                norm = torch.nn.utils.clip_grad_norm_(
                    parameters, max_grad_norm, error_if_nonfinite=True
                )
                if float(norm) <= 0:
                    raise ValueError("compile warmup adapter gradient is zero")
                torch.cuda.synchronize(device)
                measurement.update(
                    batch=step + 1,
                    seconds=time.perf_counter() - warm_start,
                    optimizer_updates=0,
                )
                report["compile_warmup"].append(measurement)
                print(f"compile_warmup={json.dumps(measurement)}", flush=True)
                publish()
            restore()
            report["compile_warmup_master_unchanged"] = (
                tensor_digest(parameters) == initial_digest
            )
            publish()
        report["timing_passes"] = []
        for pass_index in range(1 + timing_repeats):
            # Replay the identical learning trajectory, including the LR-zero
            # first step and lazy AdamW allocation, with fresh optimizer state.
            restore()
            optimizer = optimizer_factory()
            scheduler = scheduler_factory(optimizer, steps)
            current = {
                "kind": "warmup" if timing_repeats and pass_index == 0 else "measured",
                "pass_index": pass_index,
                "initial_master_sha256": tensor_digest(parameters),
                "steps": [],
            }
            if timing_repeats:
                report["timing_passes"].append(current)
            report["steps"] = current["steps"]
            graph_start = torch._dynamo.utils.counters["stats"]["unique_graphs"]
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
                if float(norm) <= 0:
                    raise ValueError("training adapter gradient is zero")
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
                    dynamo_unique_graphs=torch._dynamo.utils.counters["stats"][
                        "unique_graphs"
                    ],
                )
                if ten_step_learning_comparison:
                    measurement["common_probe"] = evaluate(dense=True)
                current["steps"].append(measurement)
                print(
                    f"precision_pass={pass_index} "
                    f"precision_step={json.dumps(measurement)}",
                    flush=True,
                )
                publish()
            current["loop_seconds"] = time.perf_counter() - start
            current["new_dynamo_graphs"] = (
                torch._dynamo.utils.counters["stats"]["unique_graphs"] - graph_start
            )
            if ten_step_learning_comparison:
                report["timing_summary"] = timing_summary([current])
            current["final_master_sha256"] = tensor_digest(parameters)
            if current["final_master_sha256"] == initial_digest:
                raise ValueError("optimizer replay produced no adapter change")
            report["training_seconds"] = current["loop_seconds"]
            publish()
            if pass_index == 0 and profile_batch is not None:
                from gleipnir.fp4_performance import profile_backward

                indices = order[(profile_batch - 1) * 32 : profile_batch * 32]

                def profile_action(indices=indices):
                    measurement = backward([features[i] for i in indices])
                    norm = torch.nn.utils.clip_grad_norm_(
                        parameters, max_grad_norm, error_if_nonfinite=True
                    )
                    measurement["gradient_norm"] = float(norm)
                    return measurement

                report["profile"] = profile_backward(profile_action, output)
                model.zero_grad(set_to_none=True)
                if tensor_digest(parameters) != current["final_master_sha256"]:
                    raise ValueError(
                        "profiling changed adapters without an optimizer step"
                    )
                publish()
        if timing_repeats:
            report["timing_summary"] = timing_summary(report["timing_passes"][1:])
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
