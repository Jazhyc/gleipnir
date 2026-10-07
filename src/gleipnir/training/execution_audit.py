"""Matched backend gradients, actual optimizer updates, and bounded trajectories."""

from __future__ import annotations

import gc
import hashlib
import json
import math
import time
from collections.abc import Callable, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import torch

from gleipnir.adaptive_microbatching import MicrobatchPolicy


def compare_tensors(
    reference: Sequence[torch.Tensor | None], candidate: Sequence[torch.Tensor | None]
) -> dict[str, float | int | None]:
    """Compare every element, with explicit missing-gradient and zero handling."""
    if len(reference) != len(candidate):
        raise ValueError("tensor collections have different lengths")
    ref_sq = actual_sq = error_sq = dot = maximum = 0.0
    compared = sign_flips = nonzero = 0
    for ref, actual in zip(reference, candidate, strict=True):
        if (ref is None) != (actual is None):
            raise ValueError("execution changed gradient participation")
        if ref is None:
            continue
        if ref.shape != actual.shape:
            raise ValueError("execution changed tensor shape")
        a, b = ref.detach().float().cpu(), actual.detach().float().cpu()
        if not bool(torch.isfinite(a).all() and torch.isfinite(b).all()):
            raise FloatingPointError("nonfinite compared tensor")
        difference = b - a
        ref_sq += float(a.square().sum())
        actual_sq += float(b.square().sum())
        error_sq += float(difference.square().sum())
        dot += float((a * b).sum())
        maximum = max(maximum, float(difference.abs().max()))
        mask = (a != 0) | (b != 0)
        nonzero += int(mask.sum())
        sign_flips += int(((torch.sign(a) != torch.sign(b)) & mask).sum())
        compared += a.numel()
    return {
        "relative_l2_error": math.sqrt(error_sq / ref_sq)
        if ref_sq > 0
        else (0.0 if error_sq == 0 else None),
        "reference_norm": math.sqrt(ref_sq),
        "candidate_norm": math.sqrt(actual_sq),
        "cosine": dot / math.sqrt(ref_sq * actual_sq)
        if ref_sq > 0 and actual_sq > 0
        else None,
        "maximum_absolute_error": maximum,
        "compared_elements": compared,
        "sign_disagreement_fraction_nonzero": sign_flips / nonzero if nonzero else 0.0,
    }


def tensor_digest(values: Sequence[torch.Tensor]) -> str:
    """Hash shapes, dtypes, and complete CPU values of FP32 master adapters."""
    digest = hashlib.sha256()
    for value in values:
        value = value.detach().contiguous().cpu()
        digest.update(str((tuple(value.shape), str(value.dtype))).encode())
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


@contextmanager
def use_forwards(forwards: Sequence[tuple[torch.nn.Module, Callable]]):
    """Switch complete module forwards for both forward and checkpoint replay."""
    previous = [(module, module.forward) for module, _ in forwards]
    try:
        for module, forward in forwards:
            module.forward = forward
        yield
    finally:
        for module, forward in previous:
            module.forward = forward


def run_execution_audit(
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
    steps: int = 10,
    logical_batch_size: int = 32,
    max_padded_tokens: int = 16384,
    max_microbatch_size: int = 8,
    max_grad_norm: float = 1.0,
    counters: Callable = lambda: {},
) -> dict[str, Any]:
    """Reset identical masters/state for four diagnostic execution conditions."""
    if len(features) != steps * logical_batch_size or steps < 1:
        raise ValueError("audit requires exactly the frozen complete logical batches")
    if max_grad_norm <= 0:
        raise ValueError("audit requires positive gradient clipping")
    named = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    parameters = [p for _, p in named]
    if not parameters or any(p.dtype != torch.float32 for p in parameters):
        raise ValueError("audit requires FP32 master adapters")
    initial = [p.detach().cpu().clone() for p in parameters]
    initial_digest = tensor_digest(initial)
    original_forwards = [(module, module.forward) for module in model.modules()]
    compiled_forwards = None
    was_training = model.training
    cuda = parameters[0].device.type == "cuda"
    generator = torch.Generator().manual_seed(seed)
    order = torch.randperm(len(features), generator=generator).tolist()
    longest = sorted(
        range(len(features)), key=lambda i: -len(features[i]["direct_input_ids"])
    )[:8]
    if len(longest) != 8:
        raise ValueError("audit requires eight probe examples")
    probe = []
    for index, length in zip(
        longest, [2048, 2048, 2048, 2048, 1024, 512, 256, 128], strict=True
    ):
        item = dict(features[index])
        item["direct_input_ids"] = item["direct_input_ids"][-length:]
        probe.append(item)
    policies = {
        "singleton": MicrobatchPolicy(max_padded_tokens, 1),
        "adaptive": MicrobatchPolicy(max_padded_tokens, max_microbatch_size),
    }
    conditions = [
        ("eager", "singleton"),
        ("eager", "adaptive"),
        ("compiled", "singleton"),
        ("compiled", "adaptive"),
    ]
    report: dict[str, Any] = {
        "status": "running",
        "initial_master_sha256": initial_digest,
        "trainable_names": [n for n, _ in named],
        "order": order,
        "order_sha256": hashlib.sha256(json.dumps(order).encode()).hexdigest(),
        "probe_dataset_indices": longest,
        "probe_lengths": [len(x["direct_input_ids"]) for x in probe],
        "trajectory_steps": steps,
        "logical_batch_size": logical_batch_size,
        "probe_updates_use_base_lr_without_scheduler": True,
        "common_evaluation": "eager_eval_singleton_bf16",
        "probes": {},
        "trajectories": {},
    }
    output.mkdir(parents=True, exist_ok=True)

    def publish() -> None:
        temporary = output / "execution_audit.tmp"
        temporary.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        temporary.replace(output / "execution_audit.json")

    def synchronize() -> None:
        if cuda:
            torch.cuda.synchronize(parameters[0].device)

    def snapshot() -> list[torch.Tensor]:
        return [p.detach().cpu().clone() for p in parameters]

    def restore() -> None:
        model.zero_grad(set_to_none=True)
        with torch.no_grad():
            for parameter, value in zip(parameters, initial, strict=True):
                parameter.copy_(value)
        if tensor_digest(snapshot()) != initial_digest:
            raise ValueError("initial master restoration failed")
        torch.manual_seed(seed)
        if cuda:
            torch.cuda.manual_seed_all(seed)
        model.train()

    def collect_gradients() -> list[torch.Tensor | None]:
        return [
            None if p.grad is None else p.grad.detach().cpu().clone()
            for p in parameters
        ]

    def backward(batch_features: list[dict], policy: MicrobatchPolicy) -> dict:
        model.zero_grad(set_to_none=True)
        lengths = [len(f["direct_input_ids"]) for f in batch_features]
        partition = policy.partition(lengths)
        total = 0.0
        for indices in partition:
            loss = loss_forward(collator([batch_features[i] for i in indices]))
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError("nonfinite execution-audit loss")
            weight = len(indices) / len(batch_features)
            total += float(loss.detach()) * weight
            (loss * weight).backward()
        return {
            "mean_loss": total,
            "physical_sizes": [len(i) for i in partition],
            "actual_tokens": sum(lengths),
            "padded_tokens": sum(
                len(i) * max(lengths[j] for j in i) for i in partition
            ),
        }

    def evaluate() -> dict:
        training = model.training
        try:
            model.eval()
            with use_forwards(original_forwards), torch.no_grad():
                losses = [
                    float(loss_forward(collator([item])).detach()) for item in probe
                ]
            if not all(math.isfinite(x) for x in losses):
                raise FloatingPointError("nonfinite common-evaluation loss")
            return {"mean_loss": sum(losses) / len(losses), "per_example_loss": losses}
        finally:
            model.train(training)

    def selected_forwards(backend: str):
        nonlocal compiled_forwards
        if backend == "eager":
            return original_forwards
        if compiled_forwards is None:
            report["compiled_layer_indices"] = install_compile()
            if not report["compiled_layer_indices"]:
                raise ValueError("compiled audit requires active compiled layers")
            compiled_forwards = [(module, module.forward) for module in model.modules()]
        return compiled_forwards

    def release(optimizer) -> None:
        model.zero_grad(set_to_none=True)
        optimizer.state.clear()
        gc.collect()
        if cuda:
            torch.cuda.empty_cache()

    references = {}
    trajectory_references = {}
    publish()
    try:
        for backend, partition_name in conditions:
            name = f"{backend}-{partition_name}"
            restore()
            forwards = selected_forwards(backend)
            optimizer = optimizer_factory()
            if optimizer.state:
                raise ValueError("probe optimizer state must start empty")
            try:
                with use_forwards(forwards):
                    baseline = evaluate()
                    first = backward(probe, policies[partition_name])
                    gradients = collect_gradients()
                    repeated = backward(probe, policies[partition_name])
                    repeat_metrics = compare_tensors(gradients, collect_gradients())
                    for p, gradient in zip(parameters, gradients, strict=True):
                        p.grad = None if gradient is None else gradient.to(p.device)
                    clip_norm = float(
                        torch.nn.utils.clip_grad_norm_(
                            parameters, max_grad_norm, error_if_nonfinite=True
                        )
                    )
                    optimizer.step()
                    delta = [
                        v - old for v, old in zip(snapshot(), initial, strict=True)
                    ]
                    result = {
                        "initial_master_sha256": initial_digest,
                        "initial_optimizer_state_entries": 0,
                        "optimizer_groups": [
                            {k: v for k, v in group.items() if k != "params"}
                            for group in optimizer.param_groups
                        ],
                        "forward": first,
                        "repeated_forward": repeated,
                        "repeat_gradient_comparison": repeat_metrics,
                        "unclipped_gradient_norm": clip_norm,
                        "baseline_common_loss": baseline,
                        "post_update_common_loss": evaluate(),
                        "update_norm": math.sqrt(
                            sum(float(v.square().sum()) for v in delta)
                        ),
                        "counters": counters(),
                    }
                    if backend == "compiled":
                        ref_gradients, ref_delta = references[("eager", partition_name)]
                        result["eager_compiled_gradient_comparison"] = compare_tensors(
                            ref_gradients, gradients
                        )
                        result["eager_compiled_update_comparison"] = compare_tensors(
                            ref_delta, delta
                        )
                    if partition_name == "adaptive":
                        ref_gradients, ref_delta = references[(backend, "singleton")]
                        result["singleton_adaptive_gradient_comparison"] = (
                            compare_tensors(ref_gradients, gradients)
                        )
                        result["singleton_adaptive_update_comparison"] = (
                            compare_tensors(ref_delta, delta)
                        )
                    references[(backend, partition_name)] = (gradients, delta)
                    report["probes"][name] = result
                    publish()
                    print(
                        f"audit_probe_done={name} "
                        f"post_loss={result['post_update_common_loss']['mean_loss']}",
                        flush=True,
                    )
            finally:
                release(optimizer)
                del optimizer
        references.clear()
        for backend, partition_name in conditions:
            name = f"{backend}-{partition_name}"
            restore()
            optimizer = optimizer_factory()
            if optimizer.state:
                raise ValueError("trajectory optimizer state must start empty")
            scheduler = scheduler_factory(optimizer, steps)
            rows = []
            report["trajectories"][name] = {
                "status": "running",
                "initial_master_sha256": initial_digest,
                "initial_optimizer_state_entries": 0,
                "steps": rows,
                "baseline_common_loss": evaluate(),
            }
            publish()
            try:
                with use_forwards(selected_forwards(backend)):
                    for step in range(steps):
                        indices = order[
                            step * logical_batch_size : (step + 1) * logical_batch_size
                        ]
                        synchronize()
                        started = time.perf_counter()
                        batch = backward(
                            [features[i] for i in indices], policies[partition_name]
                        )
                        lr = [group["lr"] for group in optimizer.param_groups]
                        grad_norm = float(
                            torch.nn.utils.clip_grad_norm_(
                                parameters, max_grad_norm, error_if_nonfinite=True
                            )
                        )
                        optimizer.step()
                        scheduler.step()
                        model.zero_grad(set_to_none=True)
                        synchronize()
                        elapsed = time.perf_counter() - started
                        rows.append(
                            {
                                "step": step + 1,
                                "logical_indices": indices,
                                "lr": lr,
                                "unclipped_gradient_norm": grad_norm,
                                "seconds": elapsed,
                                **batch,
                                "common_probe": evaluate(),
                            }
                        )
                        publish()
                        print(
                            f"audit_train_step={name}:{step + 1}/{steps} "
                            f"train_loss={batch['mean_loss']} "
                            f"common_loss={rows[-1]['common_probe']['mean_loss']} "
                            f"seconds={elapsed}",
                            flush=True,
                        )
                final = snapshot()
                result = report["trajectories"][name]
                result.update(
                    status="completed",
                    final_master_sha256=tensor_digest(final),
                    counters=counters(),
                )
                from gleipnir.systems_artifacts import save_systems_master

                result["master_artifact"] = save_systems_master(
                    {n: v for (n, _), v in zip(named, final, strict=True)}
                )
                if backend == "compiled":
                    ref_final = trajectory_references[("eager", partition_name)]
                    result["eager_compiled_final_parameter_comparison"] = (
                        compare_tensors(ref_final, final)
                    )
                    result["eager_compiled_cumulative_update_comparison"] = (
                        compare_tensors(
                            [
                                v - old
                                for v, old in zip(ref_final, initial, strict=True)
                            ],
                            [v - old for v, old in zip(final, initial, strict=True)],
                        )
                    )
                if partition_name == "adaptive":
                    ref_final = trajectory_references[(backend, "singleton")]
                    result["singleton_adaptive_cumulative_update_comparison"] = (
                        compare_tensors(
                            [
                                v - old
                                for v, old in zip(ref_final, initial, strict=True)
                            ],
                            [v - old for v, old in zip(final, initial, strict=True)],
                        )
                    )
                trajectory_references[(backend, partition_name)] = final
                publish()
            finally:
                release(optimizer)
                del scheduler, optimizer
        report["status"] = "completed"
        publish()
        return report
    except BaseException as error:
        report.update(status="failed", error=f"{type(error).__name__}: {error}")
        publish()
        raise
    finally:
        with torch.no_grad():
            for p, value in zip(parameters, initial, strict=True):
                p.copy_(value)
        model.zero_grad(set_to_none=True)
        for module, forward in original_forwards:
            module.forward = forward
        model.train(was_training)
