"""Bounded production-kernel branching parity; never launch training on failure."""

import argparse
import json
import time
from pathlib import Path


def run(
    *,
    checkpoint_segments=False,
    parent=None,
    stress_only=False,
    two_gpu=False,
    alignment=1,
    fp32_head=False,
    independent_endpoint=False,
) -> dict:
    import torch
    from peft import PeftModel, prepare_model_for_kbit_training
    from transformers import BitsAndBytesConfig, Qwen3_5ForCausalLM

    from gleipnir.branch_training import branched_decision_logits, plan_branches

    torch.manual_seed(42)
    torch.set_num_threads(1)
    device_map = {"": 0}
    if two_gpu:
        device_map = {
            "model.embed_tokens": 0,
            "model.rotary_emb": 0,
            "model.norm": 1,
            "lm_head": 0,
        }
        device_map.update({f"model.layers.{i}": int(i >= 16) for i in range(32)})
    model = Qwen3_5ForCausalLM.from_pretrained(
        "Qwen/Qwen3.5-4B",
        revision="851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a",
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
        device_map=device_map,
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        ),
    )
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=False)
    model = PeftModel.from_pretrained(
        model,
        "results/monitoring_prefix_training_low/runs/"
        "prefix-w010-lr2em05-seed0/causal_adapter",
        is_trainable=True,
    ).train()
    base = model.get_base_model()
    kernels = []
    for layer in base.model.layers:
        if hasattr(layer, "linear_attn"):
            a = layer.linear_attn
            kernel = a.chunk_gated_delta_rule.__module__
            if not kernel.startswith("fla.") or a.causal_conv1d_fn is None:
                raise RuntimeError("production kernels unavailable")
            kernels.append(kernel)
    trunk = torch.randint(100, 10000, (1024,)).tolist()
    requests = [
        trunk[:256] + [200, 201, 202],
        trunk[:513] + [200, 201, 202],
        trunk + [200, 201, 202],
    ]
    plan = plan_branches(requests)
    targets = torch.tensor([0.2, 0.7, 0.9], device="cuda")
    weights = torch.tensor([0.05, 0.05, 1.0], device="cuda") / 1.1
    if parent is not None:
        from transformers import AutoTokenizer

        from gleipnir.branch_data import BranchDataset

        tokenizer = AutoTokenizer.from_pretrained(
            "Qwen/Qwen3.5-4B", revision="851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"
        )
        dataset = BranchDataset(tokenizer)
        row = dataset[parent]
        plan = row["plan"]
        requests = plan.requests
        targets = torch.tensor(row["targets"], device="cuda")
        count = len(targets) - 1
        weights = torch.tensor([0.1 / max(count, 1)] * count + [1.0], device="cuda")
        if count:
            weights /= 1.1
    params = {n: p for n, p in model.named_parameters() if p.requires_grad}
    plan = plan_branches(requests, alignment=alignment)
    work_tokens = plan.token_work(independent_endpoint=independent_endpoint)

    def loss(logits):
        return (
            torch.nn.functional.binary_cross_entropy_with_logits(
                logits[:, 1].float() - logits[:, 0].float(),
                targets.to(logits.device),
                reduction="none",
            )
            * weights.to(logits.device)
        ).sum()

    if stress_only:
        print(
            "stress_forward", parent, len(requests), work_tokens, flush=True
        )
        for i in range(torch.cuda.device_count()):
            torch.cuda.reset_peak_memory_stats(i)
        started = time.perf_counter()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            actual = branched_decision_logits(
                model,
                plan,
                [15, 16],
                checkpoint_segments=checkpoint_segments,
                fp32_head=fp32_head,
                independent_endpoint=independent_endpoint,
            )
            objective = loss(actual)
        print("stress_backward", flush=True)
        objective.backward()
        if not torch.isfinite(objective) or any(
            p.grad is None or not torch.isfinite(p.grad).all() for p in params.values()
        ):
            raise RuntimeError("nonfinite or missing stress gradient")
        optimizer = torch.optim.AdamW(params.values(), lr=2e-5)
        optimizer.step()
        torch.cuda.synchronize()
        return {
            "passed": True,
            "gate": "memory_and_finite_update_only",
            "parent": parent,
            "prefixes": len(requests) - 1,
            "checkpoint_segments": checkpoint_segments,
            "seconds": time.perf_counter() - started,
            "processed_tokens": work_tokens,
            "max_request_tokens": max(map(len, requests)),
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
            "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
            "device_peaks": [
                torch.cuda.max_memory_allocated(i)
                for i in range(torch.cuda.device_count())
            ],
            "two_gpu_model_parallel": two_gpu,
            "loss": float(objective.detach()),
        }

    print("reference_forward", flush=True)
    reference = []
    # Independent backwards avoid retaining three reference activation graphs.
    for index, row in enumerate(requests):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            hidden = base.model(
                input_ids=torch.tensor([row], device="cuda"), use_cache=False
            ).last_hidden_state[:, -1, :]
            with torch.autocast("cuda", enabled=not fp32_head, dtype=torch.bfloat16):
                h = hidden.to(base.lm_head.weight.device)
                logits = torch.nn.functional.linear(
                    h.float() if fp32_head else h,
                    base.lm_head.weight[[15, 16]],
                )
            term = torch.nn.functional.binary_cross_entropy_with_logits(
                logits[0, 1].float() - logits[0, 0].float(),
                targets[index].to(logits.device),
            ) * weights[index].to(logits.device)
        reference.append(logits.detach()[0])
        term.backward()
    grads = {n: p.grad.detach().float().cpu().clone() for n, p in params.items()}
    model.zero_grad(set_to_none=True)
    print("branched_forward", flush=True)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        actual = branched_decision_logits(
            model,
            plan,
            [15, 16],
            checkpoint_segments=checkpoint_segments,
            fp32_head=fp32_head,
            independent_endpoint=independent_endpoint,
        )
        objective = loss(actual)
    objective.backward()
    expected = torch.stack(reference)
    difference, norm, dot, actual_norm = 0.0, 0.0, 0.0, 0.0
    for name, p in params.items():
        if p.grad is None or not torch.isfinite(p.grad).all():
            raise ValueError(f"invalid gradient: {name}")
        a, b = p.grad.detach().float().cpu(), grads[name]
        difference += float((a - b).square().sum())
        norm += float(b.square().sum())
        dot += float((a * b).sum())
        actual_norm += float(a.square().sum())
    margins = actual[:, 1].detach().float() - actual[:, 0].detach().float()
    ref_margins = expected[:, 1].float() - expected[:, 0].float()
    result = {
        "max_margin_error": float((margins - ref_margins).abs().max()),
        "max_probability_error": float(
            (margins.sigmoid() - ref_margins.sigmoid()).abs().max()
        ),
        "gradient_relative_l2": (difference / max(norm, 1e-30)) ** 0.5,
        "gradient_cosine": dot / max((norm * actual_norm) ** 0.5, 1e-30),
        "reference_gradient_norm": norm**0.5,
        "loss": float(objective.detach()),
        "kernels": sorted(set(kernels)),
        "independent_tokens": sum(map(len, requests)),
        "branched_tokens": work_tokens,
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "checkpoint_segments": checkpoint_segments,
        "two_gpu_model_parallel": two_gpu,
        "alignment": alignment,
        "fp32_head": fp32_head,
        "independent_endpoint": independent_endpoint,
    }
    result["passed"] = (
        result["max_margin_error"] <= 0.1
        and result["max_probability_error"] <= 0.02
        and result["gradient_relative_l2"] <= 0.05
        and result["gradient_cosine"] >= 0.995
        and norm > 0
    )
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint-segments", action="store_true")
    parser.add_argument("--parent", type=int)
    parser.add_argument("--stress-only", action="store_true")
    parser.add_argument("--two-gpu", action="store_true")
    parser.add_argument("--alignment", type=int, choices=(1, 64), default=1)
    parser.add_argument("--fp32-head", action="store_true")
    parser.add_argument("--independent-endpoint", action="store_true")
    args = parser.parse_args()
    result = run(
        checkpoint_segments=args.checkpoint_segments,
        parent=args.parent,
        stress_only=args.stress_only,
        two_gpu=args.two_gpu,
        alignment=args.alignment,
        fp32_head=args.fp32_head,
        independent_endpoint=args.independent_endpoint,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)
    if not result["passed"]:
        raise SystemExit("production branching parity failed; training gated off")
