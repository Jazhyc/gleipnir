"""Local random-weight CPU oracle; never loads a checkpoint or touches a GPU."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from contextlib import contextmanager
from pathlib import Path

import torch
import torch.nn.functional as F
import transformers
import yaml
from transformers import Qwen3_5TextConfig, Qwen3_5TextModel
from transformers.models.qwen3_5 import modeling_qwen3_5 as qwen

from gleipnir.packed_sequences import PackedSequenceLayout


def tiny_model(seed: int) -> Qwen3_5TextModel:
    """Use one recurrent and one full-attention layer with the installed code."""
    torch.manual_seed(seed)
    config = Qwen3_5TextConfig(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=16,
        linear_num_key_heads=2,
        linear_num_value_heads=2,
        linear_key_head_dim=8,
        linear_value_head_dim=8,
        layer_types=["linear_attention", "full_attention"],
        max_position_embeddings=256,
        rope_parameters={
            "rope_type": "default",
            "rope_theta": 10000.0,
            "partial_rotary_factor": 1.0,
            "mrope_section": [2, 3, 3],
        },
        use_cache=False,
    )
    config._attn_implementation = "sdpa"
    return Qwen3_5TextModel(config).float().cpu().eval()


@contextmanager
def reference_boundaries(model, *, convolution: bool, recurrence: bool):
    """Segment only CPU reference operators; restore bindings even on failure."""
    module = model.layers[0].linear_attn
    originals = module.causal_conv1d_fn, module.chunk_gated_delta_rule

    def convolve(x, weight, bias=None, activation=None, seq_idx=None):
        if activation != "silu":
            raise ValueError("the diagnostic expects Qwen's SiLU convolution")
        cuts = [0, x.shape[-1]]
        if convolution and seq_idx is not None:
            cuts = [
                0,
                *(
                    torch.nonzero(seq_idx[0, 1:] != seq_idx[0, :-1]).flatten() + 1
                ).tolist(),
                x.shape[-1],
            ]
        outputs = []
        for start, end in zip(cuts[:-1], cuts[1:], strict=True):
            output = F.conv1d(
                x[..., start:end],
                weight.unsqueeze(1),
                bias=bias,
                padding=weight.shape[-1] - 1,
                groups=x.shape[1],
            )[..., : end - start]
            outputs.append(F.silu(output))
        return torch.cat(outputs, dim=-1)

    def scan(query, key, value, g, beta, cu_seqlens=None, **kwargs):
        cuts = [0, query.shape[1]]
        if recurrence and cu_seqlens is not None:
            cuts = cu_seqlens.tolist()
        if kwargs.get("initial_state") is not None or kwargs.get("output_final_state"):
            raise ValueError("packed CPU oracle forbids persistent states")
        outputs = []
        for start, end in zip(cuts[:-1], cuts[1:], strict=True):
            output, _ = qwen.torch_chunk_gated_delta_rule(
                query[:, start:end],
                key[:, start:end],
                value[:, start:end],
                g[:, start:end],
                beta[:, start:end],
                **kwargs,
            )
            outputs.append(output)
        return torch.cat(outputs, dim=1), None

    try:
        module.causal_conv1d_fn = convolve
        module.chunk_gated_delta_rule = scan
        yield
    finally:
        module.causal_conv1d_fn, module.chunk_gated_delta_rule = originals


def measure_case(model, layout: PackedSequenceLayout, *, mask_mode: str) -> dict:
    """Measure A->B dependence in hidden outputs and the input Jacobian."""
    tokens = (torch.arange(layout.total_tokens) % 63 + 1).unsqueeze(0)
    split = layout.lengths[0]
    kwargs = layout.kernel_kwargs()
    if mask_mode == "dense":
        kwargs["attention_mask"] = layout.sdpa_mask_mapping()
    elif mask_mode == "all_ones":
        kwargs["attention_mask"] = torch.ones_like(tokens)
    elif mask_mode != "automatic":
        raise ValueError("unknown mask mode")

    with torch.no_grad():
        packed = model(input_ids=tokens, **kwargs).last_hidden_state
        changed = tokens.clone()
        changed[:, :split] = (changed[:, :split] + 17) % 63 + 1
        perturbed = model(input_ids=changed, **kwargs).last_hidden_state
        independent = model(
            input_ids=tokens[:, split:], use_cache=False
        ).last_hidden_state
    embeddings = model.embed_tokens(tokens).detach().requires_grad_(True)
    hidden = model(inputs_embeds=embeddings, **kwargs).last_hidden_state
    # An asymmetric readout avoids cancellations from RMSNorm or a plain sum.
    weights = torch.linspace(-1, 1, hidden.shape[-1])
    gradient = torch.autograd.grad((hidden[0, -1] * weights).sum(), embeddings)[0]
    return {
        "packed_vs_independent_max": float(
            (packed[:, split:] - independent).abs().max()
        ),
        "prefix_perturbation_max": float(
            (packed[:, split:] - perturbed[:, split:]).abs().max()
        ),
        "cross_example_input_gradient_max": float(gradient[:, :split].abs().max()),
        "within_example_input_gradient_max": float(gradient[:, split:].abs().max()),
    }


def run_audit(config: dict) -> dict:
    """Record negative controls and two isolated-mask implementations."""
    torch.set_num_threads(1)
    layout = PackedSequenceLayout(tuple(config["lengths"]))
    if len(layout.lengths) != 2 or layout.total_tokens > 128:
        raise ValueError("CPU audit requires two sequences totaling at most 128 tokens")
    model = tiny_model(config["seed"])
    cases = {}
    policies = {
        "torch_fallbacks_with_boundaries": (False, False, "automatic"),
        "only_convolution_reset": (True, False, "dense"),
        "only_recurrence_reset": (False, True, "dense"),
        "all_ones_mask_with_kernel_resets": (True, True, "all_ones"),
        "isolated_automatic_mask": (True, True, "automatic"),
        "isolated_explicit_mask": (True, True, "dense"),
    }
    for name, (convolution, recurrence, mask_mode) in policies.items():
        with reference_boundaries(
            model, convolution=convolution, recurrence=recurrence
        ):
            record = measure_case(model, layout, mask_mode=mask_mode)
        record["finite"] = all(math.isfinite(value) for value in record.values())
        record["isolated"] = (
            record["packed_vs_independent_max"] <= config["forward_tolerance"]
            and record["prefix_perturbation_max"] <= config["forward_tolerance"]
            and record["cross_example_input_gradient_max"]
            <= config["cross_gradient_tolerance"]
            and record["within_example_input_gradient_max"] > 0
        )
        cases[name] = record
    passed = all(
        cases[name]["finite"]
        and cases[name]["isolated"] == name.startswith("isolated_")
        for name in cases
    )
    source = Path(qwen.__file__)
    return {
        "passed": passed,
        "scope": "random_weight_float32_cpu_reference_only",
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "model_source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "config": config,
        "cases": cases,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=Path(__file__).with_name("config.yaml")
    )
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text())
    result = run_audit(config)
    output = Path(config["output"])
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)
    if not result["passed"]:
        raise SystemExit("CPU isolation audit failed")


if __name__ == "__main__":
    main()
