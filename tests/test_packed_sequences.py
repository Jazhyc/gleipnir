"""Check the boundary contract and causal isolation, including negative controls."""

import pytest
import torch

from experiments.monitoring_sequence_packing import audit_cpu
from experiments.monitoring_sequence_packing.audit_cpu import run_audit
from gleipnir.packed_sequences import (
    PackedSequenceLayout,
    collate_packed_monitoring,
    forward_packed_monitoring_logits,
    installed_segmented_sdpa,
    packed_partition,
    segmented_sdpa_interface,
)


@pytest.mark.parametrize("lengths", [(), (0,), (-1, 2), (True,), (1.5,), (2**31,)])
def test_layout_rejects_invalid_lengths(lengths):
    with pytest.raises(ValueError):
        PackedSequenceLayout(lengths)


def test_boundaries_share_one_source_and_preserve_decisions():
    layout = PackedSequenceLayout((3, 1, 2))
    kwargs = layout.kernel_kwargs()
    assert kwargs["position_ids"].tolist() == [[0, 1, 2, 0, 0, 1]]
    assert kwargs["seq_idx"].tolist() == [[0, 0, 0, 1, 2, 2]]
    assert kwargs["seq_idx"].dtype == torch.int32
    assert kwargs["cu_seq_lens_q"].tolist() == [0, 3, 4, 6]
    assert kwargs["cu_seq_lens_k"].dtype == torch.int32
    assert kwargs["max_length_q"] == kwargs["max_length_k"] == 3
    assert kwargs["use_cache"] is False
    assert "attention_mask" not in kwargs
    assert layout.decision_positions().tolist() == [2, 3, 5]


def test_boolean_mask_blocks_other_examples_and_future_tokens():
    mask = PackedSequenceLayout((2, 2)).dense_causal_mask()[0, 0]
    assert mask.tolist() == [
        [True, False, False, False],
        [True, True, False, False],
        [False, False, True, False],
        [False, False, True, True],
    ]


def test_dense_oracle_refuses_long_context_allocation():
    with pytest.raises(ValueError, match="token limit"):
        PackedSequenceLayout((1025, 1024)).dense_causal_mask()


@pytest.mark.parametrize("lengths", [[7, 5], [1, 3], [63, 65]])
def test_real_tiny_qwen_detects_missing_resets_and_mask_misuse(lengths):
    result = run_audit(
        dict(
            seed=0,
            lengths=lengths,
            forward_tolerance=1e-5,
            cross_gradient_tolerance=1e-8,
        )
    )
    assert result["passed"], result["cases"]


def test_audit_rejects_nonfinite_negative_controls(monkeypatch):
    monkeypatch.setattr(
        audit_cpu,
        "measure_case",
        lambda *args, **kwargs: {
            "packed_vs_independent_max": float("nan"),
            "prefix_perturbation_max": 0.0,
            "cross_example_input_gradient_max": 0.0,
            "within_example_input_gradient_max": 1.0,
        },
    )
    result = run_audit(
        dict(
            seed=0,
            lengths=[7, 5],
            forward_tolerance=1e-5,
            cross_gradient_tolerance=1e-8,
        )
    )
    assert not result["passed"]
    assert not any(case["finite"] for case in result["cases"].values())


def test_packing_keeps_complete_examples_and_every_target():
    features = [
        dict(
            direct_input_ids=list(range(n)),
            binary_label=i % 2,
            dataset_id=i,
            soft_target=i / 10,
        )
        for i, n in enumerate([11, 5, 3, 2])
    ]
    groups = packed_partition([len(f["direct_input_ids"]) for f in features], 8)
    assert groups == [[0], [1, 2], [3]]
    assert sorted(i for group in groups for i in group) == list(range(4))
    batch = collate_packed_monitoring([features[1], features[2]])
    assert batch["packed_lengths"] == (5, 3)
    assert batch["dataset_ids"].tolist() == [1, 2]
    assert batch["direct_input_ids"].shape == (1, 8)
    assert "direct_attention_mask" not in batch
    with pytest.raises(ValueError, match="direct binary"):
        collate_packed_monitoring([dict(features[0], input_ids=[1])])


def test_segmented_sdpa_matches_dense_mask_and_has_zero_cross_gradient():
    def original(module, query, key, value, mask, **kwargs):
        result = torch.nn.functional.scaled_dot_product_attention(
            query, key, value, attn_mask=mask, is_causal=kwargs.get("is_causal", False)
        )
        return result.transpose(1, 2), None

    torch.manual_seed(0)
    q, k, v = [torch.randn(1, 2, 8, 4, requires_grad=True) for _ in range(3)]
    layout = PackedSequenceLayout((3, 5))
    actual, _ = segmented_sdpa_interface(original)(
        None, q, k, v, None, **layout.kernel_kwargs()
    )
    expected, _ = original(None, q, k, v, layout.dense_causal_mask())
    torch.testing.assert_close(actual, expected)
    actual[:, 3:].sum().backward()
    assert all(t.grad[:, :, :3].abs().max() == 0 for t in [q, k, v])
    ordinary, _ = segmented_sdpa_interface(original)(
        None, q, k, v, layout.dense_causal_mask()
    )
    torch.testing.assert_close(ordinary, expected)


def test_real_qwen_packed_readout_preserves_every_example():
    from transformers import Qwen3_5ForCausalLM

    text = audit_cpu.tiny_model(0)
    model = Qwen3_5ForCausalLM(text.config).float().eval()
    ids = torch.arange(8)[None]
    with pytest.raises(ValueError, match="isolated SDPA"):
        forward_packed_monitoring_logits(model, ids, (3, 5))
    with audit_cpu.reference_boundaries(model.model, convolution=True, recurrence=True):
        with installed_segmented_sdpa(), torch.no_grad():
            logits, _ = forward_packed_monitoring_logits(model, ids, (3, 5))
            singletons = torch.cat(
                [
                    model(input_ids=ids[:, :3], logits_to_keep=1).logits[0],
                    model(input_ids=ids[:, 3:], logits_to_keep=1).logits[0],
                ]
            )
    assert logits.shape == (2, model.config.vocab_size)
    torch.testing.assert_close(logits, singletons, atol=1e-5, rtol=1e-5)


def test_packing_config_preserves_the_completed_bf16_recipe():
    from pathlib import Path

    import yaml

    from experiments.fp4_stability.run import validate_config

    root = Path(__file__).parents[1] / "experiments"
    baseline = yaml.safe_load(
        (root / "fp4_stability/bf16_flashqla_ten_step_comparison.yaml").read_text()
    )
    config = yaml.safe_load(
        (root / "monitoring_sequence_packing/bf16_gpu.yaml").read_text()
    )
    validate_config(config)
    assert {key for key in config if config[key] != baseline.get(key)} == {
        "output",
        "logs",
        "sequence_packing",
        "packing_only",
    }
    for override in [
        {"full_bf16_lora": False},
        {"gated_delta_backend": "fla"},
        {"flashqla_auto_cp": True},
    ]:
        with pytest.raises(ValueError, match="sequence packing requires"):
            validate_config(dict(config, **override))


def test_packing_learning_config_is_bounded_and_preserves_compiled_recipe():
    from pathlib import Path

    import yaml

    from experiments.fp4_stability.run import validate_config

    root = Path(__file__).parents[1] / "experiments/monitoring_sequence_packing"
    baseline = yaml.safe_load((root / "bf16_casts_gpu.yaml").read_text())
    config = yaml.safe_load((root / "bf16_learning_gpu.yaml").read_text())
    validate_config(config)
    assert {key for key in config if config[key] != baseline.get(key)} == {
        "output",
        "logs",
        "packing_learning_gradient_tolerance",
    }
    for override in [
        {"steps": 20},
        {"packing_learning_gradient_tolerance": 0.151},
        {"sequence_packing": False},
    ]:
        with pytest.raises(ValueError):
            validate_config(dict(config, **override))
    cached = yaml.safe_load((root / "bf16_learning_cached_gpu.yaml").read_text())
    validate_config(cached)
    assert {key for key in cached if cached[key] != config.get(key)} == {
        "output",
        "logs",
        "packing_compile_cache_limit",
    }
    with pytest.raises(ValueError):
        validate_config(dict(cached, packing_compile_cache_limit=129))
    no_checkpoint = yaml.safe_load((root / "bf16_no_checkpoint_gpu.yaml").read_text())
    validate_config(no_checkpoint)
    assert {key for key in no_checkpoint if no_checkpoint[key] != cached.get(key)} == {
        "output",
        "logs",
        "disable_gradient_checkpointing",
    }
    larger = yaml.safe_load((root / "bf16_larger_batch_gpu.yaml").read_text())
    validate_config(larger)
    assert {key for key in larger if larger[key] != cached.get(key)} == {
        "output",
        "logs",
        "adaptive_token_budget",
        "packing_only",
    }


@pytest.mark.parametrize("value", ["true", None, 1])
def test_packing_only_requires_boolean(value):
    from pathlib import Path

    import yaml

    from experiments.fp4_stability.run import validate_config

    config = yaml.safe_load(
        (
            Path(__file__).parents[1]
            / "experiments/monitoring_sequence_packing/bf16_larger_batch_gpu.yaml"
        ).read_text()
    )
    with pytest.raises(ValueError, match="packing_only"):
        validate_config(dict(config, packing_only=value))
