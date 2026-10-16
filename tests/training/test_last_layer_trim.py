import pytest
import torch
from transformers import Qwen3_5ForCausalLM
from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig

from gleipnir.training.last_layer_trim import (
    LastLayerTrim,
    gradient_parity,
    trimmed_layer_forward,
)
from gleipnir.training.packing import (
    forward_packed_monitoring_logits,
    installed_segmented_sdpa,
)

LENGTHS = (5, 17, 1, 9)


def tiny_model() -> Qwen3_5ForCausalLM:
    torch.manual_seed(0)
    config = Qwen3_5TextConfig(
        vocab_size=64,
        hidden_size=64,
        intermediate_size=96,
        num_hidden_layers=3,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        layer_types=["full_attention"] * 3,
        max_position_embeddings=128,
    )
    config._attn_implementation = "sdpa"
    model = Qwen3_5ForCausalLM(config).float()
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.add_(torch.randn_like(parameter) * 0.02)
    return model


def packed_forward(model, ids):
    return forward_packed_monitoring_logits(model, ids, LENGTHS)[0]


def test_trim_matches_full_layer_logits_and_gradients() -> None:
    model = tiny_model()
    ids = torch.randint(0, 64, (1, sum(LENGTHS)))
    targets = torch.tensor([0.9, 0.1, 0.6, 0.3])
    with installed_segmented_sdpa():
        trim = LastLayerTrim(model)
        logits = {}

        def loss_forward():
            logits["value"] = packed_forward(model, ids)[:, :2]
            margin = logits["value"][:, 1] - logits["value"][:, 0]
            return torch.nn.functional.binary_cross_entropy_with_logits(margin, targets)

        result = gradient_parity(
            model, trim, loss_forward, capture=lambda: logits["value"]
        )
    assert trim.calls == 1
    assert result["finite"]
    assert result["final_layer_parameters_without_gradient"] == []
    assert result["noise_floor"]["gradient_relative_l2"] == 0
    parity = result["trim_vs_reference"]
    assert parity["logit_max_abs"] < 1e-5
    assert parity["loss_abs"] < 1e-6
    assert parity["gradient_relative_l2"] < 1e-5
    assert parity["worst_parameter_relative_l2"] < 1e-4


def test_trim_falls_back_when_unpacked_and_rejects_double_install() -> None:
    model = tiny_model()
    LastLayerTrim(model)
    with pytest.raises(ValueError, match="already installed"):
        LastLayerTrim(model)
    ids = torch.randint(0, 64, (1, 8))
    trim = model.model.layers[-1]._gleipnir_last_layer_trim
    trim.enabled = False
    expected = model(input_ids=ids, use_cache=False).logits
    trim.enabled = True
    # Unpacked calls fall back to the exact untrimmed layer.
    assert torch.equal(model(input_ids=ids, use_cache=False).logits, expected)
    assert trim.unpacked_calls == 1 and trim.calls == 0


def test_disabled_trim_is_the_original_forward() -> None:
    model = tiny_model()
    ids = torch.randint(0, 64, (1, sum(LENGTHS)))
    with installed_segmented_sdpa():
        before = packed_forward(model, ids)
        trim = LastLayerTrim(model)
        trim.enabled = False
        after = packed_forward(model, ids)
    assert torch.equal(before, after) and trim.calls == 0


def test_campaign_command_and_validation_bind_the_trim() -> None:
    from pathlib import Path

    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf

    from gleipnir.campaigns.training_command import training_command
    from gleipnir.training.validation import validate_training_metadata

    root = Path(__file__).resolve().parents[2]
    with initialize_config_dir(
        config_dir=str(root / "src/gleipnir/configs/systems_screen"),
        version_base=None,
    ):
        recipe = OmegaConf.to_container(
            compose(config_name="qwen35_4b_b200_packed_sdpa"), resolve=True
        )["recipe"]
    job = {
        **recipe,
        "job_name": "trim-test",
        "output_dir": "results/trim-test",
        "student_rows": "data/test.jsonl",
        "soft_targets": "data/soft.jsonl",
        "causal_adapter_dir": "results/trim-test/causal_adapter",
        "max_steps": -1,
        "last_layer_trim": True,
    }
    assert "++student.training.last_layer_trim=true" in training_command(job)
    assert not any(
        "last_layer_trim" in a
        for a in training_command({**job, "last_layer_trim": False})
    )
    metadata = {
        "training_state": {"global_step": 1},
        "sequence_packing": {
            "initial_master_sha256": "a",
            "final_master_sha256": "b",
            **{
                k: {"passed": True}
                for k in ("eager_canary", "compiled_canary", "preflight")
            },
        },
        "losses": {
            "soft_weight": 1.0,
            "direct_weight": 0.0,
            "completion_weight": 0,
            "accumulation_policy": "sum_per_example_over_logical_batch_v1",
        },
        "gradient_checkpointing": False,
        "quantization": {"enabled": False},
        "direct_logits_mode": "selected_positions",
        "optimization": {"learning_rate": 5e-5},
        "gated_delta_backend": {"backend": "flashqla"},
        "last_layer_trim": None,
    }
    trim_job = {
        "soft_loss_weight": 1.0,
        "direct_loss_weight": 0.0,
        "learning_rate": 5e-5,
        "last_layer_trim": True,
    }
    profile = {"metadata_expectations": {}}
    with pytest.raises(ValueError, match="last-layer trim"):
        validate_training_metadata(
            metadata, trim_job, "a", expected_steps=1, profile=profile
        )
    metadata["last_layer_trim"] = {
        "canary": {"passed": True},
        "training": {"trimmed_calls": 10},
    }
    validate_training_metadata(
        metadata, trim_job, "a", expected_steps=1, profile=profile
    )


def test_compiled_trim_matches_the_eager_trim() -> None:
    model = tiny_model()
    ids = torch.randint(0, 64, (1, sum(LENGTHS)))
    with installed_segmented_sdpa():
        trim = LastLayerTrim(
            model,
            compile_function=lambda f: torch.compile(f, backend="eager", dynamic=True),
        )
        compiled = packed_forward(model, ids)
        trim.layer.forward = lambda *a, **k: trimmed_layer_forward(trim.layer, *a, **k)
        eager = packed_forward(model, ids)
    assert trim.metadata()["compiled"] and trim.calls == 1
    assert torch.allclose(compiled, eager, atol=1e-6)


def test_packed_forward_uses_host_cuts_without_device_readback() -> None:
    from gleipnir.training import packing

    model = tiny_model()
    ids = torch.randint(0, 64, (1, sum(LENGTHS)))
    with installed_segmented_sdpa():
        packing.HOST_CUT_FALLBACKS["count"] = 0
        packed_forward(model, ids)
        assert packing.HOST_CUT_FALLBACKS["count"] == 0
        layout = packing.PackedSequenceLayout(LENGTHS)
        kwargs = layout.kernel_kwargs()
        assert packing.host_cuts(kwargs["cu_seq_lens_q"]) == list(layout.offsets)
        assert packing.HOST_CUT_FALLBACKS["count"] == 1


def test_host_built_kernel_kwargs_match_the_device_formulas() -> None:
    from gleipnir.training.packing import PackedSequenceLayout

    layout = PackedSequenceLayout(LENGTHS)
    kwargs = layout.kernel_kwargs()
    expected_positions = torch.cat([torch.arange(n) for n in LENGTHS]).unsqueeze(0)
    expected_ids = torch.repeat_interleave(
        torch.arange(len(LENGTHS), dtype=torch.int32), torch.tensor(LENGTHS)
    ).unsqueeze(0)
    assert torch.equal(kwargs["position_ids"], expected_positions)
    assert torch.equal(kwargs["seq_idx"], expected_ids)
    assert kwargs["cu_seq_lens_q"].tolist() == list(layout.offsets)
    assert kwargs["cu_seq_lens_q"].dtype == torch.int32
    assert layout.decision_positions().tolist() == [o - 1 for o in layout.offsets[1:]]


def test_packed_collation_flags_soft_target_validity_on_the_host() -> None:
    from gleipnir.training.packing import collate_packed_monitoring

    good = {"direct_input_ids": [1, 2], "binary_label": 1, "dataset_id": 0}
    assert collate_packed_monitoring([{**good, "soft_target": 0.5}])[
        "soft_targets_host_valid"
    ]
    for bad in (0.0, 1.0, float("nan")):
        batch = collate_packed_monitoring([{**good, "soft_target": bad}])
        assert batch["soft_targets_host_valid"] is False
