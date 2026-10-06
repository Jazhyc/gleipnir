"""Structural safety and exact FP32-to-BF16 merge arithmetic on tiny shards."""

import json

import pytest
import torch
from safetensors.torch import load_file, save_file

from gleipnir.merged_lora import file_sha256, lora_pairs, merge_checkpoint, merge_weight


def adapter_config():
    return {"peft_type": "LORA", "bias": "none", "r": 2, "lora_alpha": 4}


def test_merge_matches_fp32_peft_safe_merge_and_preserves_master():
    from peft import LoraConfig, get_peft_model

    torch.manual_seed(3)
    base = torch.nn.Sequential(torch.nn.Linear(3, 4, bias=False)).to(torch.bfloat16)
    model = get_peft_model(base, LoraConfig(r=2, lora_alpha=4, target_modules=["0"]))
    layer = model.base_model.model[0]
    a = layer.lora_A["default"].weight
    b = layer.lora_B["default"].weight
    a.data = a.float().data
    b.data = torch.randn_like(b.float()) * 0.1
    original = layer.base_layer.weight.detach().clone()
    master_a, master_b = a.detach().clone(), b.detach().clone()
    observed = merge_weight(original, master_a, master_b, rank=2, alpha=4)
    # PEFT casts the delta to the base dtype before addition. Merge in FP32
    # to independently check our single-rounding arithmetic, then export BF16.
    model.float()
    expected = model.merge_and_unload(safe_merge=True).to(torch.bfloat16)[0].weight
    assert torch.equal(observed, expected)
    assert torch.equal(master_a, a) and torch.equal(master_b, b)


def test_shard_merge_changes_only_target_and_keeps_source(tmp_path):
    base, adapter, destination = (
        tmp_path / name for name in ["base", "adapter", "merged"]
    )
    base.mkdir()
    adapter.mkdir()
    key = "model.language_model.layer.weight"
    weight = torch.ones(4, 3, dtype=torch.bfloat16)
    untouched = torch.tensor([0.123], dtype=torch.float32)
    save_file({key: weight, "norm": untouched}, base / "model-1.safetensors")
    (base / "model.safetensors.index.json").write_text(
        json.dumps(
            {"weight_map": {key: "model-1.safetensors", "norm": "model-1.safetensors"}}
        )
    )
    (base / "config.json").write_text("{}")
    prefix = "base_model.model.model.language_model.layer"
    a = torch.full((2, 3), 0.1)
    b = torch.full((4, 2), 0.2)
    save_file(
        {prefix + ".lora_A.weight": a, prefix + ".lora_B.weight": b},
        adapter / "adapter_model.safetensors",
    )
    (adapter / "adapter_config.json").write_text(json.dumps(adapter_config()))
    before = file_sha256(base / "model-1.safetensors")
    manifest = merge_checkpoint(
        base,
        adapter,
        destination,
        expected_adapter_sha256=file_sha256(adapter / "adapter_model.safetensors"),
        model_id="test",
        revision="fixed",
    )
    merged = load_file(destination / "model-1.safetensors")
    assert torch.equal(merged[key], (weight.float() + 2 * b @ a).to(torch.bfloat16))
    assert torch.equal(merged["norm"], untouched)
    assert file_sha256(base / "model-1.safetensors") == before
    assert (
        manifest["merged_projection_count"] == manifest["changed_projection_count"] == 1
    )


def test_unsupported_layout_and_nonfinite_updates_rejected():
    prefix = "base_model.model.layer"
    keys = [prefix + ".lora_A.weight", prefix + ".lora_B.weight"]
    assert list(lora_pairs(keys, adapter_config())) == ["layer.weight"]
    with pytest.raises(ValueError, match="unsupported merge variant"):
        lora_pairs(keys, {**adapter_config(), "use_dora": True})
    with pytest.raises(ValueError, match="missing LoRA B"):
        lora_pairs(keys[:1], adapter_config())
    with pytest.raises(ValueError, match="nonfinite"):
        merge_weight(
            torch.ones(4, 3, dtype=torch.bfloat16),
            torch.full((2, 3), float("nan")),
            torch.ones(4, 2),
            rank=2,
            alpha=4,
        )


def test_merged_server_omits_lora_and_retains_engine_settings():
    import yaml

    from experiments.b200_inference_benchmark.run import EXPERIMENT, server_command

    config = yaml.safe_load((EXPERIMENT / "config.yaml").read_text())
    from pathlib import Path

    merged = server_command(config, Path("/tmp/merged"))
    assert merged[merged.index("--model") + 1] == "/tmp/merged"
    assert merged[merged.index("--served-model-name") + 1] == "monitor"
    assert not any("lora" in arg for arg in merged)
    original = server_command(config)
    for flag in (
        "--max-model-len",
        "--max-num-seqs",
        "--max-num-batched-tokens",
        "--gpu-memory-utilization",
        "--gdn-prefill-backend",
        "--seed",
    ):
        assert merged[merged.index(flag) + 1] == original[original.index(flag) + 1]


def test_pairing_rejects_prompt_drift_and_reports_baseline_instability():
    from gleipnir.inference_benchmark import paired_score_summary

    def row(score, prompt="fixed"):
        return {"id": "a", "prompt_sha256": prompt, "score": score, "margin": score}

    result = paired_score_summary([[row(0.48)], [row(0.51)]], [[row(0.52)]])
    assert result["threshold_flips"] == 1
    assert result["baseline_threshold_unstable_ids"] == ["a"]
    with pytest.raises(ValueError, match="prompt identity drift"):
        paired_score_summary([[row(0.5)]], [[row(0.5, "changed")]])
