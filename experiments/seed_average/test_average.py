import json

import pytest
import torch

from experiments.seed_average.average import averaged_config, build, stack


def adapters(count: int = 3, rank: int = 4) -> list[dict[str, torch.Tensor]]:
    torch.manual_seed(0)
    return [
        {
            "m.lora_A.weight": torch.randn(rank, 8),
            "m.lora_B.weight": torch.randn(6, rank),
        }
        for _ in range(count)
    ]


def test_stacked_update_is_the_exact_average() -> None:
    parts = adapters()
    stacked = stack(parts)
    scale = 2.0
    expected = sum(scale * p["m.lora_B.weight"] @ p["m.lora_A.weight"] for p in parts)
    expected = expected / len(parts)
    actual = scale * stacked["m.lora_B.weight"] @ stacked["m.lora_A.weight"]
    assert stacked["m.lora_A.weight"].shape == (12, 8)
    assert torch.allclose(actual, expected, atol=1e-5)


def test_config_keeps_the_scaling() -> None:
    config = averaged_config(
        [{"r": 128, "lora_alpha": 256, "target_modules": ["q_proj"]}] * 3
    )
    assert (config["r"], config["lora_alpha"]) == (384, 768)
    with pytest.raises(ValueError):
        averaged_config([{"r": 128, "lora_alpha": 256, "use_dora": True}] * 3)
    with pytest.raises(ValueError):
        averaged_config([{"r": 128, "lora_alpha": 256}, {"r": 64, "lora_alpha": 256}])


def test_non_lora_tensors_are_rejected() -> None:
    parts = adapters()
    for p in parts:
        p["head.weight"] = torch.zeros(2)
    with pytest.raises(ValueError):
        stack(parts)


def test_build_writes_an_exact_adapter(tmp_path) -> None:
    from safetensors.torch import save_file

    sources = []
    for i, part in enumerate(adapters()):
        source = tmp_path / f"s{i}"
        source.mkdir()
        save_file(part, source / "adapter_model.safetensors")
        config = {"r": 4, "lora_alpha": 8, "target_modules": ["m"]}
        (source / "adapter_config.json").write_text(json.dumps(config))
        (source / "training_metadata.json").write_text("{}")
        sources.append(source)
    receipt = build(sources, tmp_path / "out")
    assert receipt["r"] == 12 and receipt["max_relative_update_error"] < 1e-6
    assert not (tmp_path / "out/training_metadata.json").exists()
    with pytest.raises(ValueError):
        build(sources, tmp_path / "out")


def test_release_rule_uses_the_seed_mean() -> None:
    from experiments.seed_average.compare import decide

    seeds = {"seed0": 0.8959, "seed1": 0.8978, "seed2": 0.9142}
    assert decide(0.9030, seeds)["release_candidate"] == "average"
    assert decide(0.9000, seeds)["release_candidate"] == "seed2"
