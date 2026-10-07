import pytest

from gleipnir.qwen35_loftq import causal_to_checkpoint_weight


def test_qwen35_causal_weights_map_to_native_checkpoint() -> None:
    assert causal_to_checkpoint_weight("model.layers.0.mlp.gate_proj.weight") == (
        "model.language_model.layers.0.mlp.gate_proj.weight"
    )
    assert causal_to_checkpoint_weight("lm_head.weight") == "lm_head.weight"
    with pytest.raises(ValueError, match="unsupported"):
        causal_to_checkpoint_weight("visual.blocks.0.weight")
