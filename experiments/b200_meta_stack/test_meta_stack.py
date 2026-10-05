"""Meaningful source-drift and explicit native variant identity checks."""

import pytest

from gleipnir.nvidia_mxfp8_meta_variants import (
    NativeVariant,
    probability_source,
    warp_amax_source,
)


@pytest.mark.parametrize(
    "dq,coordinate,names",
    [
        (True, "tTR_cDP", ("group_amax_0", "group_amax_1")),
        (False, "tTR_cVDO", ("group_amax", "group_amax_1")),
    ],
)
def test_warp_reduction_precedes_conversion(dq, coordinate, names):
    marker = f"                dS_row = cute.get({coordinate}[0], mode=[0])"
    source = "maxima\n" + marker + "\n                convert_and_store()\n"
    changed = warp_amax_source(source, dq=dq)
    for name in names:
        assert changed.index(f"warp_redux_sync({name}") < changed.index(marker)
    assert changed.endswith("\n                convert_and_store()\n")
    with pytest.raises(ValueError, match="changed"):
        warp_amax_source(changed.replace(marker, ""), dq=dq)
    with pytest.raises(ValueError, match="ambiguous"):
        warp_amax_source(source + source, dq=dq)


def test_native_variants_are_distinct_cache_keys_and_reject_invalid_width():
    assert (
        len(
            {
                NativeVariant(),
                NativeVariant(persistent_dq=True),
                NativeVariant(ds_warp_amax=True),
                NativeVariant(dq_store_bits=128),
            }
        )
        == 4
    )
    with pytest.raises(ValueError, match="width"):
        NativeVariant(dq_store_bits=64)


def test_probability_transform_keeps_unscaled_reductions_and_matches_sf():
    source = "CFG, _TMA = make_cfg_d256_mxfp8(PARAMS)\nSF_CONST_VALUE = 0x7F\n"
    source += "\n".join(
        f"store(chunk_P_{i}.to(STORAGE_DTYPE))\nreduce(chunk_P_{i})" for i in range(16)
    )
    changed = probability_source(source, 8)
    assert "SF_CONST_VALUE = 119" in changed
    assert "RESCALE_THRESHOLD=0.0" in changed
    assert changed.count("* cutlass.Float32(256.0)") == 16
    assert all(f"reduce(chunk_P_{i})" in changed for i in range(16))
    with pytest.raises(ValueError, match="conversion source changed"):
        probability_source(source.replace("chunk_P_15.to", "other.to"), 8)
    with pytest.raises(ValueError, match="expects"):
        probability_source(source, 9)


def test_frozen_projection_weight_quantization_has_exact_power_of_two_scales():
    import torch

    from gleipnir.nvidia_mxfp8_projection_pilot import quantize_weight

    weight = torch.zeros(3, 64, dtype=torch.bfloat16)
    weight[1, :32] = 448
    weight[1, 32:] = 896
    weight[2] = 0.5
    payload, scale = quantize_weight(weight)
    assert scale[1].tolist() == [127, 128]
    assert payload[1].float().tolist() == [448.0] * 64
    assert torch.isfinite(payload.float()).all()
    recovered = (
        payload.float().reshape(3, 2, 32) * (2.0 ** (scale.float() - 127))[..., None]
    )
    assert torch.equal(recovered.reshape(weight.shape), weight.float())
    with pytest.raises(ValueError, match="divisible"):
        quantize_weight(torch.ones(2, 33, dtype=torch.bfloat16))


def test_qwen_norm_rotary_reference_preserves_passthrough_and_bf16_boundaries():
    import torch

    from experiments.b200_meta_stack.producer_reference import norm_rope

    x = torch.randn(3, 4, 256, dtype=torch.bfloat16, requires_grad=True)
    w = torch.zeros(256, dtype=torch.bfloat16)
    cos = torch.ones(3, 64, dtype=torch.bfloat16)
    sin = torch.zeros_like(cos)
    normalized = (
        x.float() * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + 1e-6)
    ).bfloat16()
    actual = norm_rope(x, w, cos, sin)
    assert torch.equal(actual, normalized)
    actual.float().square().sum().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()
    cos.zero_()
    sin.fill_(1)
    rotated = norm_rope(x, w, cos, sin)
    assert torch.equal(rotated[..., :32], -normalized[..., 32:64])
    assert torch.equal(rotated[..., 32:64], normalized[..., :32])
    assert torch.equal(rotated[..., 64:], normalized[..., 64:])


def test_meta_options_reject_misspellings_and_implicit_boolean_precision():
    from gleipnir.nvidia_mxfp8_meta_training import validate_options

    with pytest.raises(ValueError, match="unknown"):
        validate_options({"norm_rop": True})
    with pytest.raises(ValueError, match="explicit boolean"):
        validate_options({"norm_rope": "true"})
    with pytest.raises(ValueError, match="must be boolean"):
        validate_options({"persistent_dq": 1})
    norm, variant = validate_options({"norm_rope": True, "dq_store_bits": 128})
    assert norm and variant == NativeVariant(dq_store_bits=128)


def test_meta_receipt_binding_rejects_wrong_combination_and_source_drift(
    tmp_path, monkeypatch
):
    import hashlib
    import json
    from dataclasses import asdict

    from experiments.b200_meta_stack import training_screen as module

    monkeypatch.setattr(module, "ROOT", tmp_path)
    monkeypatch.setattr(
        module, "accept_native", lambda native, square: {"execution_correct": True}
    )
    spec = asdict(NativeVariant(dq_store_bits=128))
    receipts = {
        "variant": {"status": "complete", "spec": spec, "canary_sha256": "nativehash"},
        "producer": {
            "status": "complete",
            "spec": spec,
            "producer_checks": [
                {
                    "heads": h,
                    "code_agreement": 1.0,
                    "native_scales_bitexact": [True, True, True],
                    "backward_relative_l2": 0.001,
                }
                for h in (16, 4)
            ],
            "attention_comparison": {
                "finite": True,
                "forward_relative_l2": 0.0,
                "gradient_relative_l2": [0.001, 0.001, 0.0],
            },
            "isolation": {"passed": True},
            "graph_replay": {"passed": True},
        },
    }
    config = {
        "meta_attention_options": {"norm_rope": True, "dq_store_bits": 128},
        "native_reference_sha256": "nativehash",
    }
    for name, receipt in receipts.items():
        path = tmp_path / (name + ".json")
        path.write_text(json.dumps(receipt))
        config[name + "_reference"] = path.name
        config[name + "_reference_sha256"] = hashlib.sha256(
            path.read_bytes()
        ).hexdigest()
    assert module.accept_meta({}, config)["selected_options"]["dq_store_bits"] == 128
    config["meta_attention_options"]["dq_store_bits"] = 16
    with pytest.raises(ValueError, match="selected variant"):
        module.accept_meta({}, config)
    config["meta_attention_options"]["dq_store_bits"] = 128
    (tmp_path / "producer.json").write_text("{}")
    with pytest.raises(ValueError, match="checksum drift"):
        module.accept_meta({}, config)


def test_meta_scope_preserves_unpacked_reference_and_restores_on_exception(monkeypatch):
    from contextlib import nullcontext

    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5Attention

    from gleipnir import nvidia_mxfp8_meta_training as module

    monkeypatch.setattr(module, "native_variant", lambda variant: nullcontext())
    sentinel = object()

    def original(*args, **kwargs):
        return sentinel

    monkeypatch.setattr(Qwen3_5Attention, "forward", original)
    dummy = object()
    with pytest.raises(RuntimeError, match="exit"):
        with module.meta_attention_runtime({"norm_rope": True}):
            assert Qwen3_5Attention.forward(dummy, None, None, None) is sentinel
            raise RuntimeError("exit")
    assert Qwen3_5Attention.forward is original
