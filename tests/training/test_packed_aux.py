"""Auxiliary continuation targets on the packed monitoring path."""

import pytest
import torch

from experiments.monitoring_sequence_packing import audit_cpu
from gleipnir.packed_sequences import (
    aux_readout_positions,
    collate_packed_monitoring,
    forward_packed_monitoring_logits,
    forward_packed_monitoring_logits_with_aux,
    installed_segmented_sdpa,
)


def feature(n: int, aux: int, offset: int) -> dict:
    row = dict(
        direct_input_ids=list(range(n)), binary_label=1, dataset_id=0, soft_target=0.7
    )
    if aux:
        row.update(
            aux_length=aux,
            aux_top_ids=[[offset + t, offset + t + 1] for t in range(aux)],
            aux_top_logprobs=[[-0.5, -1.5] for _ in range(aux)],
            aux_mask=[t % 2 == 0 for t in range(aux)],
        )
    return row


def test_readout_positions_count_from_each_row_end() -> None:
    decisions, continuation = aux_readout_positions((5, 4), (2, 1))
    # Row 0 occupies [0, 5): decision at 2, continuation at 3, 4.
    # Row 1 occupies [5, 9): decision at 7, continuation at 8.
    assert decisions.tolist() == [2, 7]
    assert continuation.tolist() == [3, 4, 8]
    plain, empty = aux_readout_positions((3, 2), (0, 0))
    assert plain.tolist() == [2, 4] and empty.numel() == 0
    with pytest.raises(ValueError, match="no decision prompt"):
        aux_readout_positions((2,), (2,))


def test_collate_concatenates_aux_targets_in_pack_order() -> None:
    batch = collate_packed_monitoring([feature(5, 2, 10), feature(4, 1, 20)])
    assert batch["aux_lengths"] == (2, 1)
    assert batch["aux_top_ids"].tolist() == [[10, 11], [11, 12], [20, 21]]
    assert batch["aux_mask"].tolist() == [True, False, True]
    assert batch["aux_top_logprobs"].dtype == torch.float32
    assert batch["aux_targets_host_valid"] is True
    plain = collate_packed_monitoring([feature(5, 0, 0)])
    assert not any(key.startswith("aux") for key in plain)
    with pytest.raises(ValueError, match="mixed missing auxiliary"):
        collate_packed_monitoring([feature(5, 2, 0), feature(4, 0, 0)])
    bad = feature(5, 2, 0)
    bad["aux_mask"] = [True]
    with pytest.raises(ValueError, match="continuation length"):
        collate_packed_monitoring([bad])


def test_continuation_does_not_change_the_decision_logit() -> None:
    from transformers import Qwen3_5ForCausalLM

    text = audit_cpu.tiny_model(0)
    model = Qwen3_5ForCausalLM(text.config).float().eval()
    prompts = [torch.arange(3), torch.arange(4) + 3]
    continuations = [torch.tensor([7, 8]), torch.tensor([9])]
    rows = [torch.cat(pair) for pair in zip(prompts, continuations, strict=True)]
    ids = torch.cat(rows)[None]
    lengths = tuple(len(row) for row in rows)
    with audit_cpu.reference_boundaries(model.model, convolution=True, recurrence=True):
        with installed_segmented_sdpa(), torch.no_grad():
            decision, aux, _ = forward_packed_monitoring_logits_with_aux(
                model, ids, lengths, (2, 1)
            )
            prompt_only, _ = forward_packed_monitoring_logits(
                model, torch.cat(prompts)[None], (3, 4)
            )
            full = [model(input_ids=row[None]).logits[0] for row in rows]
    torch.testing.assert_close(decision, prompt_only, atol=1e-5, rtol=1e-5)
    expected_aux = torch.cat([full[0][3:5], full[1][4:5]])
    torch.testing.assert_close(aux, expected_aux, atol=1e-5, rtol=1e-5)


def test_truncation_keeps_the_continuation_whole() -> None:
    from gleipnir.packed_sequences import truncate_direct_prefix

    plain = feature(6, 0, 0)
    assert truncate_direct_prefix(plain, 3)["direct_input_ids"] == [3, 4, 5]
    row = feature(6, 2, 0)  # prompt [0..3], continuation [4, 5]
    assert truncate_direct_prefix(row, 4)["direct_input_ids"] == [2, 3, 4, 5]
    assert truncate_direct_prefix(row, 1)["direct_input_ids"] == [3, 4, 5]
    assert row["direct_input_ids"] == list(range(6))


def test_aux_continuation_feature_validates_alignment_and_label() -> None:
    from experiments.deception_distillation.train_student_sft import (
        aux_continuation_feature,
    )

    class Tokenizer:
        def encode(self, text, add_special_tokens=False):
            return [15 + int(text)]

    record = dict(
        index="r",
        label=1,
        aux_input_ids=[16, 99],
        aux_top_ids=[[99, 98], [7, 8]],
        aux_top_logprobs=[[-0.1, -2.0], [-0.2, -3.0]],
        aux_mask=[False, True],
    )
    result = aux_continuation_feature(record, Tokenizer())
    assert result["aux_length"] == 2 and result["aux_mask"] == [False, True]
    with pytest.raises(ValueError, match="start with the label"):
        aux_continuation_feature(dict(record, label=0), Tokenizer())
    with pytest.raises(ValueError, match="misaligned"):
        aux_continuation_feature(dict(record, aux_mask=[True]), Tokenizer())


def test_packed_config_rejects_aux_with_last_layer_trim() -> None:
    from gleipnir.training.packed import validate_packed_training_config
    from tests.training.test_packed_training import student_config

    config = student_config()
    config["training"].update(aux_loss_weight=0.1, soft_loss_weight=1.0)
    assert validate_packed_training_config(config)
    config["training"]["last_layer_trim"] = True
    with pytest.raises(ValueError, match="untrimmed soft packing"):
        validate_packed_training_config(config)
