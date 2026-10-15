"""GDN fast path: eligibility, per-forward metadata cache and A/B switch."""

from types import SimpleNamespace

import torch

from gleipnir.serving import gdn_fast


def metadata(**overrides):
    value = dict(
        num_prefills=2,
        num_decodes=0,
        spec_sequence_masks=None,
        prefill_state_indices=torch.tensor([3, 5]),
        prefill_has_initial_state=torch.tensor([False, False]),
        prefill_query_start_loc=torch.tensor([0, 4, 9], dtype=torch.int32),
    )
    value.update(overrides)
    return SimpleNamespace(**value)


LAYER = SimpleNamespace(
    chunk_gated_delta_rule=SimpleNamespace(gdn_prefill_backend="flashinfer")
)


def test_only_plain_flashinfer_prefill_batches_are_eligible():
    assert gdn_fast.eligible(LAYER, metadata())
    assert not gdn_fast.eligible(LAYER, None)
    assert not gdn_fast.eligible(LAYER, metadata(num_decodes=1))
    assert not gdn_fast.eligible(LAYER, metadata(num_prefills=0))
    assert not gdn_fast.eligible(LAYER, metadata(spec_sequence_masks=torch.ones(1)))
    triton = SimpleNamespace(
        chunk_gated_delta_rule=SimpleNamespace(gdn_prefill_backend="triton")
    )
    assert not gdn_fast.eligible(triton, metadata())


def test_step_metadata_is_computed_once_per_metadata_object():
    calls = []

    def fresh(count):
        calls.append(count)
        return True

    first = metadata()
    for _ in range(24):
        cu, all_fresh = gdn_fast.step_metadata(first, fresh)
    assert calls == [2] and all_fresh and cu.dtype == torch.int64
    assert cu.tolist() == [0, 4, 9]
    second = metadata(prefill_query_start_loc=torch.tensor([0, 7], dtype=torch.int32))
    cu, _ = gdn_fast.step_metadata(second, lambda count: False)
    assert cu.tolist() == [0, 7] and len(calls) == 1


def test_switch_is_reported_in_the_audit():
    assert gdn_fast.set_enabled(False)["enabled"] is False
    assert gdn_fast.set_enabled(True)["enabled"] is True
