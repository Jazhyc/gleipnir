"""Keep gate interpretation and recurrent-state layout across backend changes."""

from types import SimpleNamespace

import pytest

from gleipnir.serving_gdn_kernels import (
    install_nvvm_compatibility,
    make_flashqla_prefill,
)


def test_cute_enum_rename_preserves_values_and_existing_definition():
    group = SimpleNamespace(CTA_1=0, CTA_2=1)
    commit = object()
    module = SimpleNamespace(CTAGroupKind=group, tcgen05_commit=commit)
    assert install_nvvm_compatibility(module) == {
        "Tcgen05GroupKind": "CTAGroupKind",
        "tcgen05_commit_arrive": "tcgen05_commit",
    }
    assert module.Tcgen05GroupKind is group
    assert module.tcgen05_commit_arrive is commit
    assert install_nvvm_compatibility(module) == {}
    with pytest.raises(ValueError, match="unexpected NVVM"):
        install_nvvm_compatibility(
            SimpleNamespace(CTAGroupKind=SimpleNamespace(CTA_1=3))
        )


class FakeTensor:
    def __init__(self, value):
        self.value = value

    def float(self):
        return self

    def contiguous(self):
        return self


def test_flashqla_adapter_preserves_log_gates_and_v_first_state():
    calls = []

    def native(**kwargs):
        calls.append(kwargs)
        return "g", "A", "output", "h", "state", "cache"

    adapter = make_flashqla_prefill(
        native, lambda x: FakeTensor(("normalized", x.value)), auto_cp=False
    )
    g, beta, state = FakeTensor(-1), FakeTensor(0.5), FakeTensor(1)
    q, k, v = [FakeTensor(name) for name in ("q", "k", "v")]
    assert adapter(q, k, v, g, beta, state, True, "sequences") == (
        "output",
        "state",
    )
    call = calls[0]
    assert call["g"] is g  # No exp: FlashQLA receives log forget gates.
    assert call["initial_state"] is state
    assert call["state_v_first"] is True
    assert call["is_train"] is False
    assert not call["enable_fwd_cp_cache"] and not call["output_h"]
    assert call["q"].value == ("normalized", "q")
    assert adapter(q, k, v, g, beta, state, False, "sequences", False) == (
        "output",
        None,
    )
    assert calls[-1]["q"] is q
