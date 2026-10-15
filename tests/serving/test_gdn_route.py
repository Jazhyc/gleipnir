"""The GDN route gate forces non-CP only below its token threshold."""

from types import SimpleNamespace

import pytest

from gleipnir.serving import gdn_route


def record(**kwargs):
    return kwargs


def call(tokens, **extra):
    q = SimpleNamespace(shape=(tokens, 32, 128))
    return gdn_route.routed_call(record, 8192, q=q, backend="flashinfer", **extra)


def test_short_calls_force_non_cp_and_long_calls_keep_auto():
    before = gdn_route.audit()
    assert call(1024)["use_cp"] is False
    assert "use_cp" not in call(8192)
    assert "use_cp" not in call(32768)
    after = gdn_route.audit()
    assert after["non_cp"] - before["non_cp"] == 1
    assert after["auto"] - before["auto"] == 2


def test_gate_rejects_conflicting_or_unsupported_calls():
    with pytest.raises(ValueError, match="explicit use_cp"):
        call(1024, use_cp=True)
    with pytest.raises(ValueError, match="keyword FlashInfer"):
        gdn_route.routed_call(record, 8192, object(), backend="flashinfer")
    with pytest.raises(ValueError, match="keyword FlashInfer"):
        gdn_route.routed_call(
            record, 8192, q=SimpleNamespace(shape=(1, 1, 1)), backend="auto"
        )
