"""Lean dispatch toggles and reports its installed ops."""

from gleipnir.serving import lean_dispatch


def test_switch_round_trips_through_the_audit():
    assert lean_dispatch.set_enabled(False)["enabled"] is False
    assert lean_dispatch.set_enabled(True)["enabled"] is True
