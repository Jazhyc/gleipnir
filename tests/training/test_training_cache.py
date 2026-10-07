"""Ordinary campaigns reuse warm persistent caches without moving old artifacts."""

from pathlib import Path

from gleipnir.monitoring_campaign_runtime import training_cache


def test_adopts_populated_legacy_cache_and_reuses_across_campaigns(tmp_path: Path):
    legacy = tmp_path / ".cache/training/student_injection_awareness"
    legacy.mkdir(parents=True)
    artifact = legacy / "compiled-kernel"
    artifact.write_bytes(b"preserved")
    first = training_cache(tmp_path, "campaign-one")
    second = training_cache(tmp_path, "campaign-two")
    assert first == second == legacy
    assert (first / artifact.name).read_bytes() == b"preserved"
    assert (tmp_path / ".cache/training/shared").is_symlink()


def test_existing_shared_cache_has_priority(tmp_path: Path):
    shared = tmp_path / ".cache/training/shared"
    shared.mkdir(parents=True)
    (shared / "artifact").write_text("keep")
    (shared.parent / "student_injection_awareness").mkdir()
    assert training_cache(tmp_path, "new-campaign") == shared
    assert (shared / "artifact").read_text() == "keep"


def test_cold_cache_is_explicit_and_does_not_replace_shared(tmp_path: Path):
    shared = training_cache(tmp_path, "ordinary")
    isolated = training_cache(tmp_path, "cold-start-measurement", isolated_cache=True)
    assert isolated == tmp_path / ".cache/training/cold-start-measurement"
    assert isolated != shared
    assert training_cache(tmp_path, "another-ordinary") == shared
