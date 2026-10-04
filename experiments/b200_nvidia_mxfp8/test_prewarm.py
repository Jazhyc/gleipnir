"""Validate exact-cohort cache preparation without loading the CUDA overlay."""

import hashlib
import json

import pytest

from experiments.b200_nvidia_mxfp8.prewarm import partitions, selected_lengths


def test_manifest_identity_unique_lengths_and_singleton(tmp_path):
    manifest = tmp_path / "selection.jsonl"
    payload = "".join(
        json.dumps({"student_direct_tokens": n}) + "\n" for n in [1, 129, 65, 129]
    )
    manifest.write_text(payload)
    digest = hashlib.sha256(payload.encode()).hexdigest()
    assert selected_lengths(manifest, digest) == [129, 65]
    with pytest.raises(ValueError, match="checksum drift"):
        selected_lengths(manifest, "0" * 64)


@pytest.mark.parametrize("invalid", [0, -1, True, 3.5])
def test_reject_invalid_manifest_lengths(tmp_path, invalid):
    manifest = tmp_path / "selection.jsonl"
    payload = json.dumps({"student_direct_tokens": invalid}) + "\n"
    manifest.write_text(payload)
    with pytest.raises(ValueError, match="invalid prewarm"):
        selected_lengths(manifest, hashlib.sha256(payload.encode()).hexdigest())


def test_partition_covers_each_length_once():
    lengths = [257, 129, 127, 65, 63]
    assert partitions(lengths, 3) == [[257, 65], [129, 63], [127]]
    for workers in (0, 17):
        with pytest.raises(ValueError, match="workers"):
            partitions(lengths, workers)
