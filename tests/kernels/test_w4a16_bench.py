"""Small CPU checks for benchmark input integrity and numerical diagnostics."""

import pytest
import torch

from experiments.local_inference.w4a16_bench import checked_load, relative_error
from gleipnir.qwen35_adapter_rebase import sha256_file


def test_checked_load_fails_closed(tmp_path):
    path = tmp_path / "capture.pt"
    torch.save({"x": torch.ones(2)}, path)
    assert checked_load(path, sha256_file(path))["x"].tolist() == [1, 1]
    with pytest.raises(ValueError, match="Checksum mismatch"):
        checked_load(path, "wrong")


def test_relative_error():
    reference = torch.tensor([3.0, 4.0])
    assert relative_error(reference, reference) == 0
    assert relative_error(reference * 1.5, reference) == pytest.approx(0.5)
