"""Independent setup must run concurrently and propagate failed install jobs."""

from threading import Barrier

import pytest

from experiments.b200_frost_training import setup_training
from gleipnir.training.backends import qwen35


@pytest.mark.parametrize("failure", [False, True])
def test_independent_setup_jobs_overlap_and_fail_closed(monkeypatch, tmp_path, failure):
    barrier = Barrier(3, timeout=5)
    completed = []

    def install(name):
        def run(*args, **kwargs):
            barrier.wait()
            completed.append(name)
            if failure and name == "conv":
                raise RuntimeError("build failed")

        return run

    monkeypatch.setattr(qwen35, "ensure_fla_kernels", install("fla"))
    monkeypatch.setattr(qwen35, "ensure_causal_conv1d", install("conv"))
    monkeypatch.setattr(setup_training, "ensure_flashqla_overlay", install("qla"))
    if failure:
        with pytest.raises(RuntimeError, match="build failed"):
            setup_training.prepare_overlays(tmp_path, {})
    else:
        setup_training.prepare_overlays(tmp_path, {})
    assert sorted(completed) == ["conv", "fla", "qla"]
