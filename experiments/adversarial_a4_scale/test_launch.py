"""Launch gates and scorer receipts without network or model loading."""
import copy
import json
from pathlib import Path

import pytest

from .common import config, digest, fixed, sha
from .dependency import arm_config
from .gpu_score import check_runtime, finite_result
from .pipeline import require_live
from .selection import select
from .smoke import development_tasks
from .test_pipeline import setup  # noqa: F401


def test_teacher_search_is_symmetric(setup):
    _, manifest, pairs, dep = setup
    cfg = arm_config(manifest["config"], "27b-merged")
    dep = copy.deepcopy(dep)
    dep["monitor_identity"].update(monitor="27b-merged", instruction="teacher")
    result = select(dep, pairs, cfg)
    assert len(result["task_ids"]) == 2
    assert cfg["monitor_contracts"]["transfer"] == ["4b-v03", "kimi-k3", "gpt-6.1-sol", "claude-sonnet-5.5"]


def test_smoke_cannot_use_cohort_tasks(setup, monkeypatch):
    out, manifest, pairs, _ = setup
    cfg = manifest["config"]
    cfg["step1"] = str(out / "step1")
    fixed(out / "step1/selection.json", {"development": ["a", "b", "c", "d", "e"], "heldout": ["a"], "calibration": ["cal"]})
    with pytest.raises(ValueError, match="development"):
        development_tasks(cfg)


def test_smoke_rejects_unregistered_settings(setup):
    _, manifest, _, _ = setup
    manifest["stage"] = "smoke"
    manifest["config"]["attempts"] = 6
    with pytest.raises(RuntimeError, match="unregistered smoke"):
        require_live(manifest)


def test_runtime_and_finite_guards():
    with pytest.raises(ValueError, match="runtime differs"):
        check_runtime({"dtype": "float16"}, {"dtype": "bfloat16"})
    with pytest.raises(ValueError):
        finite_result({"score": float('nan'), "logit_0": 1, "logit_1": 2, "tokens": 4})
    with pytest.raises(ValueError):
        finite_result({"score": 0.2, "logit_0": 1, "logit_1": float('inf'), "tokens": 4})


def test_expired_deadline_precedes_model_load(setup):
    from .gpu_score import run
    _, manifest, _, _ = setup
    class Expired:
        def check_deadline(self):
            raise SystemExit("deadline")
    identity = {"monitor": "4b-v03"}
    with pytest.raises(SystemExit, match="deadline"):
        run({"monitor_identity": identity, "jobs": []}, {"monitor_identity": identity}, manifest,
            setup[0], setup[0] / 'score.json', campaign=Expired(),
            scorer_factory=lambda *a: pytest.fail("model loaded after deadline"))


def test_cost_gate_binds_smoke_and_attempts(setup):
    from .cost import check_gate
    out, manifest, _, _ = setup
    cfg = manifest["config"]
    gate = {"status": "passed", "selected_writer": cfg["writer"], "summaries": []}
    fixed(out / 'gate.json', gate)
    cost = {"smoke_gate_sha256": digest(gate), "writer": cfg["writer"], "attempts": cfg["attempts"]}
    fixed(out / 'cost.json', cost)
    cfg['smoke_gate'] = {"path": str(out / 'gate.json'), "sha256": digest(gate)}
    cfg['cost_check'] = {"path": str(out / 'cost.json'), "sha256": digest(cost)}
    assert check_gate(cfg) == cost
    cfg['attempts'] -= 1
    with pytest.raises(ValueError, match="cost receipt"):
        check_gate(cfg)
