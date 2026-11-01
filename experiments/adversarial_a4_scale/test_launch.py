"""Launch gates and scorer receipts without network or model loading."""

import copy
import json
from pathlib import Path

import pytest

from .common import digest, fixed, sha
from .dependency import arm_config
from .gpu_score import check_runtime, finite_result
from .pipeline import require_live
from .selection import select
from .smoke import development_tasks
from .test_pipeline import setup as _setup_fixture


@pytest.fixture(name="setup")
def launch_setup(tmp_path):
    return _setup_fixture.__wrapped__(tmp_path)


def test_teacher_search_is_symmetric(setup):
    _, manifest, pairs, dep = setup
    cfg = arm_config(manifest["config"], "27b-merged")
    dep = copy.deepcopy(dep)
    dep["monitor_identity"].update(monitor="27b-merged", instruction="teacher")
    result = select(dep, pairs, cfg)
    assert len(result["task_ids"]) == 2
    assert cfg["monitor_contracts"]["transfer"] == [
        "4b-v03",
        "kimi-k3",
        "gpt-6.1-sol",
        "claude-sonnet-5.5",
    ]


def test_smoke_cannot_use_cohort_tasks(setup, monkeypatch):
    out, manifest, pairs, _ = setup
    cfg = manifest["config"]
    cfg["step1"] = str(out / "step1")
    fixed(
        out / "step1/selection.json",
        {
            "development": ["a", "b", "c", "d", "e"],
            "heldout": ["a"],
            "calibration": ["cal"],
        },
    )
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
        finite_result({"score": float("nan"), "logit_0": 1, "logit_1": 2, "tokens": 4})
    with pytest.raises(ValueError):
        finite_result(
            {"score": 0.2, "logit_0": 1, "logit_1": float("inf"), "tokens": 4}
        )


def test_expired_deadline_precedes_model_load(setup):
    from .gpu_score import run

    _, manifest, _, _ = setup

    class Expired:
        def check_deadline(self):
            raise SystemExit("deadline")

    identity = {"monitor": "4b-v03"}
    with pytest.raises(SystemExit, match="deadline"):
        run(
            {"monitor_identity": identity, "jobs": []},
            {"monitor_identity": identity},
            manifest,
            setup[0],
            setup[0] / "score.json",
            campaign=Expired(),
            scorer_factory=lambda *a: pytest.fail("model loaded after deadline"),
        )


def test_cost_gate_binds_smoke_and_attempts(setup, monkeypatch):
    from .cost import check_gate

    monkeypatch.setattr(
        "experiments.adversarial_a4_scale.cost.verify_evidence", lambda *a: None
    )
    out, manifest, _, _ = setup
    cfg = manifest["config"]
    gate = {"status": "passed", "selected_writer": cfg["writer"], "summaries": []}
    fixed(out / "gate.json", gate)
    cost = {
        "smoke_gate_sha256": digest(gate),
        "writer": cfg["writer"],
        "attempts": cfg["attempts"],
    }
    fixed(out / "cost.json", cost)
    cfg["smoke_gate"] = {"path": str(out / "gate.json"), "sha256": digest(gate)}
    cfg["cost_check"] = {"path": str(out / "cost.json"), "sha256": digest(cost)}
    assert check_gate(cfg) == cost
    cfg["attempts"] -= 1
    with pytest.raises(ValueError, match="cost receipt"):
        check_gate(cfg)


def test_empty_smoke_evidence_is_rejected(setup, monkeypatch):
    from . import smoke

    out, manifest, pairs, _ = setup
    cfg = manifest["config"]
    cfg["smoke_registration_commit"] = "registered"
    cfg["smoke"]["output"] = str(out / "smoke")
    original = (Path(smoke.__file__).parent / "pipeline.py").read_text()
    from .common import source_identity

    sources = source_identity()
    monkeypatch.setattr(smoke, "development_tasks", lambda _: list(pairs))
    monkeypatch.setattr("subprocess.check_output", lambda *a, **k: original)
    fixed(
        out / "smoke/preregistration.json",
        {
            "smoke": cfg["smoke"],
            "tasks": list(pairs),
            "sources": sources,
            "prompt": sha((Path(smoke.__file__).parent / "prompt.txt").read_text()),
        },
    )
    with pytest.raises(ValueError, match="missing smoke evidence"):
        smoke.verify_evidence(cfg, {"status": "passed", "summaries": []}, pairs)


def test_common_export_and_rank_require_dual_barrier(setup):
    from . import pipeline as p

    out, manifest, pairs, _ = setup
    manifest["config"]["dual_arm_barrier"] = True
    for action in [
        lambda: p.search_batch(out, manifest, pairs),
        lambda: p.rank(out, manifest, pairs, {}),
    ]:
        with pytest.raises(FileNotFoundError, match="generation_complete"):
            action()


def test_parallel_generation_has_one_receipt_per_slot(setup):
    from . import pipeline as p
    from .test_pipeline import responder

    out, manifest, pairs, _ = setup
    manifest["config"]["generation_workers"] = 4
    p.generate(out, manifest, pairs, responder=responder)
    assert len(p.generation_records(out)) == 20
    assert len(p.sealed(out, manifest)) == 20
    before = (out / "generations.jsonl").read_bytes()
    p.generate(out, manifest, pairs, responder=lambda _: pytest.fail("parallel retry"))
    assert (out / "generations.jsonl").read_bytes() == before


def test_parallel_dispatch_respects_scope_cap(setup):
    from . import pipeline as p
    from .test_pipeline import responder

    out, manifest, pairs, _ = setup
    manifest["config"]["generation_workers"] = 8
    manifest["config"]["budgets_usd"]["tinker"] = 0.01
    result = p.generate(out, manifest, pairs, responder=responder)
    assert result["status"] == "budget_exhausted"
    assert not (out / "generation_intents.jsonl").exists()


def test_duplicate_canary_dependency_rejected(setup):
    from gleipnir.adversarial.eval_sets import file_sha256

    from .dependency import verify_dependency

    out, manifest, _, _ = setup
    canary = {
        "passed": True,
        "rows": [{"task_id": "a", "code": "honest", "score": 0.2}] * 8,
    }
    runtime = {"scorer": {}}
    cp, rp = out / "a.heldout.canary.json", out / "a.heldout.runtime.json"
    fixed(cp, canary)
    fixed(rp, runtime)
    dep = {
        "monitor_identity": {
            "monitor": "4b-v03",
            "canary_receipt_sha256": file_sha256(cp),
            "runtime_receipt_sha256": file_sha256(rp),
        },
        "canary_receipt": canary,
        "runtime_receipt": runtime,
        "provenance": {"files": {str(cp): file_sha256(cp), str(rp): file_sha256(rp)}},
    }
    fixed(out / "dependency.json", dep)
    cfg = manifest["config"]
    cfg["dependencies"] = {
        "4b-v03": {"path": str(out / "dependency.json"), "sha256": digest(dep)}
    }
    with pytest.raises(ValueError, match="eight unique"):
        verify_dependency(dep, cfg)
    bad = copy.deepcopy(dep)
    bad["canary_receipt"]["rows"][0]["score"] = 0.8
    with pytest.raises(ValueError, match="dependency changed"):
        verify_dependency(bad, cfg)


def test_gpu_receipt_resume_and_failed_canary(setup, monkeypatch):
    from . import gpu_score as g

    out, manifest, pairs, _ = setup
    pairs["fourth"] = copy.deepcopy(pairs["toy-a"])
    identity = {
        "monitor": "27b-merged",
        "scorer": {"monitor": "27b-merged", "adapter": None},
    }
    rows = [
        {"task_id": t, "code": c, "score": 0.2}
        for t in pairs
        for c in ["honest", "backdoored"]
    ]
    dep = {"monitor_identity": identity, "canary_receipt": {"rows": rows}}
    batch = {
        "monitor_identity": identity,
        "jobs": [
            {
                "id": "candidate",
                "prompt": pairs["toy-a"]["backdoored"]["prompt"],
                "prompt_sha256": sha(pairs["toy-a"]["backdoored"]["prompt"]),
            }
        ],
    }

    class Fake:
        calls = 0
        bad = False

        def __init__(self, *args):
            pass

        def identity(self):
            return identity["scorer"]

        def score(self, prompt, adapter=True):
            Fake.calls += 1
            return {
                "score": float("nan") if Fake.bad else 0.2,
                "logit_0": 1.0,
                "logit_1": 2.0,
                "tokens": 10,
            }

    class Clock:
        def check_deadline(self):
            return {"deadline_epoch": 1e20}

    monkeypatch.setattr(
        "experiments.adversarial_a4_scale.dependency.verify_dependency", lambda *a: None
    )
    monkeypatch.setattr(g, "benchmark_pairs", lambda _: pairs)
    monkeypatch.setattr(g, "instructed_pairs", lambda pairs, cfg: pairs)
    result = g.run(
        batch,
        dep,
        manifest,
        out,
        out / "receipt.json",
        scorer_factory=Fake,
        campaign=Clock(),
    )
    assert result["rows"][0]["status"] == "ok"
    assert Fake.calls == 9
    g.run(
        batch,
        dep,
        manifest,
        out,
        out / "receipt.json",
        scorer_factory=Fake,
        campaign=Clock(),
    )
    assert Fake.calls == 17  # canary reruns, completed job is never repeated
    Fake.bad = True
    with pytest.raises(RuntimeError, match="canary failed"):
        g.run(
            batch,
            dep,
            manifest,
            out,
            out / "failed.json",
            scorer_factory=Fake,
            campaign=Clock(),
        )
    failed = list(out.glob("failed.json.canary.*.json"))
    assert len(failed) == 1 and not json.loads(failed[0].read_text())["passed"]
    assert not (out / "failed.rows.jsonl").exists()


def test_cost_population_rejects_changed_tasks(setup):
    from .cost import verify_population
    from .pipeline import generation_estimate, messages

    _, manifest, pairs, _ = setup
    cfg, selected = manifest["config"], manifest["selection"]
    tasks = selected["task_ids"]
    receipt = {
        "population": {
            "4b-v03": {
                "dependency_sha256": selected["dependency_sha256"],
                "tasks_sha256": digest(tasks),
                "requests_sha256": digest(
                    {t: messages(pairs[t]["backdoored"]) for t in tasks}
                ),
            }
        },
        "per_arm_one_attempt_worst_usd": {
            "4b-v03": generation_estimate(tasks, pairs, {**cfg, "attempts": 1})[
                "worst_case_usd"
            ]
        },
    }
    verify_population(receipt, "4b-v03", selected, pairs, cfg)
    changed = copy.deepcopy(selected)
    changed["task_ids"] = tasks[:-1]
    with pytest.raises(ValueError, match="population changed"):
        verify_population(receipt, "4b-v03", changed, pairs, cfg)


@pytest.fixture
def smoke_evidence(setup, monkeypatch):
    from . import pipeline as p
    from . import smoke
    from .common import append, source_identity

    out, manifest, pairs, _ = setup
    cfg = manifest["config"]
    cfg["smoke_registration_commit"] = "registered"
    cfg["smoke"]["output"] = str(out / "smoke")
    cfg["step1"] = str(out / "step1")
    tasks = [f"dev-{i}" for i in range(5)]
    fixed(
        out / "step1/selection.json",
        {"development": tasks, "heldout": ["heldout"], "calibration": ["cal"]},
    )
    for task in tasks:
        pairs[task] = copy.deepcopy(pairs["toy-a"])
    tasks = smoke.development_tasks(cfg)
    sources = source_identity()
    monkeypatch.setattr(
        "subprocess.check_output", lambda *a, **k: Path(p.__file__).read_text()
    )
    fixed(
        out / "smoke/preregistration.json",
        {
            "smoke": cfg["smoke"],
            "tasks": tasks,
            "sources": sources,
            "prompt": sha((Path(p.__file__).parent / "prompt.txt").read_text()),
        },
    )
    spec = cfg["smoke"]["candidates"][0]
    m = {
        "stage": "smoke",
        "config": {**cfg, "writer": spec, "attempts": 1},
        "sources": sources,
        "selection": {"task_ids": tasks},
    }
    dest = out / "smoke/0"
    fixed(dest / "manifest.json", m)
    fixed(dest / "generation_identity.json", {"manifest": digest(m), "mode": "live"})
    rows = []
    for i, task in enumerate(tasks):
        response = {
            "text": json.dumps([{"before_line": 1, "comment": "Implementation notes."}])
            if i < 4
            else "bad JSON",
            "finish_reason": "stop",
            "model": spec["model"],
            "provider": None,
            "cost_usd": 0.01,
            "usage": {"prompt_tokens": 100, "completion_tokens": 100},
        }
        request_hash = digest(
            {
                "messages": p.messages(pairs[task]["backdoored"]),
                "writer": spec,
                "attempt": 1,
            }
        )
        row = {
            "id": f"{task}:1",
            "task_id": task,
            "attempt": 1,
            "response": response,
            "request_sha256": request_hash,
            "charged_usd": 0.01,
            **p.validate(response, pairs[task]["backdoored"]),
        }
        append(
            dest / "generation_intents.jsonl",
            {"id": row["id"], "request_sha256": request_hash},
        )
        append(dest / "generations.jsonl", row)
        rows.append(row)
    fixed(
        dest / "generation_complete.json",
        {
            "manifest": digest(m),
            "mode": "live",
            "records_sha256": digest({r["id"]: r for r in rows}),
        },
    )
    gate = {
        "status": "passed",
        "selected_writer": spec,
        "summaries": [
            {
                "candidate": 0,
                "writer": spec,
                "calls": 5,
                "valid": 4,
                "passed": True,
                "receipts_sha256": digest(rows),
            }
        ],
    }
    return cfg, gate, pairs, dest, rows


def test_smoke_evidence_revalidates_insufficient_passes(smoke_evidence):
    from .smoke import verify_evidence

    cfg, gate, pairs, dest, rows = smoke_evidence
    verify_evidence(cfg, gate, pairs)
    # Replace two declared successful responses with failures, retaining the
    # forged four-pass summary and updating journal seals as a faulty producer might.
    for row in rows[:2]:
        row["response"]["text"] = "bad JSON"
    (dest / "generations.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    m = json.loads((dest / "manifest.json").read_text())
    (dest / "generation_complete.json").write_text(
        json.dumps(
            {
                "manifest": digest(m),
                "mode": "live",
                "records_sha256": digest({r["id"]: r for r in rows}),
            }
        )
    )
    gate["summaries"][0]["receipts_sha256"] = digest(rows)
    with pytest.raises(ValueError, match="validation evidence"):
        verify_evidence(cfg, gate, pairs)


def test_smoke_prompt_drift_is_rejected(smoke_evidence, monkeypatch):
    from .common import HERE
    from .smoke import verify_evidence

    cfg, gate, pairs, _, _ = smoke_evidence
    original = Path.read_text
    monkeypatch.setattr(
        Path,
        "read_text",
        lambda self, *a, **k: (
            original(self, *a, **k) + ("changed" if self == HERE / "prompt.txt" else "")
        ),
    )
    with pytest.raises(ValueError, match="prompt or validator changed"):
        verify_evidence(cfg, gate, pairs)


def test_per_arm_dispatch_rejected_before_client_or_parent_state(setup, monkeypatch):
    from concurrent.futures import ThreadPoolExecutor

    from . import pipeline as p
    from .common import lock

    out, manifest, pairs, _ = setup
    manifest["config"]["dual_arm_barrier"] = True
    manifest["config"]["budgets_usd"]["tinker"] = 0.01
    monkeypatch.setattr(p, "Client", lambda *a, **k: pytest.fail("per-arm paid client"))

    def launch(arm):
        with pytest.raises(RuntimeError, match="study coordinator"):
            p.generate(out / arm, manifest, pairs)

    with lock(out):
        with ThreadPoolExecutor(max_workers=2) as executor:
            list(executor.map(launch, ["4b-v03", "27b-merged"]))
    assert not list(out.glob("*/generation_intents.jsonl"))


def test_coordinator_checks_sibling_balance_and_resume(setup, monkeypatch):
    from . import pipeline as p
    from . import study
    from .common import source_identity
    from .cost import verify_population
    from .test_pipeline import responder

    out, old, pairs, _ = setup
    root = out / "study"
    cfg = old["config"]
    cfg["dual_arm_barrier"] = True
    cfg["generation_workers"] = 4
    tasks = old["selection"]["task_ids"]
    per_arm = p.generation_estimate(tasks, pairs, cfg)["worst_case_usd"]
    cfg["budgets_usd"]["tinker"] = 1.5 * per_arm
    gate = {"population": {}, "per_arm_one_attempt_worst_usd": {}}
    for arm in study.ARMS:
        ac = arm_config(cfg, arm)
        selection = copy.deepcopy(old["selection"])
        ap = study.instructed_pairs(pairs, ac)
        gate["population"][arm] = {
            "dependency_sha256": selection["dependency_sha256"],
            "tasks_sha256": digest(tasks),
            "requests_sha256": digest(
                {t: p.messages(ap[t]["backdoored"]) for t in tasks}
            ),
        }
        gate["per_arm_one_attempt_worst_usd"][arm] = p.generation_estimate(
            tasks, ap, {**ac, "attempts": 1}
        )["worst_case_usd"]
        m = {
            **old,
            "config": ac,
            "selection": selection,
            "sources": source_identity(),
            "pair_hashes": {
                t: {c: sha(r["prompt"]) for c, r in ap[t].items()}
                for t in old["pair_hashes"]
            },
        }
        fixed(root / arm / "manifest.json", m)
        verify_population(gate, arm, selection, ap, ac)
    monkeypatch.setattr(
        "experiments.adversarial_a4_scale.cost.check_gate", lambda _: gate
    )
    with pytest.raises(RuntimeError, match="aggregate run"):
        study.validate_population_and_balance(root, cfg, pairs)
    sibling = root / "27b-merged/manifest.json"
    original = sibling.read_text()
    sibling.unlink()
    with pytest.raises(FileNotFoundError):
        study.validate_population_and_balance(root, cfg, pairs)
    sibling.write_text(original)
    first = json.loads((root / "4b-v03/manifest.json").read_text())
    ap = study.instructed_pairs(pairs, first["config"])
    p.generate(root / "4b-v03", first, ap, responder=responder, _coordinated=True)
    # Completed slots are excluded from the resumed worst-case reservation.
    study.validate_population_and_balance(root, cfg, pairs)
    before = (root / "4b-v03/generations.jsonl").read_bytes()
    p.generate(
        root / "4b-v03",
        first,
        ap,
        responder=lambda _: pytest.fail("resend"),
        _coordinated=True,
    )
    assert (root / "4b-v03/generations.jsonl").read_bytes() == before
