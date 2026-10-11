"""Focused tests for class direction, exact boundaries and invalid inputs."""

import json
import math

import pytest

from experiments.teacher_disagreement_filtering.prepare import subset
from gleipnir.data.teacher_filter import remove_disagreement


@pytest.mark.parametrize("delta", [0.05, 0.1, 0.2])
def test_positive_delta_boundaries(delta):
    lower, upper = 0.5 - delta, 0.5 + delta
    assert remove_disagreement(1, math.nextafter(lower, 0), delta)
    assert not remove_disagreement(1, lower, delta)
    assert not remove_disagreement(1, 0.49, delta)
    assert remove_disagreement(0, math.nextafter(upper, 1), delta)
    assert not remove_disagreement(0, upper, delta)
    assert not remove_disagreement(0, 0.51, delta)
    assert not remove_disagreement(0, 0.01, delta)
    assert not remove_disagreement(1, 0.99, delta)


def test_zero_delta_ties():
    assert remove_disagreement(0, 0.5, 0)
    assert not remove_disagreement(1, 0.5, 0)
    assert remove_disagreement(1, 0.499, 0)
    assert not remove_disagreement(0, 0.499, 0)


@pytest.mark.parametrize(
    "label,score,delta",
    [
        (2, 0.5, 0),
        (0, float("nan"), 0),
        (1, 1.1, 0),
        (0, 0.5, -0.1),
        (1, 0.5, 0.5),
        (1, 0.5, float("nan")),
    ],
)
def test_invalid_inputs(label, score, delta):
    with pytest.raises(ValueError):
        remove_disagreement(label, score, delta)


def test_subset_preserves_bytes_order_and_dataset_identity(tmp_path):
    rows = [
        b'{"dataset": "a", "index": "1", "target": 0.1}\n',
        b'{"dataset":"b","index":"1","target":0.2}\n',
        b'{"dataset":"a","index":"2","target":0.3}\n',
    ]
    source, target = tmp_path / "source.jsonl", tmp_path / "target.jsonl"
    source.write_bytes(b"".join(rows))
    subset(source, target, {("a", "1")})
    assert target.read_bytes() == b"".join(rows[1:])
    assert [json.loads(r)["dataset"] for r in target.read_bytes().splitlines()] == [
        "b",
        "a",
    ]
    with pytest.raises(FileExistsError):
        subset(source, target, set())


@pytest.mark.parametrize("fail_at", [None, "delta005"])
def test_queue_preflights_all_arms_and_stops_on_failure(tmp_path, monkeypatch, fail_at):
    from types import SimpleNamespace

    from experiments.teacher_disagreement_filtering import run as runner
    from experiments.teacher_disagreement_filtering.prepare import CASES

    output = tmp_path / "output"
    output.mkdir()
    manifest = {"cases": {name: {"config": name + ".yaml"} for name in CASES}}
    prepared, launched, summarized, retired, stopped = [], [], [], [], []

    def load(root, path):
        name = path.stem
        return SimpleNamespace(
            name=name,
            inventory=lambda **kwargs: {"passed": True},
            prepare=lambda: prepared.append(name),
        )

    def launch(command, **kwargs):
        name = command[-1]
        assert prepared == list(CASES)
        launched.append(name)
        return SimpleNamespace(
            pid=100 + len(launched),
            poll=lambda: 1 if name == fail_at else 0,
            returncode=1 if name == fail_at else 0,
        )

    monkeypatch.setattr(runner, "ROOT", tmp_path)
    monkeypatch.setattr(runner, "OUTPUT", output)
    monkeypatch.setattr(runner, "verify_preparation", lambda: manifest)
    monkeypatch.setattr(runner.Campaign, "load", load)
    monkeypatch.setattr(runner.subprocess, "Popen", launch)
    monkeypatch.setattr(
        runner.psutil, "Process", lambda pid: SimpleNamespace(create_time=lambda: 1.0)
    )
    monkeypatch.setattr(runner, "retire", lambda ctx: retired.append(ctx.name))
    monkeypatch.setattr(runner, "release_merge", lambda ctx: None)
    monkeypatch.setattr(runner, "summarize", lambda m: summarized.append(m))
    monkeypatch.setattr(
        runner, "terminate_owned", lambda p, owned: stopped.append(p.pid)
    )
    if fail_at:
        with pytest.raises(RuntimeError, match="delta005 exited 1"):
            runner.run()
        assert launched == ["control", "delta000", "delta005"]
        assert retired == ["control", "delta000"]
        assert stopped and not summarized
    else:
        runner.run()
        assert launched == retired == list(CASES)
        assert summarized == [manifest]
    state = json.loads((output / "status.json").read_text())
    assert state["status"] == ("failed" if fail_at else "complete")
    with pytest.raises(ValueError, match="already launched"):
        runner.run()


def test_cached_control_reuses_exact_artifacts_without_training(tmp_path, monkeypatch):
    from types import SimpleNamespace

    from experiments.teacher_disagreement_filtering import run as runner
    from gleipnir.campaigns.monitoring import training

    fields = [
        "seed",
        "learning_rate",
        "soft_loss_weight",
        "direct_loss_weight",
        "model_revision",
        "rank",
        "lora_alpha",
        "max_length",
        "gradient_checkpointing",
        "effective_batch_size",
    ]
    job = dict.fromkeys(fields, 0)
    inputs = {}
    for name in [
        "cached_job",
        "cached_metadata",
        "cached_master_config",
        "cached_master",
        "cached_export_config",
        "cached_export",
        "cached_rebase",
    ]:
        inputs[name] = tmp_path / name
        inputs[name].write_text(json.dumps(job if name == "cached_job" else {}))
    data, output, adapter = [tmp_path / n for n in ["data", "output", "adapter"]]
    data.mkdir()
    (data / "manifest.json").write_text("{}")
    ctx = SimpleNamespace(
        config={"cached_control": {"inputs": {}}},
        check=lambda: None,
        input=lambda name: inputs[name],
        job=lambda: job,
        data=data,
        output=output,
        adapter=adapter,
    )
    calls = []
    monkeypatch.setattr(
        training, "validate_completion", lambda m, c: calls.append("validated")
    )
    monkeypatch.setattr(
        runner, "worker", lambda *args: pytest.fail("control must not train")
    )
    runner.launch(ctx, "train")
    assert calls == ["validated"]
    assert (
        adapter / "causal_adapter/adapter_model.safetensors"
    ).read_bytes() == inputs["cached_master"].read_bytes()
    complete = json.loads((adapter / "complete.json").read_text())
    assert complete["training_performed_this_run"] is False
    assert complete["status"] == "reused_trained_control"
    with pytest.raises(ValueError, match="refuse to overwrite"):
        runner.launch(ctx, "train")


@pytest.mark.parametrize("changed", [None, "master", "weights", "server"])
def test_merge_slot_retirement_preserves_receipts_and_checks_identity(
    tmp_path, monkeypatch, changed
):
    from types import SimpleNamespace

    from experiments.teacher_disagreement_filtering import run as runner
    from gleipnir.data.monitoring import file_hash, write_json

    slot, output, adapter, serving = [
        tmp_path / n for n in ["slot", "output", "adapter", "serving"]
    ]
    slot.mkdir()
    serving.mkdir()
    master = adapter / "causal_adapter/adapter_model.safetensors"
    master.parent.mkdir(parents=True)
    master.write_bytes(b"persistent master")
    weights = slot / "model.safetensors"
    weights.write_bytes(b"disposable merged model")
    artifact = {"files_sha256": {weights.name: file_hash(weights)}}
    write_json(slot / "merge_manifest.json", artifact)
    write_json(output / "merged_artifact.json", artifact)
    write_json(
        output / "evaluation/complete.json",
        {"parity_passed": True, "rows": 3012, "master_sha256": file_hash(master)},
    )
    ctx = SimpleNamespace(
        config={"merged_model": str(slot)},
        output=output,
        adapter=adapter,
        serving=serving,
    )
    monkeypatch.setattr(runner, "MERGED_SLOT", slot)
    if changed == "master":
        master.write_bytes(b"changed")
    elif changed == "weights":
        weights.write_bytes(b"changed")
    elif changed == "server":
        (serving / "server.json").write_text("{}")
    if changed:
        with pytest.raises(ValueError):
            runner.release_merge(ctx)
        assert slot.exists()
    else:
        runner.release_merge(ctx)
        assert not slot.exists()
        assert master.exists()
        assert (
            json.loads((output / "archived_merge_manifest.json").read_text())
            == artifact
        )
        receipt = json.loads((output / "merged_slot_retired.json").read_text())
        assert receipt["master_preserved"] and not receipt["weights_preserved"]


@pytest.mark.parametrize("existing_status", [None, "running", "complete"])
def test_outer_launch_failure_is_recorded(tmp_path, monkeypatch, existing_status):
    from experiments.teacher_disagreement_filtering import run as runner
    from gleipnir.data.monitoring import write_json

    monkeypatch.setattr(runner, "OUTPUT", tmp_path)
    if existing_status:
        write_json(tmp_path / "status.json", {"status": existing_status})

    def fail():
        raise FileExistsError("existing launch log")

    monkeypatch.setattr(runner, "run", fail)
    with pytest.raises(FileExistsError):
        runner.guarded_run()
    state = json.loads((tmp_path / "status.json").read_text())
    assert state["status"] == (
        "complete" if existing_status == "complete" else "failed"
    )
    receipt = json.loads((tmp_path / "queue_failure.json").read_text())
    assert receipt["error_type"] == "FileExistsError"
