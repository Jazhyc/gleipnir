import json
import subprocess
import sys
from pathlib import Path

import pytest

from gleipnir.evaluation_watchdog import (
    RecoveryPolicy,
    live_group,
    prediction_counts,
    run_resumable_evaluation,
)


def policy(**overrides):
    return RecoveryPolicy(
        **{
            "startup_seconds": 0.5,
            "stall_seconds": 0.25,
            "poll_seconds": 0.03,
            "terminate_seconds": 0.2,
            "max_restarts": 1,
            **overrides,
        }
    )


def events(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def test_counts_ignore_touch_and_partial_records(tmp_path):
    path = tmp_path / "predictions.jsonl"
    path.write_text('{"id":1}\n{"id":')
    assert prediction_counts([path]) == (1,)
    path.touch()
    assert prediction_counts([path]) == (1,)
    path.write_text("invalid\n")
    with pytest.raises(json.JSONDecodeError):
        prediction_counts([path])


def test_stall_restarts_and_preserves_saved_rows(tmp_path):
    output, marker, log = [
        tmp_path / name for name in ("predictions.jsonl", "marker", "events")
    ]
    output.write_text('{"id":0}\n')
    code = """
import pathlib,sys,time
p,m=map(pathlib.Path,sys.argv[1:])
if not m.exists():
    m.touch()
    time.sleep(100)
with p.open('a') as f:
    f.write('{"id":1}\\n')
"""
    run_resumable_evaluation(
        [sys.executable, "-c", code, str(output), str(marker)],
        [output],
        log,
        policy=policy(),
    )
    assert output.read_text() == '{"id":0}\n{"id":1}\n'
    names = [r["event"] for r in events(log)]
    assert names.count("restart") == 1
    assert names[-1] == "complete"
    assert all(not live_group(r["pid"]) for r in events(log) if r["event"] == "attempt")


def test_nonzero_exit_is_not_retried(tmp_path):
    log = tmp_path / "events"
    with pytest.raises(subprocess.CalledProcessError):
        run_resumable_evaluation(
            [sys.executable, "-c", "raise SystemExit(2)"],
            [tmp_path / "predictions"],
            log,
            policy=policy(),
        )
    assert sum(r["event"] == "attempt" for r in events(log)) == 1


def test_cache_truncation_fails_without_retry(tmp_path):
    output, log = tmp_path / "predictions", tmp_path / "events"
    output.write_text("{}\n")
    code = "import pathlib,sys; pathlib.Path(sys.argv[1]).write_text('')"
    with pytest.raises(ValueError, match="decreased"):
        run_resumable_evaluation(
            [sys.executable, "-c", code, str(output)], [output], log, policy=policy()
        )
    assert not any(r["event"] == "restart" for r in events(log))


def test_retry_limit_and_sigkill_cleanup(tmp_path):
    log = tmp_path / "events"
    code = (
        "import signal,time; signal.signal(signal.SIGTERM,signal.SIG_IGN); "
        "time.sleep(100)"
    )
    with pytest.raises(TimeoutError):
        run_resumable_evaluation(
            [sys.executable, "-c", code],
            [tmp_path / "predictions"],
            log,
            policy=policy(),
        )
    attempts = [r for r in events(log) if r["event"] == "attempt"]
    assert len(attempts) == 2
    assert all(not live_group(r["pid"]) for r in attempts)


def test_real_progress_resets_stall_deadline(tmp_path):
    output, log = tmp_path / "predictions", tmp_path / "events"
    code = """
import pathlib,sys,time
p=pathlib.Path(sys.argv[1])
for i in range(6):
    with p.open('a') as f: f.write('{}\\n')
    time.sleep(.12)
"""
    run_resumable_evaluation(
        [sys.executable, "-c", code, str(output)], [output], log, policy=policy()
    )
    assert prediction_counts([output]) == (6,)
    assert not any(r["event"] == "restart" for r in events(log))


def test_orphaned_workers_are_killed_with_owned_group(tmp_path):
    log = tmp_path / "events"
    code = """
import subprocess,sys,time
subprocess.Popen([sys.executable, '-c',
    'import signal,time; signal.signal(signal.SIGTERM,signal.SIG_IGN); '
    'time.sleep(100)'])
time.sleep(100)
"""
    with pytest.raises(TimeoutError):
        run_resumable_evaluation(
            [sys.executable, "-c", code],
            [tmp_path / "predictions"],
            log,
            policy=policy(max_restarts=0),
        )
    attempt = next(r for r in events(log) if r["event"] == "attempt")
    assert not live_group(attempt["pid"])


def test_guarded_entrypoint_scopes_progress_to_selected_adapter(monkeypatch, tmp_path):
    import argparse

    import gleipnir.evaluation_watchdog as watchdog
    from experiments.tool_trajectory_monitoring import benchmark_distilled_ood as module

    args = argparse.Namespace(
        config=Path("config"),
        model_size="4b",
        output_root=tmp_path,
        only_job=["chosen"],
        include_base=False,
        watchdog_child=False,
        no_watchdog=False,
        force=False,
    )
    monkeypatch.setattr(module, "parse_args", lambda: args)
    monkeypatch.setattr(module, "load_json", lambda p: {"model_groups": {"4b": {}}})
    monkeypatch.setattr(module, "validate_config", lambda c: None)
    monkeypatch.setattr(module, "validate_jobs", lambda *a: [{"job_name": "chosen"}])
    calls = []
    monkeypatch.setattr(
        watchdog, "run_resumable_evaluation", lambda *a: calls.append(a)
    )
    module.guarded_main()
    assert calls[0][1] == [tmp_path / "4b/adapters/chosen/predictions.jsonl"]
    assert calls[0][0][-1] == "--watchdog-child"
