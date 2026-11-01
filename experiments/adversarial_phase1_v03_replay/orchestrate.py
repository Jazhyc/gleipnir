"""One existing GPU session, with durable receipts, bounded run, collection, pause."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import tarfile
import time

from experiments.adversarial_a4_feedback.orchestrate import command, jl, remote
from experiments.adversarial_phase1_v03_replay import run

ROOT = run.ROOT
LOG = ROOT / "logs/jarvislabs/adversarial_phase1_v03_replay"
STATE = LOG / "operations.json"


def collect(machine, run_id):
    archive = "/home/phase1-v03-replay.tar.gz"
    remote(
        machine,
        f"set -e; tar -czf {archive} -C /home/gleipnir-replay "
        "results/adversarial_phase1_v03_replay "
        f"-C /home jl-runs/{run_id}/output.log",
    )
    sha = remote(machine, f"sha256sum {archive}").split()[0]
    target = LOG / "artifacts.tar.gz"
    command(["jl", "download", str(machine), archive, str(target)], timeout=120)
    assert hashlib.sha256(target.read_bytes()).hexdigest() == sha
    with tarfile.open(target) as tar:
        for m in tar.getmembers():
            if not m.isfile():
                continue
            if m.name.startswith("results/adversarial_phase1_v03_replay/"):
                path = ROOT / m.name
                assert path.resolve().is_relative_to(run.OUT.resolve())
            elif m.name == f"jl-runs/{run_id}/output.log":
                path = LOG / "score.log"
            else:
                raise ValueError("Unexpected archive member")
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(tar.extractfile(m).read())
    return sha


def main(machine):
    LOG.mkdir(parents=True, exist_ok=True)
    with (LOG / "controller.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if STATE.exists():
            raise RuntimeError("Prior controller state exists; inspect, do not retry")
        run.prepare()
        live = jl("get", machine, "--json")
        assert live["status"] == "Paused" and live["gpu_type"] == "A100"
        assert not command(
            [
                "git",
                "status",
                "--porcelain",
                "--",
                "experiments/adversarial_phase1_v03_replay",
            ]
        )
        commit = command(["git", "rev-parse", "HEAD"]).strip()
        bundle = LOG / "code.bundle"
        command(["git", "bundle", "create", str(bundle), "16bc154..HEAD"])
        state = {
            "state": "resume_requested",
            "machine_id": machine,
            "protocol_commit": commit,
            "gpu_seconds": 0,
            "active_run_id": None,
        }
        run.save(STATE, state)
        started = time.monotonic()
        run_id = None
        completed = False
        # Never retry an uncertain resume. Persist its returned ID immediately.
        resumed = jl("resume", machine, "--yes", "--json")
        machine = resumed["machine_id"]
        state.update(machine_id=machine, state="setup")
        run.save(STATE, state)
        print(f"Existing A100 resumed as {machine}", flush=True)
        try:
            health = remote(
                machine,
                "nvidia-smi --query-gpu=name,memory.total,memory.used,utilization.gpu "
                "--format=csv,noheader; df -BG /home; "
                'pgrep -af "python.*experiments.adversarial" || true',
            )
            (LOG / "initial_health.log").write_text(health)
            # Refuse concurrent compute; this query returns numeric process IDs.
            active = remote(
                machine, "nvidia-smi --query-compute-apps=pid --format=csv,noheader"
            ).strip()
            if active:
                raise RuntimeError(
                    "GPU already has compute processes; refusing duplicate work"
                )
            free = int(
                remote(machine, "df --output=avail -B1 /home | tail -n 1").strip()
            )
            assert free >= 30_000_000_000, (
                "Need 30 GB free for pinned 9B base and adapter"
            )
            command(
                [
                    "jl",
                    "upload",
                    str(machine),
                    str(bundle),
                    "/home/phase1-replay-code.bundle",
                ],
                timeout=120,
            )
            remote(
                machine,
                "set -e; cd /home/gleipnir; "
                "git fetch /home/phase1-replay-code.bundle HEAD; "
                "git worktree add --detach /home/gleipnir-replay FETCH_HEAD; "
                "cd /home/gleipnir-replay; ln -s /home/gleipnir/.venv .venv; "
                "cp /home/gleipnir/.adversarial-env .adversarial-env; "
                'for d in /home/gleipnir/data/*; do ln -s "$d" data/; done; '
                'for d in /home/gleipnir/results/*; do ln -s "$d" results/; done',
            )
            assert (
                remote(
                    machine, "cd /home/gleipnir-replay && git rev-parse HEAD"
                ).strip()
                == commit
            )
            remote(
                machine,
                "mkdir -p /home/gleipnir-replay/results/adversarial_phase1_v03_replay",
            )
            for name in [
                *run.CONFIG["source_files"],
                str(run.OUT.relative_to(ROOT) / "manifest.json"),
            ]:
                command(
                    [
                        "jl",
                        "upload",
                        str(machine),
                        str(ROOT / name),
                        "/home/gleipnir-replay/" + name,
                    ],
                    timeout=120,
                )
            remote(
                machine,
                "set -e; cd /home/gleipnir-replay; "
                "uv pip install --python .venv/bin/python "
                "--target /tmp/gleipnir-qwen35-fla --no-deps --link-mode copy "
                "flash-linear-attention==0.5.2 fla-core==0.5.2; "
                'source .adversarial-env; export PYTHONPATH="src:.:$PYTHONPATH"; '
                '.venv/bin/python -c "from transformers.utils.import_utils import '
                "is_flash_linear_attention_available; "
                'assert is_flash_linear_attention_available()"',
            )
            remaining = int(
                run.CONFIG["gpu_budget_seconds"] - (time.monotonic() - started) - 180
            )
            assert remaining > 60
            script = (
                "cd /home/gleipnir-replay && source .adversarial-env && "
                'export PYTHONPATH="src:.:$PYTHONPATH" && '
                + f"timeout {remaining} .venv/bin/python -u -m "
                "experiments.adversarial_phase1_v03_replay.run score"
            )
            launch = jl(
                "run", "--on", machine, "--json", "--yes", "--", "bash", "-lc", script
            )
            run_id = launch["run_id"]
            state.update(state="scoring", active_run_id=run_id)
            run.save(STATE, state)
            print(f"Managed run {run_id}", flush=True)
            while True:
                time.sleep(30)
                status = jl("run", "status", run_id, "--json")
                phase = status.get("state", status.get("status"))
                print(
                    f"Run {phase}; allocated {time.monotonic() - started:.1f}s",
                    flush=True,
                )
                if phase in ("succeeded", "failed", "stopped", "completed"):
                    state["run_status"] = phase
                    break
                if time.monotonic() - started > run.CONFIG["gpu_budget_seconds"] - 150:
                    jl("run", "stop", run_id, "--json")
                    raise RuntimeError("GPU time cap reached")
            state["archive_sha256"] = collect(machine, run_id)
            assert state["run_status"] == "succeeded"
            assert (
                json.loads((run.OUT / "state.json").read_text())["state"] == "succeeded"
            )
            completed = True
        except BaseException as error:
            state["error_type"] = type(error).__name__
            raise
        finally:
            if run_id and "archive_sha256" not in state:
                try:
                    jl("run", "stop", run_id, "--json")
                    state["archive_sha256"] = collect(machine, run_id)
                except Exception as error:
                    state["collection_error"] = type(error).__name__
            state["state"] = "pausing"
            run.save(STATE, state)
            jl("pause", machine, "--yes", "--json")
            state["final_status"] = jl("get", machine, "--json")["status"]
            state["gpu_seconds"] = time.monotonic() - started
            state.update(
                state="complete"
                if completed and state["final_status"] == "Paused"
                else "failed",
                active_run_id=None,
            )
            run.save(STATE, state)
            print(
                f"GPU {state['final_status']}; allocated {state['gpu_seconds']:.1f}s",
                flush=True,
            )
        assert state["final_status"] == "Paused"
        run.report()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--machine", type=int, required=True)
    main(parser.parse_args().machine)
