"""Local controller: generate with GPU paused, score, collect, verify, pause."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import tarfile
import time

from experiments.adversarial_a4_feedback import run

ROOT = run.eval_sets.ROOT
LOG = ROOT / "logs/jarvislabs/adversarial_a4_feedback"
STATE = LOG / "operations.json"


def command(args, **kwargs):
    result = subprocess.run(args, cwd=ROOT, text=True, capture_output=True, **kwargs)
    if result.returncode:
        # CLI responses may contain notebook credentials: keep this error terse.
        raise RuntimeError(f"{args[0]} {args[1]} failed with exit {result.returncode}")
    return result.stdout


def jl(*args):
    value = json.loads(command(["jl", *map(str, args)]))
    if isinstance(value, dict) and value.get("error"):
        raise RuntimeError("JarvisLabs API returned an error; inspect sanitized status")
    return value


def remote(machine, script):
    result = jl("exec", machine, "--json", "--", "bash", "-lc", script)
    if result.get("exit_code", 0) != 0:
        raise RuntimeError("remote command failed: " + result.get("stderr", "")[-800:])
    return result.get("stdout", "")


def save(state):
    temp = STATE.with_suffix(".tmp")
    temp.write_text(json.dumps(state, indent=2) + "\n")
    temp.replace(STATE)


def collect(machine, run_id, round_no):
    archive = f"/home/a4-round-{round_no}.tar.gz"
    remote(
        machine,
        f"set -e; tar -czf {archive} -C /home/gleipnir "
        "results/adversarial_a4_feedback "
        f"-C /home jl-runs/{run_id}/output.log",
    )
    sha = remote(machine, f"sha256sum {archive}").split()[0]
    target = LOG / f"round_{round_no}.tar.gz"
    command(["jl", "download", str(machine), archive, str(target)])
    assert hashlib.sha256(target.read_bytes()).hexdigest() == sha
    with tarfile.open(target) as tar:
        for member in tar.getmembers():
            if not member.isfile():
                continue
            if member.name.startswith("results/adversarial_a4_feedback/"):
                path = ROOT / member.name
                assert path.resolve().is_relative_to(run.OUT.resolve())
            elif member.name == f"jl-runs/{run_id}/output.log":
                path = LOG / f"score_round_{round_no}.log"
            else:
                raise ValueError("unexpected archive member")
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(tar.extractfile(member).read())
    return sha


def main(machine):
    LOG.mkdir(parents=True, exist_ok=True)
    with run.lock("orchestrator"):
        run.prepare()
        state = (
            json.loads(STATE.read_text())
            if STATE.exists()
            else {
                "machine_id": machine,
                "gpu_seconds": 0,
                "rounds": [],
                "state": "ready",
            }
        )
        if state["state"] not in ("ready", "complete"):
            raise RuntimeError("unfinished controller session; inspect before resuming")
        if state["state"] == "complete":
            print("already complete", flush=True)
            return
        env = os.environ.copy()
        if "OPENROUTER_API_KEY" not in env:
            for line in (ROOT / "../../.env").resolve().read_text().splitlines():
                if line.startswith("OPENROUTER_API_KEY="):
                    env["OPENROUTER_API_KEY"] = (
                        line.split("=", 1)[1].strip().strip("\"'")
                    )
        if not env.get("OPENROUTER_API_KEY"):
            raise ValueError("missing OpenRouter credential")
        env["PYTHONPATH"] = "src:."
        bundle = LOG / "code.bundle"
        command(["git", "bundle", "create", str(bundle), "379d6f6..HEAD"])
        frozen_head = command(["git", "rev-parse", "HEAD"]).strip()
        for round_no in range(len(state["rounds"]) + 1, run.CONFIG["rounds"] + 1):
            machine = state["machine_id"]
            live = jl("get", machine, "--json")
            if live.get("status") != "Paused" or live.get("gpu_type") != "A100":
                raise RuntimeError(
                    "expected existing paused A100; do not duplicate active work"
                )
            state.update(state="generating", round=round_no)
            save(state)
            print(f"round {round_no}: generating locally; GPU paused", flush=True)
            with (LOG / f"generation_round_{round_no}.log").open("a") as log:
                child = subprocess.run(
                    [
                        sys.executable,
                        "-u",
                        "-m",
                        "experiments.adversarial_a4_feedback.run",
                        "generate",
                        "--round",
                        str(round_no),
                    ],
                    cwd=ROOT,
                    env=env,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                )
            if child.returncode:
                state["state"] = "generation_failed"
                save(state)
                raise RuntimeError(
                    "generation failed; GPU remains paused; inspect generation log"
                )
            remaining = run.CONFIG["gpu_budget_seconds"] - state["gpu_seconds"]
            if remaining < 240:
                state["state"] = "gpu_budget_exhausted"
                save(state)
                raise RuntimeError("insufficient GPU budget for another session")
            state["state"] = "resume_requested"
            save(state)
            gpu_start = time.monotonic()
            run_id = None
            round_ok = False
            resumed = jl("resume", machine, "--yes", "--json")
            machine = resumed["machine_id"]
            state.update(machine_id=machine, state="scoring")
            save(state)
            print(f"round {round_no}: resumed existing A100 as {machine}", flush=True)
            receipt = {
                "round": round_no,
                "machine_id": machine,
                "protocol_commit": frozen_head,
            }
            try:
                command(
                    ["jl", "upload", str(machine), str(bundle), "/home/a4-code.bundle"]
                )
                remote(
                    machine,
                    "set -e; cd /home/gleipnir; git diff --exit-code; "
                    "git fetch /home/a4-code.bundle HEAD; "
                    "git merge --ff-only FETCH_HEAD",
                )
                # Upload only the immutable manifest and local generation receipts;
                # remote score receipts persist across pauses.
                remote(
                    machine, "mkdir -p /home/gleipnir/results/adversarial_a4_feedback"
                )
                for name in ("manifest.json", "generations.jsonl", "api_intents.jsonl"):
                    command(
                        [
                            "jl",
                            "upload",
                            str(machine),
                            str(run.OUT / name),
                            "/home/gleipnir/" + run.CONFIG["output"] + "/" + name,
                        ]
                    )
                remote(
                    machine,
                    "set -e; cd /home/gleipnir; uv pip install "
                    "--python .venv/bin/python --target /tmp/gleipnir-qwen35-fla "
                    "--no-deps --link-mode copy flash-linear-attention==0.5.2 "
                    "fla-core==0.5.2; source .adversarial-env; "
                    'export PYTHONPATH="src:.:$PYTHONPATH"; .venv/bin/python -c '
                    '"from transformers.utils.import_utils import '
                    "is_flash_linear_attention_available; "
                    'assert is_flash_linear_attention_available()"',
                )
                timeout = int(remaining - (time.monotonic() - gpu_start) - 150)
                if timeout < 60:
                    raise RuntimeError("GPU setup consumed remaining budget")
                script = (
                    "cd /home/gleipnir && source .adversarial-env && "
                    'export PYTHONPATH="src:.:$PYTHONPATH" && '
                    f"timeout {timeout} .venv/bin/python -u -m "
                    f"experiments.adversarial_a4_feedback.run score --round {round_no}"
                )
                launched = jl(
                    "run",
                    "--on",
                    machine,
                    "--json",
                    "--yes",
                    "--",
                    "bash",
                    "-lc",
                    script,
                )
                run_id = launched["run_id"]
                receipt["run_id"] = run_id
                state["active_run_id"] = run_id
                save(state)
                print(f"round {round_no}: managed run {run_id}", flush=True)
                while True:
                    time.sleep(20)
                    status = jl("run", "status", run_id, "--json")
                    phase = status.get("state", status.get("status"))
                    print(f"round {round_no}: {phase}", flush=True)
                    if phase in ("succeeded", "failed", "stopped", "completed"):
                        receipt["run_status"] = status.get("state", phase)
                        break
                    if time.monotonic() - gpu_start > remaining - 100:
                        jl("run", "stop", run_id, "--json")
                        raise RuntimeError("GPU session time limit reached")
                receipt["archive_sha256"] = collect(machine, run_id, round_no)
                assert (
                    json.loads((run.OUT / f"score_round_{round_no}.json").read_text())[
                        "state"
                    ]
                    == "succeeded"
                )
                round_ok = True
            finally:
                # Stop a possibly active run before pausing on an exceptional path.
                if run_id and "archive_sha256" not in receipt:
                    try:
                        jl("run", "stop", run_id, "--json")
                        receipt["archive_sha256"] = collect(machine, run_id, round_no)
                    except Exception as error:
                        receipt["collection_error"] = type(error).__name__
                jl("pause", machine, "--yes", "--json")
                live = jl("get", machine, "--json")
                receipt["final_status"] = live.get("status")
                state["gpu_seconds"] += time.monotonic() - gpu_start
                state.update(
                    state="ready"
                    if round_ok and receipt["final_status"] == "Paused"
                    else "scoring_failed",
                    active_run_id=None,
                )
                receipt["cumulative_gpu_seconds"] = state["gpu_seconds"]
                state["rounds"].append(receipt)
                save(state)
                print(
                    f"round {round_no}: GPU {receipt['final_status']}; "
                    f"cumulative {state['gpu_seconds']:.1f}s",
                    flush=True,
                )
            if receipt["final_status"] != "Paused":
                raise RuntimeError("pause not confirmed")
            if receipt.get("run_status") != "succeeded":
                raise RuntimeError("scoring did not succeed")
        state["state"] = "complete"
        save(state)
        run.report()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--machine", type=int, required=True)
    main(parser.parse_args().machine)
