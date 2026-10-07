"""Prepare or launch an opt-in admission trial; never reclaim another workload."""

from __future__ import annotations

import argparse
import asyncio
import json
import subprocess
from pathlib import Path

from experiments.b200_inference_benchmark.run import ROOT, sha, write
from experiments.b200_inference_benchmark.run import environment as base_environment
from experiments.b200_monitor_score.run import SERVING, measure
from gleipnir.serving.length_admission import AdmissionPolicy

EXPERIMENT = Path(__file__).parent
SCHEDULER = "gleipnir.serving.vllm.length_scheduler.LengthAwareScheduler"
POLICY_SOURCE = "src/gleipnir/serving/length_admission.py"
INTEGRATION_SOURCE = "src/gleipnir/serving/vllm/length_scheduler.py"


def admission_command(command: list[str], settings: dict, hashes: dict) -> list[str]:
    """Add only the admission policy to the selected synchronous pooling recipe."""
    command = command.copy()
    if (
        command[command.index("-m") + 1] != "experiments.b200_mutation_analysis.server"
        or command[command.index("--runner") + 1] != "pooling"
        or "--enable-chunked-prefill" not in command
        or "--no-enable-prefix-caching" not in command
    ):
        raise ValueError("admission trial requires the selected cached score recipe")
    if "--scheduler-cls" in command or "--async-scheduling" in command:
        raise ValueError("parent already overrides scheduler or async scheduling")
    if "--scheduling-policy" in command and (
        command[command.index("--scheduling-policy") + 1] != "fcfs"
    ):
        raise ValueError("admission trial requires upstream FCFS policy")
    config = dict(settings["length_admission"])
    AdmissionPolicy.from_dict(config)
    config.update(
        policy_sha256=hashes[POLICY_SOURCE],
        integration_sha256=hashes[INTEGRATION_SOURCE],
    )
    index = command.index("--additional-config") + 1
    additional = json.loads(command[index])
    if not additional.get("qk_mutation_analysis") or "monitor_score" not in additional:
        raise ValueError("parent score/mutation contract missing")
    if "length_admission" in additional:
        raise ValueError("parent already has an admission policy")
    additional["length_admission"] = config
    command[index] = json.dumps(additional, sort_keys=True)
    command += ["--scheduler-cls", SCHEDULER]
    if "--no-async-scheduling" not in command:
        command += ["--no-async-scheduling"]
    return command


def require_idle_gpu(*, allowed_pids: set[int] | None = None) -> None:
    """Reject unrelated GPU work; allow only a verified server being replaced."""
    apps = subprocess.check_output(
        ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"], text=True
    ).strip()
    resident = {int(pid.strip()) for pid in apps.splitlines() if pid.strip()}
    if resident - (allowed_pids or set()):
        raise ValueError(
            "GPU is occupied; admission trial does not stop other workloads"
        )


async def run(name: str, retired_parent: Path | None) -> None:
    # A live parent is replaced only through the existing identity-checked stop
    # helper. An idle-GPU launch reconstructs its preserved public receipts.
    environment = None
    if retired_parent is not None:
        require_idle_gpu()
        from gleipnir.serving.score_runtime import resume_score_environment

        parent = json.loads(retired_parent.read_text())
        environment = resume_score_environment(ROOT, parent, base_environment())
    else:
        parent = json.loads((SERVING / "server.json").read_text())
        worker = json.loads((SERVING / "loaded_precision.json").read_text())[
            "worker_pid"
        ]
        require_idle_gpu(allowed_pids={parent["pid"], worker})
    try:
        await measure(
            name,
            experiment=EXPERIMENT,
            result_group="b200_length_admission",
            command_prepare=admission_command,
            retired_parent=retired_parent,
            resumed_environment=environment,
        )
    except BaseException as error:
        out = ROOT / "results/b200_length_admission" / name
        if out.exists():
            write(out / "failure.json", {"error": f"{type(error).__name__}: {error}"})
        # Retire only our identity-matched candidate; never restore a reference.
        if (out / "parent_server.json").exists() and (SERVING / "server.json").exists():
            current = json.loads((SERVING / "server.json").read_text())
            if (
                "--scheduler-cls" in current["command"]
                and (
                    current["command"][current["command"].index("--scheduler-cls") + 1]
                    == SCHEDULER
                )
                and current.get("score_sources", {}).get(POLICY_SOURCE)
                == sha(ROOT / POLICY_SOURCE)
                and current.get("frontend", {}).get("receipt_path")
                == str(out / "frontend.json")
            ):
                if Path(f"/proc/{current['pid']}").exists():
                    subprocess.run(
                        [
                            current["command"][0],
                            "-m",
                            "experiments.b200_attention_gdn_serving.stop_server",
                            "--archive-name",
                            name + "_failed",
                            "--reason",
                            "Leave GPU available after failed admission trial",
                        ],
                        cwd=ROOT,
                        env=environment,
                        check=True,
                    )
                else:
                    require_idle_gpu()
                    log = ROOT / "logs/runpod/b200_attention_gdn_serving/server.log"
                    target = log.with_name(name + "_failed_server.log")
                    if log.exists() and target.exists():
                        raise ValueError(
                            "refuse to overwrite failed-candidate log"
                        ) from error
                    current["status"] = "exited"
                    write(SERVING / f"{name}_failed_server_exited.json", current)
                    if log.exists():
                        log.rename(target)
                    (SERVING / "server.json").unlink()
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", required=True)
    parser.add_argument("--retired-parent", type=Path)
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    if args.name in {"", ".", ".."} or Path(args.name).name != args.name:
        raise ValueError("run name must be a directory stem")
    if args.prepare_only:
        # Read only public command metadata; do not inspect hardware or start a server.
        selection = json.loads(
            (ROOT / "experiments/b200_inference_benchmark/baseline.json").read_text()
        )
        metadata = ROOT / selection["server_metadata"]
        if (
            sha(metadata)
            != selection["artifact_bindings"][selection["server_metadata"]]
        ):
            raise ValueError("selected server metadata drift")
        parent = json.loads(metadata.read_text())
        settings = json.loads((EXPERIMENT / "config.json").read_text())
        hashes = {p: sha(ROOT / p) for p in settings["additional_sources"]}
        command = admission_command(parent["command"], settings, hashes)
        out = ROOT / "results/b200_length_admission" / args.name
        out.mkdir(parents=True, exist_ok=False)
        write(
            out / "prepared.json",
            {
                "command": command,
                "sources": hashes,
                "settings": settings,
                "launched": False,
            },
        )
        print(out / "prepared.json")
        return
    asyncio.run(run(args.name, args.retired_parent))


if __name__ == "__main__":
    main()
