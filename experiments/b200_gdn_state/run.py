"""Retire the leftover scorer, admit BF16 state, and run a bounded comparison."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import shutil
import statistics
import subprocess
from pathlib import Path

from experiments.b200_inference_benchmark.run import ROOT, write
from experiments.b200_monitor_score.run import SERVING, measure
from gleipnir.serving.gdn.state import validate_native

EXPERIMENT = Path(__file__).parent
STATE_WORKER = "experiments.b200_gdn_state.worker.GdnStateMonitorScoreAuditWorker"


def resume_environment(parent: dict) -> dict[str, str]:
    """Rebuild the pinned environment from existing helpers and public receipts."""
    from experiments.b200_inference_benchmark.run import environment

    # The deployed checkout predates the package move; these retained aliases
    # resolve the same pinned implementations in either repository layout.
    from gleipnir.native_fp4_training import native_fp4_environment
    from gleipnir.qwen35_fast_training import DEFAULT_TRITON_TARGET, triton_environment
    from gleipnir.serving_gigatoken import configure_frontend
    from gleipnir.serving_runtime import local_serving_runtime

    env = native_fp4_environment(
        triton_environment(DEFAULT_TRITON_TARGET, environment()), ROOT
    )
    env["PYTHONPATH"] += f":{ROOT / '.cache/kernels/fa4'}"
    runtime = local_serving_runtime(ROOT, env)
    if (
        runtime is None
        or runtime["manifest_sha256"] != parent["local_runtime"]["manifest_sha256"]
        or runtime["python"] != parent["command"][0]
    ):
        raise ValueError("retired-parent staged runtime identity changed")
    frontend_command = parent["command"].copy()
    frontend_command[frontend_command.index("-m") + 1] = (
        "experiments.b200_attention_gdn_serving.server"
    )
    configure_frontend(ROOT, parent["frontend"], frontend_command, env)
    env["GLEIPNIR_FROST_WRAPPER_VALIDATION"] = parent["host_wrapper"]["validation"]
    env["VLLM_SERVER_DEV_MODE"] = "1"
    selection = json.loads(
        (ROOT / "experiments/b200_inference_benchmark/baseline.json").read_text()
    )
    env["GLEIPNIR_STRIDE_VALIDATION"] = selection["mutation_validation"]
    for key, value in parent["cache_paths"].items():
        if key in {
            "HF_HUB_CACHE",
            "VLLM_CACHE_ROOT",
            "TORCHINDUCTOR_CACHE_DIR",
            "TRITON_CACHE_DIR",
            "TILELANG_CACHE_DIR",
            "TVM_CACHE_DIR",
            "CUDNN_FRONTEND_COMPILED_CACHE",
            "CUDNN_FRONTEND_COMPILED_CACHE_MAX_BYTES",
            "CUTE_DSL_CACHE_DIR",
            "FLASHINFER_CACHE_DIR",
        }:
            env[key] = value
    env["FLASHINFER_WORKSPACE_BASE"] = str(ROOT / ".cache/flashinfer")
    return env


def archive_exited_candidate(name: str) -> None:
    """Preserve an already-exited launch only after verifying no GPU workers."""
    path = SERVING / "server.json"
    current = json.loads(path.read_text())
    command = current["command"]
    if command[command.index("--worker-cls") + 1] != STATE_WORKER:
        raise ValueError("refuse to archive an unrelated server")
    if Path(f"/proc/{current['pid']}").exists():
        raise ValueError("candidate API is still live")
    apps = subprocess.check_output(
        ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"], text=True
    ).strip()
    if apps:
        raise ValueError("GPU workers remain after candidate API exit")
    log = ROOT / "logs/runpod/b200_attention_gdn_serving/server.log"
    destination = log.with_name(f"{name}_candidate_server.log")
    if log.exists() and destination.exists():
        raise ValueError("preserve the existing failed-candidate log")
    current.update(
        status="exited", retirement_reason="Failed startup; GPU verified empty"
    )
    write(SERVING / f"{name}_candidate_server_exited.json", current)
    path.unlink()
    if log.exists():
        log.rename(destination)


def state_command(
    command: list[str], settings: dict, hashes: dict, *, validation: str
) -> list[str]:
    """Change only the persistent-state flag, GDN wrapper and audited worker."""
    command = command.copy()
    if command[command.index("-m") + 1] != "experiments.b200_mutation_analysis.server":
        raise ValueError("BF16 state requires the repaired selected scorer")
    if "--mamba-ssm-cache-dtype" in command:
        raise ValueError("parent already overrides recurrent-state precision")
    command[command.index("--worker-cls") + 1] = STATE_WORKER
    command += ["--mamba-ssm-cache-dtype", "bfloat16"]
    index = command.index("--additional-config") + 1
    additional = json.loads(command[index])
    additional["serving_condition"].update(
        gdn_state_dtype="bfloat16", gdn_state_validation=validation
    )
    additional["gleipnir_frost_fp4"].update(
        {
            p: h
            for p, h in hashes.items()
            if p
            in (
                "src/gleipnir/serving/gdn/state.py",
                "experiments/b200_gdn_state/worker.py",
            )
        }
    )
    command[index] = json.dumps(additional, sort_keys=True)
    return command


def screen(candidate: dict, reference: dict, ranking: dict) -> dict:
    """Apply the frozen development screening rule without promoting a recipe."""

    def median(report, concurrency, key):
        rows = [t for t in report["trials"] if t["concurrency"] == concurrency]
        return statistics.median(
            t["latency"][key]
            if key.startswith("p") and key.endswith("_seconds")
            else t[key]
            for t in rows
        )

    throughput = (
        median(candidate, 128, "prompt_tokens_per_second")
        / median(reference, 128, "prompt_tokens_per_second")
        - 1
    )
    latency = {
        k: median(candidate, 1, k) / median(reference, 1, k) - 1
        for k in ("p50_seconds", "p95_seconds")
    }
    delta = ranking["auroc_delta"]
    passed = (
        throughput > 0.01
        and all(v <= 0.02 for v in latency.values())
        and all(abs(delta[k]) <= 0.001 for k in ("macro", "pooled"))
    )
    return {
        "passed": passed,
        "c128_throughput_fraction_change": throughput,
        "c1_latency_fraction_change": latency,
        "c128_auroc_delta": delta,
        "promoted": False,
    }


async def run(
    name: str, *, retired_parent: Path | None = None, reuse_native: Path | None = None
) -> None:
    out = ROOT / "results/b200_gdn_state" / name
    parent = json.loads((retired_parent or (SERVING / "server.json")).read_text())
    pid = parent["pid"]
    if retired_parent is not None:
        if (SERVING / "server.json").exists() or Path(f"/proc/{pid}").exists():
            raise ValueError("retired-parent retry requires no live server")
        actual = parent["command"]
        environment = resume_environment(parent)
    else:
        actual = [
            v.decode()
            for v in Path(f"/proc/{pid}/cmdline").read_bytes().split(b"\0")
            if v
        ]
        if actual != parent["command"]:
            raise ValueError("leftover server identity changed")
        # Inherited environment stays only in memory; never serialize secrets.
        environment = dict(
            v.decode().split("=", 1)
            for v in Path(f"/proc/{pid}/environ").read_bytes().split(b"\0")
            if v
        )

    def native_prepare(env: dict, destination: Path) -> None:
        if reuse_native is not None:
            receipt = json.loads(reuse_native.read_text())
            validate_native(receipt)
            for source, digest in receipt["sources"].items():
                if hashlib.sha256((ROOT / source).read_bytes()).hexdigest() != digest:
                    raise ValueError(f"reused BF16 native source changed: {source}")
            shutil.copy2(reuse_native, destination / "native.json")
            write(
                destination / "native_reuse.json",
                {
                    "source": str(reuse_native),
                    "sha256": hashlib.sha256(reuse_native.read_bytes()).hexdigest(),
                },
            )
            return
        with (destination / "native.log").open("x") as handle:
            subprocess.run(
                [
                    actual[0],
                    "-m",
                    "experiments.b200_gdn_state.canary",
                    "--output",
                    str(destination / "native.json"),
                ],
                cwd=ROOT,
                env=env,
                stdout=handle,
                stderr=subprocess.STDOUT,
                check=True,
                timeout=900,
            )

    async def ready_prepare(client, destination):
        response = await client.post(
            "/collective_rpc",
            json={"method": "gdn_state_audit", "kwargs": {}, "timeout": 60},
        )
        response.raise_for_status()
        receipt = response.json()["results"][0]
        if not receipt["passed"] or len(receipt["caches"]) != 24:
            raise ValueError("live BF16 GDN state audit failed")
        write(destination / "gdn_state.json", receipt)

    retire = False
    try:
        await measure(
            name,
            experiment=EXPERIMENT,
            result_group="b200_gdn_state",
            native_prepare=native_prepare,
            ready_prepare=ready_prepare,
            command_prepare=lambda c, s, h: state_command(
                c, s, h, validation=str((out / "native.json").relative_to(ROOT))
            ),
            retired_parent=retired_parent,
            resumed_environment=environment if retired_parent is not None else None,
        )
        from gleipnir.serving.reference import selected_score_reference

        selected = selected_score_reference(ROOT)
        reference = json.loads(
            (ROOT / selected["results"] / "summary.json").read_text()
        )
        summary = json.loads((out / "summary.json").read_text())
        ranking = json.loads((out / "c128_comparison.json").read_text())["ranking"]
        result = screen(summary, reference, ranking)
        write(out / "screen.json", result)
        retire = not result["passed"]
        print("bf16_state_screen", json.dumps(result), flush=True)
    except BaseException as error:
        if out.exists():
            write(out / "failure.json", {"error": f"{type(error).__name__}: {error}"})
        retire = True
        raise
    finally:
        if retire and (out / "parent_server.json").exists():
            if (SERVING / "server.json").exists():
                current = json.loads((SERVING / "server.json").read_text())
                if (
                    current["command"][current["command"].index("--worker-cls") + 1]
                    != STATE_WORKER
                ):
                    raise ValueError("refuse recovery over an unrelated server")
                if not Path(f"/proc/{current['pid']}").exists():
                    archive_exited_candidate(name)
                else:
                    subprocess.run(
                        [
                            actual[0],
                            "-m",
                            "experiments.b200_attention_gdn_serving.stop_server",
                            "--archive-name",
                            name + "_candidate",
                            "--reason",
                            "Leave GPU available after rejected BF16 state screen",
                        ],
                        cwd=ROOT,
                        env=environment,
                        check=True,
                    )
            write(
                out / "recovery.json",
                {
                    "reference_restored": False,
                    "candidate_retired": True,
                    "reference_restart_requires_user_request": True,
                },
            )
            print("bf16_state_candidate_retired", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", required=True)
    parser.add_argument("--retired-parent", type=Path)
    parser.add_argument("--reuse-native", type=Path)
    args = parser.parse_args()
    if Path(args.name).name != args.name:
        raise ValueError("run name must be a stem")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    asyncio.run(
        run(
            args.name,
            retired_parent=args.retired_parent,
            reuse_native=args.reuse_native,
        )
    )


if __name__ == "__main__":
    main()
