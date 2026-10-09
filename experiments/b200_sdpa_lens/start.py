"""Restore the SDPA adapter's BF16 scorer with opt-in eager Lens hooks."""

from __future__ import annotations

import argparse
import asyncio
import importlib.metadata
import inspect
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import httpx

from gleipnir.data.monitoring import file_hash, read_rows, write_json

ROOT = Path(__file__).resolve().parents[2]
CONFIG = Path(__file__).with_name("config.json")
SERVING = ROOT / "results/b200_attention_gdn_serving"
OVERLAY = Path("/tmp/gleipnir-vllm-lens-1.3.0")


def lens_command(parent: dict, model: str) -> list[str]:
    """Preserve the adapter-specific BF16 command; opt into eager Lens only."""
    command = list(parent["command"])
    condition = json.loads(command[command.index("--additional-config") + 1])[
        "serving_condition"
    ]
    if (
        parent.get("serving_precision") != "bf16"
        or command[command.index("--model") + 1] != model
        or condition["quantization"] is not None
        or any(
            condition[k] != "bf16"
            for k in (
                "attention_precision",
                "attention_projection_precision",
                "gdn_projection_precision",
                "mlp_precision",
            )
        )
        or command[command.index("--worker-cls") + 1]
        != "gleipnir.serving.bf16_worker.Bf16Worker"
    ):
        raise ValueError("Lens parent must be the matching unquantized BF16 adapter")
    command[0] = sys.executable
    command[2] = "experiments.b200_sdpa_lens.server"
    command += [
        "--enforce-eager",
        "--worker-extension-cls",
        "gleipnir.serving.lens_worker.MonitorLensExtension",
    ]
    return command


async def start(name: str, *, config_path: Path = CONFIG) -> None:
    from vllm.v1.core.sched import scheduler

    from experiments.b200_long_context.run import gpu
    from experiments.b200_vllm031.runtime import candidate_environment
    from experiments.b200_vllm_lens.smoke import smoke
    from experiments.b200_vllm_lens.verify_client import verify
    from gleipnir.campaigns.monitoring.evaluation import agreement
    from gleipnir.evaluation.http_score import score_batch
    from gleipnir.serving.reference import selected_serving_default

    config = json.loads(config_path.read_text())
    out = ROOT / "results/b200_sdpa_lens" / name
    out.mkdir(parents=True, exist_ok=False)
    log = ROOT / "logs/runpod/b200_sdpa_lens" / f"{name}_server.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    inputs = {}
    for key, spec in config["inputs"].items():
        path = ROOT / spec["path"]
        if file_hash(path) != spec["sha256"]:
            raise ValueError(f"frozen input drift: {key}")
        inputs[key] = path
    if {k: importlib.metadata.version(k) for k in config["runtime"]} != config[
        "runtime"
    ]:
        raise ValueError("restored serving packages changed")
    hardware = gpu()
    if hardware["apps"] or hardware["gpu"].split(",")[1].strip() != config["gpu_uuid"]:
        raise ValueError("unexpected GPU or active model")
    parent = json.loads(inputs["parent_server"].read_text())
    adapter = json.loads(inputs["adapter_complete"].read_text())
    merged = json.loads(inputs["merged_artifact"].read_text())
    if (
        not parent["adapter_sha256"]
        == adapter["serving_sha256"]
        == merged["adapter_sha256"]
    ):
        raise ValueError("restored adapter identity changed")
    if not json.loads(inputs["merge_parity"].read_text())["passed"]:
        raise ValueError("restored merge lacks passing master parity")
    for filename, sha in merged["files_sha256"].items():
        if file_hash(Path(config["model"]) / filename) != sha:
            raise ValueError(f"restored model drift: {filename}")
    stage = json.loads(inputs["lens_staging"].read_text())
    for filename, sha in stage["files_sha256"].items():
        if file_hash(OVERLAY / filename) != sha:
            raise ValueError(f"restored Lens drift: {filename}")
    selection, _ = selected_serving_default(ROOT)
    binding = {
        "upstream_sha256": file_hash(Path(inspect.getfile(scheduler))),
        "integration_sha256": file_hash(ROOT / "experiments/b200_vllm031/scheduler.py"),
    }
    if binding != selection["scheduler_binding"]:
        raise ValueError("pooling scheduler drift")
    sources = {}
    for filename in (
        "experiments/b200_sdpa_lens/start.py",
        "experiments/b200_sdpa_lens/server.py",
        "experiments/b200_sdpa_lens/README.md",
        "experiments/b200_vllm_lens/smoke.py",
        "src/gleipnir/serving/lens.py",
        "src/gleipnir/serving/lens_api.py",
        "src/gleipnir/serving/lens_worker.py",
        "src/gleipnir/serving/lens_projection.py",
        "src/gleipnir/serving/lens_readout.py",
        "src/gleipnir/serving/bf16_worker.py",
    ):
        sources[filename] = file_hash(ROOT / filename)
        copy = out / "executed_sources" / filename
        copy.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / filename, copy)
    write_json(
        out / "manifest.json",
        {
            "config_sha256": file_hash(config_path),
            "sources": sources,
            "inputs": config["inputs"],
        },
    )
    shutil.copyfile(config_path, out / "config.json")
    # A stopped container's PID is not a live identity on this new pod.
    if (SERVING / "server.json").exists():
        shutil.copyfile(SERVING / "server.json", out / "terminated_pod_server.json")
        (SERVING / "server.json").unlink()
    command = lens_command(parent, config["model"])
    binding_index = command.index("--additional-config") + 1
    additional = json.loads(command[binding_index])
    additional["gleipnir_frost_fp4"].update(sources)
    command[binding_index] = json.dumps(additional, sort_keys=True)

    env = candidate_environment(ROOT)
    env["PYTHONPATH"] += ":" + str(OVERLAY)
    env.update(
        VLLM_LENS_DISABLE="1",
        VLLM_GDN_DECODE_KERNEL="cuda",
        GLEIPNIR_FLASHINFER_GDN_CP="auto",
        GLEIPNIR_BF16_AUDIT=str(out / "bf16_audit.json"),
        GLEIPNIR_GIGATOKEN_RECEIPT=str(out / "frontend.json"),
        GLEIPNIR_VLLM031_SCHEDULER_BINDING=json.dumps(binding),
    )
    env.pop("GLEIPNIR_FROST_WRAPPER_VALIDATION", None)
    started = time.perf_counter()
    with log.open("x") as handle:
        process = subprocess.Popen(
            command,
            cwd=ROOT,
            env=env,
            stdout=handle,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    active = {
        **parent,
        "pid": process.pid,
        "command": command,
        "status": "starting",
        "log": str(log),
        "lens_research": True,
        "enforce_eager": True,
        "serving_default": "sdpa_bf16_eager_lens",
        "launched_at_unix": time.time(),
        "frontend": parent["frontend"] | {"receipt_path": str(out / "frontend.json")},
    }
    active.pop("ready_at_unix", None)
    write_json(SERVING / "server.json", active)
    write_json(out / "server.json", active)
    try:
        async with httpx.AsyncClient(
            base_url="http://127.0.0.1:8010", trust_env=False, timeout=300
        ) as client:
            while True:
                if process.poll() is not None:
                    raise RuntimeError(f"Lens server exited {process.returncode}")
                try:
                    if (await client.get("/health")).status_code == 200:
                        break
                except httpx.HTTPError:
                    pass
                if time.perf_counter() - started > 1200:
                    raise TimeoutError("Lens startup exceeded twenty minutes")
                await asyncio.sleep(2)
            response = await client.get("/v1/monitor/lens/info")
            response.raise_for_status()
            info = response.json()
            if info["layers"] != list(range(32)) or info["hidden_size"] != 2560:
                raise ValueError("Lens model geometry changed")
            write_json(out / "lens_info.json", info)
        active.update(status="ready", ready_at_unix=time.time())
        write_json(SERVING / "server.json", active)
        write_json(out / "server.json", active)
        print("sdpa_lens_ready", process.pid, flush=True)
        workload = read_rows(inputs["canary"])
        master = json.loads(inputs["master_reference"].read_text())
        previous = json.loads(inputs["compiled_predictions"].read_text())
        if [r["prompt_sha256"] for r in workload] != master["prompt_sha256"]:
            raise ValueError("master canary identity changed")
        values, timing = await score_batch(
            workload, {"port": 8010, "concurrency": 4, "timeout_seconds": 300}
        )
        if [(v["id"], v["prompt_sha256"], v["prompt_tokens"]) for v in values] != [
            (v["id"], v["prompt_sha256"], v["prompt_tokens"]) for v in previous
        ]:
            raise ValueError("compiled/eager canary identity changed")
        gate = agreement(
            [v["score"] for v in values],
            master["adapter"],
            master["base"],
            config["parity"],
        )
        gate["versus_compiled_bf16"] = agreement(
            [v["score"] for v in values],
            [v["score"] for v in previous],
            master["base"],
            config["parity"],
        )
        gate.update(
            research_eager_finite=gate["finite"] and gate["adapter_effect"] > 0,
            mode="user_authorized_bf16_eager_lens",
        )
        write_json(out / "canary_predictions.json", values)
        write_json(out / "parity.json", gate)
        write_json(out / "canary_timing.json", timing)
        native = json.loads((out / "bf16_audit.json").read_text())
        if (
            not gate["passed"]
            or not native["passed"]
            or len(native["attention_calls"]) != 8
        ):
            raise ValueError("BF16 eager master/native gate failed")
        write_json(out / "smoke.json", await smoke(out, reproduction=gate))
        await verify(out)
        for filename, sha in sources.items():
            if file_hash(ROOT / filename) != sha:
                raise ValueError(f"source changed during smoke: {filename}")
        write_json(
            out / "complete.json",
            {
                "passed": True,
                "master_sha256": adapter["master_sha256"],
                "adapter_sha256": adapter["serving_sha256"],
                "server_pid": process.pid,
                "startup_seconds": active["ready_at_unix"] - active["launched_at_unix"],
                "server_retained_warm": True,
            },
        )
        print("sdpa_lens_complete", flush=True)
    except BaseException as error:
        write_json(
            out / "failure.json", {"type": type(error).__name__, "message": str(error)}
        )
        if process.poll() is None:
            subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "experiments.b200_vllm031.stop",
                    "--archive-name",
                    "sdpa_lens_" + name + "_failed",
                ],
                env=env,
                cwd=ROOT,
                check=True,
            )
        else:
            (SERVING / "server.json").unlink(missing_ok=True)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", required=True)
    parser.add_argument("--config", type=Path, default=CONFIG)
    args = parser.parse_args()
    if not args.name or args.name in {".", ".."} or Path(args.name).name != args.name:
        raise ValueError("startup name must be a stem")
    sys.path.insert(0, str(OVERLAY))
    os.environ["VLLM_LENS_DISABLE"] = "1"
    asyncio.run(start(args.name, config_path=args.config))


if __name__ == "__main__":
    main()
