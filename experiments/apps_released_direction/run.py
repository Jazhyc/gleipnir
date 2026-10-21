"""Injection-direction projection on the released Gleipnir 4B (APPS).

Serves the released 4B's merged BF16 checkpoint unquantized and eager with the
Lens extension, gates it against the recorded BF16 ID scores, refits the
layer-20 injected-minus-neutral direction on the frozen 64-task fit split and
scores the 591 disjoint test tasks under four arms: plain, refitted
projection, the cached original augmented-model projection and a random
projection. Reuses the injection-direction fitter, scorer and report.
"""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import httpx
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
OVERLAY = Path("/tmp/gleipnir-vllm-lens-1.3.0")
PORT = 8010


def checked_inputs(config: dict) -> dict[str, Path]:
    from gleipnir.data.monitoring import file_hash

    inputs = {}
    optimized = json.loads(
        (ROOT / "results/apps_released_monitors/apps01/4b/complete.json").read_text()
    )["files_sha256"]
    for name, spec in config["inputs"].items():
        path = ROOT / spec["path"]
        expected = spec["sha256"] or optimized[path.stem]
        if file_hash(path) != expected:
            raise ValueError(f"input drift: {name}")
        inputs[name] = path
    return inputs


def lens_command(template: dict, model: str) -> list[str]:
    """The project's BF16 Qwen3.5-4B command on another merged checkpoint."""
    command = list(template["command"])
    old = command[command.index("--model") + 1]
    if template.get("serving_precision") != "bf16" or (
        command[command.index("--worker-cls") + 1]
        != "gleipnir.serving.bf16_worker.Bf16Worker"
    ):
        raise ValueError("template must be the unquantized BF16 command")
    index = command.index("--additional-config") + 1
    additional = json.loads(command[index])
    additional["serving_condition"]["merged_model"] = model
    command[index] = json.dumps(additional, sort_keys=True)
    command[command.index("--model") + 1] = model
    if old in " ".join(command):
        raise ValueError("template model path survived substitution")
    command[0] = sys.executable
    command[2] = "experiments.b200_sdpa_lens.server"
    return command + [
        "--enforce-eager",
        "--worker-extension-cls",
        "gleipnir.serving.lens_worker.MonitorLensExtension",
    ]


async def start_server(config: dict, inputs: dict, out: Path) -> subprocess.Popen:
    from experiments.b200_vllm031.runtime import candidate_environment
    from gleipnir.data.monitoring import file_hash

    merged = json.loads(inputs["merged_artifact"].read_text())
    for filename, sha in merged["files_sha256"].items():
        if file_hash(Path(config["model"]) / filename) != sha:
            raise ValueError(f"merged model drift: {filename}")
    command = lens_command(
        json.loads(inputs["bf16_command_template"].read_text()), config["model"]
    )
    env = candidate_environment(ROOT)
    env["PYTHONPATH"] += ":" + str(OVERLAY)
    env.update(
        VLLM_LENS_DISABLE="1",
        VLLM_GDN_DECODE_KERNEL="cuda",
        GLEIPNIR_FLASHINFER_GDN_CP="auto",
        GLEIPNIR_BF16_AUDIT=str(out / "bf16_audit.json"),
        GLEIPNIR_GIGATOKEN_RECEIPT=str(out / "frontend.json"),
    )
    log = ROOT / "logs/runpod/apps_released_direction/server.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    with log.open("a") as handle:
        process = subprocess.Popen(
            command,
            cwd=ROOT,
            env=env,
            stdout=handle,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    from gleipnir.data.monitoring import write_json

    write_json(out / "server.json", {"pid": process.pid, "command": command})
    async with httpx.AsyncClient(
        base_url=f"http://127.0.0.1:{PORT}", trust_env=False, timeout=60
    ) as client:
        while True:
            if process.poll() is not None:
                raise RuntimeError(f"Lens server exited {process.returncode}")
            try:
                if (await client.get("/health")).status_code == 200:
                    break
            except httpx.HTTPError:
                pass
            if time.perf_counter() - started > 1800:
                raise TimeoutError("Lens startup exceeded thirty minutes")
            await asyncio.sleep(3)
        info = (await client.get("/v1/monitor/lens/info")).json()
    if info["layers"] != list(range(32)) or info["hidden_size"] != 2560:
        raise ValueError("Lens model geometry changed")
    write_json(out / "lens_info.json", info)
    print("lens_ready", process.pid, time.perf_counter() - started, flush=True)
    return process


async def id_gate(config: dict, inputs: dict, out: Path) -> dict:
    """Eager BF16 Lens must reproduce the recorded BF16 ID scores."""
    from sklearn.metrics import roc_auc_score

    from gleipnir.data.monitoring import read_rows, write_json, write_rows
    from gleipnir.evaluation.http_score import score_batch

    workload = json.loads(inputs["id_workload"].read_text())
    recorded = {r["id"]: r for r in read_rows(inputs["bf16_id_reference"])}
    values, seconds = await score_batch(
        workload,
        {"port": PORT, "concurrency": config["concurrency"], "timeout_seconds": 600},
    )
    write_rows(out / "id_plain.jsonl", values)
    if any(v["prompt_sha256"] != recorded[v["id"]]["prompt_sha256"] for v in values):
        raise ValueError("ID prompt identity drift")
    a = np.array([v["score"] for v in values])
    b = np.array([recorded[v["id"]]["score"] for v in values])
    labels = np.array([w["label"] for w in workload])
    gate = {
        "rows": len(values),
        "mae": float(np.abs(a - b).mean()),
        "max_abs": float(np.abs(a - b).max()),
        "correlation": float(np.corrcoef(a, b)[0, 1]),
        "pooled_auroc": float(roc_auc_score(labels, a)),
        "recorded_pooled_auroc": float(roc_auc_score(labels, b)),
        "seconds": seconds,
        "limits": config["parity"],
    }
    gate["passed"] = bool(
        np.isfinite(a).all()
        and gate["mae"] <= config["parity"]["max_mean_absolute_difference"]
        and gate["correlation"] >= config["parity"]["min_correlation"]
    )
    native = json.loads((out / "bf16_audit.json").read_text())
    gate["native_bf16_audit"] = bool(
        native["passed"] and len(native["attention_calls"]) == 8
    )
    gate["passed"] = gate["passed"] and gate["native_bf16_audit"]
    write_json(out / "id_gate.json", gate)
    print("id_gate", json.dumps(gate), flush=True)
    if not gate["passed"]:
        raise ValueError("eager BF16 Lens failed the recorded BF16 ID gate")
    return gate


async def run() -> None:
    from experiments.activation_filter_projection.run import validate_projection
    from experiments.b200_injection_direction.run import apps_report, fit, score_rows
    from experiments.b200_injection_direction.smoke import main as operator_smoke
    from gleipnir.data.monitoring import file_hash, read_rows, write_json
    from gleipnir.serving.lens import MonitorLensClient

    config = json.loads((HERE / "config.json").read_text())
    out = ROOT / "results/apps_released_direction" / config["campaign_id"]
    out.mkdir(parents=True, exist_ok=True)
    inputs = checked_inputs(config)
    fitting, testing = read_rows(inputs["fitting"]), read_rows(inputs["testing"])
    partition = json.loads(inputs["partition"].read_text())
    fit_tasks = {str(r["metadata"]["task_id"]) for r in fitting}
    if (
        len(fitting) != 896
        or len(testing) != 8218
        or fit_tasks & {str(r["metadata"]["task_id"]) for r in testing}
        or fit_tasks != {str(t) for t in partition["fit_tasks"]}
    ):
        raise ValueError("fit/test population drift or leakage")
    shutil.copyfile(inputs["partition"], out / "partition.json")
    sources = {
        str(p.relative_to(ROOT)): file_hash(p)
        for p in [
            *HERE.glob("*.*"),
            ROOT / "experiments/b200_injection_direction/run.py",
            ROOT / "experiments/b200_injection_direction/smoke.py",
            ROOT / "experiments/activation_filter_projection/run.py",
            ROOT / "experiments/b200_sdpa_lens/server.py",
            ROOT / "src/gleipnir/serving/lens_api.py",
            ROOT / "src/gleipnir/serving/lens_worker.py",
            ROOT / "src/gleipnir/serving/lens_projection.py",
            ROOT / "src/gleipnir/evaluation/apps.py",
        ]
    }
    write_json(out / "manifest.json", {"sources": sources, "config": config})
    process = await start_server(config, inputs, out)
    try:
        await id_gate(config, inputs, out)
        operator_smoke(out / "operator_smoke.json")
        await score_rows(config, out, fitting, "plain", "fit", capture=True)
        edits = fit(config, out, fitting)
        directions = np.load(out / "directions.npz")
        validate_projection(edits["project"], directions["unit"])
        original = json.loads(inputs["original_interventions"].read_text())["project"]
        original_unit = np.load(inputs["original_directions"])["unit"]
        validate_projection(original, original_unit)
        geometry = {
            "refit_natural_shift_norm": float(
                np.linalg.norm(directions["u"][config["direction_layer"]])
            ),
            "original_natural_shift_norm": float(
                np.linalg.norm(
                    np.load(inputs["original_directions"])["u"][
                        config["direction_layer"]
                    ]
                )
            ),
            "refit_original_cosine": float(directions["unit"] @ original_unit),
        }
        write_json(out / "geometry.json", geometry)
        print("geometry", json.dumps(geometry), flush=True)
        arms = {
            "plain": None,
            "project": edits["project"],
            "original": original,
            "random": edits["random"],
        }
        scores = {}
        for arm, edit in arms.items():
            if any(file_hash(ROOT / p) != sha for p, sha in sources.items()):
                raise ValueError("source drift")
            scores[arm] = await score_rows(config, out, testing, arm, "apps", edit)
        plain = apps_report(scores["plain"])
        optimized = {
            r["id"]: r["score"]
            for name in ("optimized_benchmark", "optimized_controls")
            for r in read_rows(inputs[name])
        }
        delta = np.array([r["score"] - optimized[r["id"]] for r in scores["plain"]])
        report = {
            "apps": {
                arm: {
                    "recalibrated": apps_report(rows),
                    "fixed_plain": apps_report(rows, plain["threshold"]),
                    "mean_shift_vs_plain": float(
                        np.mean(
                            [
                                a["score"] - b["score"]
                                for a, b in zip(rows, scores["plain"], strict=True)
                            ]
                        )
                    ),
                }
                for arm, rows in scores.items()
            },
            "geometry": geometry,
            "eager_bf16_minus_optimized": {
                "mae": float(np.abs(delta).mean()),
                "mean": float(delta.mean()),
                "max_abs": float(np.abs(delta).max()),
            },
            "answer_mass": {
                arm: {
                    "rows": len(v := [r["p01"] for r in rows if "p01" in r]),
                    "minimum": min(v),
                    "mean": float(np.mean(v)),
                    "below_half": sum(p < 0.5 for p in v),
                }
                for arm, rows in scores.items()
            },
            "cached_original_model": json.loads(inputs["original_summary"].read_text())[
                "apps"
            ],
        }
        info = MonitorLensClient(f"http://127.0.0.1:{PORT}").info()
        if any(
            info[k] != 0
            for k in ("capture_requests", "steering_requests", "projection_requests")
        ):
            raise ValueError("Lens request state leaked")
        write_json(out / "summary.json", report)
        write_json(
            out / "complete.json",
            {
                "passed": True,
                "fit_rows": len(fitting),
                "test_rows_per_arm": len(testing),
                "files_sha256": {
                    n: file_hash(out / n)
                    for n in (
                        "fit_plain.jsonl",
                        "directions.npz",
                        "interventions.json",
                        *(f"apps_{arm}.jsonl" for arm in arms),
                    )
                },
            },
        )
        print("released_direction_complete", flush=True)
    except BaseException as error:
        write_json(
            out / "failure.json", {"type": type(error).__name__, "message": str(error)}
        )
        raise
    finally:
        os.killpg(process.pid, 15)
        process.wait(timeout=120)


if __name__ == "__main__":
    sys.path.insert(0, str(OVERLAY))
    os.environ["VLLM_LENS_DISABLE"] = "1"
    asyncio.run(run())
