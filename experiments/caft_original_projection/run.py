"""Score the original fixed projection on the warm CAFT Lens model."""

from __future__ import annotations

import asyncio
import importlib.metadata
import json
import os
import shutil
import sys
from pathlib import Path

import numpy as np

from experiments.activation_filter_projection.run import validate_projection
from experiments.b200_injection_direction.run import apps_report, score_rows
from gleipnir.data.monitoring import file_hash, read_rows, write_json
from gleipnir.evaluation.http_score import score_batch

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent


async def run() -> None:
    config = json.loads((HERE / "config.json").read_text())
    out = ROOT / "results/caft_original_projection" / config["campaign_id"]
    out.mkdir(parents=True, exist_ok=False)
    try:
        inputs = {}
        for name, spec in config["inputs"].items():
            path = ROOT / spec["path"]
            if file_hash(path) != spec["sha256"]:
                raise ValueError(f"input drift: {name}")
            inputs[name] = path
        server = json.loads(inputs["server"].read_text())
        active = json.loads(
            (ROOT / "results/b200_attention_gdn_serving/server.json").read_text()
        )
        actual = [
            v.decode()
            for v in Path(f"/proc/{server['pid']}/cmdline").read_bytes().split(b"\0")
            if v
        ]
        if (
            active != server
            or actual != server["command"]
            or os.getpgid(server["pid"]) != server["pid"]
        ):
            raise ValueError("warm Lens server identity drift")
        startup = json.loads(inputs["startup_config"].read_text())
        if {k: importlib.metadata.version(k) for k in startup["runtime"]} != startup[
            "runtime"
        ]:
            raise ValueError("serving runtime drift")
        if not json.loads(inputs["startup_complete"].read_text())["passed"]:
            raise ValueError("warm startup incomplete")
        for name, expected in json.loads(inputs["startup_manifest"].read_text())[
            "sources"
        ].items():
            if file_hash(ROOT / name) != expected:
                raise ValueError(f"Lens source drift: {name}")
        rows = read_rows(inputs["workload"])
        partition = json.loads(inputs["partition"].read_text())
        if (
            len(rows) != 8218
            or len({r["id"] for r in rows}) != 8218
            or {str(r["metadata"]["task_id"]) for r in rows}
            != set(partition["test_tasks"])
            or set(partition["test_tasks"]) & set(partition["fit_tasks"])
        ):
            raise ValueError("test population/fit leakage")
        edit = json.loads(inputs["interventions"].read_text())["project"]
        validate_projection(edit, np.load(inputs["directions"])["unit"])
        training = json.loads(inputs["training_metadata"].read_text())[
            "concept_ablation"
        ]
        if training["archive_sha256"] != config["inputs"]["directions"][
            "sha256"
        ] or training["applied_layers"] != list(range(32)):
            raise ValueError("original projection differs from training direction")
        shutil.copyfile(inputs["partition"], out / "partition.json")
        paths = [
            *HERE.glob("*.*"),
            ROOT / "experiments/b200_injection_direction/run.py",
            ROOT / "experiments/activation_filter_projection/run.py",
            ROOT / "src/gleipnir/evaluation/apps.py",
        ]
        sources = {str(p.relative_to(ROOT)): file_hash(p) for p in paths}
        for name in sources:
            copy = out / "executed_sources" / name
            copy.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(ROOT / name, copy)
        write_json(
            out / "manifest.json",
            {"sources": sources, "config": config, "server": server},
        )
        old = json.loads(inputs["canary_predictions"].read_text())
        values, seconds = await score_batch(
            read_rows(inputs["canary"]),
            {"port": 8010, "concurrency": 4, "timeout_seconds": 300},
        )
        if [(r["id"], r["prompt_sha256"], r["prompt_tokens"]) for r in values] != [
            (r["id"], r["prompt_sha256"], r["prompt_tokens"]) for r in old
        ]:
            raise ValueError("no-op prompt drift")
        x, y = (np.array([r["score"] for r in v]) for v in [values, old])
        mae, corr = float(np.mean(abs(x - y))), float(np.corrcoef(x, y)[0, 1])
        gate = {
            "passed": bool(np.isfinite(x).all() and mae <= 0.005 and corr >= 0.995),
            "mae": mae,
            "correlation": corr,
            "seconds": seconds,
        }
        write_json(out / "no_op.json", gate)
        if not gate["passed"]:
            raise ValueError("warm no-op failed")
        print("original_projection_no_op", mae, corr, flush=True)
        projected = await score_rows(config, out, rows, "project", "apps", edit)
        scores = {
            "plain": read_rows(inputs["plain"]),
            "new_project": read_rows(inputs["new_project"]),
            "original_project": projected,
        }
        for values in scores.values():
            if [(r["id"], r["prompt_sha256"], r["prompt_tokens"]) for r in values] != [
                (r["id"], r["prompt_sha256"], r["prompt_tokens"]) for r in rows
            ]:
                raise ValueError("cached/new score population drift")
        threshold = apps_report(scores["plain"])["threshold"]
        report = {
            "apps": {
                arm: {
                    "recalibrated": apps_report(values),
                    "fixed_unedited": apps_report(values, threshold),
                }
                for arm, values in scores.items()
            }
        }
        report["original_minus_plain_mean_score"] = float(
            np.mean(
                [
                    a["score"] - b["score"]
                    for a, b in zip(projected, scores["plain"], strict=True)
                ]
            )
        )
        mass = [r["p01"] for r in projected if "p01" in r]
        report["answer_mass"] = {
            "rows": len(mass),
            "minimum": min(mass),
            "mean": float(np.mean(mass)),
            "below_half": sum(p < 0.5 for p in mass),
        }
        from gleipnir.serving.lens import MonitorLensClient

        info = MonitorLensClient("http://127.0.0.1:8010").info()
        if any(
            info[k] != 0
            for k in ("capture_requests", "steering_requests", "projection_requests")
        ):
            raise ValueError("request state leaked")
        if any(file_hash(ROOT / p) != expected for p, expected in sources.items()):
            raise ValueError("source drift")
        write_json(out / "summary.json", report)
        write_json(
            out / "complete.json",
            {
                "passed": True,
                "test_rows": 8218,
                "server_pid": server["pid"],
                "project_sha256": file_hash(out / "apps_project.jsonl"),
                "same_direction_as_training": True,
                "reused_same_engine_baselines": True,
            },
        )
        write_json(out / "status.json", {"stage": "complete"})
        print("caft_original_projection_complete", flush=True)
    except BaseException as error:
        write_json(
            out / "failure.json", {"type": type(error).__name__, "message": str(error)}
        )
        raise


if __name__ == "__main__":
    sys.path.insert(0, "/tmp/gleipnir-vllm-lens-1.3.0")
    os.environ["VLLM_LENS_DISABLE"] = "1"
    asyncio.run(run())
