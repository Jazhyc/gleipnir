"""Fit a model-specific paired injection axis, then evaluate its projection."""

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
from experiments.b200_injection_direction.run import apps_report, fit, score_rows
from gleipnir.data.monitoring import file_hash, read_rows, write_json

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent


def check_groups(fitting: list[dict], testing: list[dict], partition: dict) -> None:
    """Keep every derived task variant on the frozen side of the split."""
    a = {str(r["metadata"]["task_id"]) for r in fitting}
    b = {str(r["metadata"]["task_id"]) for r in testing}
    if a & b or a != set(partition["fit_tasks"]) or b != set(partition["test_tasks"]):
        raise ValueError("fit/test task membership or leakage")
    for rows in [fitting, testing]:
        if len({r["id"] for r in rows}) != len(rows):
            raise ValueError("duplicate prompt identity")


async def run() -> None:
    config = json.loads((HERE / "config.json").read_text())
    out = ROOT / "results/activation_filter_refit" / config["campaign_id"]
    out.mkdir(parents=True, exist_ok=False)
    try:
        inputs = {}
        for name, spec in config["inputs"].items():
            p = ROOT / spec["path"]
            if file_hash(p) != spec["sha256"]:
                raise ValueError(f"input drift: {name}")
            inputs[name] = p
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
            raise ValueError("warm server identity drift")
        startup = json.loads(inputs["startup_config"].read_text())
        if {k: importlib.metadata.version(k) for k in startup["runtime"]} != startup[
            "runtime"
        ]:
            raise ValueError("serving runtime drift")
        if not json.loads(inputs["startup_complete"].read_text())["passed"]:
            raise ValueError("warm startup lacks passing gates")
        for path, sha in json.loads(inputs["startup_manifest"].read_text())[
            "sources"
        ].items():
            if file_hash(ROOT / path) != sha:
                raise ValueError("warm integration source drift")
        fitting, testing = read_rows(inputs["fitting"]), read_rows(inputs["testing"])
        partition = json.loads(inputs["partition"].read_text())
        check_groups(fitting, testing, partition)
        if len(fitting) != 896 or len(testing) != 8218:
            raise ValueError("population drift")
        shutil.copyfile(inputs["partition"], out / "partition.json")
        sources = {
            str(p.relative_to(ROOT)): file_hash(p)
            for p in [
                *HERE.glob("*.*"),
                ROOT / "experiments/b200_injection_direction/run.py",
                ROOT / "experiments/activation_filter_projection/run.py",
                ROOT / "src/gleipnir/evaluation/apps.py",
            ]
        }
        for filename in sources:
            p = out / "executed_sources" / filename
            p.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(ROOT / filename, p)
        write_json(
            out / "manifest.json",
            {"sources": sources, "inputs": config["inputs"], "server": server},
        )
        from gleipnir.evaluation.http_score import score_batch

        old = json.loads(inputs["canary_predictions"].read_text())
        values, timing = await score_batch(
            read_rows(inputs["canary"]),
            {"port": 8010, "concurrency": 4, "timeout_seconds": 300},
        )
        if [(r["id"], r["prompt_sha256"], r["prompt_tokens"]) for r in values] != [
            (r["id"], r["prompt_sha256"], r["prompt_tokens"]) for r in old
        ]:
            raise ValueError("no-op prompt drift")
        x, y = (
            np.array([r["score"] for r in values]),
            np.array([r["score"] for r in old]),
        )
        mae, corr = float(np.mean(abs(x - y))), float(np.corrcoef(x, y)[0, 1])
        gate = {
            "passed": bool(np.isfinite(x).all() and mae <= 0.005 and corr >= 0.995),
            "mae": mae,
            "correlation": corr,
            "timing": timing,
        }
        write_json(out / "no_op.json", gate)
        if not gate["passed"]:
            raise ValueError("warm no-op failed")
        print("refit_no_op_passed", mae, corr, flush=True)
        await score_rows(config, out, fitting, "plain", "fit", capture=True)
        edits = fit(config, out, fitting)
        unit = np.load(out / "directions.npz")["unit"]
        validate_projection(edits["project"], unit)
        old_dirs = np.load(inputs["old_directions"])
        geometry = {
            "unit_cosine_to_old": float(unit @ old_dirs["unit"]),
            "new_natural_shift_norm": float(
                np.linalg.norm(np.load(out / "directions.npz")["u"][20])
            ),
            "old_natural_shift_norm": float(np.linalg.norm(old_dirs["u"][20])),
        }
        write_json(out / "geometry.json", geometry)
        print("refit_geometry", json.dumps(geometry), flush=True)
        if any(file_hash(ROOT / p) != sha for p, sha in sources.items()):
            raise ValueError("source drift")
        projected = await score_rows(
            config, out, testing, "project", "apps", edits["project"]
        )
        scores = {
            "plain": read_rows(inputs["plain"]),
            "old_project": read_rows(inputs["old_project"]),
            "new_project": projected,
        }
        for rows in scores.values():
            if [(r["id"], r["prompt_sha256"], r["prompt_tokens"]) for r in rows] != [
                (r["id"], r["prompt_sha256"], r["prompt_tokens"]) for r in testing
            ]:
                raise ValueError("cached/new population drift")
        plain = apps_report(scores["plain"])
        report = {
            "apps": {
                arm: {
                    "recalibrated": apps_report(rows),
                    "fixed_unedited": apps_report(rows, plain["threshold"]),
                }
                for arm, rows in scores.items()
            },
            "geometry": geometry,
        }
        v = [r["p01"] for r in projected if "p01" in r]
        report["answer_mass"] = {
            "rows": len(v),
            "minimum": min(v),
            "mean": float(np.mean(v)),
            "below_half": sum(p < 0.5 for p in v),
        }
        report["new_minus_plain_mean_score"] = float(
            np.mean(
                [
                    a["score"] - b["score"]
                    for a, b in zip(projected, scores["plain"], strict=True)
                ]
            )
        )
        from gleipnir.serving.lens import MonitorLensClient

        info = MonitorLensClient("http://127.0.0.1:8010").info()
        if any(
            info[k] != 0
            for k in ["capture_requests", "steering_requests", "projection_requests"]
        ):
            raise ValueError("request state leaked")
        write_json(out / "summary.json", report)
        write_json(
            out / "complete.json",
            {
                "passed": True,
                "fit_rows": 896,
                "test_rows": 8218,
                "files_sha256": {
                    n: file_hash(out / n)
                    for n in [
                        "fit_plain.jsonl",
                        "fit_activations.npz",
                        "directions.npz",
                        "interventions.json",
                        "apps_project.jsonl",
                    ]
                },
                "reused_same_engine_baselines": True,
                "server_pid": server["pid"],
            },
        )
        write_json(out / "status.json", {"stage": "complete"})
        print("activation_filter_refit_complete", flush=True)
    except BaseException as error:
        write_json(
            out / "failure.json", {"type": type(error).__name__, "message": str(error)}
        )
        raise


if __name__ == "__main__":
    sys.path.insert(0, "/tmp/gleipnir-vllm-lens-1.3.0")
    os.environ["VLLM_LENS_DISABLE"] = "1"
    asyncio.run(run())
