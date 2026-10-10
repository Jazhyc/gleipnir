"""Fit and project a CAFT-model injection direction on frozen APPS tasks."""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

from experiments.activation_filter_projection.run import validate_projection
from experiments.activation_filter_refit.run import check_groups
from experiments.b200_injection_direction.run import apps_report, fit, score_rows
from experiments.b200_injection_direction.smoke import main as operator_smoke
from experiments.b200_sdpa_lens.start import OVERLAY, start
from gleipnir.data.monitoring import file_hash, read_rows, write_json

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent


async def run() -> None:
    config = json.loads((HERE / "config.json").read_text())
    out = ROOT / "results/caft_direction_refit" / config["campaign_id"]
    out.mkdir(parents=True, exist_ok=False)
    try:
        inputs = {}
        for name, spec in config["inputs"].items():
            path = ROOT / spec["path"]
            if file_hash(path) != spec["sha256"]:
                raise ValueError(f"input drift: {name}")
            inputs[name] = path
        fitting, testing = read_rows(inputs["fitting"]), read_rows(inputs["testing"])
        partition = json.loads(inputs["partition"].read_text())
        check_groups(fitting, testing, partition)
        if len(fitting) != 896 or len(testing) != 8218:
            raise ValueError("fit/test population drift")
        startup = json.loads(inputs["startup"].read_text())
        parent = json.loads(
            (ROOT / startup["inputs"]["parent_server"]["path"]).read_text()
        )
        active = json.loads(
            (ROOT / "results/b200_attention_gdn_serving/server.json").read_text()
        )
        actual = [
            v.decode()
            for v in Path(f"/proc/{parent['pid']}/cmdline").read_bytes().split(b"\0")
            if v
        ]
        if active != parent or actual != parent["command"]:
            raise ValueError("compiled parent server identity drift")
        shutil.copyfile(inputs["partition"], out / "partition.json")
        paths = [
            *HERE.glob("*.*"),
            ROOT / "experiments/activation_filter_projection/run.py",
            ROOT / "experiments/activation_filter_refit/run.py",
            ROOT / "experiments/b200_injection_direction/run.py",
            ROOT / "experiments/b200_injection_direction/smoke.py",
            ROOT / "src/gleipnir/evaluation/apps.py",
        ]
        sources = {str(p.relative_to(ROOT)): file_hash(p) for p in paths}
        for filename in sources:
            copy = out / "executed_sources" / filename
            copy.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(ROOT / filename, copy)
        write_json(out / "manifest.json", {"sources": sources, "inputs": config})
        subprocess.run(
            [
                sys.executable,
                "-m",
                "experiments.b200_vllm031.stop",
                "--archive-name",
                config["campaign_id"] + "_before_lens",
            ],
            check=True,
        )
        await start(config["startup_name"], config_path=inputs["startup"])
        operator_smoke(out / "operator_smoke.json")
        server = json.loads(
            (ROOT / "results/b200_attention_gdn_serving/server.json").read_text()
        )
        write_json(out / "server.json", server)
        await score_rows(config, out, fitting, "plain", "fit", capture=True)
        edits = fit(config, out, fitting)
        directions = np.load(out / "directions.npz")
        unit = directions["unit"]
        validate_projection(edits["project"], unit)
        geometry = {
            "new_natural_shift_norm": float(np.linalg.norm(directions["u"][20]))
        }
        for name in ("original", "filtered"):
            previous = np.load(inputs[name + "_directions"])
            geometry[name] = {
                "unit_cosine": float(unit @ previous["unit"]),
                "natural_shift_norm": float(np.linalg.norm(previous["u"][20])),
            }
        write_json(out / "geometry.json", geometry)
        print("caft_refit_geometry", json.dumps(geometry), flush=True)
        scores = {}
        for arm, intervention in [("plain", None), ("project", edits["project"])]:
            if any(file_hash(ROOT / p) != sha for p, sha in sources.items()):
                raise ValueError("source drift")
            scores[arm] = await score_rows(
                config, out, testing, arm, "apps", intervention
            )
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
        cached = {
            r["id"]: r
            for name in ("compiled_benchmark", "compiled_controls")
            for r in read_rows(inputs[name])
        }
        if any(
            r["prompt_sha256"] != cached[r["id"]]["prompt_sha256"]
            for r in scores["plain"]
        ):
            raise ValueError("compiled/eager prompt drift")
        delta = [r["score"] - cached[r["id"]]["score"] for r in scores["plain"]]
        report["eager_minus_compiled"] = {
            "mae": float(np.mean(np.abs(delta))),
            "mean": float(np.mean(delta)),
            "maximum_absolute_difference": float(np.max(np.abs(delta))),
        }
        report["project_minus_plain_mean_score"] = float(
            np.mean(
                [
                    a["score"] - b["score"]
                    for a, b in zip(scores["project"], scores["plain"], strict=True)
                ]
            )
        )
        report["answer_mass"] = {}
        for arm, rows in scores.items():
            v = [r["p01"] for r in rows if "p01" in r]
            report["answer_mass"][arm] = {
                "rows": len(v),
                "minimum": min(v),
                "mean": float(np.mean(v)),
                "below_half": sum(p < 0.5 for p in v),
            }
        report["cached_comparisons"] = {
            "original_augmented": json.loads(inputs["original_summary"].read_text())[
                "apps"
            ],
            "activation_filtered": json.loads(inputs["filtered_summary"].read_text())[
                "apps"
            ],
        }
        from gleipnir.serving.lens import MonitorLensClient

        info = MonitorLensClient("http://127.0.0.1:8010").info()
        if any(
            info[k] != 0
            for k in ("capture_requests", "steering_requests", "projection_requests")
        ):
            raise ValueError("request state leaked")
        write_json(out / "summary.json", report)
        write_json(
            out / "complete.json",
            {
                "passed": True,
                "fit_rows": 896,
                "test_rows_per_arm": 8218,
                "server_pid": server["pid"],
                "files_sha256": {
                    n: file_hash(out / n)
                    for n in (
                        "fit_plain.jsonl",
                        "fit_activations.npz",
                        "directions.npz",
                        "interventions.json",
                        "apps_plain.jsonl",
                        "apps_project.jsonl",
                    )
                },
            },
        )
        write_json(out / "status.json", {"stage": "complete"})
        print("caft_direction_refit_complete", flush=True)
    except BaseException as error:
        write_json(
            out / "failure.json", {"type": type(error).__name__, "message": str(error)}
        )
        raise


if __name__ == "__main__":
    sys.path.insert(0, str(OVERLAY))
    os.environ["VLLM_LENS_DISABLE"] = "1"
    asyncio.run(run())
