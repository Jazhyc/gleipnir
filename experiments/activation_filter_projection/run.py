"""Transfer the fixed APPS projection to the activation-filtered checkpoint."""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

from experiments.b200_injection_direction.run import apps_report, score_rows
from experiments.b200_injection_direction.smoke import main as operator_smoke
from experiments.b200_sdpa_lens.start import OVERLAY, start
from gleipnir.data.monitoring import file_hash, read_rows, write_json

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent


def validate_projection(edit: dict, unit: np.ndarray) -> None:
    """Require the exact frozen unit axis and all-layer zero-centered removal."""
    if (
        len(edit["directional_edits"]) != 1
        or edit["directional_edits"][0]["layer_indices"] != list(range(32))
        or edit["directional_edits"][0]["beta"] != 1
        or any(
            edit["directional_edits"][0][k] != [0.0] * 32
            for k in ["decision_centers", "span_centers"]
        )
        or unit.shape != (2560,)
        or not np.isfinite(unit).all()
        or not np.isclose(np.linalg.norm(unit), 1, atol=1e-6)
        or not np.array_equal(
            np.array(edit["directional_edits"][0]["direction"], dtype=np.float32), unit
        )
    ):
        raise ValueError("projection geometry or frozen direction drift")


async def run() -> None:
    config = json.loads((HERE / "config.json").read_text())
    out = ROOT / "results/activation_filter_projection" / config["campaign_id"]
    out.mkdir(parents=True, exist_ok=False)
    try:
        inputs = {}
        for name, spec in config["inputs"].items():
            p = ROOT / spec["path"]
            if file_hash(p) != spec["sha256"]:
                raise ValueError(f"input drift: {name}")
            inputs[name] = p
        rows = read_rows(inputs["workload"])
        partition = json.loads(inputs["partition"].read_text())
        fit = set(partition["fit_tasks"])
        if (
            len(rows) != 8218
            or len({r["id"] for r in rows}) != 8218
            or any(str(r["metadata"]["task_id"]) in fit for r in rows)
        ):
            raise ValueError("APPS population/fit-task leakage")
        edit = json.loads(inputs["interventions"].read_text())["project"]
        validate_projection(edit, np.load(inputs["directions"])["unit"])
        shutil.copyfile(inputs["partition"], out / "partition.json")
        sources = {
            str(p.relative_to(ROOT)): file_hash(p)
            for p in [
                *HERE.glob("*.*"),
                ROOT / "experiments/b200_injection_direction/run.py",
                ROOT / "experiments/b200_injection_direction/smoke.py",
                ROOT / "src/gleipnir/evaluation/apps.py",
            ]
        }
        for filename in sources:
            p = out / "executed_sources" / filename
            p.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(ROOT / filename, p)
        write_json(
            out / "manifest.json",
            {"inputs": config["inputs"], "sources": sources, "intervention": edit},
        )
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
        await start(config["startup_name"], config_path=HERE / "startup.json")
        operator_smoke(out / "operator_smoke.json")
        server = json.loads(
            (ROOT / "results/b200_attention_gdn_serving/server.json").read_text()
        )
        write_json(out / "server.json", server)
        scores = {}
        for arm, intervention in [("plain", None), ("project", edit)]:
            if any(file_hash(ROOT / p) != sha for p, sha in sources.items()):
                raise ValueError("source drift during scoring")
            scores[arm] = await score_rows(config, out, rows, arm, "apps", intervention)
        plain = apps_report(scores["plain"])
        report = {
            "apps": {},
            "qualification": (
                "Fixed historical axis, new activation-filtered checkpoint, "
                "same held-out tasks and matched eager baseline; "
                "no refitting or promotion."
            ),
        }
        for arm in scores:
            report["apps"][arm] = {
                "recalibrated": apps_report(scores[arm]),
                "fixed_unedited": apps_report(scores[arm], plain["threshold"]),
            }
        cached = {
            r["id"]: r
            for key in ["compiled_benchmark", "compiled_controls"]
            for r in read_rows(inputs[key])
        }
        diffs = [r["score"] - cached[r["id"]]["score"] for r in scores["plain"]]
        for row in scores["plain"]:
            if row["prompt_sha256"] != cached[row["id"]]["prompt_sha256"]:
                raise ValueError("compiled/eager prompt mismatch")
        report["eager_minus_compiled"] = {
            "mean_absolute_difference": float(np.mean(np.abs(diffs))),
            "mean_difference": float(np.mean(diffs)),
            "maximum_absolute_difference": float(np.max(np.abs(diffs))),
        }
        report["project_minus_plain"] = {
            "mean_difference": float(
                np.mean(
                    [
                        a["score"] - b["score"]
                        for a, b in zip(scores["project"], scores["plain"], strict=True)
                    ]
                )
            )
        }
        report["answer_mass"] = {
            arm: {
                "rows": len(v := [r["p01"] for r in scores[arm] if "p01" in r]),
                "minimum": min(v),
                "mean": float(np.mean(v)),
                "below_half": sum(x < 0.5 for x in v),
            }
            for arm in scores
        }
        from gleipnir.serving.lens import MonitorLensClient

        info = MonitorLensClient("http://127.0.0.1:8010").info()
        if any(
            info[k] != 0
            for k in ["capture_requests", "steering_requests", "projection_requests"]
        ):
            raise ValueError("Lens request state leaked")
        write_json(out / "summary.json", report)
        write_json(
            out / "complete.json",
            {
                "passed": True,
                "rows_per_arm": 8218,
                "files_sha256": {
                    f"apps_{arm}.jsonl": file_hash(out / f"apps_{arm}.jsonl")
                    for arm in scores
                },
                "server_pid": server["pid"],
            },
        )
        write_json(out / "status.json", {"stage": "complete"})
        print("activation_filter_projection_complete", flush=True)
    except BaseException as error:
        write_json(
            out / "failure.json", {"type": type(error).__name__, "message": str(error)}
        )
        raise


if __name__ == "__main__":
    sys.path.insert(0, str(OVERLAY))
    os.environ["VLLM_LENS_DISABLE"] = "1"
    asyncio.run(run())
