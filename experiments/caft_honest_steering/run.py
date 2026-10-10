"""Add the frozen augmented axis to clean/neutral honest CAFT inputs."""

from __future__ import annotations

import asyncio
import importlib.metadata
import json
import os
import shutil
import sys
from pathlib import Path

import numpy as np
import torch

from experiments.b200_injection_direction.run import score_rows
from experiments.caft_honest_steering.metrics import report
from gleipnir.data.monitoring import file_hash, read_rows, write_json
from gleipnir.evaluation.http_score import score_batch

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent


async def run() -> None:
    from vllm_lens import SteeringVector

    config = json.loads((HERE / "config.json").read_text())
    out = ROOT / "results/caft_honest_steering" / config["campaign_id"]
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
            raise ValueError("warm server identity drift")
        startup = json.loads(inputs["startup_config"].read_text())
        if {k: importlib.metadata.version(k) for k in startup["runtime"]} != startup[
            "runtime"
        ] or not json.loads(inputs["startup_complete"].read_text())["passed"]:
            raise ValueError("startup/runtime drift")
        for name, expected in json.loads(inputs["startup_manifest"].read_text())[
            "sources"
        ].items():
            if file_hash(ROOT / name) != expected:
                raise ValueError(f"Lens source drift: {name}")

        def select(row: dict) -> bool:
            meta = row.get("metadata", row)
            return (
                meta["ground_truth"] == 0 and meta["condition"] in config["conditions"]
            )

        rows = [r for r in read_rows(inputs["workload"]) if select(r)]
        baseline = [r for r in read_rows(inputs["plain"]) if select(r)]
        partition = json.loads(inputs["partition"].read_text())
        if (
            len(rows) != 2364
            or {str(r["metadata"]["task_id"]) for r in rows}
            != set(partition["test_tasks"])
            or set(partition["fit_tasks"]) & set(partition["test_tasks"])
        ):
            raise ValueError("honest population/fit leakage")
        threshold = json.loads(inputs["plain_summary"].read_text())["apps"]["plain"][
            "recalibrated"
        ]["threshold"]
        report(baseline, baseline, threshold)
        shutil.copyfile(inputs["partition"], out / "partition.json")
        paths = [
            *HERE.glob("*.*"),
            ROOT / "experiments/b200_injection_direction/run.py",
            ROOT / "src/gleipnir/serving/lens.py",
            ROOT / "src/gleipnir/evaluation/calibration.py",
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
            raise ValueError("canary prompt drift")
        x, y = (np.array([r["score"] for r in v]) for v in (values, old))
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
        print("honest_steering_no_op", mae, corr, flush=True)
        dirs = np.load(inputs["directions"])
        shift = dirs["u"][20]
        norm = float(np.linalg.norm(shift))
        positive = json.loads(inputs["interventions"].read_text())["add_positive"][
            "steering_vectors"
        ][0]
        vector = SteeringVector.model_validate(positive)
        if (
            not np.array_equal(vector.activations.float().numpy(), shift[None])
            or vector.layer_indices != [20]
            or vector.scale != 1
            or vector.norm_match
            or vector.position_indices is not None
        ):
            raise ValueError("original additive geometry drift")
        random = SteeringVector(
            activations=torch.from_numpy((dirs["random"] * norm)[None]),
            layer_indices=[20],
            scale=1,
            norm_match=False,
        ).model_dump(mode="json")
        edits = {
            "positive_1": {"steering_vectors": [positive]},
            "positive_2": {"steering_vectors": [positive | {"scale": 2}]},
            "random_1": {"steering_vectors": [random]},
            "random_2": {"steering_vectors": [random | {"scale": 2}]},
        }
        if not np.isfinite(norm) or not np.isclose(
            norm, config["natural_shift_norm"], atol=1e-6
        ):
            raise ValueError("additive norm drift")
        write_json(out / "interventions.json", edits)
        result = {
            "threshold": threshold,
            "natural_shift_norm": norm,
            "arms": {"plain": report(baseline, baseline, threshold)},
            "answer_mass": {},
        }
        for arm, edit in edits.items():
            if any(file_hash(ROOT / p) != expected for p, expected in sources.items()):
                raise ValueError("source drift")
            scored = await score_rows(config, out, rows, arm, "apps", edit)
            result["arms"][arm] = report(scored, baseline, threshold)
            mass = [r["p01"] for r in scored if "p01" in r]
            if len(mass) != 512:
                raise ValueError("answer-mass population drift")
            result["answer_mass"][arm] = {
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
        write_json(out / "summary.json", result)
        write_json(
            out / "complete.json",
            {
                "passed": True,
                "rows_per_arm": 2364,
                "scored_rows": 9456,
                "server_pid": server["pid"],
                "files_sha256": {
                    arm: file_hash(out / f"apps_{arm}.jsonl") for arm in edits
                },
                "baseline_reused": True,
            },
        )
        write_json(out / "status.json", {"stage": "complete"})
        print("caft_honest_steering_complete", flush=True)
    except BaseException as error:
        write_json(
            out / "failure.json", {"type": type(error).__name__, "message": str(error)}
        )
        raise


if __name__ == "__main__":
    sys.path.insert(0, "/tmp/gleipnir-vllm-lens-1.3.0")
    os.environ["VLLM_LENS_DISABLE"] = "1"
    asyncio.run(run())
