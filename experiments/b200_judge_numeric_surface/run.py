"""Run fixed numeric judging surfaces on the unchanged warm monitor."""

from __future__ import annotations

import asyncio
import json
import math
import os
import shutil
import sys
from pathlib import Path

import httpx
import numpy as np

from gleipnir.data.monitoring import file_hash, read_rows, write_json
from gleipnir.evaluation.lens_capture import capture_rows

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).parent
CONFIG = HERE / "config.json"
ACTIVE = ROOT / "results/b200_attention_gdn_serving/server.json"


def sources() -> dict:
    paths = (
        list(HERE.glob("*.py"))
        + [
            HERE / "README.md",
            ROOT / "src/gleipnir/evaluation/lens_capture.py",
            ROOT / "src/gleipnir/evaluation/preferences.py",
            ROOT / "src/gleipnir/evaluation/metrics.py",
        ]
        + list((ROOT / "src/gleipnir/serving").glob("lens*.py"))
    )
    return {str(p.relative_to(ROOT)): file_hash(p) for p in paths}


def checked(config: dict, out: Path) -> None:
    m = json.loads((out / "manifest.json").read_text())
    if file_hash(CONFIG) != m["config_sha256"] or sources() != m["sources"]:
        raise ValueError("source/config drift")
    for spec in config["inputs"].values():
        if file_hash(ROOT / spec["path"]) != spec["sha256"]:
            raise ValueError("input drift " + spec["path"])
    for name, digest in m["workloads"].items():
        if file_hash(out / name) != digest:
            raise ValueError("workload/receipt drift")
    server = json.loads(ACTIVE.read_text())
    actual = [
        v.decode()
        for v in Path(f"/proc/{server['pid']}/cmdline").read_bytes().split(b"\0")
        if v
    ]
    if (
        server != m["server"]
        or actual != server["command"]
        or os.getpgid(server["pid"]) != server["pid"]
    ):
        raise ValueError("resident identity drift")


async def run() -> None:
    config = json.loads(CONFIG.read_text())
    out = ROOT / "results/b200_judge_numeric_surface" / config["campaign_id"]
    if not (out / "prepare_receipt.json").exists():
        raise ValueError("prepare before scoring")
    for spec in config["inputs"].values():
        if file_hash(ROOT / spec["path"]) != spec["sha256"]:
            raise ValueError("input drift")
    for key in ["readout_gate", "startup"]:
        if not json.loads((ROOT / config["inputs"][key]["path"]).read_text())["passed"]:
            raise ValueError("parent gate failed")
    parent = json.loads(
        (ROOT / config["inputs"]["parent_manifest"]["path"]).read_text()
    )
    if json.loads(ACTIVE.read_text()) != parent["server"]:
        raise ValueError("requires unchanged engine")
    for path, digest in parent["sources"].items():
        if (
            path.startswith("src/gleipnir/serving/lens")
            and file_hash(ROOT / path) != digest
        ):
            raise ValueError("resident Lens source drift")
    receipt = json.loads((out / "prepare_receipt.json").read_text())
    if not receipt["candidate_and_attack_bytes_preserved"]:
        raise ValueError("transformation failed")
    for name, digest in receipt["workloads"].items():
        if file_hash(out / name) != digest:
            raise ValueError("prepared transformation drift")
    if not (out / "manifest.json").exists():
        m = {
            "config_sha256": file_hash(CONFIG),
            "sources": sources(),
            "server": parent["server"],
            "workloads": receipt["workloads"]
            | {"prepare_receipt.json": file_hash(out / "prepare_receipt.json")},
        }
        for path in m["sources"]:
            target = out / "executed_sources" / path
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(ROOT / path, target)
        write_json(out / "manifest.json", m)
        shutil.copyfile(CONFIG, out / "config.json")
    checked(config, out)
    unit = np.load(ROOT / config["inputs"]["directions"]["path"])["unit"]
    edit = json.loads((ROOT / config["inputs"]["interventions"]["path"]).read_text())[
        "project"
    ]
    e = edit["directional_edits"][0]
    if (
        e["beta"] != 1
        or e["layer_indices"] != list(range(32))
        or any(e["decision_centers"] + e["span_centers"])
        or not np.array_equal(e["direction"], unit)
    ):
        raise ValueError("projection differs from completed arm")

    async def capture(rows, pop, intervention=None):
        return await capture_rows(
            rows,
            out,
            pop,
            surface="01",
            unit=unit,
            intervention=intervention,
            concurrency=config["concurrency"],
            batch_rows=config["batch_rows"],
            timeout_seconds=config["timeout_seconds"],
            max_projected_component_fraction=config["projected_max_component_fraction"]
            if intervention and intervention["directional_edits"][0]["beta"] == 1
            else None,
        )

    canary = read_rows(ROOT / config["inputs"]["canary_workload"]["path"])
    previous = read_rows(ROOT / config["inputs"]["canary_scores"]["path"])
    actual = await capture(canary, "canary")
    if [(r["id"], r["prompt_sha256"]) for r in actual] != [
        (r["id"], r["prompt_sha256"]) for r in previous
    ]:
        raise ValueError("canary identity drift")
    x = np.array([r["score"] for r in actual])
    y = np.array([r["monitor_score"] for r in previous])
    mae = float(np.mean(abs(x - y)))
    corr = float(np.corrcoef(x, y)[0, 1])
    gate = {
        "mae": mae,
        "correlation": corr,
        "passed": math.isfinite(corr)
        and mae <= config["canary_max_mae"]
        and corr >= config["canary_min_correlation"],
    }
    write_json(out / "canary_gate.json", gate)
    if not gate["passed"]:
        raise ValueError("numeric cached-head reproduction failed")
    short = [
        min(
            read_rows(out / "fully_numeric_workload.jsonl"),
            key=lambda r: r["prompt_tokens"],
        )
    ]
    base = await capture(short, "smoke_base")
    zero = await capture(
        short, "smoke_zero", {"directional_edits": [e | {"beta": 0.0}]}
    )
    restored = await capture(short, "smoke_plain")
    with (
        np.load(out / "batches/smoke_base/000000.npz") as a,
        np.load(out / "batches/smoke_zero/000000.npz") as b,
        np.load(out / "batches/smoke_plain/000000.npz") as c,
    ):
        if (
            not np.array_equal(a["residual"], b["residual"])
            or not np.array_equal(a["residual"], c["residual"])
            or base[0]["logits"] != zero[0]["logits"]
            or base[0]["logits"] != restored[0]["logits"]
        ):
            raise ValueError("numeric zero/restoration smoke failed")
    write_json(
        out / "smoke.json", {"passed": True, "zero_and_following_plain_exact": True}
    )
    for variant in config["variants"]:
        workload = read_rows(out / (variant + "_workload.jsonl"))
        for arm, intervention in [("plain", None), ("project", edit)]:
            checked(config, out)
            await capture(workload, variant + "_" + arm, intervention)
    checked(config, out)
    write_json(out / "status.json", {"stage": "analyzing"})
    from experiments.b200_judge_numeric_surface.analyze import analyze

    analyze(out, ROOT, config)
    async with httpx.AsyncClient(trust_env=False, timeout=30) as client:
        response = await client.get("http://127.0.0.1:8010/v1/monitor/lens/info")
        response.raise_for_status()
        info = response.json()
    if any(
        info[k]
        for k in ["capture_requests", "steering_requests", "projection_requests"]
    ):
        raise ValueError("request state retained")
    write_json(out / "closure_info.json", info)
    write_json(out / "status.json", {"stage": "complete"})
    print("numeric_judge_complete", flush=True)


def main() -> None:
    sys.path.insert(0, "/tmp/gleipnir-vllm-lens-1.3.0")
    out = (
        ROOT
        / "results/b200_judge_numeric_surface"
        / json.loads(CONFIG.read_text())["campaign_id"]
    )
    try:
        asyncio.run(run())
    except BaseException as error:
        out.mkdir(parents=True, exist_ok=True)
        write_json(
            out / "failure.json", {"type": type(error).__name__, "message": str(error)}
        )
        raise


if __name__ == "__main__":
    main()
