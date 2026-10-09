"""Capture an unchanged direction and fixed controls on the existing warm engine."""

from __future__ import annotations

import asyncio
import json
import math
import os
import shutil
import sys
import time
from pathlib import Path

import httpx
import numpy as np

from gleipnir.data.monitoring import file_hash, read_rows, write_json, write_rows
from gleipnir.serving.monitor_score import validate_score_response

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).parent
CONFIG = HERE / "config.json"
ACTIVE = ROOT / "results/b200_attention_gdn_serving/server.json"


def sources() -> dict:
    paths = (
        list(HERE.glob("*.py"))
        + [HERE / "README.md"]
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
    for name, sha in m["workloads"].items():
        if file_hash(out / name) != sha:
            raise ValueError("workload/control receipt drift")
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


async def capture(
    config: dict,
    out: Path,
    rows: list[dict],
    population: str,
    unit: np.ndarray,
    edit: dict | None,
) -> list[dict]:
    from vllm_lens._helpers._serialize import deserialize_tensor

    folder = out / "batches" / population
    folder.mkdir(parents=True, exist_ok=True)
    sem = asyncio.Semaphore(config["concurrency"])
    values = []
    async with httpx.AsyncClient(
        base_url="http://127.0.0.1:8010",
        trust_env=False,
        timeout=config["timeout_seconds"],
        limits=httpx.Limits(max_connections=config["concurrency"]),
    ) as client:

        async def one(row):
            payload = {
                "model": "monitor",
                "prompt": row["prompt"],
                "capture_layers": [20, 31],
                "capture_positions": "last",
                "full_readout": True,
            }
            if edit:
                payload.update(edit)
            started = time.perf_counter()
            async with sem:
                response = await client.post("/v1/monitor/lens", json=payload)
            response.raise_for_status()
            body = response.json()
            validate_score_response(body, row["prompt_tokens"])
            if body["activation_layers"] != [20, 31] or body[
                "activation_positions"
            ] != [row["prompt_tokens"] - 1]:
                raise ValueError("capture layer/token coverage changed")
            residual = (
                deserialize_tensor(body["activations"]["residual_stream"])
                .float()
                .numpy()[:, 0]
            )
            if residual.shape != (2, 2560) or not np.isfinite(residual).all():
                raise ValueError("invalid captured residual")
            # FP32 preserves every BF16 value, including exponents outside FP16.
            saved = residual.copy()
            j = body["judge_readout"]
            if (
                j["token_ids"] != [32, 33]
                or not all(math.isfinite(v) for v in j["logits"])
                or not math.isfinite(j["probability_mass"])
                or not 0 <= j["probability_mass"] <= 1
            ):
                raise ValueError("invalid A/B readout")
            margin = j["logits"][1] - j["logits"][0]
            probability = (
                1 / (1 + math.exp(-margin))
                if margin >= 0
                else math.exp(margin) / (1 + math.exp(margin))
            )
            value = {
                k: row[k] for k in ["id", "prompt_sha256", "prompt_tokens"]
            } | row.get("metadata", {})
            value.update(
                logits=j["logits"],
                margin=margin,
                score=probability,
                pab=j["probability_mass"],
                latency_seconds=time.perf_counter() - started,
            )
            if "label" in value:
                value["correct_margin_ab"] = (2 * value["label"] - 1) * margin
            for i, layer in enumerate([20, 31]):
                z = float(residual[i] @ unit)
                norm = float(np.linalg.norm(residual[i]))
                value[f"z{layer}"] = z
                value[f"norm{layer}"] = norm
                if norm <= 0 or not math.isfinite(z) or not math.isfinite(norm):
                    raise ValueError("invalid direction component")
                if (
                    population == "project"
                    and abs(z) / norm > config["projected_max_component_fraction"]
                ):
                    raise ValueError("projection component exceeds BF16 allowance")
            return value, saved

        for offset in range(0, len(rows), config["batch_rows"]):
            batch = rows[offset : offset + config["batch_rows"]]
            path = folder / f"{offset:06d}.json"
            tensor = path.with_suffix(".npz")
            if path.exists():
                saved = json.loads(path.read_text())
                batch_values = saved["values"]
                if file_hash(tensor) != saved["tensor_sha256"] or [
                    (r["id"], r["prompt_sha256"], r["prompt_tokens"])
                    for r in batch_values
                ] != [(r["id"], r["prompt_sha256"], r["prompt_tokens"]) for r in batch]:
                    raise ValueError("resumed batch drift")
            else:
                started = time.perf_counter()
                results = await asyncio.gather(*(one(row) for row in batch))
                batch_values = [r for r, a in results]
                np.savez(tensor, residual=np.stack([a for r, a in results]))
                write_json(
                    path,
                    {
                        "values": batch_values,
                        "seconds": time.perf_counter() - started,
                        "tokens": sum(r["prompt_tokens"] for r in batch),
                        "tensor_sha256": file_hash(tensor),
                    },
                )
            values.extend(batch_values)
            write_json(
                out / "status.json",
                {
                    "stage": "capture",
                    "population": population,
                    "rows": len(values),
                    "total": len(rows),
                },
            )
            print(
                "judge_signal_capture", population, len(values), len(rows), flush=True
            )
    write_rows(out / (population + ".jsonl"), values)
    return values


async def run() -> None:
    config = json.loads(CONFIG.read_text())
    out = ROOT / "results/b200_judge_direction_signal" / config["campaign_id"]
    if not (out / "prepare_receipt.json").exists():
        raise ValueError("prepare before capture")
    for spec in config["inputs"].values():
        if file_hash(ROOT / spec["path"]) != spec["sha256"]:
            raise ValueError("input drift " + spec["path"])
    for key in ["startup", "readout_gate"]:
        if not json.loads((ROOT / config["inputs"][key]["path"]).read_text())["passed"]:
            raise ValueError("parent gate failed")
    parent = json.loads((ROOT / config["inputs"]["judge_manifest"]["path"]).read_text())
    if json.loads(ACTIVE.read_text()) != parent["server"]:
        raise ValueError("requires unchanged passing engine")
    for path, sha in parent["sources"].items():
        if (
            path.startswith("src/gleipnir/serving/lens")
            and file_hash(ROOT / path) != sha
        ):
            raise ValueError("resident Lens source drift")
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
    prepared = json.loads((out / "prepare_receipt.json").read_text())
    for name, sha in prepared["workload_sha256"].items():
        if file_hash(out / name) != sha:
            raise ValueError("prepared workload changed")
    if not (out / "manifest.json").exists():
        m = {
            "config_sha256": file_hash(CONFIG),
            "sources": sources(),
            "server": parent["server"],
            "workloads": prepared["workload_sha256"]
            | {"prepare_receipt.json": file_hash(out / "prepare_receipt.json")},
        }
        for path in m["sources"]:
            dest = out / "executed_sources" / path
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(ROOT / path, dest)
        write_json(out / "manifest.json", m)
        shutil.copyfile(CONFIG, out / "config.json")
    checked(config, out)
    previous = read_rows(ROOT / config["inputs"]["judge_canary"]["path"])
    canary = read_rows(ROOT / config["inputs"]["canary_workload"]["path"])
    if [r["id"] for r in canary] != [r["id"] for r in previous]:
        raise ValueError("canary identity changed")
    actual = await capture(config, out, canary, "canary", unit, None)
    x = np.array([r["score"] for r in actual])
    y = np.array([r["score"] for r in previous])
    corr = float(np.corrcoef(x, y)[0, 1])
    mae = float(np.mean(abs(x - y)))
    gate = {
        "mae": mae,
        "correlation": corr,
        "passed": math.isfinite(corr)
        and mae <= config["canary_max_mae"]
        and corr >= config["canary_min_correlation"],
    }
    write_json(out / "capture_gate.json", gate)
    if not gate["passed"]:
        raise ValueError("capture no-op reproduction failed")
    short = [min(canary, key=lambda r: r["prompt_tokens"])]
    base = await capture(config, out, short, "smoke_base", unit, None)
    zero = await capture(
        config,
        out,
        short,
        "smoke_zero",
        unit,
        {"directional_edits": [e | {"beta": 0.0}]},
    )
    restored = await capture(config, out, short, "smoke_plain", unit, None)
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
            raise ValueError("capture beta-zero/restoration changed")
    write_json(
        out / "smoke.json", {"passed": True, "zero_and_following_plain_exact": True}
    )
    for name, workload, intervention in [
        ("plain", "original", None),
        ("project", "original", edit),
        ("natural", "natural", None),
        ("whitespace", "whitespace", None),
    ]:
        checked(config, out)
        await capture(
            config,
            out,
            read_rows(out / (workload + "_workload.jsonl")),
            name,
            unit,
            intervention,
        )
    checked(config, out)
    write_json(out / "status.json", {"stage": "analyzing"})
    from experiments.b200_judge_direction_signal.analyze import analyze

    analyze(out)
    async with httpx.AsyncClient(trust_env=False, timeout=30) as client:
        response = await client.get("http://127.0.0.1:8010/v1/monitor/lens/info")
        response.raise_for_status()
        info = response.json()
    if any(
        info[k]
        for k in ["capture_requests", "steering_requests", "projection_requests"]
    ):
        raise ValueError("request state not empty")
    write_json(out / "closure_info.json", info)
    write_json(out / "status.json", {"stage": "complete"})
    print("judge_signal_complete", flush=True)


def main() -> None:
    sys.path.insert(0, "/tmp/gleipnir-vllm-lens-1.3.0")
    out = (
        ROOT
        / "results/b200_judge_direction_signal"
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
