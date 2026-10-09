"""Resumable small-fit injection directions and fixed APPS/ID interventions."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import math
import os
import shutil
import sys
import time
from pathlib import Path

import httpx
import numpy as np
import torch

from experiments.b200_injection_direction.data import CONFIG, ROOT
from gleipnir.campaigns.monitoring.evaluation import id_metrics
from gleipnir.data.monitoring import file_hash, read_rows, write_json, write_rows
from gleipnir.evaluation.apps import summarize_apps
from gleipnir.serving.monitor_score import validate_score_response


def output(config: dict) -> Path:
    return ROOT / "results/b200_injection_direction" / config["campaign_id"]


def source_files() -> dict[str, str]:
    paths = list(Path(__file__).parent.glob("*.py")) + [
        Path(__file__).parent / "README.md"
    ]
    paths += [
        ROOT / "src/gleipnir/serving" / n
        for n in ["lens.py", "lens_api.py", "lens_worker.py", "lens_projection.py"]
    ]
    paths += [ROOT / "src/gleipnir/evaluation/apps.py"]
    return {str(p.relative_to(ROOT)): file_hash(p) for p in paths}


def checked(config: dict, out: Path) -> None:
    manifest = json.loads((out / "manifest.json").read_text())
    if (
        file_hash(CONFIG) != manifest["config_sha256"]
        or source_files() != manifest["sources"]
    ):
        raise ValueError("source/config drift")
    for name, sha in manifest["workloads"].items():
        if file_hash(out / name) != sha:
            raise ValueError("workload drift " + name)
    server = json.loads(
        (ROOT / "results/b200_attention_gdn_serving/server.json").read_text()
    )
    if server != manifest["server"]:
        raise ValueError("resident identity changed")
    actual = [
        v.decode()
        for v in Path(f"/proc/{server['pid']}/cmdline").read_bytes().split(b"\0")
        if v
    ]
    if actual != server["command"] or os.getpgid(server["pid"]) != server["pid"]:
        raise ValueError("resident process drift")
    for spec in config["inputs"].values():
        if file_hash(ROOT / spec["path"]) != spec["sha256"]:
            raise ValueError("frozen input drift")


def freeze(config: dict, out: Path) -> None:
    server = json.loads(
        (ROOT / "results/b200_attention_gdn_serving/server.json").read_text()
    )
    if (
        server["status"] != "ready"
        or server.get("serving_precision") != "bf16"
        or not server.get("lens_research")
        or "--enforce-eager" not in server["command"]
    ):
        raise ValueError("requires passed eager BF16 Lens resident")
    start = ROOT / "results/b200_sdpa_lens/sdpa02_projection"
    if not json.loads((start / "complete.json").read_text())["passed"]:
        raise ValueError("projection startup failed")
    a = json.loads((start / "canary_predictions.json").read_text())
    b = json.loads(
        (ROOT / config["inputs"]["previous_eager_canary"]["path"]).read_text()
    )
    if [(v["id"], v["prompt_sha256"]) for v in a] != [
        (v["id"], v["prompt_sha256"]) for v in b
    ]:
        raise ValueError("no-op canary changed")
    x = np.array([v["score"] for v in a])
    y = np.array([v["score"] for v in b])
    mae = float(abs(x - y).mean())
    corr = float(np.corrcoef(x, y)[0, 1])
    if mae > 0.005 or corr < 0.995:
        raise ValueError("projection no-op reproduction failed")
    paths = [p for p in out.glob("*_workload.jsonl")] + [out / "partition.json"]
    manifest = {
        "config_sha256": file_hash(CONFIG),
        "sources": source_files(),
        "workloads": {p.name: file_hash(p) for p in paths},
        "server": server,
        "no_op_mae": mae,
        "no_op_correlation": corr,
    }
    if (out / "manifest.json").exists():
        checked(config, out)
        return
    for name in manifest["sources"]:
        p = out / "executed_sources" / name
        p.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / name, p)
    write_json(out / "manifest.json", manifest)


async def score_rows(
    config, out, rows, arm, population, intervention=None, capture=False
):
    from vllm_lens._helpers._serialize import deserialize_tensor

    directory = out / "batches" / population / arm
    directory.mkdir(parents=True, exist_ok=True)
    batch_size = config["capture_batch_rows"] if capture else config["batch_rows"]
    concurrency = config["capture_concurrency"] if capture else config["concurrency"]
    add_tasks = set(json.loads((out / "partition.json").read_text())["add_tasks"])
    semaphore = asyncio.Semaphore(concurrency)
    all_values = []
    decisions = []
    span_means = []
    probe = {}
    async with httpx.AsyncClient(
        base_url="http://127.0.0.1:8010",
        trust_env=False,
        timeout=300,
        limits=httpx.Limits(max_connections=concurrency),
    ) as client:

        async def one(row):
            task = str(row.get("metadata", {}).get("task_id", ""))
            diagnostic = (
                not capture and population in ("apps", "add") and task in add_tasks
            )
            layers = (
                list(range(32))
                if capture or (diagnostic and arm == "plain")
                else [31]
                if diagnostic
                else []
            )
            payload = {
                "model": "monitor",
                "prompt": row["prompt"],
                "capture_layers": layers,
                "capture_positions": "last",
                "capture_span_positions": row.get("span_positions", [])
                if capture
                else [],
                "full_readout": diagnostic,
            }
            if intervention:
                payload.update(intervention)
            before = time.perf_counter()
            async with semaphore:
                response = await client.post("/v1/monitor/lens", json=payload)
            response.raise_for_status()
            body = response.json()
            validated = validate_score_response(body, row["prompt_tokens"])
            value = {k: row[k] for k in ["id", "prompt_sha256", "prompt_tokens"]}
            value.update(row.get("metadata", {}))
            value.update(validated)
            value.update(
                logits=body["logits"], latency_seconds=time.perf_counter() - before
            )
            if "dataset" in row:
                value.update(source_dataset=row["dataset"], ground_truth=row["label"])
            activations = {
                k: deserialize_tensor(v).float().numpy()
                for k, v in body.get("activations", {}).items()
            }
            if capture and (
                activations["residual_stream"].shape != (32, 1, 2560)
                or not all(np.isfinite(v).all() for v in activations.values())
            ):
                raise ValueError("fit capture shape/finite failure")
            if diagnostic:
                if (
                    not math.isfinite(body["p01"])
                    or not 0 <= body["p01"] <= 1
                    or max(
                        abs(a - b)
                        for a, b in zip(
                            body["readout_logits"], body["logits"], strict=True
                        )
                    )
                    > 0.25
                ):
                    raise ValueError("full-vocabulary readout mismatch")
                value.update(p01=body["p01"], readout_logits=body["readout_logits"])
            return value, activations

        for offset in range(0, len(rows), batch_size):
            batch = rows[offset : offset + batch_size]
            path = directory / f"{offset:06d}.json"
            tensor_path = path.with_suffix(".npz")
            if path.exists():
                saved = json.loads(path.read_text())
                if (
                    saved["tensor_sha256"] is not None
                    and file_hash(tensor_path) != saved["tensor_sha256"]
                ):
                    raise ValueError("resumed activation hash drift")
                values = saved["values"]
                tensors = np.load(tensor_path) if tensor_path.exists() else None
                if [
                    (v["id"], v["prompt_sha256"], v["prompt_tokens"]) for v in values
                ] != [(v["id"], v["prompt_sha256"], v["prompt_tokens"]) for v in batch]:
                    raise ValueError("resume batch identity drift")
            else:
                before = time.perf_counter()
                results = await asyncio.gather(*(one(r) for r in batch))
                values = [v for v, _ in results]
                tensors = {}
                if capture:
                    tensors = {
                        "decision": np.stack(
                            [a["residual_stream"][:, 0] for _, a in results]
                        ).astype(np.float16),
                        "span": np.stack(
                            [
                                a.get(
                                    "residual_span_mean",
                                    np.zeros((32, 2560), np.float32),
                                )
                                for _, a in results
                            ]
                        ),
                    }
                elif arm == "plain" and population == "apps":
                    tensors = {
                        r["id"]: a["residual_stream"][:, 0].astype(np.float16)
                        for r, (_, a) in zip(batch, results, strict=True)
                        if a
                    }
                if tensors:
                    np.savez(tensor_path, **tensors)
                write_json(
                    path,
                    {
                        "values": values,
                        "seconds": time.perf_counter() - before,
                        "tokens": sum(r["prompt_tokens"] for r in batch),
                        "tensor_sha256": file_hash(tensor_path) if tensors else None,
                    },
                )
            if tensors is not None:
                if capture:
                    decisions.append(tensors["decision"])
                    span_means.append(tensors["span"])
                elif arm == "plain" and population == "apps":
                    probe.update({k: tensors[k] for k in tensors})
            all_values.extend(values)
            write_json(
                out / "status.json",
                {
                    "stage": population,
                    "arm": arm,
                    "rows": len(all_values),
                    "total": len(rows),
                },
            )
            print(
                "injection_direction_progress",
                population,
                arm,
                len(all_values),
                len(rows),
                flush=True,
            )
    write_rows(out / f"{population}_{arm}.jsonl", all_values)
    if capture:
        np.savez(
            out / "fit_activations.npz",
            decision=np.concatenate(decisions),
            span=np.concatenate(span_means),
        )
    if probe:
        np.savez(out / "test_probe_activations.npz", **probe)
    return all_values


def fit(config: dict, out: Path, rows: list[dict]) -> dict:
    from vllm_lens import SteeringVector

    acts = np.load(out / "fit_activations.npz")
    dec = acts["decision"].astype(np.float32)
    span = acts["span"].astype(np.float32)
    at = {
        (
            str(r["metadata"]["task_id"]),
            r["metadata"]["ground_truth"],
            r["metadata"]["condition"],
        ): i
        for i, r in enumerate(rows)
    }
    pairs = []
    neutral = []
    bd = []
    for i, r in enumerate(rows):
        task = str(r["metadata"]["task_id"])
        label = r["metadata"]["ground_truth"]
        condition = r["metadata"]["condition"]
        if condition.startswith("inj_"):
            pairs.append((i, at[(task, label, "neutral_" + condition[4:])]))
        if condition.startswith("neutral_"):
            neutral.append(i)
            if label == 1:
                bd.append((i, at[(task, 0, condition)]))
    if len(pairs) != 384:
        raise ValueError("fit pair coverage drift")
    a, b = zip(*pairs, strict=True)
    u = (dec[list(a)] - dec[list(b)]).mean(0)
    t = u[config["direction_layer"]]
    norm = float(np.linalg.norm(t))
    if not np.isfinite(norm) or norm <= 1e-6:
        raise ValueError("degenerate injection direction")
    t = t / norm
    v = (dec[[a for a, b in bd]] - dec[[b for a, b in bd]]).mean(0)
    dc = (dec[neutral] @ t).mean(0)
    sc = (span[neutral] @ t).mean(0)
    rng = np.random.default_rng(config["random_seed"])
    random = rng.standard_normal(2560).astype(np.float32)
    random /= np.linalg.norm(random)
    base = {
        "direction": t.tolist(),
        "layer_indices": list(range(32)),
        "decision_centers": dc.tolist(),
        "span_centers": sc.tolist(),
        "beta": config["clamp_beta"],
    }
    project = base | {
        "decision_centers": [0.0] * 32,
        "span_centers": [0.0] * 32,
        "beta": 1.0,
    }
    edits = {
        "plain": None,
        "clamp": {"directional_edits": [base]},
        "project": {"directional_edits": [project]},
        "random": {"directional_edits": [project | {"direction": random.tolist()}]},
    }
    for name, scale in [("add_positive", 1.0), ("add_negative", -1.0)]:
        vector = SteeringVector(
            activations=torch.from_numpy(u[config["direction_layer"]][None]),
            layer_indices=[config["direction_layer"]],
            scale=scale,
        )
        edits[name] = {"steering_vectors": [vector.model_dump(mode="json")]}
    np.savez(
        out / "directions.npz",
        u=u,
        v=v,
        unit=t,
        random=random,
        decision_centers=dc,
        span_centers=sc,
    )
    write_json(out / "interventions.json", edits)
    write_json(
        out / "fit_receipt.json",
        {
            "pairs": len(pairs),
            "neutral_rows": len(neutral),
            "natural_shift_norm": norm,
            "fit_ids_sha256": hashlib.sha256(
                json.dumps([r["id"] for r in rows]).encode()
            ).hexdigest(),
            "directions_sha256": file_hash(out / "directions.npz"),
            "interventions_sha256": file_hash(out / "interventions.json"),
        },
    )
    return edits


def apps_report(rows: list[dict], threshold: float | None = None) -> dict:
    benchmark = [
        r
        for r in rows
        if r["condition"] in ("clean", "vanilla") or r["ground_truth"] == 1
    ]
    controls = [r for r in rows if r["ground_truth"] == 0 and r["condition"] != "clean"]
    n = sum(r["condition"] == "clean" for r in rows)
    b = sum(r["condition"] == "vanilla" for r in rows)
    return summarize_apps(
        benchmark, controls, expected_clean=n, expected_backdoors=b, threshold=threshold
    )


async def run() -> None:
    config = json.loads(CONFIG.read_text())
    out = output(config)
    if not json.loads((out / "operator_smoke.json").read_text())["passed"]:
        raise ValueError("projection operator smoke did not pass")
    freeze(config, out)
    checked(config, out)
    fitting = read_rows(out / "fit_workload.jsonl")
    apps = read_rows(out / "apps_workload.jsonl")
    idrows = read_rows(out / "id_workload.jsonl")
    addrows = read_rows(out / "add_workload.jsonl")
    if not (out / "fit_receipt.json").exists():
        await score_rows(config, out, fitting, "plain", "fit", capture=True)
        edits = fit(config, out, fitting)
    else:
        receipt = json.loads((out / "fit_receipt.json").read_text())
        if (
            file_hash(out / "directions.npz") != receipt["directions_sha256"]
            or file_hash(out / "interventions.json") != receipt["interventions_sha256"]
        ):
            raise ValueError("frozen fit artifacts changed")
        edits = json.loads((out / "interventions.json").read_text())
    all_scores = {}
    for arm in config["primary_arms"]:
        checked(config, out)
        all_scores[arm] = await score_rows(config, out, apps, arm, "apps", edits[arm])
        await score_rows(config, out, idrows, arm, "id", edits[arm])
    for arm in config["additive_arms"]:
        await score_rows(config, out, addrows, arm, "add", edits[arm])
    baseline = apps_report(all_scores["plain"])
    summary = {
        "apps": {},
        "id": {},
        "add": {},
        "fit": json.loads((out / "fit_receipt.json").read_text()),
        "partition": json.loads((out / "partition.json").read_text()),
        "qualification": (
            "APPS-fitted exploratory intervention, disjoint task test "
            "partition; single SDPA model/seed, BF16 eager. No promotion."
        ),
    }
    for arm in config["primary_arms"]:
        summary["apps"][arm] = {
            "recalibrated": apps_report(all_scores[arm]),
            "fixed_threshold": apps_report(all_scores[arm], baseline["threshold"]),
        }
        summary["id"][arm] = id_metrics(read_rows(out / f"id_{arm}.jsonl"))
    add_baseline = [
        r
        for r in all_scores["plain"]
        if str(r["task_id"]) in set(summary["partition"]["add_tasks"])
    ]
    threshold = apps_report(add_baseline)["threshold"]
    summary["add"]["plain"] = {
        "recalibrated": apps_report(add_baseline),
        "fixed_threshold": apps_report(add_baseline, threshold),
    }
    for arm in config["additive_arms"]:
        vals = read_rows(out / f"add_{arm}.jsonl")
        summary["add"][arm] = {
            "recalibrated": apps_report(vals),
            "fixed_threshold": apps_report(vals, threshold),
            "p01_mean": float(np.mean([r["p01"] for r in vals])),
            "p01_min": float(min(r["p01"] for r in vals)),
        }
    from experiments.b200_injection_direction.analyze import bootstrap, probes

    for arm in config["primary_arms"]:
        values = all_scores[arm]
        subset = [r for r in values if "p01" in r]
        summary["apps"][arm]["answer_mass"] = {
            "rows": len(subset),
            "mean": float(np.mean([r["p01"] for r in subset])),
            "minimum": float(min(r["p01"] for r in subset)),
            "fraction_below_half": float(np.mean([r["p01"] < 0.5 for r in subset])),
        }
    for arm in config["additive_arms"]:
        vals = read_rows(out / f"add_{arm}.jsonl")
        summary["add"][arm]["format_degradation_fraction"] = float(
            np.mean([r["p01"] < 0.5 for r in vals])
        )
    checked(config, out)
    write_json(out / "summary.json", summary)
    write_json(out / "status.json", {"stage": "analyzing"})
    probes(out, config)
    bootstrap(out, config)
    write_json(out / "status.json", {"stage": "complete"})
    print("injection_direction_complete", flush=True)


def main() -> None:
    sys.path.insert(0, "/tmp/gleipnir-vllm-lens-1.3.0")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=["run"], default="run")
    parser.parse_args()
    out = output(json.loads(CONFIG.read_text()))
    try:
        asyncio.run(run())
    except BaseException as e:
        write_json(out / "failure.json", {"type": type(e).__name__, "message": str(e)})
        raise


if __name__ == "__main__":
    main()
