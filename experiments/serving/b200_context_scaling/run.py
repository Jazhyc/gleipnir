"""Measure exact 2K windows using the already-running selected scorer."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import subprocess
from pathlib import Path

import httpx
import numpy as np

from experiments.b200_inference_benchmark.run import DATA, ROOT, sha, write
from experiments.b200_monitor_score.run import SERVING, trial
from gleipnir.inference_benchmark import measurement_summary

EXPERIMENT = Path(__file__).parent


def digest(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def build_windows(source: list[dict], tokenizer, target: int, count: int) -> list[dict]:
    """Crop distinct deterministic windows, binding their exact source lineage."""
    donors = []
    for row in source:
        if row["prompt_tokens"] < target + 64:
            continue
        if digest(row["prompt"]) != row["prompt_sha256"]:
            raise ValueError("source prompt hash drift")
        ids = tokenizer.encode(row["prompt"], add_special_tokens=False)
        if len(ids) != row["prompt_tokens"]:
            raise ValueError("source token count drift")
        donors.append((row, ids))
    if not donors:
        raise ValueError("no sufficiently long source prompts")
    windows, seen = [], set()
    for index in range(count):
        row, ids = donors[index % len(donors)]
        salt = int(digest(f"{row['prompt_sha256']}:{index}")[:16], 16)
        for attempt in range(64):
            start = (salt + 13 * attempt) % (len(ids) - target - 63)
            text = tokenizer.decode(ids[start : start + target + 64])
            for _ in range(8):
                encoded = tokenizer.encode(text, add_special_tokens=False)
                if len(encoded) == target:
                    break
                text = tokenizer.decode(encoded[:target])
            actual = len(tokenizer.encode(text, add_special_tokens=False))
            text_hash = digest(text)
            if actual == target and text_hash not in seen:
                break
        else:
            raise ValueError("could not construct a unique exact-length window")
        seen.add(text_hash)
        windows.append(
            {
                "id": f"window{index:03d}",
                "prompt": text,
                "prompt_tokens": actual,
                "prompt_sha256": text_hash,
                "source_id": row["id"],
                "source_prompt_sha256": row["prompt_sha256"],
                "source_prompt_tokens": row["prompt_tokens"],
                "source_token_start": start,
                "source_candidate_tokens": target + 64,
            }
        )
    return windows


def bind_reference(current: dict, prior: dict) -> None:
    """Reject changed process or serving recipe before benchmarking."""
    if prior["status"] != "complete" or current["status"] != "ready":
        raise ValueError("reference is not ready/complete")
    for key in ("pid", "command", "config_sha256", "score_sources"):
        if current[key] != prior["server"][key]:
            raise ValueError(f"reference identity drift: {key}")
    if "--scheduler-cls" in current["command"]:
        raise ValueError("reference scheduler changed")
    os.kill(current["pid"], 0)


async def run(name: str) -> None:
    settings = json.loads((EXPERIMENT / "config.json").read_text())
    out = ROOT / "results/b200_context_scaling" / name
    out.mkdir(parents=True, exist_ok=False)
    write(out / "config.json", settings)
    executed = out / "executed_sources"
    executed.mkdir()
    for filename in ("run.py", "config.json", "README.md"):
        (executed / filename).write_bytes((EXPERIMENT / filename).read_bytes())
    report = {
        "status": "preparing",
        "trials": [],
        "control_trials": [],
        "promoted": False,
    }
    write(out / "summary.json", report)
    try:
        reference_path = ROOT / settings["reference_run"] / "summary.json"
        prior = json.loads(reference_path.read_text())
        server_path = SERVING / "server.json"
        server_bytes = server_path.read_bytes()
        server = json.loads(server_bytes)
        bind_reference(server, prior)
        gpu = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=name,uuid,memory.used,utilization.gpu",
                "--format=csv,noheader",
            ],
            text=True,
        ).strip()
        apps = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-compute-apps=pid,process_name,used_memory",
                "--format=csv,noheader",
            ],
            text=True,
        ).strip()
        if settings["gpu_uuid"] not in gpu or len(apps.splitlines()) != 1:
            raise ValueError("GPU identity or exclusive worker count changed")
        write(
            out / "reference_binding.json",
            {
                "server": server,
                "gpu": gpu,
                "compute_apps": apps,
                "prior_summary_sha256": sha(reference_path),
                "server_sha256": sha(server_path),
            },
        )
        from transformers import AutoTokenizer

        model = server["command"][server["command"].index("--model") + 1]
        tokenizer = AutoTokenizer.from_pretrained(model, local_files_only=True)
        source_path = DATA / "full.json"
        windows = build_windows(
            json.loads(source_path.read_text()),
            tokenizer,
            settings["target_tokens"],
            settings["count"],
        )
        write(out / "workload.json", windows)
        write(
            out / "workload_binding.json",
            {
                "source_sha256": sha(source_path),
                "workload_sha256": sha(out / "workload.json"),
                "rows": len(windows),
                "total_prompt_tokens": sum(r["prompt_tokens"] for r in windows),
                "min_prompt_tokens": min(r["prompt_tokens"] for r in windows),
                "max_prompt_tokens": max(r["prompt_tokens"] for r in windows),
                "distinct_source_prompts": len({r["source_id"] for r in windows}),
                "kind": "cropped_text_systems_only_no_labels",
            },
        )
        async with httpx.AsyncClient(
            base_url=f"http://127.0.0.1:{settings['port']}", trust_env=False, timeout=60
        ) as client:
            (await client.get("/health")).raise_for_status()
            response = await client.post(
                "/collective_rpc",
                json={"method": "frost_wrapper_state", "kwargs": {}, "timeout": 60},
            )
            response.raise_for_status()
            native = response.json()
            if native["results"][0]["mode"] != "direct":
                raise ValueError("direct FROST runtime changed")
            write(out / "native.json", native)
        control_path = ROOT / settings["canary_reference"]
        control = json.loads(control_path.read_text())
        if not control["evaluation_passed"]:
            raise ValueError("accepted adapter canary missing")
        values, _ = await trial(
            json.loads((DATA / "canary.json").read_text()), 4, settings
        )
        observed = np.array([v["score"] for v in values])
        accepted = np.array(control["served"]["adapter"])
        mean = float(np.mean(np.abs(observed - accepted)))
        correlation = float(np.corrcoef(observed, accepted)[0, 1])
        effect = float(np.max(np.abs(observed - np.array(control["served"]["base"]))))
        passed = (
            mean <= settings["canary_mean_error_limit"]
            and correlation >= settings["canary_correlation_floor"]
            and effect > 0
        )
        write(
            out / "canary.json",
            {
                "passed": bool(passed),
                "mean_absolute_difference": mean,
                "correlation": correlation,
                "adapter_effect": effect,
                "reference_sha256": sha(control_path),
            },
        )
        write(out / "canary_predictions.json", values)
        if not passed:
            raise ValueError("real adapter canary failed")
        print("workload_frozen_canary_passed", len(windows), mean, flush=True)
        report["status"] = "running"
        original = json.loads(source_path.read_text())
        await trial(original, 8, settings)
        for repeat in range(settings["repeats"]):
            values, seconds = await trial(original, 8, settings)
            write(out / f"control_c8_repeat{repeat}.json", values)
            result = {
                "concurrency": 8,
                "repeat": repeat,
                **measurement_summary(values, seconds),
            }
            report["control_trials"].append(result)
            write(out / "summary.json", report)
            print("control_c8", repeat, result["prompt_tokens_per_second"], flush=True)
        for concurrency in settings["concurrencies"]:
            for _ in range(settings["warmups"]):
                await trial(windows, concurrency, settings)
            for repeat in range(settings["repeats"]):
                values, seconds = await trial(windows, concurrency, settings)
                write(out / f"c{concurrency}_repeat{repeat}.json", values)
                result = {
                    "concurrency": concurrency,
                    "repeat": repeat,
                    **measurement_summary(values, seconds),
                }
                report["trials"].append(result)
                write(out / "summary.json", report)
                print(
                    "measured",
                    concurrency,
                    repeat,
                    result["prompt_tokens_per_second"],
                    flush=True,
                )
        if server_path.read_bytes() != server_bytes:
            raise ValueError("live server metadata changed during timing")
        bind_reference(json.loads(server_path.read_text()), prior)
        report["status"] = "complete"
        write(out / "summary.json", report)
    except BaseException as error:
        report.update(status="failed", error=f"{type(error).__name__}: {error}")
        write(out / "summary.json", report)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", required=True)
    args = parser.parse_args()
    if args.name in {"", ".", ".."} or Path(args.name).name != args.name:
        raise ValueError("run name must be a stem")
    asyncio.run(run(args.name))


if __name__ == "__main__":
    main()
