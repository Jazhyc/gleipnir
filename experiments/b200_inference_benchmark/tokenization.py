"""Measure frontend contribution using exact token IDs on a resident server."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import importlib.metadata
import json
import os
import statistics
import time
from pathlib import Path

from experiments.b200_inference_benchmark.run import (
    DATA,
    ROOT,
    resolve_kernel_baseline,
    sha,
    trial,
    write,
)
from gleipnir.inference_benchmark import (
    measurement_summary,
    paired_score_summary,
    ranking_comparison,
)


def token_rows(rows: list[dict], tokens: dict[str, list[int]]) -> list[dict]:
    """Keep rendered-text identity while replacing only the HTTP prompt payload."""
    result = []
    for row in rows:
        ids = tokens[row["id"]]
        if len(ids) != row["prompt_tokens"] or any(
            type(value) is not int or value < 0 for value in ids
        ):
            raise ValueError(f"token payload drift: {row['id']}")
        result.append({**row, "prompt": ids})
    return result


def cpu_seconds(pid: int) -> float:
    fields = Path(f"/proc/{pid}/stat").read_text().rsplit(") ", 1)[1].split()
    return (int(fields[11]) + int(fields[12])) / os.sysconf("SC_CLK_TCK")


async def measure(name: str, port: int, repeats: int) -> None:
    import httpx
    from transformers import AutoTokenizer

    output = ROOT / "results/b200_inference_benchmark" / name
    output.mkdir(exist_ok=False)
    server_root = ROOT / "results/b200_attention_gdn_serving"
    server = json.loads((server_root / "server.json").read_text())
    worker = json.loads((server_root / "loaded_precision.json").read_text())[
        "worker_pid"
    ]
    resolve_kernel_baseline({"baseline": "selected"})
    args = server["command"]
    source_hashes = json.loads(args[args.index("--additional-config") + 1])[
        "gleipnir_frost_fp4"
    ]
    for path, digest in source_hashes.items():
        if sha(ROOT / path) != digest:
            raise ValueError(f"loaded server source drift: {path}")
    manifest = json.loads((DATA / "manifest.json").read_text())
    full = json.loads((DATA / "full.json").read_text())
    quick = json.loads((DATA / "quick.json").read_text())
    for key, rows in [("full", full), ("quick", quick)]:
        if sha(DATA / f"{key}.json") != manifest["files"][key]:
            raise ValueError(f"frozen workload drift: {key}")
        for row in rows:
            if (
                hashlib.sha256(row["prompt"].encode()).hexdigest()
                != row["prompt_sha256"]
            ):
                raise ValueError("rendered prompt drift")
    model = Path(args[args.index("--model") + 1])
    report = {
        "status": "running",
        "server": server,
        "worker_pid": worker,
        "manifest_sha256": sha(DATA / "manifest.json"),
        "baseline_sha256": sha(Path(__file__).parent / "baseline.json"),
        "executed_source_sha256": sha(Path(__file__)),
        "tokenizer_files": {str(p): sha(p) for p in model.glob("*token*.json")},
        "runtime": {
            k: importlib.metadata.version(k)
            for k in ["transformers", "tokenizers", "vllm", "httpx"]
        },
        "tokenizers_parallelism": os.environ.get("TOKENIZERS_PARALLELISM"),
        "trials": [],
        "interpretation": (
            "Token-ID HTTP ablation also changes payload serialization and "
            "scheduling; client pretokenization is excluded."
        ),
    }
    write(output / "summary.json", report)
    tokenizer = AutoTokenizer.from_pretrained(model, local_files_only=True)
    if not tokenizer.is_fast:
        raise ValueError("serving benchmark expects the fast tokenizer")
    tokens = {
        row["id"]: tokenizer.encode(row["prompt"], add_special_tokens=False)
        for row in full
    }
    token_rows(full, tokens)
    write(output / "token_ids.json", tokens)
    report["tokenizer_class"] = type(tokenizer).__name__
    report["cpu_encode"] = []
    for repeat in range(repeats):
        values = []
        start = time.perf_counter()
        for row in full:
            before = time.perf_counter()
            ids = tokenizer.encode(row["prompt"], add_special_tokens=False)
            elapsed = time.perf_counter() - before
            if ids != tokens[row["id"]]:
                raise ValueError("repeated tokenization drift")
            values.append(
                {"id": row["id"], "prompt_tokens": len(ids), "latency_seconds": elapsed}
            )
        seconds = time.perf_counter() - start
        write(output / f"cpu_encode_{repeat}.json", values)
        report["cpu_encode"].append(measurement_summary(values, seconds))
    write(output / "summary.json", report)
    print("cpu_encode_complete", flush=True)
    async with httpx.AsyncClient(
        base_url=f"http://127.0.0.1:{port}",
        timeout=300,
        trust_env=False,
        limits=httpx.Limits(max_connections=256, max_keepalive_connections=128),
    ) as client:
        (await client.get("/health")).raise_for_status()
        endpoint = []
        start = time.perf_counter()
        for row in full:
            before = time.perf_counter()
            response = await client.post(
                "/tokenize",
                json={
                    "model": "monitor",
                    "prompt": row["prompt"],
                    "add_special_tokens": False,
                },
            )
            response.raise_for_status()
            value = response.json()
            elapsed = time.perf_counter() - before
            if (
                value["tokens"] != tokens[row["id"]]
                or value["count"] != row["prompt_tokens"]
            ):
                raise ValueError("server/local tokenizer mismatch")
            endpoint.append(
                {
                    "id": row["id"],
                    "prompt_tokens": value["count"],
                    "latency_seconds": elapsed,
                }
            )
        report["tokenize_endpoint"] = measurement_summary(
            endpoint, time.perf_counter() - start
        )
        write(output / "tokenize_endpoint.json", endpoint)
        write(output / "summary.json", report)
        print("endpoint_parity_complete rows=320", flush=True)
        for concurrency, rows in [(1, quick), (128, full)]:
            variants = {"text": rows, "ids": token_rows(rows, tokens)}
            results = {"text": [], "ids": []}
            for mode in ["text", "ids"]:
                values, seconds = await trial(
                    client, variants[mode], manifest["token_ids"], concurrency
                )
                write(
                    output / f"warmup_c{concurrency}_{mode}.json",
                    {"seconds": seconds, "values": values},
                )
            for repeat in range(repeats):
                for mode in ["text", "ids"] if repeat % 2 == 0 else ["ids", "text"]:
                    before = {
                        "api": cpu_seconds(server["pid"]),
                        "engine": cpu_seconds(worker),
                    }
                    values, seconds = await trial(
                        client, variants[mode], manifest["token_ids"], concurrency
                    )
                    used = {
                        k: cpu_seconds(pid) - before[k]
                        for k, pid in [("api", server["pid"]), ("engine", worker)]
                    }
                    results[mode].append(values)
                    write(output / f"c{concurrency}_{mode}_{repeat}.json", values)
                    item = {
                        "concurrency": concurrency,
                        "mode": mode,
                        "repeat": repeat,
                        **measurement_summary(values, seconds),
                        "process_cpu_seconds": used,
                    }
                    report["trials"].append(item)
                    write(output / "summary.json", report)
                    print(
                        "trial_complete",
                        concurrency,
                        mode,
                        repeat,
                        round(seconds, 3),
                        flush=True,
                    )
            a = [
                t
                for t in report["trials"]
                if t["concurrency"] == concurrency and t["mode"] == "text"
            ]
            b = [
                t
                for t in report["trials"]
                if t["concurrency"] == concurrency and t["mode"] == "ids"
            ]
            report[f"c{concurrency}_comparison"] = {
                "tokens_s": {
                    m: statistics.median(t["prompt_tokens_per_second"] for t in ts)
                    for m, ts in [("text", a), ("ids", b)]
                },
                "latency_p50": {
                    m: statistics.median(t["latency"]["p50_seconds"] for t in ts)
                    for m, ts in [("text", a), ("ids", b)]
                },
                "latency_p95": {
                    m: statistics.median(t["latency"]["p95_seconds"] for t in ts)
                    for m, ts in [("text", a), ("ids", b)]
                },
                "paired_scores": paired_score_summary(results["text"], results["ids"]),
                "ranking": ranking_comparison(rows, results["text"], results["ids"]),
            }
            write(output / "summary.json", report)
        if (
            json.loads((server_root / "server.json").read_text())["pid"]
            != server["pid"]
        ):
            raise ValueError("resident server changed during measurement")
        report.update(status="complete", server_reused=True, kernels_changed=False)
        write(output / "summary.json", report)
        print("tokenization_measurement_complete", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", required=True)
    parser.add_argument("--port", type=int, default=8010)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    if Path(args.name).name != args.name or args.repeats < 1:
        raise ValueError("use a directory stem and positive repeat count")
    if (ROOT / "results/b200_inference_benchmark" / args.name).exists():
        raise FileExistsError("preserve the existing measurement artifacts")
    try:
        asyncio.run(measure(args.name, args.port, args.repeats))
    except BaseException as error:
        write(
            ROOT / "results/b200_inference_benchmark" / args.name / "failure.json",
            {"error": f"{type(error).__name__}: {error}"},
        )
        raise


if __name__ == "__main__":
    main()
