"""Probe exact Gigatoken parity and include live encoding in serving latency."""

from __future__ import annotations

import argparse
import asyncio
import functools
import hashlib
import importlib.metadata
import json
import statistics
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

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

GIGATOKEN_VERSION = "0.10.0"


class LiveTokenizingClient:
    """Encode on every request within the existing trial's response timer."""

    def __init__(
        self,
        client: Any,
        tokenizer: Any,
        executor: ThreadPoolExecutor,
        expected: dict[str, list[int]],
    ) -> None:
        self.client = client
        self.tokenizer = tokenizer
        self.executor = executor
        self.expected = expected
        self.encodes: list[dict] = []

    def encode(self, text: str) -> tuple[list[int], float]:
        before = time.perf_counter()
        ids = self.tokenizer.encode(
            text, add_special_tokens=False, truncation=True, max_length=32768
        )
        seconds = time.perf_counter() - before
        if ids != self.expected[text]:
            raise ValueError("live Gigatoken ID drift")
        return ids, seconds

    async def post(self, path: str, *, json: dict) -> Any:
        before = time.perf_counter()
        ids, encode_seconds = await asyncio.get_running_loop().run_in_executor(
            self.executor, functools.partial(self.encode, json["prompt"])
        )
        self.encodes.append(
            {
                "encode_seconds": encode_seconds,
                "encode_and_queue_seconds": time.perf_counter() - before,
                "prompt_tokens": len(ids),
            }
        )
        return await self.client.post(path, json={**json, "prompt": ids})


def edge_cases() -> list[str]:
    fixtures = [
        "",
        " ",
        "\n\n",
        "hello\r\nworld\t  end",
        "foo_bar42 = {'x': -1.23e+4}; // comment",
        "你好世界 日本語 한국어 العربية русский",
        "é e\u0301 😀👨‍👩‍👧‍👦 🇳🇱 \u00a0\u200b",
        "<|im_start|>assistant\n<think>why?</think><|im_end|>",
        "ordinary <|endoftext|> <|vision_start|> <|image_pad|> text",
        "a" * 10000,
        "1234567890" * 1000,
    ]
    return [
        prefix + text + suffix
        for text in fixtures
        for prefix, suffix in [
            ("", ""),
            (" ", "\n"),
            ("<|im_start|>user\n", "<|im_end|>\n"),
        ]
    ]


async def measure(name: str, repeats: int) -> None:
    import gigatoken as gt
    import httpx
    from vllm.tokenizers.registry import get_tokenizer

    if importlib.metadata.version("gigatoken") != GIGATOKEN_VERSION:
        raise ValueError("Gigatoken version drift")
    output = ROOT / "results/b200_inference_benchmark" / name
    output.mkdir(exist_ok=False)
    write(output / "executed_gigatoken.json", {"source_sha256": sha(Path(__file__))})
    (output / "executed_gigatoken.py").write_bytes(Path(__file__).read_bytes())
    server_root = ROOT / "results/b200_attention_gdn_serving"
    server = json.loads((server_root / "server.json").read_text())
    args = server["command"]
    resolve_kernel_baseline({"baseline": "selected"})
    for path, digest in json.loads(args[args.index("--additional-config") + 1])[
        "gleipnir_frost_fp4"
    ].items():
        if sha(ROOT / path) != digest:
            raise ValueError(f"loaded kernel source drift: {path}")
    manifest = json.loads((DATA / "manifest.json").read_text())
    full = json.loads((DATA / "full.json").read_text())
    quick = json.loads((DATA / "quick.json").read_text())
    for key in ["quick", "full"]:
        if sha(DATA / f"{key}.json") != manifest["files"][key]:
            raise ValueError("frozen workload drift")
    for row in full:
        if hashlib.sha256(row["prompt"].encode()).hexdigest() != row["prompt_sha256"]:
            raise ValueError("prompt bytes drift")
    expected_ids = json.loads(
        (
            ROOT / "results/b200_inference_benchmark/tokenization01/token_ids.json"
        ).read_text()
    )
    expected = {row["prompt"]: expected_ids[row["id"]] for row in full}
    model = args[args.index("--model") + 1]
    hf = get_tokenizer(model, tokenizer_mode="auto", runner_type="generate")
    before = time.perf_counter()
    gig = gt.Tokenizer(hf).as_hf()
    gig.truncation_side = hf.truncation_side
    kwargs = dict(add_special_tokens=False, truncation=True, max_length=32768)
    report = {
        "status": "running",
        "server": server,
        "manifest_sha256": sha(DATA / "manifest.json"),
        "baseline_sha256": sha(Path(__file__).parent / "baseline.json"),
        "gigatoken_version": GIGATOKEN_VERSION,
        "gigatoken_module": gt.__file__,
        "construction_seconds": time.perf_counter() - before,
        "encoding_threads": 1,
        "encoding_in_request_latency": True,
        "integration": "caller live encoding then token-ID HTTP; server unmodified",
        "cpu": [],
        "trials": [],
    }
    for mode, tokenizer in [("gigatoken_cold", gig), ("hf", hf)]:
        values = []
        before = time.perf_counter()
        for row in full:
            start = time.perf_counter()
            ids = tokenizer.encode(row["prompt"], **kwargs)
            seconds = time.perf_counter() - start
            if ids != expected_ids[row["id"]] or len(ids) != row["prompt_tokens"]:
                raise ValueError(f"{mode} frozen ID mismatch: {row['id']}")
            values.append(
                {"id": row["id"], "prompt_tokens": len(ids), "latency_seconds": seconds}
            )
        write(output / f"cpu_{mode}_first.json", values)
        report["cpu"].append(
            {
                "mode": mode,
                "repeat": "first",
                **measurement_summary(values, time.perf_counter() - before),
            }
        )
    fixtures = edge_cases()
    for text in fixtures:
        for options in [
            kwargs,
            {"add_special_tokens": True},
            {"add_special_tokens": False, "truncation": True, "max_length": 3},
        ]:
            if gig.encode(text, **options) != hf.encode(text, **options):
                raise ValueError(
                    "Gigatoken synthetic special/unicode/truncation mismatch"
                )
    report["parity"] = {
        "frozen_rows": len(full),
        "edge_cases": len(fixtures) * 3,
        "exact": True,
    }
    print("exact_token_parity_passed", report["parity"], flush=True)
    for repeat in range(repeats):
        modes = [("hf", hf), ("gigatoken", gig)]
        if repeat % 2:
            modes.reverse()
        for mode, tokenizer in modes:
            values = []
            before = time.perf_counter()
            for row in full:
                start = time.perf_counter()
                ids = tokenizer.encode(row["prompt"], **kwargs)
                seconds = time.perf_counter() - start
                if ids != expected_ids[row["id"]]:
                    raise ValueError("warm encoding drift")
                values.append(
                    {
                        "id": row["id"],
                        "prompt_tokens": len(ids),
                        "latency_seconds": seconds,
                    }
                )
            write(output / f"cpu_{mode}_{repeat}.json", values)
            report["cpu"].append(
                {
                    "mode": mode,
                    "repeat": repeat,
                    **measurement_summary(values, time.perf_counter() - before),
                }
            )
            write(output / "summary.json", report)
            print("cpu_pass", mode, repeat, flush=True)
    with ThreadPoolExecutor(max_workers=1) as executor:
        for concurrency, rows in [(1, quick), (128, full)]:
            results = {"text": [], "gigatoken": []}

            async def run_pass(mode, rows=rows, concurrency=concurrency):
                async with httpx.AsyncClient(
                    base_url="http://127.0.0.1:8010",
                    timeout=300,
                    trust_env=False,
                    limits=httpx.Limits(
                        max_connections=256, max_keepalive_connections=128
                    ),
                ) as client:
                    selected = (
                        client
                        if mode == "text"
                        else LiveTokenizingClient(client, gig, executor, expected)
                    )
                    values, seconds = await trial(
                        selected, rows, manifest["token_ids"], concurrency
                    )
                    return values, seconds, getattr(selected, "encodes", [])

            for mode in ["text", "gigatoken"]:
                values, seconds, encodes = await run_pass(mode)
                write(
                    output / f"warmup_c{concurrency}_{mode}.json",
                    {"seconds": seconds, "values": values, "encodes": encodes},
                )
            for repeat in range(repeats):
                for mode in (
                    ["text", "gigatoken"] if repeat % 2 == 0 else ["gigatoken", "text"]
                ):
                    values, seconds, encodes = await run_pass(mode)
                    results[mode].append(values)
                    write(output / f"c{concurrency}_{mode}_{repeat}.json", values)
                    write(
                        output / f"c{concurrency}_{mode}_{repeat}_encodes.json", encodes
                    )
                    report["trials"].append(
                        {
                            "concurrency": concurrency,
                            "mode": mode,
                            "repeat": repeat,
                            **measurement_summary(values, seconds),
                        }
                    )
                    write(output / "summary.json", report)
                    print(
                        "serving_pass",
                        concurrency,
                        mode,
                        repeat,
                        round(seconds, 3),
                        flush=True,
                    )
            ts = {
                mode: [
                    t
                    for t in report["trials"]
                    if t["concurrency"] == concurrency and t["mode"] == mode
                ]
                for mode in results
            }
            report[f"c{concurrency}_comparison"] = {
                "tokens_s": {
                    mode: statistics.median(
                        t["prompt_tokens_per_second"] for t in values
                    )
                    for mode, values in ts.items()
                },
                "latency_p50": {
                    mode: statistics.median(t["latency"]["p50_seconds"] for t in values)
                    for mode, values in ts.items()
                },
                "latency_p95": {
                    mode: statistics.median(t["latency"]["p95_seconds"] for t in values)
                    for mode, values in ts.items()
                },
                "paired_scores": paired_score_summary(
                    results["text"], results["gigatoken"]
                ),
                "ranking": ranking_comparison(
                    rows, results["text"], results["gigatoken"]
                ),
            }
            write(output / "summary.json", report)
    if json.loads((server_root / "server.json").read_text())["pid"] != server["pid"]:
        raise ValueError("server identity changed")
    report.update(status="complete", kernels_changed=False, server_reused=True)
    write(output / "summary.json", report)
    print("gigatoken_measurement_complete", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", required=True)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    if Path(args.name).name != args.name or args.repeats < 1:
        raise ValueError("use a directory stem and positive repeat count")
    output = ROOT / "results/b200_inference_benchmark" / args.name
    if output.exists():
        raise FileExistsError("preserve existing run artifacts")
    try:
        asyncio.run(measure(args.name, args.repeats))
    except BaseException as error:
        write(output / "failure.json", {"error": f"{type(error).__name__}: {error}"})
        raise


if __name__ == "__main__":
    main()
