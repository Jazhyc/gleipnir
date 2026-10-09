"""Score frozen APPS populations with the already passing BF16 monitor."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
from pathlib import Path

import httpx

from experiments.b200_apps.run import score_batch
from gleipnir.campaigns.monitoring.contract import Campaign
from gleipnir.campaigns.monitoring.evaluation import (
    attach_inputs,
    frozen_workloads,
    require_evaluation_gate,
    summarize,
)
from gleipnir.data.monitoring import file_hash, read_rows, write_json, write_rows

ROOT = Path(__file__).resolve().parents[2]
CONFIG = Path(__file__).with_name("config.yaml")


class Completed(Campaign):
    @property
    def adapter(self) -> Path:
        return self.root / self.config["completed_adapter"]


def validate_resident(server: dict, expected: dict, native: dict) -> None:
    """Reject stale or different resident weights, settings and precision."""
    if server != expected or server["status"] != "ready":
        raise ValueError("resident server identity changed")
    if (
        server.get("serving_precision") != "bf16"
        or "--enforce-eager" in server["command"]
    ):
        raise ValueError("APPS follow-up requires the compiled BF16 control")
    if (
        not native["passed"]
        or native["quantization"] is not None
        or native["projection_counts"] != {"attention": 16, "gdn": 48, "mlp": 64}
        or len(native["attention_calls"]) != 8
        or any(
            x["query_dtype"] != "torch.bfloat16"
            or x["cache_dtype"] != "torch.bfloat16"
            or not x["causal"]
            for x in native["attention_calls"]
        )
    ):
        raise ValueError("resident BF16 native audit incomplete")


def checked_resident(ctx: Campaign) -> tuple[dict, dict]:
    ctx.check()
    server = json.loads((ctx.serving / "server.json").read_text())
    expected = json.loads(ctx.input("resident_server").read_text())
    native = json.loads((ctx.serving / "loaded_precision.json").read_text())
    validate_resident(server, expected, native)
    if native != json.loads(ctx.input("resident_audit").read_text()):
        raise ValueError("resident worker/native evidence changed")
    actual = [
        x.decode()
        for x in Path(f"/proc/{server['pid']}/cmdline").read_bytes().split(b"\0")
        if x
    ]
    if actual != server["command"] or os.getpgid(server["pid"]) != server["pid"]:
        raise ValueError("resident process command/session changed")
    if not Path(f"/proc/{native['worker_pid']}").exists():
        raise ValueError("resident GPU worker missing")
    complete = json.loads(ctx.input("completed_training").read_text())
    if server["adapter_sha256"] != complete["serving_sha256"]:
        raise ValueError("resident adapter changed")
    model = server["command"][server["command"].index("--model") + 1]
    if model != ctx.config["merged_model"]:
        raise ValueError("resident merged model changed")
    merged = json.loads(ctx.input("completed_merge").read_text())
    for name, expected_hash in merged["files_sha256"].items():
        if file_hash(Path(model) / name) != expected_hash:
            raise ValueError(f"resident merged file changed: {name}")
    gate = json.loads(ctx.input("resident_parity").read_text())
    require_evaluation_gate(gate, diagnostic=False)
    return server, gate


async def score(ctx: Campaign) -> None:
    if (ctx.output / "evaluation").exists():
        raise ValueError("APPS scoring already attempted; preserve its receipts")
    server, gate = checked_resident(ctx)
    write_json(ctx.output / "server.json", server)
    write_json(ctx.output / "optimized_parity.json", gate)
    write_json(
        ctx.output / "bf16_audit.json",
        json.loads(ctx.input("resident_audit").read_text()),
    )
    async with httpx.AsyncClient(
        base_url="http://127.0.0.1:8010", trust_env=False
    ) as client:
        (await client.get("/health")).raise_for_status()
    workloads = frozen_workloads(ctx)
    populations = {}
    for split, rows in workloads.items():
        values, times = [], []
        for start in range(0, len(rows), ctx.config["evaluation"]["batch_rows"]):
            group = rows[start : start + ctx.config["evaluation"]["batch_rows"]]
            batch, seconds = await score_batch(group, ctx.config["evaluation"])
            write_json(
                ctx.output / "evaluation/batches" / split / f"{start:05d}.json", batch
            )
            values.extend(batch)
            times.append(seconds)
            write_json(
                ctx.output / "status.json",
                {
                    "stage": "scoring",
                    "split": split,
                    "rows": len(values),
                    "total": len(rows),
                },
            )
            print("bf16_apps_progress", split, len(values), len(rows), flush=True)
        populations[split] = attach_inputs(values, read_rows(ctx.input(split)))
        write_rows(ctx.output / "evaluation" / f"{split}.jsonl", populations[split])
        write_json(
            ctx.output / "evaluation" / f"{split}_timing.json",
            {
                "seconds": sum(times),
                "batch_seconds": times,
                "rows": len(rows),
                "tokens": sum(r["prompt_tokens"] for r in rows),
            },
        )
    checked_resident(ctx)
    write_json(ctx.output / "summary.json", summarize(populations, ctx))
    write_json(
        ctx.output / "evaluation/complete.json",
        {
            "rows": sum(map(len, populations.values())),
            "config_sha256": file_hash(ctx.config_path),
            "master_sha256": json.loads(ctx.input("completed_training").read_text())[
                "master_sha256"
            ],
            "parity_passed": True,
            "evaluation_scope": "parity_gated",
            "files_sha256": {
                s: file_hash(ctx.output / "evaluation" / f"{s}.jsonl")
                for s in populations
            },
        },
    )
    write_json(ctx.output / "status.json", {"stage": "complete", "rows": 9114})
    print("bf16_apps_complete", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("prepare", "score"), required=True)
    parser.add_argument("--config", type=Path, default=CONFIG)
    args = parser.parse_args()
    ctx = Campaign.load(ROOT, args.config)
    if args.stage == "prepare":
        ctx.prepare()
        for name, source in (
            ("merged_artifact.json", "completed_merge"),
            ("merged_parity.json", "completed_merged_parity"),
        ):
            path = ctx.output / name
            contents = ctx.input(source).read_bytes()
            if path.exists() and path.read_bytes() != contents:
                raise ValueError("prepared model reference changed")
            path.write_bytes(contents)
        print("bf16_apps_prepared", flush=True)
        return
    ctx = Completed(ctx.root, ctx.config_path, ctx.config, ctx.source_root)
    try:
        asyncio.run(score(ctx))
    except BaseException as error:
        write_json(
            ctx.output / "failure.json",
            {"type": type(error).__name__, "message": str(error)},
        )
        write_json(ctx.output / "status.json", {"stage": "failed"})
        raise


if __name__ == "__main__":
    main()
