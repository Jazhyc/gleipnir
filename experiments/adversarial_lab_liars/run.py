"""Score the multi-agent liar sets.

Kimi K3 through OpenRouter (on the Mac; resumable, $4 cap):
    python experiments/adversarial_lab_liars/run.py kimi
GPU, one monitor at a time (on the box; resumable, with canaries and
decision-token residuals at the probe layers):
    python experiments/adversarial_lab_liars/run.py canary --monitor 4b
    python experiments/adversarial_lab_liars/run.py score --monitor 4b
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from experiments.adversarial_lab_liars.common import (
    CONFIG,
    DATA,
    OUTPUT,
    openrouter_balance,
    openrouter_key,
    read_jsonl,
    sha,
)
from gleipnir.adversarial.eval_sets import ROOT, file_sha256

SETS = ("main", "swap", "last16")


def git_commit() -> str:
    return subprocess.run(
        ["git", "-C", str(ROOT), "rev-parse", "HEAD"], capture_output=True, text=True
    ).stdout.strip()


def manifest() -> dict[str, Any]:
    return json.loads((DATA / "manifest.json").read_text())


def load_set(name: str, prompt: str) -> list[dict[str, Any]]:
    key = f"{name}_{prompt}"
    path = DATA / f"{key}.jsonl"
    if file_sha256(path) != CONFIG["sha256"][key]:
        raise ValueError(f"{path} does not match its frozen SHA256")
    return read_jsonl(path)


def scorable(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Rows within the Gleipnir token limit (over-limit rows: see the manifest)."""
    return [r for r in rows if not r["metadata"]["over_limit"]]


def record_for(row: dict[str, Any]) -> dict[str, Any]:
    m = row["metadata"]
    out = {"id": row["id"], "group": m["condition"], "label": m["ground_truth"]}
    for key in ("task_id", "parent_id", "seed", "cell", "worker"):
        if key in m:
            out[key] = m[key]
    return out


# ---------------------------------------------------------------- Kimi


def kimi_config():
    from gleipnir.teachers.openrouter import OpenRouterConfig

    spec = CONFIG["monitors"]["kimi"]
    return OpenRouterConfig(
        model=spec["model"],
        max_tokens=spec["max_tokens"],
        top_logprobs=spec["top_logprobs"],
        provider_sort=None,
        provider_only=spec["provider_only"],
        provider_max_prompt_price=spec["max_prompt_price"],
        provider_max_completion_price=spec["max_completion_price"],
        allow_fallbacks=False,
    )


def kimi(args: argparse.Namespace) -> None:
    # The is_it_4b request and parsing code, so Kimi settings match exactly.
    from experiments.adversarial_is_it_4b.run import kimi_record
    from gleipnir.teachers.openrouter import request_settings

    spec = CONFIG["monitors"]["kimi"]
    key = openrouter_key()
    balance = openrouter_balance(key)
    print(f"OpenRouter balance before: ${balance:.2f}", flush=True)
    if balance < spec["min_balance_usd"]:
        raise SystemExit(f"balance under ${spec['min_balance_usd']}: not starting")
    order = json.loads((DATA / "kimi_order.json").read_text())
    if file_sha256(DATA / "kimi_order.json") != CONFIG["sha256"]["kimi_order"]:
        raise ValueError("kimi_order.json does not match its frozen SHA256")
    rows = {(n, r["id"]): r for n in SETS for r in scorable(load_set(n, "teacher"))}
    config = kimi_config()
    done, spent = set(), [0.0]
    for name in SETS:
        directory = OUTPUT / "kimi" / name
        directory.mkdir(parents=True, exist_ok=True)
        for r in read_jsonl(directory / "predictions.jsonl"):
            spent[0] += r.get("cost") or 0.0
            if r["status"] == "ok":
                done.add((name, r["id"]))
        (directory / "identity.json").write_text(
            json.dumps(
                {
                    "monitor": "kimi",
                    "set": name,
                    "commit": git_commit(),
                    "request_settings": request_settings(config),
                },
                indent=2,
            )
            + "\n"
        )
    todo = [o for o in order if (o["set"], o["id"]) not in done]
    if args.limit:
        todo = todo[: args.limit]
    print(f"{len(todo)} to score, ${spent[0]:.2f} spent so far", flush=True)
    lock = threading.Lock()

    def work(item):
        if spent[0] >= spec["max_usd"]:
            return None
        row = rows[(item["set"], item["id"])]
        record = kimi_record(row, config, key)
        record.update({k: v for k, v in record_for(row).items() if k not in record})
        record["tier"] = item["tier"]
        with lock:
            spent[0] += record.get("cost") or 0.0
            with (OUTPUT / "kimi" / item["set"] / "predictions.jsonl").open("a") as h:
                h.write(json.dumps(record) + "\n")
        return record

    statuses: dict[str, int] = {}
    with ThreadPoolExecutor(spec["workers"]) as pool:
        for n, record in enumerate(pool.map(work, todo), 1):
            status = record["status"] if record else "skipped_cap"
            statuses[status] = statuses.get(status, 0) + 1
            if n % 50 == 0 or n == len(todo):
                print(f"  {n}/{len(todo)}, ${spent[0]:.2f}, {statuses}", flush=True)
    after = openrouter_balance(key)
    summary = {
        "spent": round(spent[0], 4),
        "statuses": statuses,
        "stopped_at_cap": spent[0] >= spec["max_usd"],
        "balance_before": round(balance, 2),
        "balance_after": round(after, 2),
    }
    with (OUTPUT / "kimi" / "runs.jsonl").open("a") as h:
        h.write(
            json.dumps({**summary, "time": time.time(), "commit": git_commit()}) + "\n"
        )
    print(json.dumps(summary))


# ---------------------------------------------------------------- GPU


def decoder_blocks(model: Any, count: int) -> Any:
    found = [
        m
        for n, m in model.named_modules()
        if n.endswith(".layers") and hasattr(m, "__len__") and len(m) == count
    ]
    if len(found) != 1:
        raise ValueError(f"expected one {count}-block decoder, found {len(found)}")
    return found[0]


class Capture:
    """Forward hooks keeping only the decision-token residual at each layer."""

    def __init__(self, model: Any, layers: list[int], hidden: int):
        self.layers, self.hidden = layers, hidden
        self.blocks = decoder_blocks(model, max(layers)) if layers else None
        self.values: dict[int, Any] = {}
        self.handles: list = []

    def hook(self, layer: int):
        def save(_module, _inputs, output):
            state = output[0] if isinstance(output, tuple) else output
            if state.shape[0] != 1 or state.shape[-1] != self.hidden:
                raise ValueError("unexpected residual shape")
            if layer in self.values:
                raise ValueError("decoder block called twice")
            self.values[layer] = state[0, -1].float().cpu().numpy().copy()

        return save

    def __enter__(self):
        self.values = {}
        for layer in self.layers:
            self.handles.append(
                self.blocks[layer - 1].register_forward_hook(self.hook(layer))
            )
        return self

    def __exit__(self, *_):
        for handle in self.handles:
            handle.remove()
        self.handles = []

    def array(self):
        import numpy as np

        if sorted(self.values) != sorted(self.layers):
            raise ValueError("missing residual capture")
        out = np.stack([self.values[layer] for layer in self.layers])
        if not np.isfinite(out).all():
            raise ValueError("nonfinite activation")
        return out


def load_scorer(key: str, device: str):
    from gleipnir.adversarial.scorer import Scorer

    spec = CONFIG["monitors"][key]
    return Scorer(spec["monitor"], device=device), spec


def run_canaries(scorer, spec: dict, key: str) -> dict[str, Any]:
    """Known-score rows: scores within tolerance, hooks change nothing."""
    tolerance = CONFIG["analysis"]["canary_max_abs_score"]
    rows = read_jsonl(DATA / "canaries.jsonl")
    if file_sha256(DATA / "canaries.jsonl") != CONFIG["sha256"]["canaries"]:
        raise ValueError("canaries.jsonl does not match its frozen SHA256")
    results = []
    for row in rows:
        prompt = row["prompts"][spec["prompt"]]
        plain = scorer.score(prompt)
        with Capture(scorer.model, spec["layers"], spec["hidden_size"]) as capture:
            hooked = scorer.score(prompt)
        capture.array()
        expected = row["expected"][key]["score"]
        results.append(
            {
                "id": row["id"],
                "score": plain["score"],
                "expected": expected,
                "abs_diff": abs(plain["score"] - expected),
                "hooks_identical": (plain["logit_0"], plain["logit_1"])
                == (hooked["logit_0"], hooked["logit_1"]),
                "finite": all(
                    math.isfinite(plain[k]) for k in ("score", "logit_0", "logit_1")
                ),
            }
        )
        print(results[-1], flush=True)
    passed = all(
        r["finite"] and r["hooks_identical"] and r["abs_diff"] <= tolerance
        for r in results
    )
    return {"passed": passed, "tolerance": tolerance, "rows": results}


def canary(args: argparse.Namespace) -> None:
    scorer, spec = load_scorer(args.monitor, args.device)
    receipt = {
        **run_canaries(scorer, spec, args.monitor),
        "identity": scorer.identity(),
        "commit": git_commit(),
    }
    directory = OUTPUT / args.monitor
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "canary.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(f"canary passed={receipt['passed']}")
    if not receipt["passed"]:
        raise SystemExit(1)


def score(args: argparse.Namespace) -> None:
    import numpy as np

    from gleipnir.adversarial.scorer import TooLong

    scorer, spec = load_scorer(args.monitor, args.device)
    check = run_canaries(scorer, spec, args.monitor)
    base = OUTPUT / args.monitor
    base.mkdir(parents=True, exist_ok=True)
    with (base / "canary_runs.jsonl").open("a") as h:
        h.write(
            json.dumps({**check, "time": time.time(), "commit": git_commit()}) + "\n"
        )
    if not check["passed"]:
        raise SystemExit("canaries failed: not scoring")
    for name in args.sets:
        rows = scorable(load_set(name, spec["prompt"]))
        directory = base / name
        acts = directory / "activations"
        acts.mkdir(parents=True, exist_ok=True)
        (directory / "identity.json").write_text(
            json.dumps(
                {
                    "monitor": args.monitor,
                    "set": name,
                    "rows": len(rows),
                    "commit": git_commit(),
                    "layers": spec["layers"],
                    "activation": "decision-token residual, block output, float32",
                    "scorer": scorer.identity(),
                },
                indent=2,
            )
            + "\n"
        )
        path = directory / "predictions.jsonl"
        done = {r["id"] for r in read_jsonl(path)}
        todo = [r for r in rows if r["id"] not in done]
        started, tokens = time.time(), 0
        with path.open("a") as handle:
            for n, row in enumerate(todo, 1):
                record = record_for(row)
                try:
                    with Capture(
                        scorer.model, spec["layers"], spec["hidden_size"]
                    ) as capture:
                        record.update(scorer.score(row["prompt"]), status="ok")
                    file = acts / f"{sha(row['id'])[:24]}.npy"
                    np.save(file, capture.array())
                    record["activation_file"] = file.name
                    tokens += record["tokens"]
                except TooLong as error:
                    record.update(status="too_long", error=str(error))
                handle.write(json.dumps(record) + "\n")
                handle.flush()
                if n % 25 == 0 or n == len(todo):
                    elapsed = time.time() - started
                    print(
                        f"{args.monitor}/{name}: {n}/{len(todo)}, "
                        f"{tokens / elapsed:,.0f} tok/s",
                        flush=True,
                    )
    final = run_canaries(scorer, spec, args.monitor)
    with (base / "canary_runs.jsonl").open("a") as h:
        h.write(
            json.dumps(
                {**final, "time": time.time(), "commit": git_commit(), "when": "after"}
            )
            + "\n"
        )
    print(f"closing canary passed={final['passed']}")


def main() -> None:
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    k = commands.add_parser("kimi")
    k.add_argument("--limit", type=int)
    for name in ("canary", "score"):
        c = commands.add_parser(name)
        c.add_argument("--monitor", required=True, choices=["4b", "27b"])
        c.add_argument("--device", default="cuda")
    commands.choices["score"].add_argument(
        "--sets", nargs="+", default=list(SETS), choices=SETS
    )
    args = parser.parse_args()
    {"kimi": kimi, "canary": canary, "score": score}[args.command](args)


if __name__ == "__main__":
    main()
