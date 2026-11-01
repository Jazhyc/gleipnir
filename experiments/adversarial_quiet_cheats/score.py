"""Score arm 1's rows (modelled on monitor-aware's score.py, which it reuses).

GPU, on the A100 `gleipnir-qwen-gpu` in ~/gleipnir-coc (vm_score.sh): one
monitor at a time, resumable, monitor-aware's canaries before and after, and
the decision-token residual saved for every row at monitor-aware's layers:

    $PY experiments/adversarial_quiet_cheats/score.py canary --monitor 4b
    $PY experiments/adversarial_quiet_cheats/score.py score --monitor 4b
    $PY experiments/adversarial_quiet_cheats/score.py score --monitor 27b

Amendment 2 adds, for the 4B only, a raw-transcript capture pass (`raw`): the
rendered trajectory alone, with no monitor prompt and no chat template, one
forward pass per row with no generation, once with the v0.3 adapter and once
with it disabled (the frozen base). It saves the last-token residual and the
mean over all tokens at the 4B's layers in FP32. The prompted pass is
unchanged:

    $PY experiments/adversarial_quiet_cheats/score.py raw --monitor 4b

Kimi K3 through OpenRouter (on the Mac), frozen order, capped at $1.00 and
started only with a balance of at least $2.00. The 5 drift originals run
first; if any differs from monitor-aware's Kimi score by more than 0.05, it
stops before scoring anything else (Amendment 1):

    PYTHONPATH=src:. python experiments/adversarial_quiet_cheats/score.py kimi

Rows must match `arm1_deletion.rows_sha256` in config.json (set at the rows
freeze); nothing scores before that.
"""

from __future__ import annotations

import argparse
import inspect
import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from experiments.adversarial_quiet_cheats.common import (
    ARM1,
    DATA,
    MA_CONFIG,
    MA_OUT,
    OUT,
    file_sha256,
    read_jsonl,
    sha,
)
from gleipnir.adversarial.contract import MAX_TOKENS, split_prompt

SCORING = OUT / "scoring"
GPU = MA_CONFIG["scoring"]["gpu_monitors"]
KIMI = ARM1["kimi"]
RAW = ARM1["raw_capture"]


def frozen(name: str) -> str:
    expected = (ARM1.get("rows_sha256") or {}).get(name)
    if not expected:
        raise SystemExit(f"{name}: no frozen SHA256 in config (rows freeze not done)")
    path = DATA / (f"{name}.jsonl" if name.startswith("rows_") else f"{name}.json")
    if file_sha256(path) != expected:
        raise ValueError(f"{path} does not match its frozen SHA256")
    return str(path)


def load_rows(prompt: str) -> list[dict[str, Any]]:
    frozen(f"rows_{prompt}")
    return read_jsonl(DATA / f"rows_{prompt}.jsonl")


def scorable(rows: list[dict], monitor: str) -> list[dict]:
    limit_of = "27b" if monitor == "kimi" else monitor
    return [r for r in rows if not r["metadata"]["too_long"][limit_of]]


def record_for(row: dict[str, Any]) -> dict[str, Any]:
    m = row["metadata"]
    return {
        "id": row["id"],
        "group": m["condition"],
        "label": m["ground_truth"],
        "role": m["role"],
        "setup": m["setup"],
        "task_id": m["task_id"],
        "parent_id": m["parent_id"],
    }


# ---------------------------------------------------------------- GPU


def gpu_receipt(passed: bool, tag: str, monitor: str, check: dict) -> None:
    from experiments.adversarial_monitor_aware.score import git_commit

    base = SCORING / monitor
    base.mkdir(parents=True, exist_ok=True)
    with (base / "canary_runs.jsonl").open("a") as h:
        h.write(
            json.dumps(
                {**check, "when": tag, "time": time.time(), "commit": git_commit()}
            )
            + "\n"
        )


def canary(args: argparse.Namespace) -> None:
    from experiments.adversarial_monitor_aware.score import load_scorer, run_canaries

    scorer, spec = load_scorer(args.monitor, args.device)
    check = run_canaries(scorer, spec, args.monitor)
    gpu_receipt(check["passed"], "standalone", args.monitor, check)
    print(f"canary passed={check['passed']}")
    if not check["passed"]:
        raise SystemExit(1)


def score(args: argparse.Namespace) -> None:
    import numpy as np

    from experiments.adversarial_lab_liars.run import Capture
    from experiments.adversarial_monitor_aware.score import (
        git_commit,
        load_scorer,
        run_canaries,
    )
    from gleipnir.adversarial.scorer import TooLong

    scorer, spec = load_scorer(args.monitor, args.device)
    check = run_canaries(scorer, spec, args.monitor)
    gpu_receipt(check["passed"], "before", args.monitor, check)
    if not check["passed"]:
        raise SystemExit("canaries failed: not scoring")
    rows = scorable(load_rows(spec["prompt"]), args.monitor)
    directory = SCORING / args.monitor / "rows"
    acts = directory / "activations"
    acts.mkdir(parents=True, exist_ok=True)
    (directory / "identity.json").write_text(
        json.dumps(
            {
                "monitor": args.monitor,
                "rows": len(rows),
                "input_sha256": ARM1["rows_sha256"][f"rows_{spec['prompt']}"],
                "commit": git_commit(),
                "layers": spec["layers"],
                "capture": "prompted",
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
            record = {**record_for(row), "capture": "prompted"}
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
                rate = tokens / max(time.time() - started, 1e-9)
                print(f"{args.monitor}: {n}/{len(todo)}, {rate:,.0f} tok/s", flush=True)
    final = run_canaries(scorer, spec, args.monitor)
    gpu_receipt(final["passed"], "after", args.monitor, final)
    print(f"closing canary passed={final['passed']}")
    if not final["passed"]:
        raise SystemExit(1)


# ---------------------------------------------------------------- raw capture


def raw_text(prompt: str) -> str:
    """The raw-capture input: the rendered trajectory alone (Amendment 2).

    The row's prompt is `instruction + OPEN + trajectory + CLOSE`; this returns
    the trajectory bytes between the tags, without the tags, the instruction,
    the chat template or `Prediction:`.
    """
    return split_prompt(prompt).trajectory


def raw_format_sha256() -> str:
    """SHA256 of `raw_text`'s source, frozen in config (`raw_capture`)."""
    return sha(inspect.getsource(raw_text))


def raw_ids(tokenizer: Any, prompt: str) -> list[int]:
    """Token ids of `raw_text(prompt)`. Qwen3.5's tokenizer adds no BOS or other
    special token to plain text; this is checked, so the ids are the text's."""
    text = raw_text(prompt)
    ids = tokenizer.encode(text, add_special_tokens=False)
    if tokenizer.encode(text, add_special_tokens=True) != ids:
        raise ValueError("tokenizer adds special tokens to raw text")
    if set(ids) & set(tokenizer.all_special_ids):
        raise ValueError("raw text tokenizes to a special token")
    if len(ids) > MAX_TOKENS:
        raise ValueError(f"raw input {len(ids)} tokens, limit {MAX_TOKENS}")
    return ids


class RawCapture:
    """Forward hooks keeping the last-token residual and the mean over all
    tokens at each layer (block output, FP32). They change no output."""

    def __init__(self, model: Any, layers: list[int], hidden: int):
        from experiments.adversarial_lab_liars.run import decoder_blocks

        self.layers, self.hidden = layers, hidden
        self.blocks = decoder_blocks(model, max(layers))
        self.values: dict[str, dict[int, Any]] = {"last": {}, "mean": {}}
        self.handles: list = []

    def hook(self, layer: int):
        def save(_module, _inputs, output):
            state = output[0] if isinstance(output, tuple) else output
            if (
                state.dim() != 3
                or state.shape[0] != 1
                or state.shape[-1] != self.hidden
            ):
                raise ValueError("unexpected residual shape")
            if layer in self.values["last"]:
                raise ValueError("decoder block called twice")
            hidden = state[0].float()
            self.values["last"][layer] = hidden[-1].cpu().numpy().copy()
            self.values["mean"][layer] = hidden.mean(dim=0).cpu().numpy().copy()

        return save

    def __enter__(self):
        self.values = {"last": {}, "mean": {}}
        for layer in self.layers:
            self.handles.append(
                self.blocks[layer - 1].register_forward_hook(self.hook(layer))
            )
        return self

    def __exit__(self, *_):
        for handle in self.handles:
            handle.remove()
        self.handles = []

    def arrays(self) -> dict[str, Any]:
        """{"last": (layers, hidden), "mean": (layers, hidden)}, float32; the
        same shape as the prompted pass's decision-token array."""
        import numpy as np

        out = {}
        for pool, values in self.values.items():
            if sorted(values) != sorted(self.layers):
                raise ValueError("missing residual capture")
            out[pool] = np.stack([values[layer] for layer in self.layers])
            if not np.isfinite(out[pool]).all():
                raise ValueError("nonfinite activation")
        return out


def raw_forward(scorer: Any, ids: list[int], adapter: bool) -> None:
    """One forward pass on raw ids, no generation (logits are not used)."""
    torch = scorer.torch
    with torch.inference_mode():
        inputs = torch.tensor([ids], device=scorer.device)
        if adapter or scorer.spec.adapter is None:
            scorer.model(input_ids=inputs, logits_to_keep=1)
        else:
            with scorer.model.disable_adapter():
                scorer.model(input_ids=inputs, logits_to_keep=1)


def raw(args: argparse.Namespace) -> None:
    import numpy as np

    from experiments.adversarial_monitor_aware.score import (
        git_commit,
        load_scorer,
        run_canaries,
    )

    if args.monitor not in RAW["monitors"]:
        raise SystemExit(f"raw capture is for {RAW['monitors']} only")
    if raw_format_sha256() != RAW["format_sha256"]:
        raise SystemExit("raw_text differs from its frozen SHA256")
    scorer, spec = load_scorer(args.monitor, args.device)
    check = run_canaries(scorer, spec, args.monitor)
    gpu_receipt(check["passed"], "raw_before", args.monitor, check)
    if not check["passed"]:
        raise SystemExit("canaries failed: not capturing")
    rows = scorable(load_rows(spec["prompt"]), args.monitor)
    directory = SCORING / args.monitor / "raw"
    acts = directory / "activations"
    acts.mkdir(parents=True, exist_ok=True)
    (directory / "identity.json").write_text(
        json.dumps(
            {
                "monitor": args.monitor,
                "capture": "raw",
                "rows": len(rows),
                "weights": RAW["weights"],
                "input": RAW["input"],
                "format_function": "score.raw_text",
                "format_sha256": raw_format_sha256(),
                "input_sha256": ARM1["rows_sha256"][f"rows_{spec['prompt']}"],
                "commit": git_commit(),
                "layers": spec["layers"],
                "activation": RAW["activation"],
                "scorer": scorer.identity(),
            },
            indent=2,
        )
        + "\n"
    )
    path = directory / "captures.jsonl"
    done = {(r["id"], r["weights"]) for r in read_jsonl(path)}
    started, tokens = time.time(), 0
    with path.open("a") as handle:
        for weights in RAW["weights"]:
            todo = [r for r in rows if (r["id"], weights) not in done]
            for n, row in enumerate(todo, 1):
                ids = raw_ids(scorer.tokenizer, row["prompt"])
                with RawCapture(
                    scorer.model, spec["layers"], spec["hidden_size"]
                ) as capture:
                    raw_forward(scorer, ids, adapter=weights == "adapter")
                arrays = capture.arrays()
                file = acts / f"{sha(row['id'])[:24]}.{weights}.npz"
                np.savez(file, **arrays)
                record = {
                    **record_for(row),
                    "capture": "raw",
                    "weights": weights,
                    "status": "ok",
                    "tokens": len(ids),
                    "input_sha256": sha(raw_text(row["prompt"])),
                    "token_ids_sha256": sha(json.dumps(ids)),
                    "activation_file": file.name,
                    "activation_sha256": file_sha256(file),
                    "shape": list(arrays["last"].shape),
                }
                tokens += len(ids)
                handle.write(json.dumps(record) + "\n")
                handle.flush()
                if n % 25 == 0 or n == len(todo):
                    rate = tokens / max(time.time() - started, 1e-9)
                    print(
                        f"raw {weights}: {n}/{len(todo)}, {rate:,.0f} tok/s",
                        flush=True,
                    )
    final = run_canaries(scorer, spec, args.monitor)
    gpu_receipt(final["passed"], "raw_after", args.monitor, final)
    print(f"closing canary passed={final['passed']}")
    if not final["passed"]:
        raise SystemExit(1)


# ---------------------------------------------------------------- Kimi


def ma_kimi_scores() -> dict[str, float]:
    """monitor-aware's Kimi scores by parent run id (status ok)."""
    path = MA_OUT / "scoring" / "kimi" / "rows" / "predictions.jsonl"
    return {r["parent_id"]: r["score"] for r in read_jsonl(path) if r["status"] == "ok"}


def drift_verdict(
    records: list[dict], reference: dict[str, float], tolerance: float
) -> dict[str, Any]:
    """Pure: each drift row scored ok and within tolerance of monitor-aware's."""
    rows = []
    for r in records:
        ref = reference.get(r.get("parent_id"))
        ok = (
            r.get("status") == "ok"
            and ref is not None
            and abs(r["score"] - ref) <= tolerance
        )
        rows.append(
            {"id": r["id"], "score": r.get("score"), "reference": ref, "ok": ok}
        )
    return {
        "passed": len(rows) == KIMI["drift_rows"] and all(x["ok"] for x in rows),
        "tolerance": tolerance,
        "rows": rows,
    }


def kimi(args: argparse.Namespace) -> None:
    from experiments.adversarial_is_it_4b.run import kimi_record
    from experiments.adversarial_lab_liars.common import (
        openrouter_balance,
        openrouter_key,
    )
    from experiments.adversarial_monitor_aware.score import git_commit, kimi_config
    from gleipnir.teachers.openrouter import request_settings

    frozen("kimi_order")
    order = json.loads((DATA / "kimi_order.json").read_text())
    rows = {r["id"]: r for r in scorable(load_rows("teacher"), "kimi")}
    key = openrouter_key()
    balance = openrouter_balance(key)
    print(f"OpenRouter balance before: ${balance:.2f}", flush=True)
    if balance < KIMI["min_balance_usd"]:
        raise SystemExit(f"balance under ${KIMI['min_balance_usd']}: not starting")
    config = kimi_config()
    directory = SCORING / "kimi" / "rows"
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "predictions.jsonl"
    done, spent = {}, [0.0]
    for r in read_jsonl(path):
        spent[0] += r.get("cost") or 0.0
        if r["status"] == "ok":
            done[r["id"]] = r
    (directory / "identity.json").write_text(
        json.dumps(
            {
                "monitor": "kimi",
                "commit": git_commit(),
                "request_settings": request_settings(config),
            },
            indent=2,
        )
        + "\n"
    )
    lock = threading.Lock()

    def work(item):
        if spent[0] >= KIMI["max_usd"]:
            return None
        row = rows[item["id"]]
        record = kimi_record(row, config, key)
        record.update({k: v for k, v in record_for(row).items() if k not in record})
        record["tier"] = item["tier"]
        with lock:
            spent[0] += record.get("cost") or 0.0
            with path.open("a") as h:
                h.write(json.dumps(record) + "\n")
        return record

    def run_tier(items):
        todo = [o for o in items if o["id"] not in done and o["id"] in rows]
        with ThreadPoolExecutor(KIMI["workers"]) as pool:
            for record in pool.map(work, todo):
                if record and record["status"] == "ok":
                    done[record["id"]] = record

    drift_items = [o for o in order if o["tier"] == "drift"]
    run_tier(drift_items)
    verdict = drift_verdict(
        [done[o["id"]] for o in drift_items if o["id"] in done],
        ma_kimi_scores(),
        KIMI["drift_tolerance"],
    )
    (SCORING / "kimi" / "drift.json").write_text(json.dumps(verdict, indent=2) + "\n")
    print(f"drift check passed={verdict['passed']}", flush=True)
    if verdict["passed"]:
        rest = [o for o in order if o["tier"] != "drift"]
        if args.limit:
            rest = rest[: args.limit]
        run_tier(rest)
    summary = {
        "spent": round(spent[0], 4),
        "drift_passed": verdict["passed"],
        "scored_ok": len(done),
        "order_rows": len(order),
        "stopped_at_cap": spent[0] >= KIMI["max_usd"],
        "balance_before": round(balance, 2),
        "balance_after": round(openrouter_balance(key), 2),
    }
    with (SCORING / "kimi" / "runs.jsonl").open("a") as h:
        h.write(
            json.dumps({**summary, "time": time.time(), "commit": git_commit()}) + "\n"
        )
    print(json.dumps(summary))
    if not verdict["passed"]:
        raise SystemExit("drift check failed: originals can't be reused; stopped")


def main() -> None:
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    k = commands.add_parser("kimi")
    k.add_argument("--limit", type=int)
    for name in ("canary", "score", "raw"):
        c = commands.add_parser(name)
        c.add_argument("--monitor", required=True, choices=sorted(GPU))
        c.add_argument("--device", default="cuda")
    args = parser.parse_args()
    {"kimi": kimi, "canary": canary, "score": score, "raw": raw}[args.command](args)


if __name__ == "__main__":
    main()
