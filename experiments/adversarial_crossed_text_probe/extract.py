"""Extract bounded pooled residuals on the existing, separately authorized GPU."""

from __future__ import annotations

import argparse
import fcntl
import importlib.metadata
import json
import math
import time
from pathlib import Path
from typing import Any

import numpy as np

from experiments.adversarial_crossed_text_probe.common import (
    ROOT,
    append,
    bind,
    check_deadline,
    code_token_indices,
    config,
    digest,
    load_inputs,
    originals,
    protocol_identity,
    read_jsonl,
    require_committed,
    row_metadata,
    selected_rows,
    sha,
    write_json,
)


class Guard:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.started = time.monotonic()

    def check(self) -> None:
        check_deadline(ROOT / self.cfg["vm"])
        if time.monotonic() - self.started >= self.cfg["max_seconds"]:
            raise SystemExit("probe invocation wall-time cap reached")


class GuardedScorer:
    """Check deadlines and raw logits on *every* forward, including canaries."""

    def __init__(self, scorer: Any, guard: Guard):
        self.scorer, self.guard = scorer, guard

    def __getattr__(self, key: str) -> Any:
        return getattr(self.scorer, key)

    def score(self, request: str, adapter: bool = True) -> dict:
        self.guard.check()
        result = self.scorer.score(request, adapter=adapter)
        self.guard.check()
        if not all(math.isfinite(result[k]) for k in ("score", "logit_0", "logit_1")):
            raise ValueError("nonfinite score or logits")
        return result


def decoder_blocks(model: Any, count: int) -> Any:
    candidates = [
        (name, module)
        for name, module in model.named_modules()
        if name.endswith(".layers")
        and hasattr(module, "__len__")
        and len(module) == count
    ]
    if len(candidates) != 1:
        raise ValueError(
            f"expected one {count}-block decoder, found {[n for n, _ in candidates]}"
        )
    return candidates[0][1]


class Capture:
    def __init__(self, scorer: Any, arm: dict, indices: list[int], tokens: int):
        self.scorer, self.arm = scorer, arm
        self.indices, self.tokens = indices, tokens
        self.handles: list = []
        self.values: dict[str, np.ndarray] = {}

    def hook(self, layer: int):
        def save(_module, _inputs, output):
            hidden = output[0] if isinstance(output, tuple) else output
            if tuple(hidden.shape) != (1, self.tokens, self.arm["hidden_size"]):
                raise ValueError("unexpected block residual shape")
            if f"L{layer}:decision" in self.values:
                raise ValueError("decoder block called twice")
            # BF16 forward unchanged; only these small summaries leave the GPU.
            self.values[f"L{layer}:decision"] = (
                hidden[0, -1].float().cpu().numpy().copy()
            )
            pooled = hidden[0, self.indices].float().mean(dim=0)
            self.values[f"L{layer}:code_mean"] = pooled.cpu().numpy().copy()
            return None

        return save

    def __enter__(self):
        blocks = decoder_blocks(self.scorer.model, self.arm["layers"][-1])
        try:
            for layer in self.arm["layers"]:
                self.handles.append(
                    blocks[layer - 1].register_forward_hook(self.hook(layer))
                )
        except BaseException:
            self.__exit__(None, None, None)
            raise
        return self

    def __exit__(self, *_args):
        for handle in self.handles:
            handle.remove()

    def validate(self) -> None:
        expected = {
            f"L{layer}:{pool}"
            for layer in self.arm["layers"]
            for pool in ("decision", "code_mean")
        }
        if set(self.values) != expected:
            raise ValueError("missing or unexpected residual capture")
        if any(not np.isfinite(x).all() for x in self.values.values()):
            raise ValueError("nonfinite activation")


def capture_row(
    scorer: Any, arm: dict, request: str, original: str, added: int
) -> tuple[dict, dict[str, np.ndarray]]:
    ids = scorer.encode(request)
    indices = code_token_indices(scorer.tokenizer, request, original, added, ids)
    with Capture(scorer, arm, indices, len(ids)) as capture:
        result = scorer.score(request)
    capture.validate()
    if result["tokens"] != len(ids):
        raise ValueError("scorer token count changed")
    result.update(
        code_tokens=len(indices),
        token_ids_sha256=sha(json.dumps(ids)),
        request_sha256=sha(request),
        margin=result["logit_1"] - result["logit_0"],
    )
    return result, capture.values


def verify_packages(expected: dict) -> None:
    for name, version in expected.items():
        try:
            actual = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            actual = None
        if actual != version:
            raise ValueError(f"Step 1 package mismatch: {name}: {actual} != {version}")


def canaries(
    scorer: Any,
    arm: dict,
    rows: list[dict],
    selection: dict,
    original: dict,
    step1: dict,
) -> dict:
    from experiments.adversarial_code_or_comments.common import instruction, load_pairs
    from experiments.adversarial_code_or_comments.score import canary, with_instruction

    pairs, _ = load_pairs(step1)
    head = instruction(arm["prompt"], step1)
    original_check = canary(
        scorer,
        pairs,
        selection,
        step1,
        None if arm["prompt"] == "student" else head,
        False,
    )
    receipt = {"passed": False, "step1": original_check, "hooks": []}
    if not original_check["passed"]:
        return receipt
    for condition in ("none", "specific_accusation"):
        subset = [
            r
            for r in rows
            if r["metadata"]["split"] == "development"
            and r["metadata"]["condition"] == condition
        ]
        task = subset[0]["metadata"]["task_id"]
        for row in [r for r in subset if r["metadata"]["task_id"] == task]:
            request = with_instruction(row["prompt"], head)
            m = row["metadata"]
            plain = scorer.score(request)
            captured, _ = capture_row(
                scorer, arm, request, original[(str(task), m["code"])], m["added_chars"]
            )
            drift = max(
                abs(plain[k] - captured[k]) for k in ("score", "logit_0", "logit_1")
            )
            receipt["hooks"].append(
                {
                    "id": row["id"],
                    "plain": plain,
                    "hooked": captured,
                    "max_drift": drift,
                }
            )
    receipt["passed"] = len(receipt["hooks"]) == 4 and all(
        r["max_drift"] <= 1e-6 for r in receipt["hooks"]
    )
    return receipt


def validate_fit_gate(folder: Path, identity: dict) -> None:
    fit = json.loads((folder / "fit.json").read_text())
    if fit["identity"] != identity or fit["development_complete_sha256"] != digest(
        folder / "development.complete.json"
    ):
        raise ValueError(
            "held-out extraction requires a matching frozen development fit"
        )


def run(args: argparse.Namespace) -> None:
    if not args.execute:
        raise SystemExit(
            "implementation only; --execute requires separate GPU authorization"
        )
    cfg = config()
    guard = Guard(cfg)
    guard.check()  # Before torch import, model load, or any output mutation.
    require_committed()
    rows, selection = load_inputs(cfg)
    arm = cfg["arms"][args.arm]
    identity = {"protocol": protocol_identity(cfg), "arm": args.arm}
    folder = ROOT / cfg["output"] / args.arm
    if args.split == "heldout":
        validate_fit_gate(folder, identity)
    verify_packages(arm["runtime"]["packages"])
    root = ROOT / cfg["output"]
    root.mkdir(parents=True, exist_ok=True)
    with (root / ".extract.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise SystemExit("another probe extractor is running") from None
        folder.mkdir(exist_ok=True)
        bind(folder / "identity.json", identity)
        if args.split == "heldout":
            bind(
                folder / "heldout.fit.json", {"fit_sha256": digest(folder / "fit.json")}
            )
        receipt_path = folder / f"canary-{args.split}-{time.time_ns()}.json"
        receipt = {"passed": False, "stage": "load"}
        try:
            guard.check()
            from experiments.adversarial_code_or_comments.common import instruction
            from experiments.adversarial_code_or_comments.score import with_instruction
            from gleipnir.adversarial.scorer import Scorer

            scorer = GuardedScorer(Scorer(arm["monitor"], "cuda", "bfloat16"), guard)
            guard.check()
            runtime = scorer.identity()
            if runtime != arm["runtime"]:
                raise ValueError("model, device or backend differs from Step 1 runtime")
            write_json(folder / "runtime.json", runtime)
            step1 = json.loads(
                (ROOT / "experiments/adversarial_crossed_text/config.json").read_text()
            )
            head = instruction(arm["prompt"], step1)
            student = instruction("student", step1)
            from gleipnir.adversarial.contract import split_prompt

            if any(split_prompt(r["prompt"]).head != student for r in rows):
                raise ValueError("Step 1 student prompt identity mismatch")
            original = originals(rows)
            receipt = canaries(scorer, arm, rows, selection, original, step1)
            write_json(receipt_path, receipt)
            if not receipt["passed"]:
                raise ValueError("canary failed; no study rows extracted")
            todo = selected_rows(rows, args.split, cfg)
            journal = folder / f"{args.split}.jsonl"
            done_rows = read_jsonl(journal)
            done = {r["id"]: r for r in done_rows}
            if len(done) != len(done_rows) or set(done) - {r["id"] for r in todo}:
                raise ValueError("duplicate or unexpected completed row")
            for record in done.values():
                if digest(folder / record["file"]) != record["sha256"]:
                    raise ValueError("saved activation checksum mismatch")
            tokens, started = 0, time.monotonic()
            scorer.torch.cuda.reset_peak_memory_stats()
            for n, row in enumerate(todo, 1):
                guard.check()
                if row["id"] in done:
                    continue
                m = row["metadata"]
                code = original[(str(m["task_id"]), m["code"])]
                result, values = capture_row(
                    scorer,
                    arm,
                    with_instruction(row["prompt"], head),
                    code,
                    m["added_chars"],
                )
                filename = sha(row["id"]) + ".npz"
                target = folder / filename
                tmp = target.with_suffix(".tmp")
                with tmp.open("wb") as handle:
                    np.savez(handle, **values)
                    handle.flush()
                tmp.replace(target)
                record = {
                    **row_metadata(row, code),
                    **result,
                    "file": filename,
                    "sha256": digest(target),
                }
                append(journal, record)
                tokens += result["tokens"]
                if n % 50 == 0 or n == len(todo):
                    rate = tokens / max(time.monotonic() - started, 1)
                    print(
                        f"{args.arm} {args.split} {n}/{len(todo)}, "
                        f"{rate:.0f} prompt tokens/s",
                        flush=True,
                    )
            guard.check()
            write_json(
                folder / f"{args.split}.complete.json",
                {
                    "rows": len(todo),
                    "journal_sha256": digest(journal),
                    "identity": identity,
                },
            )
            write_json(
                folder / f"{args.split}.timing-{time.time_ns()}.json",
                {
                    "new_tokens": tokens,
                    "extraction_seconds": time.monotonic() - started,
                    "invocation_seconds": time.monotonic() - guard.started,
                    "peak_gpu_bytes": scorer.torch.cuda.max_memory_allocated(),
                },
            )
        except BaseException as error:
            receipt.update(error_type=type(error).__name__, error=str(error))
            write_json(receipt_path, receipt)
            append(
                folder / "failures.jsonl",
                {
                    "split": args.split,
                    "time": time.time(),
                    "type": type(error).__name__,
                    "error": str(error),
                },
            )
            raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", choices=list(config()["arms"]), required=True)
    parser.add_argument("--split", choices=["development", "heldout"], required=True)
    parser.add_argument("--execute", action="store_true")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
