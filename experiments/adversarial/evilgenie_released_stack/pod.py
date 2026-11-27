"""Render, count and score EvilGenie runs on a Pod running the release image.

Run on the Pod with the image runtime, while the model's server is up:

    /tmp/gleipnir-vllm031-runtime/bin/python -m \
        experiments.adversarial.evilgenie_released_stack.pod workload --model 4b
    /tmp/gleipnir-vllm031-runtime/bin/python -m \
        experiments.adversarial.evilgenie_released_stack.pod score --model 4b

``workload`` renders every run with the model's released instruction and chat
template, exactly as the recorded OOD workload (checked byte for byte on its
first rows), and counts prompt tokens with the server's Gigatoken encoder.
``score`` sends the workload in the recorded campaign's batching: workload
order, sequential batches of 128 rows at concurrency 128.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import statistics
from pathlib import Path

ROOT = next(
    p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").is_file()
)
CONFIG = json.loads(Path(__file__).with_name("config.json").read_text())


def read_jsonl(path: Path) -> list[dict]:
    with path.open() as handle:
        return [json.loads(line) for line in handle]


def write_jsonl(path: Path, rows: list[dict]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = "".join(json.dumps(r) + "\n" for r in rows)
    path.write_text(text)
    return hashlib.sha256(text.encode()).hexdigest()


# The released prompt contract (``gleipnir.adversarial.contract``), copied
# here because the image's repository snapshot predates that module.
OPEN = "\n<agent_trajectory>\n"
CLOSE = "\n</agent_trajectory>\n"
ASSISTANT_SUFFIX = "<|im_start|>assistant\n<think>\n\n</think>\n\n"


def render_user(head: str, trajectory: str) -> str:
    return f"{head}{OPEN}{trajectory}{CLOSE}"


def render(tokenizer, user_prompt: str) -> str:
    rendered = tokenizer.apply_chat_template(
        [{"role": "user", "content": user_prompt}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    if not rendered.endswith(ASSISTANT_SUFFIX):
        raise ValueError("chat template no longer ends at the non-thinking boundary")
    return rendered + "Prediction:"


def workload(model: str) -> None:
    import gigatoken
    from transformers import AutoTokenizer

    from experiments.release_serving.targets import TARGETS

    spec = CONFIG["models"][model]
    tokenizer = AutoTokenizer.from_pretrained(TARGETS[model].merged_path)
    head = (ROOT / spec["instruction"]).read_text()
    # The renderer must reproduce the recorded OOD workload byte for byte
    # (rows prepared locally by ``check_rows``, so the Pod needs no OOD files).
    for row in read_jsonl(ROOT / CONFIG["workload_dir"] / f"check_{model}.jsonl"):
        if render(tokenizer, render_user(head, row["trajectory"])) != row["prompt"]:
            raise ValueError(f"{model}: rendering differs from the OOD workload")
    states = read_jsonl(ROOT / CONFIG["states"])
    native = gigatoken.Tokenizer(tokenizer)
    rows = []
    for state in states:
        prompt = render(tokenizer, render_user(head, state["state"]))
        rows.append(
            {
                "id": state["id"],
                "prompt": prompt,
                "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
                "prompt_tokens": len(native.encode(prompt)),
            }
        )
    path = ROOT / CONFIG["workload_dir"] / f"{model}.jsonl"
    digest = write_jsonl(path, rows)
    tokens = [r["prompt_tokens"] for r in rows]
    print(
        json.dumps(
            {
                "model": model,
                "rows": len(rows),
                "prompt_tokens": sum(tokens),
                "max_prompt_tokens": max(tokens),
                "sha256": digest,
            }
        ),
        flush=True,
    )


def check_rows() -> None:
    """Local step: the first OOD workload rows with their trajectories."""
    from gleipnir.adversarial.contract import split_prompt

    ood = {r["id"]: r for r in read_jsonl(ROOT / CONFIG["ood_prompts"])}
    for model, spec in CONFIG["models"].items():
        rows = []
        with (ROOT / spec["ood_workload"]).open() as handle:
            for line in handle:
                row = json.loads(line)
                body = split_prompt(ood[row["id"]]["prompt"]).trajectory
                rows.append(
                    {"id": row["id"], "trajectory": body, "prompt": row["prompt"]}
                )
                if len(rows) == CONFIG["render_check_rows"]:
                    break
        print(
            model,
            write_jsonl(ROOT / CONFIG["workload_dir"] / f"check_{model}.jsonl", rows),
        )


def score(model: str, port: int) -> None:
    from gleipnir.evaluation.http_score import score_batch

    rows = read_jsonl(ROOT / CONFIG["workload_dir"] / f"{model}.jsonl")
    config = {
        "port": port,
        "concurrency": CONFIG["concurrency"],
        "timeout_seconds": 1800,
    }
    scored, seconds = [], 0.0
    step = CONFIG["batch_rows"]
    for start in range(0, len(rows), step):
        values, elapsed = asyncio.run(score_batch(rows[start : start + step], config))
        scored += values
        seconds += elapsed
    out = ROOT / CONFIG["results_dir"]
    write_jsonl(
        out / f"{model}.jsonl",
        [
            {
                k: r[k]
                for k in (
                    "id",
                    "prompt_sha256",
                    "prompt_tokens",
                    "score",
                    "margin",
                    "logits",
                    "latency_seconds",
                )
            }
            for r in scored
        ],
    )
    tokens = sum(r["prompt_tokens"] for r in rows)
    summary = {
        "model": model,
        "rows": len(scored),
        "seconds": seconds,
        "prompt_tokens": tokens,
        "prompt_tokens_per_second": tokens / seconds,
        "median_latency_seconds": statistics.median(
            r["latency_seconds"] for r in scored
        ),
    }
    (out / f"{model}_timing.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("step", choices=("check-rows", "workload", "score"))
    parser.add_argument("--model", choices=("4b", "27b"))
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument(
        "--config", type=Path, help="another experiment's config (same keys)"
    )
    args = parser.parse_args()
    if args.config:
        CONFIG.clear()
        CONFIG.update(json.loads((ROOT / args.config).read_text()))
    if args.step == "check-rows":
        check_rows()
    elif args.step == "workload":
        workload(args.model)
    else:
        score(args.model, args.port)


if __name__ == "__main__":
    main()
