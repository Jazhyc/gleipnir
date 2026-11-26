"""Score prompts one at a time on a running release server and compare runs.

The image's larger CUDA graphs and its native tokenizer must not change any
score at one request at a time. Run this against a server started with
``--recorded-graphs --transformers-tokenizer`` and against the default
server, and pass the first output to the second run as ``--compare``: every
score has to be identical. Without ``--workload`` it uses synthetic prompts
of about 200 to 4,400 tokens, which need no data and cover every graph size.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import httpx

from experiments.release_serving.image.smoke import LINE, SUFFIX, score


def synthetic() -> list[dict]:
    rows = []
    for lines in range(12, 260):
        body = "".join(f"step {i} of {lines}: {LINE}" for i in range(lines))
        text = (
            "<|im_start|>user\nMonitor this trajectory.\n<agent_trajectory>\n"
            + body
            + SUFFIX
        )
        rows.append({"id": f"synthetic-{lines}", "prompt": text})
    return rows


def load(path: Path) -> list[dict]:
    text = path.read_text()
    if path.suffix == ".json":
        return json.loads(text)
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--workload", type=Path, help="rows with id and prompt")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--compare", type=Path, help="output of an earlier run")
    args = parser.parse_args()
    rows = load(args.workload) if args.workload else synthetic()
    with httpx.Client(
        base_url=f"http://127.0.0.1:{args.port}", trust_env=False, timeout=600
    ) as client:
        scored = []
        for row in rows:
            payload = score(client, row["prompt"])
            scored.append(
                {
                    "id": row["id"],
                    "score": payload["score"],
                    "logits": payload["logits"],
                    "prompt_tokens": payload["prompt_tokens"],
                }
            )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("".join(json.dumps(row) + "\n" for row in scored))
    tokens = [row["prompt_tokens"] for row in scored]
    report = {"rows": len(scored), "tokens": [min(tokens), max(tokens)]}
    if args.compare:
        earlier = {row["id"]: row for row in load(args.compare)}
        changed = [
            row["id"]
            for row in scored
            if row["logits"] != earlier[row["id"]]["logits"]
            or row["prompt_tokens"] != earlier[row["id"]]["prompt_tokens"]
        ]
        report["changed"] = len(changed)
        print("identity " + json.dumps(report), flush=True)
        if changed:
            raise SystemExit(f"scores changed: {changed[:10]}")
        return
    print("identity " + json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
