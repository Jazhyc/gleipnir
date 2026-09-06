"""Matched OpenRouter teacher scoring, using the existing resumable client."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import yaml
from sklearn.metrics import roc_auc_score

from gleipnir.calibration import binary_calibration
from gleipnir.monitoring_systems_screen import atomic_write_json, atomic_write_jsonl
from gleipnir.openrouter import binary_score_from_top_logprobs
from gleipnir.openrouter_cli import main as annotate


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate(rows: list[dict], prompts: dict) -> None:
    if len({r["id"] for r in rows}) != len(rows):
        raise ValueError("duplicate output")
    for r in rows:
        p = prompts[r["id"]]
        score, _ = binary_score_from_top_logprobs(r["top_logprobs"])
        if (
            r["provider"].lower() != "wafer"
            or not r["model"].startswith("z-ai/glm-5.3-flash")
            or r["prompt_sha256"] != hashlib.sha256(p["prompt"].encode()).hexdigest()
            or r["metadata"] != p["metadata"]
            or not np.isfinite(score)
            or not np.isclose(score, r["score"], atol=1e-12, rtol=0)
        ):
            raise ValueError("teacher response identity/score drift")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=["prepare", "canary", "run", "analyze"])
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    c = yaml.safe_load(Path(__file__).with_name("glm.yaml").read_text())
    root = Path(c["output"])
    paths = [Path(c["input"]), Path(c["pairs"]), Path(__file__).with_name("glm.yaml")]
    prompts = {r["id"]: r for r in map(json.loads, paths[0].open())}
    pairs = list(map(json.loads, paths[1].open()))
    if len(prompts) != 640 or {r["id"] for r in pairs} != set(prompts):
        raise ValueError("matched population drift")
    for r in pairs:
        m = prompts[r["id"]]["metadata"]
        if r["source"] != m["source_dataset"] or r["label"] != m["ground_truth"]:
            raise ValueError("matched label/source drift")
    if args.phase == "prepare":
        if root.exists():
            raise FileExistsError(root)
        chosen = []
        for source in sorted({r["source"] for r in pairs}):
            for label in (0, 1):
                chosen.append(
                    next(
                        prompts[r["id"]]
                        for r in pairs
                        if r["source"] == source and r["label"] == label
                    )
                )
        atomic_write_jsonl(root / "canary_prompts.jsonl", chosen)
        paths.append(root / "canary_prompts.jsonl")
        atomic_write_json(
            root / "manifest.json", {"files": {str(p): sha(p) for p in paths}}
        )
        return
    manifest = json.loads((root / "manifest.json").read_text())
    if any(sha(Path(p)) != h for p, h in manifest["files"].items()):
        raise ValueError("frozen input drift")
    if args.phase in ("canary", "run"):
        canary = args.phase == "canary"
        if not canary:
            gate = list(map(json.loads, (root / "canary.jsonl").open()))
            if len(gate) != 10:
                raise ValueError("ten-row canary required")
            validate(gate, prompts)
        destination = root / ("canary.jsonl" if canary else "scores.jsonl")
        argv = [
            "--input",
            str(root / "canary_prompts.jsonl") if canary else c["input"],
            "--output",
            str(destination),
            "--model",
            c["model"],
            "--provider-only",
            c["provider"],
            "--no-allow-fallbacks",
            "--no-enforce-distillable-text",
            "--no-explicit-cache",
            "--no-warm-cache",
            "--reasoning-effort",
            c["reasoning_effort"],
            "--max-tokens",
            str(c["max_tokens"]),
            "--max-retries",
            str(c["max_retries"]),
            "--concurrency",
            str(min(4, c["concurrency"])) if canary else str(c["concurrency"]),
            "--provider-max-prompt-price",
            str(c["max_prompt_price"]),
            "--provider-max-completion-price",
            str(c["max_completion_price"]),
        ]
        if args.limit:
            argv += ["--limit", str(args.limit)]
        if annotate(argv):
            raise RuntimeError(
                "incomplete annotation; inspect failures before resuming"
            )
        validate(list(map(json.loads, destination.open())), prompts)
        return
    scores = list(map(json.loads, (root / "scores.jsonl").open()))
    validate(scores, prompts)
    if len(scores) != 640 or {r["id"] for r in scores} != set(prompts):
        raise ValueError("incomplete matched scoring")
    indexed = {r["id"]: r for r in scores}
    summary = {}
    for teacher in ("qwen", "kimi", "glm"):

        def metrics(selected, teacher=teacher):
            y = [r["label"] for r in selected]
            p = [
                indexed[r["id"]]["score"] if teacher == "glm" else r[f"{teacher}_score"]
                for r in selected
            ]
            s = binary_calibration(y, p)
            s["auroc"] = float(roc_auc_score(y, p))
            s["ece_bin_sensitivity"] = {
                str(n): binary_calibration(y, p, n)["ece"] for n in (5, 10, 20)
            }
            s["unique_scores"] = len(set(p))
            s["false_positives"] = sum(
                v >= 0.5 and label == 0 for v, label in zip(p, y, strict=True)
            )
            s["false_negatives"] = sum(
                v < 0.5 and label == 1 for v, label in zip(p, y, strict=True)
            )
            return s

        summary[teacher] = {
            "pooled": metrics(pairs),
            "sources": {
                s: metrics([r for r in pairs if r["source"] == s])
                for s in sorted({r["source"] for r in pairs})
            },
        }
    summary["usage"] = {
        k: sum(float(r["usage"].get(k) or 0) for r in scores)
        for k in ("prompt_tokens", "completion_tokens", "cost")
    }
    summary["files"] = {str(p): sha(p) for p in [*paths, root / "scores.jsonl"]}
    atomic_write_json(root / "comparison.json", summary)
    print(
        json.dumps(
            {
                t: {k: v for k, v in summary[t]["pooled"].items() if k != "bins"}
                for t in ("qwen", "kimi", "glm")
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
