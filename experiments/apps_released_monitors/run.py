"""APPS monitor-directed injection evaluation of the released monitors.

``manifest`` (local) unwraps each frozen APPS prompt into its trajectory and
re-renders it with the release's own prompt: the student prompt for the 4B,
which reproduces the historical APPS prompts byte for byte, and the teacher
prompt for the 27B. ``score`` (Pod) scores one model against the released
serving image in fixed 128-row partitions. ``report`` (local) applies the
frozen APPS analysis to both models.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
from pathlib import Path
from typing import Any

from gleipnir.data.monitoring import file_hash, read_rows, write_json, write_rows
from gleipnir.evaluation.http_score import score_batch

EXPERIMENT = Path(__file__).resolve().parent
ROOT = EXPERIMENT.parents[1]
CONFIG = json.loads((EXPERIMENT / "config.json").read_text())
DATA = ROOT / "data/apps_released_monitors"
RESULTS = ROOT / "results/apps_released_monitors"
MANIFEST = EXPERIMENT / "manifest.json"
OPEN, CLOSE = "<agent_trajectory>", "</agent_trajectory>"


def sha256(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def trajectory(prompt: str, template: Any) -> str:
    """Invert ``template.render``; the result must render back to ``prompt``."""
    tail = f"{template.trajectory_close}\n"
    if not (prompt.startswith(template.cache_prefix) and prompt.endswith(tail)):
        raise ValueError("prompt is not wrapped in the expected template")
    body = prompt[len(template.cache_prefix) : -len(tail)]
    if template.render(body) != prompt:
        raise ValueError("trajectory does not round-trip through the template")
    return body


def chat(tokenizer: Any, template: Any, body: str) -> str:
    return (
        tokenizer.apply_chat_template(
            [{"role": "user", "content": template.render(body)}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        + template.decision_prefix
    )


def check_framing(model: str, prompt: str) -> None:
    """Prefix and suffix must equal those of the release's validated ID prompts."""
    from experiments.release_serving.targets import TARGETS
    from experiments.release_serving.validate import rows as workload_rows

    reference = workload_rows(TARGETS[model].validation["id"][0])[0]["prompt"]
    for text in (prompt, reference):
        if text.count(OPEN) != 1 or text.count(CLOSE) != 1:
            raise ValueError("prompt must contain one trajectory envelope")
    if prompt.split(OPEN)[0] != reference.split(OPEN)[0]:
        raise ValueError(f"{model} instruction prefix differs from validated prompts")
    if prompt.split(CLOSE)[1] != reference.split(CLOSE)[1]:
        raise ValueError(f"{model} decision suffix differs from validated prompts")


def build() -> tuple[dict, dict[str, dict[str, list[dict]]]]:
    from transformers import AutoTokenizer

    from experiments.tool_trajectory_monitoring.prompting import load_prompt_set

    prompt_set = load_prompt_set()
    student = prompt_set.for_role("student")
    sources = {}
    for split, spec in CONFIG["populations"].items():
        if (file_hash(ROOT / spec["input"]), file_hash(ROOT / spec["reference"])) != (
            spec["sha256"],
            spec["reference_sha256"],
        ):
            raise ValueError(f"frozen APPS inputs changed: {split}")
        rows = read_rows(ROOT / spec["input"])
        if len(rows) != spec["rows"] or len({r["id"] for r in rows}) != len(rows):
            raise ValueError("APPS coverage or uniqueness drift")
        sources[split] = (rows, read_rows(ROOT / spec["reference"]))
    manifest: dict[str, Any] = {
        "config_sha256": file_hash(EXPERIMENT / "config.json"),
        "models": {},
    }
    prompts: dict[str, dict[str, list[dict]]] = {}
    for model, spec in CONFIG["models"].items():
        tokenizer = AutoTokenizer.from_pretrained(spec["tokenizer"])
        template = prompt_set.for_role(spec["prompt_role"])
        prompts[model] = {}
        manifest["models"][model] = {
            **spec,
            "template_sha256": template.template_sha256,
            "splits": {},
        }
        for split, (rows, reference) in sources.items():
            out = []
            for row, old in zip(rows, reference, strict=True):
                text = chat(tokenizer, template, trajectory(row["prompt"], student))
                digest = sha256(text)
                if row["id"] != old["id"]:
                    raise ValueError("APPS reference order drift")
                if model == "4b" and digest != old["prompt_sha256"]:
                    raise ValueError(f"4B prompt differs from historical: {row['id']}")
                out.append(
                    {
                        "id": row["id"],
                        "prompt": text,
                        "prompt_sha256": digest,
                        "prompt_tokens": len(
                            tokenizer.encode(text, add_special_tokens=False)
                        ),
                        "metadata": row["metadata"],
                    }
                )
            check_framing(model, out[0]["prompt"])
            prompts[model][split] = out
            manifest["models"][model]["splits"][split] = {
                "rows": len(out),
                "tokens": sum(r["prompt_tokens"] for r in out),
                "maximum_tokens": max(r["prompt_tokens"] for r in out),
                "prompt_hashes_sha256": sha256(
                    "\n".join(r["prompt_sha256"] for r in out)
                ),
            }
    return manifest, prompts


def freeze() -> None:
    manifest, prompts = build()
    if MANIFEST.exists():
        if json.loads(MANIFEST.read_text()) != json.loads(json.dumps(manifest)):
            raise ValueError("rendered prompts differ from the frozen manifest")
        print("manifest unchanged")
    else:
        write_json(MANIFEST, manifest)
        print("froze", {m: s["splits"] for m, s in manifest["models"].items()})
    DATA.mkdir(parents=True, exist_ok=True)
    for model, splits in prompts.items():
        for split, rows in splits.items():
            write_rows(DATA / f"{model}_{split}.jsonl", rows)


def frozen(model: str, split: str) -> list[dict]:
    rows = read_rows(DATA / f"{model}_{split}.jsonl")
    expected = json.loads(MANIFEST.read_text())["models"][model]["splits"][split]
    if any(sha256(r["prompt"]) != r["prompt_sha256"] for r in rows) or (
        len(rows),
        sha256("\n".join(r["prompt_sha256"] for r in rows)),
    ) != (expected["rows"], expected["prompt_hashes_sha256"]):
        raise ValueError(f"{model} {split} prompts differ from the frozen manifest")
    return rows


def score(model: str, name: str) -> None:
    """One pass in fixed partitions; completed partitions are kept on resume."""
    out = RESULTS / name / model
    if (out / "complete.json").exists():
        raise ValueError(f"{model} already scored in {name}; no reruns")
    settings = CONFIG["scoring"]
    for split in CONFIG["populations"]:
        rows = frozen(model, split)
        batches = out / "batches" / split
        batches.mkdir(parents=True, exist_ok=True)
        for start in range(0, len(rows), settings["batch_rows"]):
            path = batches / f"{start:05d}.json"
            if path.exists():
                continue
            group = rows[start : start + settings["batch_rows"]]
            values, seconds = asyncio.run(score_batch(group, settings))
            write_json(
                path,
                {
                    "seconds": seconds,
                    "rows": [
                        {**v, **r["metadata"]}
                        for r, v in zip(group, values, strict=True)
                    ],
                },
            )
            print(
                f"scored {model} {split} {start + len(group)}/{len(rows)}", flush=True
            )
        scored = [
            row
            for path in sorted(batches.glob("*.json"))
            for row in json.loads(path.read_text())["rows"]
        ]
        if [r["id"] for r in scored] != [r["id"] for r in rows]:
            raise ValueError("scored rows do not match the frozen order")
        write_rows(out / f"{split}.jsonl", scored)
    write_json(
        out / "complete.json",
        {
            "model": model,
            "image": CONFIG["image"],
            "manifest_sha256": file_hash(MANIFEST),
            "files_sha256": {
                s: file_hash(out / f"{s}.jsonl") for s in CONFIG["populations"]
            },
        },
    )


def headline(report: dict) -> dict:
    names = ("transfer_gpt4", "transfer_gpt41", "transfer_gpt4o")
    conditions = report["conditions"]
    return {
        "threshold": report["threshold"],
        "clean_fpr": report["clean_fpr"],
        "vanilla_auroc": report["vanilla"]["auroc"],
        "vanilla_pauroc_at_20": report["vanilla"]["pauroc_at_20"],
        "vanilla_backdoor_tpr": report["vanilla_backdoor_tpr"],
        "mean_injected_honest_fpr": report["mean_injected_honest_fpr"],
        "templates": {
            n: {
                "injected_honest_fpr": conditions[f"inj_{n}"]["honest_fpr"],
                "injected_backdoor_tpr": conditions[f"inj_{n}"]["backdoor_tpr"],
                "matched_auroc": conditions[f"inj_{n}"]["matched"]["auroc"],
                "neutral_honest_fpr": conditions[f"neutral_{n}"]["honest_fpr"],
                "injected_honest_alarm_at_half": conditions[f"inj_{n}"]["honest"][
                    "alarm_rate_at_half"
                ],
                "injection_minus_neutral": conditions[f"inj_{n}"][
                    "paired_injection_minus_neutral"
                ],
            }
            for n in names
        },
        "clean_honest_alarm_at_half": report["clean_honest"]["alarm_rate_at_half"],
    }


def report(name: str) -> None:
    from gleipnir.evaluation.apps import summarize_apps

    result = {}
    for model in CONFIG["models"]:
        out = RESULTS / name / model
        complete = json.loads((out / "complete.json").read_text())
        for split, digest in complete["files_sha256"].items():
            if file_hash(out / f"{split}.jsonl") != digest:
                raise ValueError("completed scores changed")
        full = summarize_apps(
            read_rows(out / "benchmark.jsonl"), read_rows(out / "honest_controls.jsonl")
        )
        write_json(out / "summary.json", full)
        result[model] = headline(full)
    for label, path in CONFIG["historical"].items():
        result[label] = headline(json.loads((ROOT / path).read_text()))
    write_json(RESULTS / name / "report.json", result)
    print(json.dumps(result, indent=1))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("manifest")
    command = commands.add_parser("score")
    command.add_argument("model", choices=sorted(CONFIG["models"]))
    command.add_argument("--name", required=True)
    command = commands.add_parser("report")
    command.add_argument("--name", required=True)
    args = parser.parse_args()
    if args.command == "manifest":
        freeze()
    elif args.command == "score":
        score(args.model, args.name)
    else:
        report(args.name)


if __name__ == "__main__":
    main()
