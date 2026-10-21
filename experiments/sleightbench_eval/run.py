"""Single-pass SleightBench evaluation of the released Gleipnir monitors.

Subcommands, in order:

- ``decrypt --checkout PATH``: decrypt a pinned upstream checkout into the
  ignored data root (no upstream code is executed).
- ``manifest``: render every prompt for both models and views, check the
  framing against each model's validated ID workload, and freeze hashes and
  token counts in ``manifest.json`` (no benchmark text).
- ``score MODEL --name NAME --validation GATE``: score once against a live
  release server whose repeat-relative ID gate (``release_serving/gate.py``)
  passed; resumable, never repeated.
- ``report --name NAME``: frozen metrics for every scored model.
"""

from __future__ import annotations

import argparse
import asyncio
import dataclasses
import json
import subprocess
from collections import defaultdict
from pathlib import Path
from typing import Any

from experiments.release_serving.targets import TARGETS
from experiments.release_serving.validate import rows as workload_rows
from experiments.sleightbench_eval import benchmark
from experiments.tool_trajectory_monitoring.prompting import load_prompt_set
from gleipnir.data.monitoring import read_rows, write_json
from gleipnir.evaluation.http_score import score_batch

EXPERIMENT = Path(__file__).parent
ROOT = EXPERIMENT.parents[1]
CONFIG = json.loads((EXPERIMENT / "config.json").read_text())
RELEASE = "release"
DATA = ROOT / CONFIG["upstream"]["data_root"]
RESULTS = ROOT / "results/sleightbench_eval"
OPEN, CLOSE = "<agent_trajectory>\n", "</agent_trajectory>"


def decrypt(checkout: Path) -> None:
    head = subprocess.run(
        ["git", "-C", str(checkout), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if head != CONFIG["upstream"]["commit"]:
        raise ValueError(f"checkout is at {head}, not the pinned commit")
    hashes = benchmark.decrypt_checkout(checkout, DATA)
    write_json(DATA / "decrypted_sha256.json", hashes)
    print(f"decrypted {len(hashes)} files into {DATA}")


def render(tokenizer: Any, template: Any, trajectory: str) -> str:
    return (
        tokenizer.apply_chat_template(
            [{"role": "user", "content": template.render(trajectory)}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        + template.decision_prefix
    )


def manifest_path(variant: str) -> Path:
    suffix = "" if variant == RELEASE else f"_{variant}"
    return EXPERIMENT / f"manifest{suffix}.json"


def prompts_path(model: str, variant: str) -> Path:
    suffix = "" if variant == RELEASE else f"_{variant}"
    return DATA / f"prompts_{model}{suffix}.json"


def check_framing(model: str, prompt: str, instruction_replaced: bool) -> None:
    """Our prefix and suffix must equal those of the model's validated ID prompts.

    A prompt variant replaces the instruction, so only the decision suffix and
    the chat-template opening are compared.
    """
    reference = workload_rows(TARGETS[model].validation["id"][0])[0]["prompt"]
    for text in (prompt, reference):
        if text.count(OPEN) != 1 or text.count(CLOSE) != 1:
            raise ValueError("prompt must contain one trajectory envelope")
    opening = reference.split("\n", 1)[0]
    if instruction_replaced:
        if not prompt.startswith(opening + "\n"):
            raise ValueError(f"{model} chat opening differs from validated prompts")
    elif prompt.split(OPEN)[0] != reference.split(OPEN)[0]:
        raise ValueError(f"{model} instruction prefix differs from validated prompts")
    if prompt.split(CLOSE)[1] != reference.split(CLOSE)[1]:
        raise ValueError(f"{model} decision suffix differs from validated prompts")


def build_prompts(variant: str) -> tuple[dict, dict[str, list[dict]]]:
    from transformers import AutoTokenizer

    hashes = json.loads((DATA / "decrypted_sha256.json").read_text())
    excluded = set(CONFIG["excluded_attacks"])
    items = benchmark.transcripts(DATA, excluded)
    prompt_set = load_prompt_set()
    manifest: dict[str, Any] = {
        "upstream": CONFIG["upstream"],
        "decrypted_sha256": hashes,
        "transcripts": [
            {
                "id": t.id,
                "attack": t.attack,
                "category": t.category,
                "label": t.label,
                "tags": json.loads((t.path.parent / "metadata.json").read_text()).get(
                    "tags", []
                ),
            }
            for t in items
        ],
        "models": {},
    }
    if variant != RELEASE:
        manifest["variant"] = variant
    instruction = None
    if variant != RELEASE:
        spec = CONFIG["variants"][variant]
        instruction = (EXPERIMENT / spec["instruction"]).read_text()
    prompts: dict[str, list[dict]] = {}
    for model, spec in CONFIG["models"].items():
        tokenizer = AutoTokenizer.from_pretrained(spec["tokenizer"])
        template = prompt_set.for_role(spec["prompt_role"])
        if instruction is not None:
            template = dataclasses.replace(template, instruction=instruction)
        rows = []
        for view in CONFIG["views"]:
            for t in items:
                messages = benchmark.VIEWS[view](benchmark.read_messages(t.path))
                prompt = render(tokenizer, template, benchmark.body(messages))
                check_framing(model, prompt, instruction is not None)
                tokens = len(tokenizer.encode(prompt, add_special_tokens=False))
                if tokens > CONFIG["context_limit"]:
                    raise ValueError(f"{t.id} exceeds the context limit")
                rows.append(
                    {
                        "id": f"{view}/{t.id}",
                        "view": view,
                        "transcript": t.id,
                        "prompt": prompt,
                        "prompt_sha256": benchmark.sha256(prompt.encode()),
                        "prompt_tokens": tokens,
                    }
                )
        prompts[model] = rows
        manifest["models"][model] = {
            **spec,
            "template_sha256": template.template_sha256,
            "prompts": [{k: v for k, v in r.items() if k != "prompt"} for r in rows],
        }
    return manifest, prompts


def freeze(variant: str) -> None:
    manifest, prompts = build_prompts(variant)
    path = manifest_path(variant)
    if path.exists():
        if json.loads(path.read_text()) != json.loads(json.dumps(manifest)):
            raise ValueError("rendered prompts differ from the frozen manifest")
        print("manifest unchanged")
    else:
        write_json(path, manifest)
        print(f"froze {sum(len(p) for p in prompts.values())} prompts")
    for model, rows in prompts.items():
        write_json(prompts_path(model, variant), rows)


def frozen_prompts(model: str, variant: str) -> list[dict]:
    manifest = json.loads(manifest_path(variant).read_text())
    expected = manifest["models"][model]["prompts"]
    rows = json.loads(prompts_path(model, variant).read_text())
    for row, frozen in zip(rows, expected, strict=True):
        if (
            benchmark.sha256(row["prompt"].encode()) != frozen["prompt_sha256"]
            or {k: v for k, v in row.items() if k != "prompt"} != frozen
        ):
            raise ValueError(f"{row['id']} differs from the frozen manifest")
    return rows


def score(model: str, name: str, validation: Path, port: int, variant: str) -> None:
    report = json.loads(validation.read_text())
    if (
        report.get("model") != model
        or report.get("split") != "id"
        or not report.get("passed")
    ):
        raise ValueError("the release server's repeat-relative ID gate did not pass")
    out = RESULTS / name / f"{model}.jsonl"
    complete = out.with_suffix(".complete.json")
    if complete.exists():
        raise ValueError(f"{model} was already scored in {name}; no reruns")
    out.parent.mkdir(parents=True, exist_ok=True)
    rows = frozen_prompts(model, variant)
    done = {r["id"] for r in read_rows(out)} if out.exists() else set()
    settings = {"port": port, **CONFIG["scoring"]}
    with out.open("a") as handle:
        for row in rows:
            if row["id"] in done:
                continue
            values, _ = asyncio.run(score_batch([row], settings))
            handle.write(json.dumps(values[0]) + "\n")
            handle.flush()
            print(f"scored {row['id']} tokens={row['prompt_tokens']}", flush=True)
    scored = read_rows(out)
    if [r["id"] for r in scored] != [r["id"] for r in rows]:
        raise ValueError("scored rows do not match the frozen order")
    write_json(
        complete,
        {
            "model": model,
            "variant": variant,
            "rows": len(scored),
            "validation": report,
        },
    )


def view_report(
    view: str, scores: dict[str, float], tokens: dict[str, int], transcripts: list
) -> dict:
    metrics = CONFIG["metrics"]
    pairs = CONFIG["multi_session_pairs"]
    merged = {b: a for a, b in pairs}
    by_attack = defaultdict(dict)
    for t in transcripts:
        by_attack[t["attack"]][t["label"]] = scores[f"{view}/{t['id']}"]
    unit = {a: merged.get(a, a) for a in by_attack}
    attack = benchmark.attack_units({a: s[1] for a, s in by_attack.items()}, pairs)
    paired = benchmark.attack_units({a: s[0] for a, s in by_attack.items()}, pairs)
    benign_flat = {a: s[0] for a, s in by_attack.items()}
    benign_by_unit = defaultdict(list)
    for a, s in by_attack.items():
        benign_by_unit[unit[a]].append(s[0])
    args = (metrics["target_fpr"], metrics["pauroc_max_fpr"])
    long_units = {
        unit[t["attack"]]
        for t in transcripts
        if tokens[f"{view}/{t['id']}"] > metrics["long_context_split"]
    }
    short = sorted(set(attack) - long_units)
    category = {unit[t["attack"]]: t["category"] for t in transcripts}
    tags = {unit[t["attack"]]: t["tags"] for t in transcripts if t["label"] == 1}
    result = {
        "units": benchmark.summary(attack, benign_flat, paired, *args),
        "transcripts": benchmark.summary(
            {a: s[1] for a, s in by_attack.items()},
            benign_flat,
            benign_flat,
            *args,
        ),
        "intervals_95": benchmark.paired_bootstrap(
            attack,
            benign_by_unit,
            paired,
            target_fpr=metrics["target_fpr"],
            max_fpr=metrics["pauroc_max_fpr"],
            draws=metrics["bootstrap_draws"],
            seed=metrics["bootstrap_seed"],
        ),
        "short_context": benchmark.summary(
            {k: attack[k] for k in short},
            {a: s for a, s in benign_flat.items() if unit[a] in short},
            {k: paired[k] for k in short},
            *args,
        ),
        "long_context_units": {
            k: {"attack": attack[k], "paired_benign": paired[k]}
            for k in sorted(long_units)
        },
    }
    threshold = result["units"]["threshold"]
    groups: dict[str, dict[str, list[str]]] = {"category": {}, "tag": {}}
    for k in attack:
        groups["category"].setdefault(category[k], []).append(k)
        for tag in tags[k]:
            groups["tag"].setdefault(tag, []).append(k)
    for kind, members in groups.items():
        result[f"by_{kind}"] = {
            g: {
                "attacks": len(keys),
                "caught": sum(attack[k] > threshold for k in keys),
                "pair_wins": sum(attack[k] > paired[k] for k in keys),
            }
            for g, keys in sorted(members.items())
        }
    result["per_unit"] = {
        k: {"attack": attack[k], "paired_benign": paired[k], "category": category[k]}
        for k in sorted(attack)
    }
    return result


def report(name: str, variant: str) -> None:
    manifest = json.loads(manifest_path(variant).read_text())
    output: dict[str, Any] = {"config": CONFIG, "variant": variant, "models": {}}
    for model in CONFIG["models"]:
        path = RESULTS / name / f"{model}.jsonl"
        if not path.with_suffix(".complete.json").exists():
            continue
        scored = {r["id"]: float(r["score"]) for r in read_rows(path)}
        tokens = {
            r["id"]: r["prompt_tokens"] for r in manifest["models"][model]["prompts"]
        }
        output["models"][model] = {
            view: view_report(view, scored, tokens, manifest["transcripts"])
            for view in CONFIG["views"]
        }
        for view, result in output["models"][model].items():
            units, ci = result["units"], result["intervals_95"]
            print(
                f"{model} {view}: catch@1%FPR {units['catch_rate']:.3f} "
                f"[{ci['catch_rate'][0]:.3f}, {ci['catch_rate'][1]:.3f}] "
                f"AUROC {units['auroc']:.3f} pAUROC@20 {units['pauroc_at_20']:.3f} "
                f"pair wins {units['pair_win_rate']:.3f}"
            )
    write_json(RESULTS / name / "report.json", output)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    parser.add_argument(
        "--variant",
        default=RELEASE,
        choices=[RELEASE, *CONFIG["variants"]],
        help="prompt variant; 'release' is each model's validated instruction",
    )
    command = commands.add_parser("decrypt")
    command.add_argument("--checkout", type=Path, required=True)
    commands.add_parser("manifest")
    command = commands.add_parser("score")
    command.add_argument("model", choices=sorted(CONFIG["models"]))
    command.add_argument("--name", required=True)
    command.add_argument("--validation", type=Path, required=True)
    command.add_argument("--port", type=int, default=8000)
    command = commands.add_parser("report")
    command.add_argument("--name", required=True)
    args = parser.parse_args()
    if args.command == "decrypt":
        decrypt(args.checkout)
    elif args.command == "manifest":
        freeze(args.variant)
    elif args.command == "score":
        score(args.model, args.name, args.validation, args.port, args.variant)
    else:
        report(args.name, args.variant)


if __name__ == "__main__":
    main()
