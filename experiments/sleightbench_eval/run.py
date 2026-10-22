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


def render(
    tokenizer: Any,
    template: Any,
    trajectory: str,
    *,
    enable_thinking: bool = False,
    decision_prefix: bool = True,
    reasoning_effort: str | None = None,
) -> str:
    """Render one prompt; reasoning variants end at the assistant opening.

    ``reasoning_effort`` is passed only when set: Qwen3.8's template adds a
    system message for its default ``xhigh`` effort but none for ``medium``,
    and Qwen3.5's template ignores the argument.
    """
    extra = {"reasoning_effort": reasoning_effort} if reasoning_effort else {}
    rendered = tokenizer.apply_chat_template(
        [{"role": "user", "content": template.render(trajectory)}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=enable_thinking,
        **extra,
    )
    return rendered + (template.decision_prefix if decision_prefix else "")


def manifest_path(variant: str) -> Path:
    suffix = "" if variant == RELEASE else f"_{variant}"
    return EXPERIMENT / f"manifest{suffix}.json"


def prompts_path(model: str, variant: str) -> Path:
    suffix = "" if variant == RELEASE else f"_{variant}"
    return DATA / f"prompts_{model}{suffix}.json"


def render_paper(
    tokenizer: Any,
    instruction: str,
    trajectory: str,
    *,
    enable_thinking: bool,
    decision_prefix: bool = False,
    reasoning_effort: str | None = None,
) -> str:
    """Upstream ``render_prompt``: system prompt, transcript in boundary tags."""
    if decision_prefix:
        raise ValueError("the paper protocol generates its own score")
    extra = {"reasoning_effort": reasoning_effort} if reasoning_effort else {}
    rendered = tokenizer.apply_chat_template(
        [
            {"role": "system", "content": instruction},
            {"role": "user", "content": f"<transcript>\n{trajectory}\n</transcript>"},
        ],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=enable_thinking,
        **extra,
    )
    opening = f"<|im_start|>system\n{instruction.strip()}"
    closing = "<|im_start|>assistant\n<think>\n" + (
        "" if enable_thinking else "\n</think>\n\n"
    )
    if not rendered.startswith(opening) or not rendered.endswith(closing):
        raise ValueError("paper prompt framing differs from the upstream protocol")
    return rendered


def check_framing(
    model: str, prompt: str, instruction_replaced: bool, decision_prefix: bool = True
) -> None:
    """Our prefix and suffix must equal those of the model's validated ID prompts.

    A prompt variant replaces the instruction, so only the decision suffix and
    the chat-template opening are compared. Reasoning variants end before the
    decision, so their suffix must be the validated suffix without the
    non-thinking block and ``Prediction:``.
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
    suffix = reference.split(CLOSE)[1]
    if not decision_prefix:
        suffix = suffix.split("<think>")[0]
        if not prompt.split(CLOSE)[1].startswith(suffix):
            raise ValueError(
                f"{model} assistant opening differs from validated prompts"
            )
    elif prompt.split(CLOSE)[1] != suffix:
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
    options = {"enable_thinking": False, "decision_prefix": True}
    if variant != RELEASE:
        spec = CONFIG["variants"][variant]
        instruction = (EXPERIMENT / spec["instruction"]).read_text()
        options = {k: spec[k] for k in options}
        if "reasoning_effort" in spec:
            options["reasoning_effort"] = spec["reasoning_effort"]
    paper = variant != RELEASE and CONFIG["variants"][variant].get("format") == "paper"
    views = (
        CONFIG["variants"][variant].get("views", CONFIG["views"])
        if (variant != RELEASE)
        else CONFIG["views"]
    )
    prompts: dict[str, list[dict]] = {}
    for model, spec in CONFIG["models"].items():
        tokenizer = AutoTokenizer.from_pretrained(spec["tokenizer"])
        template = prompt_set.for_role(spec["prompt_role"])
        if instruction is not None:
            template = dataclasses.replace(template, instruction=instruction)
        rows = []
        for view in views:
            for t in items:
                messages = benchmark.VIEWS[view](benchmark.read_messages(t.path))
                if paper:
                    prompt = render_paper(
                        tokenizer, instruction, benchmark.body(messages), **options
                    )
                else:
                    prompt = render(
                        tokenizer, template, benchmark.body(messages), **options
                    )
                    check_framing(
                        model,
                        prompt,
                        instruction is not None,
                        options["decision_prefix"],
                    )
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
    result["robustness"] = benchmark.threshold_robustness(
        attack, dict(benign_by_unit), metrics["target_fpr"]
    )
    result["per_unit"] = {
        k: {"attack": attack[k], "paired_benign": paired[k], "category": category[k]}
        for k in sorted(attack)
    }
    return result


def export_api(variant: str) -> None:
    """Write user-message prompts (no chat template) for an API teacher.

    The rows are what ``gleipnir-openrouter`` sends as the user turn: the
    instruction plus the trajectory envelope, as in teacher annotation. Their
    hashes are frozen in ``manifest_api_<variant>.json``.
    """
    spec = CONFIG["variants"][variant]
    if spec.get("format") == "paper" or spec["enable_thinking"]:
        raise ValueError("the API export covers the binary logit variants only")
    instruction = (EXPERIMENT / spec["instruction"]).read_text()
    template = dataclasses.replace(
        load_prompt_set().for_role("teacher"), instruction=instruction
    )
    excluded = set(CONFIG["excluded_attacks"])
    rows = []
    for view in CONFIG["views"]:
        for t in benchmark.transcripts(DATA, excluded):
            messages = benchmark.VIEWS[view](benchmark.read_messages(t.path))
            prompt = template.render(benchmark.body(messages))
            rows.append({"id": f"{view}/{t.id}", "prompt": prompt, "metadata": {}})
    frozen = {r["id"]: benchmark.sha256(r["prompt"].encode()) for r in rows}
    path = EXPERIMENT / f"manifest_api_{variant}.json"
    if path.exists() and json.loads(path.read_text()) != frozen:
        raise ValueError("API prompts differ from the frozen manifest")
    write_json(path, frozen)
    out = DATA / f"api_{variant}.jsonl"
    out.write_text("".join(json.dumps(r) + "\n" for r in rows))
    print(f"wrote {len(rows)} API prompts to {out}")


def report(name: str, variant: str) -> None:
    manifest = json.loads(manifest_path(variant).read_text())
    output: dict[str, Any] = {"config": CONFIG, "variant": variant, "models": {}}
    api = CONFIG.get("api_models", {})
    for model in [*CONFIG["models"], *api]:
        path = RESULTS / name / f"{model}.jsonl"
        if not path.with_suffix(".complete.json").exists():
            continue
        scored = {r["id"]: float(r["score"]) for r in read_rows(path)}
        source = api[model]["tokens_from"] if model in api else model
        tokens = {
            r["id"]: r["prompt_tokens"] for r in manifest["models"][source]["prompts"]
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


def report_trials(name: str, variant: str) -> None:
    """Paper-protocol metrics over sampled trials, per model and view."""
    manifest = json.loads(manifest_path(variant).read_text())
    metrics = CONFIG["metrics"]
    pairs = CONFIG["multi_session_pairs"]
    merged = {b: a for a, b in pairs}
    output: dict[str, Any] = {"config": CONFIG, "variant": variant, "models": {}}
    api = CONFIG.get("api_models", {})
    for model in [*CONFIG["models"], *api]:
        path = RESULTS / name / f"{model}.jsonl"
        if not path.with_suffix(".complete.json").exists():
            continue
        rows = {r["id"]: r["trials"] for r in read_rows(path)}
        output["models"][model] = {}
        source = api[model]["tokens_from"] if model in api else model
        for view in sorted({r["view"] for r in manifest["models"][source]["prompts"]}):
            attack, benign = {}, {}
            for t in manifest["transcripts"]:
                (attack if t["label"] else benign)[t["attack"]] = rows[
                    f"{view}/{t['id']}"
                ]
            paired = benchmark.merge_trials(benign, pairs)
            units = benchmark.merge_trials(attack, pairs)
            result = benchmark.trial_summary(
                units,
                benign,
                paired,
                metrics["target_fpr"],
                metrics["pauroc_max_fpr"],
            )
            result["merged_parts"] = sorted(merged)
            output["models"][model][view] = result
            print(
                f"{model} {view}: per-trial catch {result['per_trial_catch_rate']:.3f} "
                f"any-trial {result['any_trial_catch_rate']:.3f} "
                f"AUROC(mean) {result['mean_score_auroc']:.3f} "
                f"pair wins {result['pair_win_rate']:.3f} "
                f"unparsed {result['unparsed_trials']}"
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
    commands.add_parser("export-api")
    command = commands.add_parser("score")
    command.add_argument("model", choices=sorted(CONFIG["models"]))
    command.add_argument("--name", required=True)
    command.add_argument("--validation", type=Path, required=True)
    command.add_argument("--port", type=int, default=8000)
    command = commands.add_parser("report")
    command.add_argument("--name", required=True)
    command = commands.add_parser("report-trials")
    command.add_argument("--name", required=True)
    args = parser.parse_args()
    if args.command == "decrypt":
        decrypt(args.checkout)
    elif args.command == "manifest":
        freeze(args.variant)
    elif args.command == "export-api":
        export_api(args.variant)
    elif args.command == "score":
        score(args.model, args.name, args.validation, args.port, args.variant)
    elif args.command == "report-trials":
        report_trials(args.name, args.variant)
    else:
        report(args.name, args.variant)


if __name__ == "__main__":
    main()
