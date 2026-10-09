"""Freeze the selected activation exclusions and materialize matched campaigns."""

from __future__ import annotations

import json
import math
from collections import Counter
from pathlib import Path

import yaml

from gleipnir.data.activation_filter import activation_exclusions, matched_exclusions
from gleipnir.data.injection_audit import audit_rows
from gleipnir.data.monitoring import file_hash, read_rows, validate_targets, write_json

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
ACTIVATIONS = (
    ROOT / "results/b200_training_clean_direction/clean_direction01/paired_models.jsonl"
)
FRACTION = 0.20
CONTROL_SEED = 20261010
CASES = {
    "ranked": "activation-filter-ranked20-sdpa01",
    "random": "activation-filter-matched20-sdpa01",
}


def subset(source: Path, target: Path, excluded: set[str]) -> None:
    """Retain exact JSONL bytes and order; filter only original identities."""
    target.parent.mkdir(parents=True, exist_ok=True)
    with source.open() as inp, target.open("w") as out:
        for line in inp:
            if json.loads(line)["index"] not in excluded:
                out.write(line)


def prepare() -> None:
    base = yaml.safe_load(
        (ROOT / "experiments/b200_augmented_sdpa_replay/retry_config.yaml").read_text()
    )
    for spec in base["inputs"].values():
        path = ROOT / spec["path"]
        if path.is_file() and file_hash(path) != spec["sha256"]:
            raise ValueError(f"baseline input drift: {path}")
    expected_activation_sha = json.loads(
        (ROOT / "experiments/training_firewall_census/config.json").read_text()
    )["activation_input_sha256"]
    if file_hash(ACTIVATIONS) != expected_activation_sha:
        raise ValueError("activation census identity drift")
    original_path = ROOT / base["inputs"]["clean_training"]["path"]
    training_path = ROOT / base["inputs"]["training"]["path"]
    teacher_path = ROOT / base["inputs"]["teacher"]["path"]
    ledger_path = ROOT / base["inputs"]["ledger"]["path"]
    original, training, teacher, ledger = map(
        read_rows, [original_path, training_path, teacher_path, ledger_path]
    )
    paired = read_rows(ACTIVATIONS)
    if len(original) != 8688 or {r["index"] for r in original} != {
        r["index"] for r in paired
    }:
        raise ValueError("original activation population mismatch")
    original_by_id = {r["index"]: r for r in original}
    for row in paired:
        source = original_by_id[row["index"]]
        if row["view"] != "original" or any(
            row[k] != source[k]
            for k in ["label", "trajectory_sha256", "student_prompt_sha256"]
        ):
            raise ValueError("activation/original evidence drift")
    selected = activation_exclusions(paired, FRACTION)
    if len(selected) != 869:
        raise ValueError("selected count differs after content closure")
    injected = {r["index"] for r in ledger}
    random, matching = matched_exclusions(paired, selected, injected, CONTROL_SEED)
    bank = json.loads((ROOT / base["inputs"]["templates"]["path"]).read_text())
    validate_targets(training, teacher)
    audit_rows(original, training, ledger, bank)

    # Use the training engine's actual HF tokenizer and direct prompt boundary.
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        str(ROOT / "data/judge_injection_continuation/tokenizer")
    )
    lengths = []
    for start in range(0, len(training), 64):
        prompts = [
            tokenizer.apply_chat_template(
                [{"role": "user", "content": r["student_prompt"]}],
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
            + "Prediction:"
            for r in training[start : start + 64]
        ]
        lengths.extend(
            len(ids)
            for ids in tokenizer(prompts, add_special_tokens=False)["input_ids"]
        )
        if start % 1024 == 0:
            print(f"token_audit {min(start + 64, len(training))}/8688", flush=True)
    if sum(lengths) != base["expected_training_tokens"] or max(lengths) > 29696:
        raise ValueError(
            f"baseline tokenizer/count drift: {sum(lengths)}, {max(lengths)}"
        )
    length_by_id = {r["index"]: n for r, n in zip(training, lengths, strict=True)}
    report = {
        "fraction": FRACTION,
        "eligible_rows": 4344,
        "criterion": (
            "Descending delta_z20 within label=0; ties by original index; "
            "exact-content closure"
        ),
        "activation_input_sha256": file_hash(ACTIVATIONS),
        "axis_sha256": (
            "8c3c6e50ee1d3880d854754f9899f5684ae2e2c686479c3ceaaf3677d9eecfa7"
        ),
        "cutoff": min(r["delta_z20"] for r in paired if r["index"] in selected),
        "control_seed": CONTROL_SEED,
        "matching": matching,
        "cases": {},
        "teacher_and_firewall_scores_used_for_selection": False,
    }
    baseline_apps = ROOT / "results/b200-augmented-sdpa-apps01/summary.json"
    baseline_id = ROOT / "results/b200-augmented-sdpa-replay02/evaluation/id.jsonl"
    for case, excluded in [("ranked", selected), ("random", random)]:
        campaign = CASES[case]
        data = ROOT / "data" / campaign
        if data.exists():
            raise ValueError(f"materialized data already exists: {data}")
        for name, source in [
            ("training", training_path),
            ("clean_training", original_path),
            ("teacher", teacher_path),
            ("ledger", ledger_path),
        ]:
            subset(source, data / (name + ".jsonl"), excluded)
        retained = read_rows(data / "training.jsonl")
        clean = read_rows(data / "clean_training.jsonl")
        targets = read_rows(data / "teacher.jsonl")
        kept_ledger = read_rows(data / "ledger.jsonl")
        validate_targets(retained, targets)
        augmentation = audit_rows(clean, retained, kept_ledger, bank)
        remaining_lengths = [length_by_id[r["index"]] for r in retained]
        token_audit = {
            "configured_cap": 29696,
            "maximum": max(remaining_lengths),
            "rows": len(retained),
            "total": sum(remaining_lengths),
            "truncated": 0,
        }
        write_json(data / "token_audit.json", token_audit)
        write_json(
            data / "exclusions.json",
            {
                "ids": sorted(excluded),
                "criterion": report["criterion"] if case == "ranked" else matching,
                "source_label_counts": dict(
                    Counter(
                        f"{r['source']}/{r['label']}"
                        for r in paired
                        if r["index"] in excluded
                    )
                ),
            },
        )
        config = dict(base)
        # Keep only operational inputs and the primary matched baseline receipts.
        names = {
            "training",
            "clean_training",
            "teacher",
            "ledger",
            "templates",
            "token_audit",
            "id",
            "id_manifest",
            "id_workload",
            "benchmark",
            "honest_controls",
            "benchmark_workload",
            "honest_controls_workload",
            "startup_reference",
            "serving_selection",
            "initial_adapter_config",
            "initial_weights",
        }
        config["inputs"] = {k: v for k, v in base["inputs"].items() if k in names}
        for name in ["training", "clean_training", "teacher", "ledger", "token_audit"]:
            path = data / (
                name + ".jsonl" if name != "token_audit" else "token_audit.json"
            )
            config["inputs"][name] = {
                "path": str(path.relative_to(ROOT)),
                "sha256": file_hash(path),
            }
        for name, path in [
            ("baseline_id", baseline_id),
            ("baseline_apps", baseline_apps),
            ("activation_census", ACTIVATIONS),
            ("exclusions", data / "exclusions.json"),
            (
                "direction_partition",
                ROOT / "results/b200_injection_direction"
                / "sdpa_injection_direction01/partition.json",
            ),
        ]:
            config["inputs"][name] = {
                "path": str(path.relative_to(ROOT)),
                "sha256": file_hash(path),
            }
        config.update(
            campaign_id=campaign,
            training_rows=len(retained),
            injected_rows=len(kept_ledger),
            expected_steps=math.ceil(len(retained) / 32),
            expected_training_tokens=sum(remaining_lengths),
            merged_model=f"/tmp/gleipnir-merged/{campaign}",
            evaluations=["id", "apps"],
            hypothesis=(
                "Activation-ranked removal reduces injected-honest false alarms "
                "relative to matched random removal; failure is diagnostic."
            ),
            intervention=(
                f"Remove {len(excluded)} harmless originals using {case} selection; "
                "retain all remaining targets and views."
            ),
            selection=(
                "Fixed final one-epoch checkpoint, fraction 0.20 frozen before "
                "ID/APPS; no tuning or promotion."
            ),
            stop_condition=(
                "Stop on contract/runtime/source drift, nonfinite/missing gradients, "
                "OOM, new-adapter parity/native failure, incomplete coverage "
                "or completion."
            ),
            extra_sources=[
                "experiments/activation_filter_training/README.md",
                "experiments/activation_filter_training/prepare.py",
                "experiments/activation_filter_training/run.py",
                "experiments/activation_filter_training/analyze.py",
                "experiments/activation_filter_training/test_contract.py",
            ],
        )
        config["baselines"] = {
            "unfiltered_sdpa_id": {
                "kind": "id",
                "input": "baseline_id",
                "qualification": (
                    "Existing same-host seed-0 BF16/SDPA replay, same initializer "
                    "and BF16 serving; data/exposure/schedule and unresolved "
                    "execution variance differ."
                ),
            },
            "unfiltered_sdpa_apps": {
                "kind": "apps",
                "input": "baseline_apps",
                "keys": ["evaluations", "apps"],
                "qualification": (
                    "Existing same SDPA checkpoint on current BF16 backend and "
                    "frozen APPS populations; filtered training changes "
                    "exposure/schedule."
                ),
            },
        }
        (HERE / f"{case}.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
        report["cases"][case] = {
            "campaign_id": campaign,
            "excluded": len(excluded),
            "tokens_removed": sum(length_by_id[i] for i in excluded),
            "token_audit": token_audit,
            "augmentation": augmentation,
        }
    write_json(ROOT / "results/activation_filter_training/selection.json", report)
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    prepare()
