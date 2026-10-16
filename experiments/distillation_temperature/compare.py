"""Apply the frozen ID advancement rule to the temperature arms."""

from __future__ import annotations

from experiments.distillation_temperature.prepare import ARMS, ROOT
from experiments.teacher_seed_variation.compare import load, score_difference, summary
from gleipnir.data.monitoring import file_hash, write_json

CONTROL = "results/teacher-seed-sdpa-s1-01/evaluation/id.jsonl"
CONTEXT = {
    "seed0": "results/teacher-seed-sdpa-s0-rescore01/evaluation/id.jsonl",
    "seed2": "results/teacher-seed-sdpa-s2-01/evaluation/id.jsonl",
}
RULE = {"min_macro_gain": 0.010, "max_source_loss": 0.010, "max_brier_rise": 0.005}


def decide(arm: dict, control: dict) -> dict:
    gain = arm["macro"]["pauroc_at_20"] - control["macro"]["pauroc_at_20"]
    sources = {
        s: arm["by_source"][s] - control["by_source"][s] for s in control["by_source"]
    }
    brier = arm["macro"]["brier"] - control["macro"]["brier"]
    return {
        "macro_pauroc_at_20_delta": gain,
        "macro_auroc_delta": arm["macro"]["auroc"] - control["macro"]["auroc"],
        "macro_brier_delta": brier,
        "by_source_pauroc_delta": sources,
        "advance": gain >= RULE["min_macro_gain"]
        and min(sources.values()) >= -RULE["max_source_loss"]
        and brier <= RULE["max_brier_rise"],
    }


def compare() -> dict:
    control_rows = load(CONTROL, "id")
    control = summary(control_rows)
    paths = {name: f"results/{name}/evaluation/id.jsonl" for name in ARMS}
    report = {
        "rule": RULE,
        "control": control,
        "context": {k: summary(load(p, "id")) for k, p in CONTEXT.items()},
        "arms": {},
    }
    for name, path in paths.items():
        if not (ROOT / path).is_file():
            continue
        rows = load(path, "id")
        arm = summary(rows)
        report["arms"][name] = {
            "knob": ARMS[name],
            "metrics": arm,
            "decision": decide(arm, control),
            "scores_vs_control": score_difference(rows, control_rows),
            "sha256": file_hash(ROOT / path),
        }
        d = report["arms"][name]["decision"]
        sources = ", ".join(
            f"{k} {v:+.4f}" for k, v in d["by_source_pauroc_delta"].items()
        )
        print(
            f"{name:22} pAUROC@20 {arm['macro']['pauroc_at_20']:.4f} "
            f"d {d['macro_pauroc_at_20_delta']:+.4f} "
            f"dBrier {d['macro_brier_delta']:+.4f} sources {sources} "
            f"advance {d['advance']}"
        )
    write_json(ROOT / "results/distillation_temperature/comparison.json", report)
    return report


if __name__ == "__main__":
    compare()
