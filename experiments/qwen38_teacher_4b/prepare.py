"""Freeze the one-seed 4B arm distilled from Qwen3.8-27B instead of Kimi K3."""

from __future__ import annotations

import argparse
import copy
import json
import math
from pathlib import Path

import yaml

from gleipnir.data.monitoring import (
    file_hash,
    read_rows,
    validate_targets,
    write_json,
    write_rows,
)
from gleipnir.data.teacher_filter import remove_disagreement

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
CONTROL = "teacher-seed-sdpa-s1-01"
SELECTION = "experiments/b200_inference_benchmark/serving_default.json"
ARM = "qwen38-teacher-s1-01"
TEACHER_MODEL = "Qwen/Qwen3.8-27B"
# Unfiltered pool of the strict-filter sweep; the campaign re-filters it on
# the new teacher, as the recipe filters on Kimi.
POOL = {
    "training": "data/student_injection_awareness/regular/student_rows.jsonl",
    "teacher": "data/student_injection_awareness/soft_targets.jsonl",
    "token_audit": "results/student_injection_awareness/4b/regular/token_audit.json",
}
POOL_ROWS, POOL_STEPS, POOL_TOKENS = 8688, 272, 83816369
PREDICTIONS = "results/qwen27b_generations_ood/qwen38_27b_train/predictions.jsonl"
FIELDS = {
    "hypothesis": (
        "A same-family teacher (unadapted Qwen3.8-27B) transfers more to the "
        "Qwen3.5-4B student than Kimi K3: higher held-out ID macro pAUROC@20 than "
        "the seed-matched Kimi-distilled control."
    ),
    "intervention": (
        "Seed-1 BF16/SDPA campaign with Qwen3.8-27B binary 0/1 soft targets "
        "(full teacher prompt, non-thinking) replacing Kimi K3; strict "
        "teacher/label filtering is applied to the new teacher; LR, schedule, "
        "initialization and loss unchanged."
    ),
    "selection": (
        "Final one-epoch checkpoint; ID-only paired comparison with "
        "teacher-seed-sdpa-s1-01; advance to more seeds, never promote, on one seed."
    ),
    "stop_condition": (
        "Stop on input/runtime drift, nonfinite or missing gradients, failed "
        "master/merge/serving parity, incomplete scoring or completion."
    ),
}


def qwen_targets(kimi: list[dict], predictions: list[dict]) -> list[dict]:
    """Replace only the teacher distribution, keeping Kimi's row identity/order."""
    # Some rows share an identical prompt; join by row identity, then check it.
    by_id = {p["id"]: p for p in predictions}
    if len(by_id) != len(predictions) or len(predictions) != len(kimi):
        raise ValueError("Qwen prediction coverage or uniqueness drift")
    rows = []
    for target in kimi:
        p = by_id[str(target["index"])]
        margin = float(p["logit_margin_1_minus_0"])
        if (
            p["source_prompt_sha256"] != target["rendered_prompt_sha256"]
            or p["label"] != target["label"]
            or not target["dataset"].endswith("/" + p["source"])
        ):
            raise ValueError("Qwen prediction prompt, label or source drift")
        if not math.isfinite(margin):
            raise ValueError("nonfinite Qwen margin")
        positive = 1.0 / (1.0 + math.exp(-margin))
        if not math.isclose(positive, p["score"], abs_tol=1e-9):
            raise ValueError("Qwen score is not sigmoid(margin)")
        rows.append(
            {
                "dataset": target["dataset"],
                "index": target["index"],
                "label": target["label"],
                "provider": "local-vllm",
                "rendered_prompt_sha256": target["rendered_prompt_sha256"],
                "soft_target": positive,
                "target_logprobs": {
                    "negative": p["logprob_0"],
                    "positive": p["logprob_1"],
                },
                "target_probs": {"negative": 1.0 - positive, "positive": positive},
                "logit_margin_1_minus_0": margin,
                "teacher_config_sha256": p["config_sha256"],
                "teacher_margin_prompt_sha256": p["margin_prompt_sha256"],
                "teacher_model": TEACHER_MODEL,
            }
        )
    return rows


def arm_config(control: dict, gpu_uuid: str, selection: str, teacher: dict) -> dict:
    """Change identity, GPU, merge slot, serving hash, text and the teacher pool."""
    config = copy.deepcopy(control)
    config.update(
        campaign_id=ARM,
        gpu_uuid=gpu_uuid,
        merged_model=f"/tmp/gleipnir-merged/{ARM}",
        extra_sources=sorted(
            str(p.relative_to(ROOT)) for p in HERE.iterdir() if p.suffix == ".py"
        ),
        teacher_filter="strict",
        training_rows=POOL_ROWS,
        expected_steps=POOL_STEPS,
        expected_training_tokens=POOL_TOKENS,
        **FIELDS,
    )
    config["inputs"].pop("filter")
    for name in ("training", "token_audit"):
        config["inputs"][name] = {
            "path": POOL[name],
            "sha256": file_hash(ROOT / POOL[name]),
        }
    config["inputs"]["teacher"] = teacher
    config["inputs"]["serving_selection"]["sha256"] = selection
    return config


def prepare(gpu_uuid: str) -> dict:
    if not gpu_uuid.startswith("GPU-"):
        raise ValueError("a verified live GPU UUID is required")
    control = yaml.safe_load((ROOT / "data" / CONTROL / "config.yaml").read_text())
    for name, spec in control["inputs"].items():
        if (
            name != "serving_selection"
            and file_hash(ROOT / spec["path"]) != spec["sha256"]
        ):
            raise ValueError(f"control input drift: {name}")
    data = ROOT / "data" / ARM
    if data.exists():
        raise ValueError(f"refuse to overwrite {data}")
    rows = read_rows(ROOT / POOL["training"])
    targets = qwen_targets(
        read_rows(ROOT / POOL["teacher"]), read_rows(ROOT / PREDICTIONS)
    )
    validate_targets(rows, targets)
    removed = sum(remove_disagreement(t["label"], t["soft_target"], 0) for t in targets)
    write_rows(data / "teacher.jsonl", targets)
    teacher = {
        "path": str((data / "teacher.jsonl").relative_to(ROOT)),
        "sha256": file_hash(data / "teacher.jsonl"),
    }
    config = arm_config(control, gpu_uuid, file_hash(ROOT / SELECTION), teacher)
    (data / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
    report = {
        "status": "prepared",
        "gpu_uuid": gpu_uuid,
        "control_config_sha256": file_hash(ROOT / "data" / CONTROL / "config.yaml"),
        "predictions_sha256": file_hash(ROOT / PREDICTIONS),
        "teacher_sha256": teacher["sha256"],
        "pool_rows": len(rows),
        "expected_strict_removed": removed,
        "expected_strict_retained": len(rows) - removed,
        "config": str((data / "config.yaml").relative_to(ROOT)),
    }
    write_json(ROOT / "results/qwen38_teacher_4b/preparation.json", report)
    print("qwen38_teacher_prepared", json.dumps(report), flush=True)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu-uuid", required=True)
    prepare(parser.parse_args().gpu_uuid)
