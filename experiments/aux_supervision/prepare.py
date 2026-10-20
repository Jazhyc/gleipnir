"""Freeze the seed-1 4B arm with Position/Reason auxiliary distillation.

Rows, order, Kimi K3 soft targets, LR, schedule and initialization are the
seed-1 control's (`teacher-seed-sdpa-s1-01`). Two arms exist:

- ``aux-position-reason-s1-01``: the compact student prompt gains the
  three-line output interface (`targets.student_aux_template`), in training
  and in ID scoring;
- ``aux-oneline-position-reason-s1-01``: the control's one-line student
  prompt, unchanged, so only the loss differs from the control.

In both arms every row carries an auxiliary continuation from the Gleipnir 27B
  extraction (`extract.py --name full01`), distilled with a top-64-plus-tail
  KL at weight ``AUX_LOSS_WEIGHT`` alongside the unchanged soft BCE.

The continuation is the teacher's own token sequence after
``Prediction:{label}``: the forced ``\\nPosition:`` tokens (unsupervised: the
teacher has no distribution there), then every generated token, all
supervised (user decision: no grounding mask). Grounding is recorded per row
as a diagnostic only.
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import yaml

from experiments.aux_supervision.targets import (
    aux_template,
    grounding,
    parse_continuation,
    student_aux_template,
)
from gleipnir.data.monitoring import (
    file_hash,
    read_rows,
    rerender_evaluation,
    rerender_training,
    trajectory_from_prompt,
    write_json,
    write_rows,
)

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
OUT = ROOT / "results/aux_supervision"
CONTROL = "teacher-seed-sdpa-s1-01"
ARM = "aux-position-reason-s1-01"
ONE_LINE_ARM = "aux-oneline-position-reason-s1-01"
ARMS = (ARM, ONE_LINE_ARM)
EXTRACT = OUT / "full01/gleipnir"
SELECTION = "experiments/b200_inference_benchmark/serving_default.json"
AUX_LOSS_WEIGHT = 0.1
TOP = 64
# Unsupervised placeholder rows: all mass on one id, the rest below float range.
PAD_LOGPROB = -1.0e4
FIELDS = {}
FIELDS[ARM] = {
    "hypothesis": (
        "Distilling Gleipnir 27B's Position quote and Reason sentence after the "
        "decision improves the 4B's decision representation: higher held-out ID "
        "macro pAUROC@20 than the seed-matched Kimi-distilled control, with "
        "inference still a single prefill read at Prediction:."
    ),
    "intervention": (
        "Seed-1 BF16/SDPA campaign on the control's rows and Kimi K3 soft "
        "targets; the compact student prompt gains a three-line output interface "
        "(training and ID); a top-64-plus-tail KL toward Gleipnir 27B's "
        "continuation distributions is added at weight 0.1; LR, schedule, "
        "initialization and decision loss unchanged."
    ),
    "selection": (
        "Final one-epoch checkpoint; ID-only paired comparison with "
        "teacher-seed-sdpa-s1-01 under the frozen distillation_temperature rule; "
        "advance to more seeds, never promote, on one seed."
    ),
    "stop_condition": (
        "Stop on input/runtime drift, nonfinite or missing gradients, failed "
        "master/merge/serving parity, incomplete scoring or completion."
    ),
}
FIELDS[ONE_LINE_ARM] = {
    **FIELDS[ARM],
    "hypothesis": (
        "The same auxiliary distillation under the control's unchanged one-line "
        "student prompt improves held-out ID macro pAUROC@20 over the "
        "seed-matched control; it also separates the three-line arm's loss "
        "(2026-10-20: -3.7 points) into prompt and loss effects."
    ),
    "intervention": (
        "Seed-1 BF16/SDPA campaign on the control's rows, Kimi K3 soft targets "
        "and one-line student prompt (training and ID byte-identical to the "
        "control); a top-64-plus-tail KL toward Gleipnir 27B's continuation "
        "distributions is added at weight 0.1; LR, schedule, initialization "
        "and decision loss unchanged."
    ),
}


def receipt(arm: str, name: str) -> Path:
    """Arm receipts; the first arm keeps its original top-level paths."""
    return OUT / name if arm == ARM else OUT / arm / name


def student_template(arm: str):
    from experiments.tool_trajectory_monitoring.prompting import load_prompt_set

    return student_aux_template() if arm == ARM else load_prompt_set().student


def position_grounded(label: int, check: dict) -> bool:
    """Diagnostic: the quote points at the agent's own action (or none on 0)."""
    if label == 0:
        return check["status"] == "none"
    return (
        check["status"] in {"unique", "ambiguous"} and check.get("speaker") == "agent"
    )


def aux_fields(extracted: dict, trajectory: str, tokenizer) -> dict:
    """Continuation inputs, top-k targets and mask, one entry per input token."""
    label = int(extracted["label"])
    forced = tokenizer.encode(f"{label}\nPosition:", add_special_tokens=False)
    generated = [step["token"] for step in extracted["steps"]]
    if not generated:
        raise ValueError(f"empty teacher continuation: {extracted['id']}")
    target = parse_continuation(extracted["text"])
    check = grounding(trajectory, label, target)
    top_ids, top_logprobs, mask = [], [], []
    # Positions 0..len(forced)-2 predict the rest of the forced prefix.
    for token in forced[1:]:
        top_ids.append([token] * TOP)
        top_logprobs.append([0.0] + [PAD_LOGPROB] * (TOP - 1))
        mask.append(False)
    for step in extracted["steps"]:
        if len(step["top_ids"]) != TOP:
            raise ValueError(f"teacher top-k width drift: {extracted['id']}")
        top_ids.append(step["top_ids"])
        top_logprobs.append(step["top_logprobs"])
        mask.append(True)
    return {
        "aux_input_ids": forced + generated[:-1],
        "aux_top_ids": top_ids,
        "aux_top_logprobs": top_logprobs,
        "aux_mask": mask,
        "aux_grounding": check,
        "aux_position_grounded": position_grounded(label, check),
        "aux_well_formed": target.well_formed,
        "aux_teacher_text": extracted["text"],
    }


def check_forced_boundary(tokenizer, render, rows: list[dict], count: int = 64):
    """The forced prefix tokenizes the same in context as alone."""
    for row in rows[:count]:
        context = render(row["trajectory"])
        forced = f"{row['label']}\nPosition:"
        whole = tokenizer.encode(context + forced, add_special_tokens=False)
        parts = tokenizer.encode(context, add_special_tokens=False)
        parts += tokenizer.encode(forced, add_special_tokens=False)
        if whole != parts:
            raise ValueError(f"forced prefix merges at the boundary: {row['index']}")


def bound(path: Path) -> dict:
    return {"path": str(path.relative_to(ROOT)), "sha256": file_hash(path)}


def prepare(gpu_uuid: str, arm: str = ARM) -> dict:
    from transformers import AutoTokenizer

    from experiments.gleipnir27b_teacher_4b.prepare import id_workload, token_lengths
    from experiments.tool_trajectory_monitoring.benchmark_qwen_ood import (
        QWEN_NON_THINKING_ASSISTANT_SUFFIX,
        render_margin_prompt,
    )
    from gleipnir.campaigns.monitoring.contract import Campaign

    if not gpu_uuid.startswith("GPU-"):
        raise ValueError("a verified live GPU UUID is required")
    control_path = ROOT / "data" / CONTROL / "config.yaml"
    control = yaml.safe_load(control_path.read_text())
    for name, spec in control["inputs"].items():
        if (
            name != "serving_selection"
            and file_hash(ROOT / spec["path"]) != spec["sha256"]
        ):
            raise ValueError(f"control input drift: {name}")
    if arm not in ARMS:
        raise ValueError(f"unknown arm: {arm}")
    data = ROOT / "data" / arm
    if data.exists():
        raise ValueError(f"refuse to overwrite data/{arm}")
    cap = Campaign.load(ROOT, control_path).job()["max_length"]
    tokenizer = AutoTokenizer.from_pretrained(
        ROOT / control["base_model"], local_files_only=True
    )

    def render(prompt: str) -> str:
        return render_margin_prompt(
            tokenizer,
            prompt,
            enable_thinking=False,
            assistant_suffix=QWEN_NON_THINKING_ASSISTANT_SUFFIX,
            decision_prefix="Prediction:",
        )

    rows = read_rows(ROOT / control["inputs"]["training"]["path"])
    teacher = read_rows(ROOT / control["inputs"]["teacher"]["path"])
    extracted = {r["id"]: r for r in read_rows(EXTRACT / "rows.jsonl")}
    if len(extracted) != len(rows) or set(extracted) != {r["index"] for r in rows}:
        raise ValueError("extraction does not cover the control rows exactly")
    template = student_template(arm)
    teacher_template = aux_template()
    for row in rows:
        row["trajectory"] = trajectory_from_prompt(
            row["student_prompt"], row["trajectory_sha256"]
        )
    check_forced_boundary(tokenizer, lambda t: render(teacher_template.render(t)), rows)
    arm_rows = rerender_training(
        [{k: v for k, v in r.items() if k != "trajectory"} for r in rows], template
    )
    if arm == ONE_LINE_ARM and any(
        a["student_prompt"] != r["student_prompt"]
        for a, r in zip(arm_rows, rows, strict=True)
    ):
        raise ValueError("one-line arm prompts differ from the control's")
    for row, source in zip(arm_rows, rows, strict=True):
        teacher_row = extracted[row["index"]]
        if teacher_row["label"] != row["label"]:
            raise ValueError(f"label drift: {row['index']}")
        row.update(aux_fields(teacher_row, source["trajectory"], tokenizer))
    lengths = token_lengths(tokenizer, arm_rows)
    totals = [
        n + len(r["aux_input_ids"]) for n, r in zip(lengths, arm_rows, strict=True)
    ]
    if max(totals) > cap:
        raise ValueError("a continuation would exceed the context cap")
    write_rows(data / "training.jsonl", arm_rows)
    write_rows(data / "teacher.jsonl", teacher)
    write_json(
        data / "token_audit.json",
        {
            "rows": len(arm_rows),
            "total": sum(totals),
            "maximum": max(totals),
            "configured_cap": cap,
            "truncated": 0,
            "continuation_tokens": sum(len(r["aux_input_ids"]) for r in arm_rows),
        },
    )
    canonical_id = read_rows(ROOT / control["inputs"]["id"]["path"])
    id_rows = rerender_evaluation(canonical_id, template)
    if arm == ONE_LINE_ARM and [r["prompt"] for r in id_rows] != [
        r["prompt"] for r in canonical_id
    ]:
        raise ValueError("one-line arm ID prompts differ from the control's")
    write_rows(data / "id_prompts.jsonl", id_rows)
    (data / "id_workload.json").write_text(
        json.dumps(id_workload(tokenizer, render, id_rows)) + "\n"
    )
    inputs = {
        name: bound(data / filename)
        for name, filename in (
            ("training", "training.jsonl"),
            ("teacher", "teacher.jsonl"),
            ("token_audit", "token_audit.json"),
            ("id", "id_prompts.jsonl"),
            ("id_workload", "id_workload.json"),
        )
    }
    config = copy.deepcopy(control)
    config.update(
        campaign_id=arm,
        gpu_uuid=gpu_uuid,
        merged_model=f"/tmp/gleipnir-merged/{arm}",
        extra_sources=sorted(
            str(p.relative_to(ROOT)) for p in HERE.iterdir() if p.suffix == ".py"
        ),
        aux_loss_weight=AUX_LOSS_WEIGHT,
        # A new objective family: rerun every startup gate (training guide).
        full_startup_validation=True,
        expected_training_tokens=sum(totals),
        **FIELDS[arm],
    )
    config["inputs"].pop("filter", None)
    config["inputs"].pop("startup_reference", None)
    config["inputs"].update(inputs)
    config["inputs"]["serving_selection"]["sha256"] = file_hash(ROOT / SELECTION)
    (data / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
    supervised = [m for r in arm_rows for m in r["aux_mask"]]
    report = {
        "status": "prepared",
        "campaign_id": arm,
        "gpu_uuid": gpu_uuid,
        "control_config_sha256": file_hash(control_path),
        "extraction_rows_sha256": file_hash(EXTRACT / "rows.jsonl"),
        "student_template_sha256": template.template_sha256,
        "teacher_template_sha256": teacher_template.template_sha256,
        "rows": len(arm_rows),
        "aux_loss_weight": AUX_LOSS_WEIGHT,
        "position_grounded_rows": sum(r["aux_position_grounded"] for r in arm_rows),
        "well_formed_rows": sum(r["aux_well_formed"] for r in arm_rows),
        "continuation_tokens": len(supervised),
        "supervised_tokens": sum(supervised),
        "training_tokens": sum(totals),
        "maximum_tokens": max(totals),
        "context_cap": cap,
        "expected_steps": config["expected_steps"],
    }
    write_json(receipt(arm, "preparation.json"), report)
    print("aux_supervision_prepared", json.dumps(report), flush=True)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu-uuid", required=True)
    parser.add_argument("--arm", choices=ARMS, default=ARM)
    args = parser.parse_args()
    prepare(args.gpu_uuid, args.arm)
