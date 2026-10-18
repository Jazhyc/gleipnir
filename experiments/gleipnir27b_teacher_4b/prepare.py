"""Freeze two seed-1 4B arms distilled from Gleipnir 27B under the teacher prompt.

Targets are Gleipnir 27B's binary 0/1 soft targets on the 8,688-row pool,
read from the vocab-distillation extraction (full teacher prompt, stock vLLM
BF16, FP32 LM head): ``soft_target = sigmoid(logit_1 - logit_0)``. The recipe's
strict filter runs on the new teacher. Both arms then share one row set and
one target file; only the 4B's input prompt differs:

- ``student``: the compact student prompt, as every 4B campaign so far;
- ``teacher``: the same trajectories rendered with the full teacher prompt
  (byte-identical to the prompts Kimi K3 and Gleipnir 27B scored), and ID
  scored on the canonical CoT-removed ID trajectories rendered the same way.

The teacher prompt adds about 800 tokens per row; rows whose teacher-prompt
input exceeds the profile's context cap are removed from both arms (no
truncation), so the arms stay row-matched.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
from pathlib import Path

import yaml

from gleipnir.data.monitoring import (
    digest,
    file_hash,
    read_rows,
    rerender_evaluation,
    rerender_training,
    validate_targets,
    write_json,
    write_rows,
)
from gleipnir.data.teacher_filter import remove_disagreement

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
OUT = ROOT / "results/gleipnir27b_teacher_4b"
CONTROL = "teacher-seed-sdpa-s1-01"
SELECTION = "experiments/b200_inference_benchmark/serving_default.json"
ARMS = {
    "student": "gleipnir27b-teacher-student-s1-01",
    "teacher": "gleipnir27b-teacher-teacherprompt-s1-01",
}
TEACHER_MODEL = "Jazhyc/Gleipnir-27B (qwen38-27b-s0-01)"
POOL = {
    "training": "data/student_injection_awareness/regular/student_rows.jsonl",
    "teacher": "data/student_injection_awareness/soft_targets.jsonl",
}
POOL_ROWS = 8688
# Raw source tags in the teacher-prompt pool vs. the campaign's dataset names.
SOURCES = {
    "crh": "cot_red_handed",
    "s45": "bash_arena",
    "o46": "bash_arena",
    "ctrl": "bash_bench",
    "stride": "stride",
    "gloom": "gloom",
}
EXTRACT = "results/gleipnir27b_vocab_distillation/extract01/gleipnir_teacher_train"
TEACHER_PROMPTS = (
    "data/tool_trajectory_monitoring/teacher_training/prompts.jsonl",
    "c06833bc0c5e8893ea45e92ec642a71e751749ee1cd0796eb88f7ab38402763f",
)
FIELDS = {
    "hypothesis": (
        "Gleipnir 27B scored under the full teacher prompt (0.9352 Mean-OOD, "
        "above Kimi K3) is a better teacher for the Qwen3.5-4B student than Kimi "
        "K3: higher held-out ID macro pAUROC@20 than the seed-matched "
        "Kimi-distilled control."
    ),
    "intervention": (
        "Seed-1 BF16/SDPA campaign with Gleipnir 27B binary 0/1 soft targets "
        "(full teacher prompt, FP32 head) replacing Kimi K3; strict teacher/label "
        "filtering on the new teacher; rows over the context cap under the "
        "teacher prompt removed from both prompt arms; LR, schedule, "
        "initialization and loss unchanged."
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
PROMPT_NOTE = {
    "student": " The student reads the compact student prompt.",
    "teacher": (
        " The student reads the full teacher prompt in training and ID scoring "
        "(canonical CoT-removed ID trajectories, teacher template)."
    ),
}


def sigmoid(margin: float) -> float:
    return 1.0 / (1.0 + math.exp(-margin))


def gleipnir_targets(
    kimi: list[dict], extracted: list[dict], prompts: dict[str, str], render
) -> list[dict]:
    """Replace only the teacher distribution, keeping Kimi's row identity/order.

    ``render`` maps a raw teacher prompt to the exact text the 27B scored.
    """
    by_id = {r["id"]: r for r in extracted}
    if len(by_id) != len(extracted) or len(extracted) != len(kimi):
        raise ValueError("Gleipnir extraction coverage or uniqueness drift")
    rows = []
    for target in kimi:
        key = str(target["index"])
        row, prompt = by_id[key], prompts[key]
        margin = float(row["margin"])
        if (
            digest(prompt) != target["rendered_prompt_sha256"]
            or digest(render(prompt)) != row["rendered_prompt_sha256"]
            or row["label"] != target["label"]
            or target["dataset"] != "tool_trajectory/" + SOURCES[row["source"]]
        ):
            raise ValueError(f"Gleipnir target prompt, label or source drift: {key}")
        if not math.isfinite(margin):
            raise ValueError("nonfinite Gleipnir margin")
        positive = sigmoid(margin)
        rows.append(
            {
                "dataset": target["dataset"],
                "index": target["index"],
                "label": target["label"],
                "provider": "local-vllm",
                "rendered_prompt_sha256": target["rendered_prompt_sha256"],
                "soft_target": positive,
                # Full-vocabulary FP32 log-probabilities of the two tokens.
                "target_logprobs": {
                    "negative": row["logprob_0"],
                    "positive": row["logprob_1"],
                },
                "target_probs": {"negative": 1.0 - positive, "positive": positive},
                "logit_margin_1_minus_0": margin,
                "outside_01_mass": row["outside_01_mass"],
                "teacher_margin_prompt_sha256": row["rendered_prompt_sha256"],
                "teacher_model": TEACHER_MODEL,
            }
        )
    return rows


def token_lengths(tokenizer, rows: list[dict]) -> list[int]:
    """The campaign's decision-boundary input length (filtering.token_lengths)."""
    lengths = []
    for start in range(0, len(rows), 64):
        prompts = [
            tokenizer.apply_chat_template(
                [{"role": "user", "content": row["student_prompt"]}],
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
            + "Prediction:"
            for row in rows[start : start + 64]
        ]
        lengths.extend(
            map(len, tokenizer(prompts, add_special_tokens=False)["input_ids"])
        )
    return lengths


def id_workload(tokenizer, render, rows: list[dict]) -> list[dict]:
    """The frozen ID workload format (results/b200_optimized_id/id02)."""
    workload = []
    for row in rows:
        text = render(row["prompt"])
        workload.append(
            {
                "id": row["id"],
                "dataset": row["metadata"]["source_dataset"],
                "label": row["metadata"]["ground_truth"],
                "prompt": text,
                "prompt_sha256": digest(text),
                "prompt_tokens": len(tokenizer.encode(text, add_special_tokens=False)),
                "source_prompt_sha256": digest(row["prompt"]),
            }
        )
    return workload


def arm_config(
    control: dict,
    arm: str,
    gpu_uuid: str,
    selection: str,
    inputs: dict[str, dict],
    rows: int,
    tokens: int,
) -> dict:
    """Change identity, GPU, merge slot, serving hash, text, rows and inputs."""
    config = copy.deepcopy(control)
    config.update(
        campaign_id=ARMS[arm],
        gpu_uuid=gpu_uuid,
        merged_model=f"/tmp/gleipnir-merged/{ARMS[arm]}",
        extra_sources=sorted(
            str(p.relative_to(ROOT)) for p in HERE.iterdir() if p.suffix == ".py"
        ),
        teacher_filter="strict",
        training_rows=rows,
        expected_steps=math.ceil(rows / control["logical_batch_size"]),
        expected_training_tokens=tokens,
        **FIELDS,
    )
    config["intervention"] += PROMPT_NOTE[arm]
    config["inputs"].pop("filter")
    config["inputs"].update(inputs)
    config["inputs"]["serving_selection"]["sha256"] = selection
    return config


def bound(path: Path) -> dict:
    return {"path": str(path.relative_to(ROOT)), "sha256": file_hash(path)}


def prepare(gpu_uuid: str) -> dict:
    from transformers import AutoTokenizer

    from experiments.tool_trajectory_monitoring.benchmark_qwen_ood import (
        QWEN_NON_THINKING_ASSISTANT_SUFFIX,
        render_margin_prompt,
    )
    from experiments.tool_trajectory_monitoring.prompting import load_prompt_set
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
    for arm in ARMS.values():
        if (ROOT / "data" / arm).exists():
            raise ValueError(f"refuse to overwrite data/{arm}")
    if file_hash(ROOT / TEACHER_PROMPTS[0]) != TEACHER_PROMPTS[1]:
        raise ValueError("frozen teacher-prompt input changed")
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

    pool = read_rows(ROOT / POOL["training"])
    if len(pool) != POOL_ROWS:
        raise ValueError("pool population changed")
    prompts = {str(r["id"]): r["prompt"] for r in read_rows(ROOT / TEACHER_PROMPTS[0])}
    targets = gleipnir_targets(
        read_rows(ROOT / POOL["teacher"]),
        read_rows(ROOT / EXTRACT / "rows.jsonl"),
        prompts,
        render,
    )
    validate_targets(pool, targets)
    by_key = {(t["dataset"], str(t["index"])): t for t in targets}
    template = load_prompt_set().teacher
    teacher_pool = rerender_training(pool, template)
    for row in teacher_pool:
        if row["student_prompt_sha256"] != row["teacher_rendered_prompt_sha256"]:
            raise ValueError("teacher-template rerender differs from the scored prompt")
    agree = [
        not remove_disagreement(
            r["label"], by_key[(r["dataset"], str(r["index"]))]["soft_target"], 0
        )
        for r in pool
    ]
    lengths = {
        "student": token_lengths(tokenizer, pool),
        "teacher": token_lengths(tokenizer, teacher_pool),
    }
    keep = [i for i in range(len(pool)) if agree[i] and lengths["teacher"][i] <= cap]
    over_cap = [
        str(pool[i]["index"])
        for i in range(len(pool))
        if agree[i] and lengths["teacher"][i] > cap
    ]
    canonical_id = read_rows(ROOT / control["inputs"]["id"]["path"])
    teacher_id = rerender_evaluation(canonical_id, template)
    selection = file_hash(ROOT / SELECTION)
    report = {
        "status": "prepared",
        "gpu_uuid": gpu_uuid,
        "control_config_sha256": file_hash(control_path),
        "extraction_rows_sha256": file_hash(ROOT / EXTRACT / "rows.jsonl"),
        "pool_rows": len(pool),
        "strict_removed": agree.count(False),
        "context_cap": cap,
        "teacher_prompt_over_cap_removed": over_cap,
        "retained": len(keep),
        "arms": {},
    }
    for arm, source in (("student", pool), ("teacher", teacher_pool)):
        data = ROOT / "data" / ARMS[arm]
        rows = [source[i] for i in keep]
        write_rows(data / "training.jsonl", rows)
        write_rows(
            data / "teacher.jsonl",
            [by_key[(r["dataset"], str(r["index"]))] for r in rows],
        )
        arm_lengths = [lengths[arm][i] for i in keep]
        write_json(
            data / "token_audit.json",
            {
                "rows": len(rows),
                "total": sum(arm_lengths),
                "maximum": max(arm_lengths),
                "configured_cap": cap,
                "truncated": 0,
            },
        )
        inputs = {
            name: bound(data / filename)
            for name, filename in (
                ("training", "training.jsonl"),
                ("teacher", "teacher.jsonl"),
                ("token_audit", "token_audit.json"),
            )
        }
        if arm == "teacher":
            write_rows(data / "id_prompts.jsonl", teacher_id)
            (data / "id_workload.json").write_text(
                json.dumps(id_workload(tokenizer, render, teacher_id)) + "\n"
            )
            inputs["id"] = bound(data / "id_prompts.jsonl")
            inputs["id_workload"] = bound(data / "id_workload.json")
        config = arm_config(
            control, arm, gpu_uuid, selection, inputs, len(rows), sum(arm_lengths)
        )
        (data / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
        report["arms"][arm] = {
            "campaign_id": ARMS[arm],
            "config": str((data / "config.yaml").relative_to(ROOT)),
            "training_tokens": sum(arm_lengths),
            "maximum_tokens": max(arm_lengths),
        }
    write_json(OUT / "preparation.json", report)
    print("gleipnir27b_teacher_prepared", json.dumps(report), flush=True)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu-uuid", required=True)
    prepare(parser.parse_args().gpu_uuid)
