"""Score the 4B release candidate on OOD through its optimized serving stack.

The merge is rebuilt from the FP32 master on the scoring host. FP32 ``b @ a``
accumulation differs across CPU BLAS kernels (an AMD EPYC host does not
reproduce an Intel host's BF16 shards bit for bit), so instead of requiring
identity with the parent merge, the merged checkpoint gets a fresh
master-to-merged parity check on the parent's saved canary prompts, with the
same scorer and gates as the campaign's ``merged-reference`` stage.
"""

from __future__ import annotations

import argparse
import fcntl
import json
import os
from pathlib import Path

from experiments.optimized_gap.prepare import OPT_4B, OUT, ROOT
from experiments.teacher_disagreement_filtering.run import retire
from experiments.teacher_seed_variation.run import scoring_context, subprocess_stage
from gleipnir.campaigns.runtime import training_environment
from gleipnir.data.monitoring import (
    digest,
    file_hash,
    read_rows,
    write_json,
    write_rows,
)


def fresh_merge(name: str) -> None:
    """Merge on this host, then check BF16 merged scores against the FP32 master."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from experiments.deception_distillation.train_student_sft import (
        gated_delta_kernel_modules,
    )
    from experiments.training_procedure_screen.evaluate_causal import score_adapter
    from gleipnir.campaigns.monitoring.evaluation import agreement
    from gleipnir.campaigns.monitoring.training import merge

    ctx = scoring_context(name)
    ctx.check()
    if Path(ctx.config["merged_model"]).exists():
        raise ValueError("scoring merge slot already exists")
    merge(ctx)
    parent = ROOT / ctx.config["parent_campaign"]
    workload = read_rows(parent / "canary_workload.jsonl")
    master = json.loads((ctx.adapter / "parity_reference.json").read_text())
    if [r["prompt_sha256"] for r in workload] != master["prompt_sha256"] or any(
        digest(r["prompt"]) != r["prompt_sha256"] for r in workload
    ):
        raise ValueError("parent canary prompts differ from the master reference")
    if master["master_sha256"] != file_hash(
        ctx.adapter / "causal_adapter/adapter_model.safetensors"
    ):
        raise ValueError("FP32 master identity changed")
    merged = ctx.config["merged_model"]
    tokenizer = AutoTokenizer.from_pretrained(merged, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        merged,
        local_files_only=True,
        dtype="bfloat16",
        device_map={"": 0},
        attn_implementation="sdpa",
    )
    kernels = gated_delta_kernel_modules(model)
    if not kernels or any(not k.startswith("fla.ops.") for k in kernels):
        raise ValueError("merged reference requires original pinned FLA kernels")
    tokenized = [
        tokenizer.encode(r["prompt"], add_special_tokens=False) for r in workload
    ]
    scores, details = score_adapter(
        model,
        tokenizer,
        tokenized,
        [15, 16],
        batch_size=1,
        decision_head_mode="token_logits",
    )
    receipt = agreement(scores, master["adapter"], master["base"], ctx.config["parity"])
    receipt.update(
        scores=scores,
        master_sha256=master["master_sha256"],
        details=details,
        merged_artifact_sha256=file_hash(ctx.output / "merged_artifact.json"),
        policy="fresh_merge_and_parity_on_scoring_host",
    )
    write_json(ctx.output / "merged_parity.json", receipt)
    write_rows(ctx.output / "canary_workload.jsonl", workload)
    print("optimized_gap_merged_parity", receipt["passed"], flush=True)
    if not receipt["passed"]:
        raise ValueError("master/merged BF16 parity failed")


def score(name: str) -> None:
    ctx = scoring_context(name)
    ctx.output.mkdir(parents=True, exist_ok=True)
    subprocess_stage(
        ["-m", "experiments.optimized_gap.run", "--stage", "merge", name],
        ctx.config["training_python"],
        training_environment(ROOT, name),
        ctx.logs / "merge.log",
    )
    subprocess_stage(
        [
            "-m",
            "experiments.b200_vllm031.runtime",
            "-m",
            "experiments.teacher_seed_variation.run",
            "--stage",
            "score",
            name,
        ],
        ctx.config["serving_python"],
        dict(os.environ),
        ctx.logs / "score.log",
    )
    retire(ctx)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=["merge"])
    parser.add_argument("name", nargs="?", default=OPT_4B)
    args = parser.parse_args()
    if args.stage == "merge":
        fresh_merge(args.name)
    else:
        OUT.mkdir(parents=True, exist_ok=True)
        with (OUT / "queue.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            score(args.name)
