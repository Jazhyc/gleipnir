"""Exact full-prefix requests from previously validated teacher artifacts."""

import hashlib
import json
from collections import defaultdict
from pathlib import Path

from gleipnir.branch_training import plan_branches
from gleipnir.prefix_sampling import attach_sampled_prefix


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class BranchDataset:
    """Materialize one parent's exact requests on demand, not a billion-token list."""

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        previous = json.loads(
            Path("results/monitoring_prefix_training/manifest.json").read_text()
        )
        self.provenance = previous["paired_training"]
        parent_path = Path(
            "data/tool_trajectory_monitoring/distillation_scaling/student_rows.jsonl"
        )
        cache_path = Path(
            "results/monitoring_prefix_supervision/qwen35_flashinfer_cache/logits.jsonl"
        )
        refs_path = Path("data/monitoring_prefix_supervision/prefix_references.jsonl")
        for path, expected in (
            (parent_path, self.provenance["source_sha256"]),
            (cache_path, self.provenance["cache_sha256"]),
            (refs_path, self.provenance["references_sha256"]),
        ):
            if sha(path) != expected:
                raise ValueError(f"all-prefix provenance drift: {path}")
        self.parents = [json.loads(s) for s in parent_path.open()]
        refs = {r["id"]: r for r in map(json.loads, refs_path.open())}
        self.prefixes = defaultdict(list)
        seen = set()
        for r in map(json.loads, cache_path.open()):
            if (
                r["id"] in seen
                or r["contract_sha256"] != self.provenance["cache_contract_sha256"]
            ):
                raise ValueError("duplicate or mismatched prefix cache")
            seen.add(r["id"])
            ref = refs[r["id"]]
            self.prefixes[ref["parent_prompt_id"]].append({**r, **ref})
        if seen != set(refs) or len(seen) != 133947 or len(self.parents) != 8688:
            raise ValueError("all-prefix population drift")
        for values in self.prefixes.values():
            values.sort(key=lambda r: (r["end_character"], r["id"]))
        soft = Path(
            "data/tool_trajectory_monitoring/distillation_scaling/soft_targets.jsonl"
        )
        job_path = Path("results/monitoring_prefix_training/jobs.jsonl")
        if sha(job_path) != previous["files"][str(job_path)]:
            raise ValueError("previous frozen jobs drift")
        jobs = [json.loads(s) for s in job_path.open()]
        if sha(soft) != jobs[0]["soft_targets_sha256"]:
            raise ValueError("full target provenance drift")
        self.soft = {
            (r["dataset"], r["index"]): r for r in map(json.loads, soft.open())
        }

    def __len__(self):
        return len(self.parents)

    def __getitem__(self, index):
        parent = self.parents[index]
        full = self.soft[(parent["dataset"], parent["index"])]
        if full["rendered_prompt_sha256"] != parent["teacher_rendered_prompt_sha256"]:
            raise ValueError("Kimi full prompt drift")
        prompts, targets = [], []
        for prefix in self.prefixes[parent["prompt_id"]]:
            attached = attach_sampled_prefix(parent, prefix)
            prompts.append(attached["prefix_student_prompt"])
            targets.append(attached["prefix_soft_target"])
        prompts.append(parent["student_prompt"])
        targets.append(full["soft_target"])
        requests = [
            self.tokenizer.encode(
                self.tokenizer.apply_chat_template(
                    [{"role": "user", "content": prompt}],
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=False,
                )
                + "Prediction:",
                add_special_tokens=False,
            )
            for prompt in prompts
        ]
        if max(map(len, requests)) > 29696:
            raise ValueError("all-prefix context overflow; no truncation")
        return {
            "parent_id": parent["prompt_id"],
            "plan": plan_branches(requests),
            "targets": targets,
        }
