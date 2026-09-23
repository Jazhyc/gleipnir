"""Reconstruct source-grounded CoT removal and freeze a 512-row local subset."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from transformers import AutoTokenizer

from experiments.id_assistant_prose_audit.thinking import source_records
from experiments.id_cot_only_evaluation.prepare import strip_cot
from experiments.local_inference.core import (
    CONFIG,
    DATA,
    ROOT,
    canary_rows,
    digest,
    select_subset,
    write_json,
)
from experiments.tool_trajectory_monitoring import (
    prepare_qwen_reasoning_id_benchmark as source,
)
from experiments.tool_trajectory_monitoring.prompting import load_prompt_set
from gleipnir.qwen35_adapter_rebase import sha256_file


def main() -> None:
    config = json.loads(CONFIG.read_text())
    if (DATA / "manifest.json").exists():
        manifest = json.loads((DATA / "manifest.json").read_text())
        if sha256_file(DATA / "subset.jsonl") != manifest["subset_sha256"]:
            raise ValueError("frozen subset changed")
        if (
            manifest["seed"] != config["seed"]
            or manifest["source_label_counts"] != config["subset_counts"]
            or manifest["tokenizer_revision"] != config["base_revision"]
        ):
            raise ValueError("frozen selection contract changed")
        print("Reusing frozen subset", flush=True)
        return
    directory = Path("data/tool_trajectory_monitoring/source/id_evaluation")
    for name, expected in [
        ("stride_test.parquet", source.STRIDE_SHA256),
        ("gloom_exfiltration.parquet", source.GLOOM_SHA256),
    ]:
        if sha256_file(directory / name) != expected:
            raise ValueError(f"source drift: {name}")
    selection_tokenizer = AutoTokenizer.from_pretrained(
        source.TOKENIZER_ID, revision=source.TOKENIZER_REVISION, local_files_only=True
    )
    rows = source.load_stride(directory / "stride_test.parquet")
    rows += source.load_gloom(
        directory / "gloom_exfiltration.parquet", selection_tokenizer
    )
    tokenizer = AutoTokenizer.from_pretrained(ROOT / "base", local_files_only=True)
    template = load_prompt_set().student
    contract = json.loads((ROOT / "adapter/prompt_contract.json").read_text())
    if template.template_sha256 != contract["template_sha256"]:
        raise ValueError("released prompt mismatch")
    if template.instruction != (ROOT / "adapter/student_prompt.txt").read_text():
        raise ValueError("released instruction mismatch")
    typed = source_records()
    audit = []
    for row in rows:
        trajectory = row.pop("trajectory")
        original_hash = digest(trajectory)
        calls = blocks = 0
        if row["id"] in typed:
            trajectory, calls, blocks = strip_cot(trajectory, typed[row["id"]])
        user_prompt = template.render(trajectory)
        prompt = (
            tokenizer.apply_chat_template(
                [{"role": "user", "content": user_prompt}],
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
            + "Prediction:"
        )
        row.update(
            prompt=prompt,
            prompt_sha256=digest(prompt),
            trajectory_sha256=digest(trajectory),
            original_trajectory_sha256=original_hash,
            tokens=len(tokenizer.encode(prompt, add_special_tokens=False)),
        )
        audit.append(
            {"id": row["id"], "removed_calls": calls, "removed_blocks": blocks}
        )
    if len(rows) != 3012 or sum(r["tokens"] for r in rows) != 33_750_959:
        raise ValueError("reconstructed population or historical token total differs")
    subset = select_subset(rows, config["subset_counts"], config["seed"])
    if max(r["tokens"] for r in subset) >= config["engine"]["max_model_len"]:
        raise ValueError("context limit exceeded")
    DATA.mkdir(parents=True, exist_ok=True)
    subset_path = DATA / "subset.jsonl"
    subset_path.write_text(
        "".join(json.dumps(r, sort_keys=True) + "\n" for r in subset)
    )
    write_json(DATA / "removal_audit.json", audit)
    manifest = {
        "subset_sha256": sha256_file(subset_path),
        "config_sha256": sha256_file(CONFIG),
        "rows": len(subset),
        "prompt_tokens": sum(r["tokens"] for r in subset),
        "min_tokens": min(r["tokens"] for r in subset),
        "max_tokens": max(r["tokens"] for r in subset),
        "source_label_counts": dict(
            Counter(f"{r['source']}:{r['label']}" for r in subset)
        ),
        "canary_ids": [r["id"] for r in canary_rows(subset)],
        "selection": "source/label quotas; four length bins; seeded ID hash",
        "seed": config["seed"],
        "tokenizer_revision": config["base_revision"],
        "template_sha256": template.template_sha256,
        "source_sha256": {"stride": source.STRIDE_SHA256, "gloom": source.GLOOM_SHA256},
        "source_license": "unspecified upstream; internal research; no redistribution",
        "reconstruction": "existing ID loaders and CoT remover; new metadata schema",
        "canonical_full_jsonl_hash_verified": False,
        "population_prompt_tokens": sum(r["tokens"] for r in rows),
        "population_rows": len(rows),
    }
    write_json(DATA / "manifest.json", manifest)
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()
