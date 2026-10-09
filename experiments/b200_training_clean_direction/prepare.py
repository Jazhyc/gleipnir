"""Freeze every original trajectory once, with no synthetic views."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path

from gleipnir.data.monitoring import file_hash, read_rows, write_json, write_rows

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).parent
CONFIG = HERE / "config.json"


def join_originals(
    clean: list[dict], teacher: list[dict], census: list[dict]
) -> list[dict]:
    """Reject synthetic inputs, missing lineage, duplicate IDs or stale targets."""
    targets = {(r["dataset"], str(r["index"])): r for r in teacher}
    flags = {r["id"]: r for r in census}
    if len(targets) != len(teacher) or len(flags) != len(census):
        raise ValueError("duplicate teacher/census identity")
    if len({str(r["index"]) for r in clean}) != len(clean):
        raise ValueError("non-unique original identity")
    result = []
    for r in clean:
        if "augmentation" in r:
            raise ValueError("synthetic augmentation in original population")
        t = targets[r["dataset"], str(r["index"])]
        f = flags[str(r["index"])]
        if r["trajectory_sha256"] != f["trajectory_sha256"] or r["label"] != f["label"]:
            raise ValueError("census parent/label drift")
        if r["teacher_rendered_prompt_sha256"] != t["rendered_prompt_sha256"]:
            raise ValueError("teacher prompt drift")
        result.append(r | {"soft_target": t["soft_target"], "census_score": f["score"]})
    return sorted(result, key=lambda r: (r["dataset"], str(r["index"])))


def prepare() -> None:
    from transformers import AutoTokenizer

    c = json.loads(CONFIG.read_text())
    out = ROOT / c["artifact_root"] / c["campaign_id"]
    if out.exists():
        raise ValueError("already prepared")
    for s in c["inputs"].values():
        if file_hash(ROOT / s["path"]) != s["sha256"]:
            raise ValueError("input drift " + s["path"])
    rs = join_originals(
        *[
            read_rows(ROOT / c["inputs"][k]["path"])
            for k in ["clean", "teacher", "census"]
        ]
    )
    if len(rs) != 8688 or len({r["lineage_group"] for r in rs}) != 8688:
        raise ValueError("original population/lineage drift")
    tokenizer = AutoTokenizer.from_pretrained(
        ROOT / c["tokenizer"], local_files_only=True
    )
    effective = tokenizer.backend_tokenizer.to_str()
    if (
        hashlib.sha256(effective.encode()).hexdigest()
        != c["effective_tokenizer_sha256"]
    ):
        raise ValueError("effective tokenizer drift")
    workload = []
    for r in rs:
        if (
            hashlib.sha256(r["student_prompt"].encode()).hexdigest()
            != r["student_prompt_sha256"]
        ):
            raise ValueError("original prompt drift")
        prompt = (
            tokenizer.apply_chat_template(
                [{"role": "user", "content": r["student_prompt"]}],
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
            + "Prediction:"
        )
        tokens = len(tokenizer.encode(prompt, add_special_tokens=False))
        if not 0 < tokens < 32768:
            raise ValueError("context overflow")
        workload.append(
            {
                "id": r["dataset"] + ":" + str(r["index"]) + ":original",
                "prompt": prompt,
                "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
                "prompt_tokens": tokens,
                "metadata": {
                    k: r[k]
                    for k in [
                        "dataset",
                        "label",
                        "lineage_group",
                        "student_prompt_sha256",
                        "trajectory_sha256",
                        "soft_target",
                        "census_score",
                    ]
                }
                | {
                    "index": str(r["index"]),
                    "source": r["raw_source"],
                    "view": "original",
                    "teacher_prompt_sha256": r["teacher_rendered_prompt_sha256"],
                },
            }
        )
    out.mkdir(parents=True)
    write_rows(out / "workload.jsonl", workload)
    (out / "effective_tokenizer.json").write_text(effective)
    write_json(
        out / "prepare_receipt.json",
        {
            "originals": len(workload),
            "synthetic_views": 0,
            "tokens_per_model": sum(r["prompt_tokens"] for r in workload),
            "maximum_tokens": max(r["prompt_tokens"] for r in workload),
            "source_counts": dict(Counter(r["raw_source"] for r in rs)),
            "workload_sha256": file_hash(out / "workload.jsonl"),
            "effective_tokenizer_sha256": file_hash(out / "effective_tokenizer.json"),
        },
    )
    print((out / "prepare_receipt.json").read_text(), flush=True)


if __name__ == "__main__":
    prepare()
