"""Resumable scoring extracted from the audited continuation evaluator.

Engine settings and status destination are explicit to keep campaigns isolated.
"""

from __future__ import annotations

import json
import math
import os
import time
from pathlib import Path
from typing import Any

from gleipnir.evaluation.decision_surface import decision_token_ids
from gleipnir.evaluation.probabilities import (
    logprob_value,
    normalized_binary_probability,
)
from gleipnir.monitoring_campaign_data import digest, file_hash, read_rows, write_json


def rendered(tokenizer: Any, row: dict, surface: str) -> str:
    text = row["student_prompt"] if surface == "AB" else row["prompt"]
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": text}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    ) + ("" if surface == "AB" else "Prediction:")


def completed_predictions(path: Path, inputs: list[dict], ident: dict) -> list[dict]:
    """Reporting requires complete, identity-matched scores and raw-logprob evidence."""
    contract = json.loads(path.with_suffix(".contract.json").read_text())
    complete = json.loads(path.with_suffix(".complete.json").read_text())
    rows = read_rows(path)
    expected = {str(r.get("id", r.get("index"))): r for r in inputs}
    if (
        contract["identity"] != ident
        or not complete["passed"]
        or complete["sha256"] != file_hash(path)
        or complete["contract_sha256"] != contract["sha256"]
        or len(rows) != len(inputs)
        or complete["rows"] != len(inputs)
        or contract["rows"] != len(inputs)
        or len({r["id"] for r in rows}) != len(rows)
        or {r["id"] for r in rows} != set(expected)
    ):
        raise ValueError("completed score identity/coverage drift")
    ids = contract["decision_ids"]
    for row in rows:
        source = expected[row["id"]]
        metadata = source.get("metadata", {})
        if (
            row["label"] != source.get("label", metadata.get("ground_truth"))
            or any(row[key] != value for key, value in metadata.items())
            or row["contract_sha256"] != contract["sha256"]
        ):
            raise ValueError("completed score metadata drift")
        raw = row["raw_decision_logprobs"]
        if any(
            str(token) not in raw or not math.isfinite(raw[str(token)]) for token in ids
        ):
            raise ValueError("completed score raw logprobs invalid")
        score = normalized_binary_probability(raw[str(ids[0])], raw[str(ids[1])])
        if not math.isfinite(row["score"]) or abs(row["score"] - score) > 1e-12:
            raise ValueError("completed score disagrees with raw logprobs")
    return rows


def predictions_for(
    llm: Any,
    sampling: Any,
    request: Any,
    tokenizer: Any,
    rows: list[dict],
    surface: str,
    path: Path,
    ident: dict,
    cell: str,
    *,
    engine: dict,
    status_dir: Path,
) -> list[dict]:
    ids = decision_token_ids(tokenizer, list(surface))
    prompts = [rendered(tokenizer, row, surface) for row in rows]
    encoded = [tokenizer.encode(p, add_special_tokens=False) for p in prompts]
    lengths = list(map(len, encoded))
    if max(lengths) >= engine["max_model_len"]:
        raise ValueError("evaluation input would truncate")
    expected = {str(r.get("id", r.get("index"))): r for r in rows}
    prompt_hashes = {
        str(r.get("id", r.get("index"))): digest(p)
        for r, p in zip(rows, prompts, strict=True)
    }
    if len(expected) != len(rows):
        raise ValueError("duplicate input identity")
    contract = digest(
        json.dumps(
            {
                "identity": ident,
                "cell": cell,
                "surface": surface,
                "input_prompt_hashes": list(map(digest, prompts)),
                "decision_ids": ids,
                "raw_logprobs": True,
            },
            sort_keys=True,
        )
    )
    saved = read_rows(path) if path.exists() else []
    done = {}
    for row in saved:
        key = str(row["id"])
        if (
            key not in expected
            or key in done
            or row["contract_sha256"] != contract
            or row["label"]
            != expected[key].get(
                "label", expected[key].get("metadata", {}).get("ground_truth")
            )
            or not math.isfinite(row["score"])
            or row["prompt_sha256"] != prompt_hashes[key]
        ):
            raise ValueError("cached prediction identity drift")
        raw = row["raw_decision_logprobs"]
        if any(str(t) not in raw or not math.isfinite(raw[str(t)]) for t in ids):
            raise ValueError("cached decision logprobs invalid")
        score = normalized_binary_probability(raw[str(ids[0])], raw[str(ids[1])])
        if abs(score - row["score"]) > 1e-12:
            raise ValueError("cached score disagrees with logprobs")
        done[key] = row
    write_json(
        path.with_suffix(".contract.json"),
        {
            "sha256": contract,
            "identity": ident,
            "surface": surface,
            "rows": len(rows),
            "decision_ids": ids,
            "maximum_tokens": max(lengths),
            "total_tokens": sum(lengths),
            "truncated": 0,
        },
    )
    remaining = [
        i for i, r in enumerate(rows) if str(r.get("id", r.get("index"))) not in done
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as handle:
        for offset in range(0, len(remaining), engine["batch_rows"]):
            indices = remaining[offset : offset + engine["batch_rows"]]
            outputs = llm.generate(
                [{"prompt_token_ids": encoded[i]} for i in indices],
                sampling,
                lora_request=request,
                use_tqdm=False,
            )
            for i, out in zip(indices, outputs, strict=True):
                if (
                    len(out.outputs[0].token_ids) != 1
                    or out.outputs[0].token_ids[0] not in ids
                    or len(out.prompt_token_ids) != lengths[i]
                ):
                    raise ValueError(
                        "one-token response or no-truncation contract failed"
                    )
                raw = {
                    str(k): logprob_value(v)
                    for k, v in out.outputs[0].logprobs[0].items()
                }
                if any(
                    str(token) not in raw or not math.isfinite(raw[str(token)])
                    for token in ids
                ):
                    raise ValueError("missing/nonfinite decision logprobs")
                score = normalized_binary_probability(
                    raw[str(ids[0])], raw[str(ids[1])]
                )
                original = rows[i]
                metadata = original.get("metadata", {})
                row = {
                    **{
                        k: v
                        for k, v in original.items()
                        if k
                        not in (
                            "prompt",
                            "student_prompt",
                            "student_target",
                            "metadata",
                        )
                    },
                    **metadata,
                    "id": str(original.get("id", original.get("index"))),
                    "label": original.get("label", metadata.get("ground_truth")),
                    "score": score,
                    "prompt_tokens": lengths[i],
                    "completion_tokens": 1,
                    "generated_token_id": out.outputs[0].token_ids[0],
                    "p_B" if surface == "AB" else "p_harmful": score,
                    "raw_decision_logprobs": {
                        str(token): raw[str(token)] for token in ids
                    },
                    "raw_returned_logprobs": raw,
                    "prompt_sha256": digest(prompts[i]),
                    "contract_sha256": contract,
                    "timestamp_unix": time.time(),
                }
                handle.write(json.dumps(row, allow_nan=False) + "\n")
                done[row["id"]] = row
            handle.flush()
            os.fsync(handle.fileno())
            write_json(
                status_dir / "status.json",
                {
                    "stage": "scoring",
                    "cell": cell,
                    "rows": len(done),
                    "total": len(rows),
                },
            )
            print(f"evaluation_progress {cell} {len(done)}/{len(rows)}", flush=True)
    result = [done[str(r.get("id", r.get("index")))] for r in rows]
    write_json(
        path.with_suffix(".complete.json"),
        {
            "rows": len(result),
            "passed": True,
            "sha256": file_hash(path),
            "contract_sha256": contract,
        },
    )
    return result
