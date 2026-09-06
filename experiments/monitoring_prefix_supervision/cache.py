"""Full prefix annotation using an already-preflighted persistent engine."""

import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any

from gleipnir.prefix_cache import binary_score, validate_resume


def run_cache(
    llm: Any,
    tokenizer: Any,
    sampling: Any,
    token_ids: list[int],
    grouped: dict,
    source: Path,
    instruction: str,
    config: dict,
    manifest: dict,
    output: Path,
    canary: dict,
) -> None:
    """Interleave eight parent trajectories, advancing one prefix per parent."""
    if not canary["passed"]:
        raise ValueError("cache preflight did not pass")
    output.mkdir(parents=True, exist_ok=True)
    contract = {
        "model": config["model"],
        "manifest": manifest,
        "prompt": {
            "instruction_sha256": manifest["instruction_sha256"],
            "assistant_suffix": config["prompt"]["assistant_suffix"],
            "decision_prefix": "Prediction:",
            "enable_thinking": False,
        },
        "sampling": {
            "temperature": 0,
            "max_tokens": 1,
            "decision_token_ids": token_ids,
        },
        "engine": {
            "tp": 2,
            "dtype": "bfloat16",
            "quantization": "fp8",
            "prefix_caching": True,
            "mamba_cache_mode": "all",
            "gdn_prefill_backend": "triton",
            "max_model_len": 32768,
            "max_num_seqs": 8,
            "max_num_batched_tokens": 8192,
            "gpu_memory_utilization": 0.9,
            "seed": 20260906,
            "enforce_eager": bool(config.get("engine", {}).get("enforce_eager", False)),
        },
    }
    contract_json = json.dumps(contract, sort_keys=True)
    contract_hash = hashlib.sha256(contract_json.encode()).hexdigest()
    contract_path = output / "contract.json"
    if contract_path.exists() and json.loads(contract_path.read_text()) != contract:
        raise ValueError("existing cache contract differs")
    contract_path.write_text(contract_json + "\n")
    references = {r["id"]: r for rows in grouped.values() for r in rows}
    cache_path = output / "logits.jsonl"
    completed = validate_resume(cache_path, references, contract_hash)
    trajectories = {}
    for line in source.open():
        row = json.loads(line)
        if row["prompt_id"] in grouped:
            trajectories[row["prompt_id"]] = (
                row["student_prompt"]
                .split("<agent_trajectory>\n", 1)[1]
                .rsplit("</agent_trajectory>", 1)[0]
            )
    parents = iter(grouped)
    active = []
    started = time.time()
    with cache_path.open("a") as stream:
        while True:
            while len(active) < 8:
                parent = next(parents, None)
                if parent is None:
                    break
                pending = iter(r for r in grouped[parent] if r["id"] not in completed)
                active.append(pending)
            batch, remaining = [], []
            for pending in active:
                row = next(pending, None)
                if row is not None:
                    batch.append(row)
                    remaining.append(pending)
            active = remaining
            if not batch:
                if not active:
                    # Refill once more if this cohort consisted of completed parents.
                    parent = next(parents, None)
                    if parent is None:
                        break
                    active.append(
                        iter(r for r in grouped[parent] if r["id"] not in completed)
                    )
                continue
            prompts = []
            for row in batch:
                text = trajectories[row["parent_prompt_id"]][: row["end_character"]]
                user = (
                    f"{instruction}\n<agent_trajectory>\n{text}"
                    + ("" if text.endswith("\n") else "\n")
                    + "</agent_trajectory>\n"
                )
                if (
                    hashlib.sha256(user.encode()).hexdigest()
                    != row["rendered_user_prompt_sha256"]
                ):
                    raise ValueError("full cache prompt drift")
                prompt = tokenizer.apply_chat_template(
                    [{"role": "user", "content": user}],
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=False,
                )
                if not prompt.endswith(config["prompt"]["assistant_suffix"]):
                    raise ValueError("assistant wrapper drift")
                prompt += "Prediction:"
                if len(tokenizer.encode(prompt, add_special_tokens=False)) >= 32768:
                    raise ValueError(f"prefix exceeds context: {row['id']}")
                prompts.append(prompt)
            try:
                outputs = llm.generate(prompts, sampling, use_tqdm=False)
            except Exception as error:
                with (output / "failures.jsonl").open("a") as failures:
                    failures.write(
                        json.dumps(
                            {
                                "ids": [r["id"] for r in batch],
                                "contract_sha256": contract_hash,
                                "timestamp_unix": time.time(),
                                "error": repr(error),
                            }
                        )
                        + "\n"
                    )
                raise
            for row, prompt, result in zip(batch, prompts, outputs, strict=True):
                if (
                    len(result.outputs) != 1
                    or len(result.outputs[0].token_ids) != 1
                    or result.outputs[0].token_ids[0] not in token_ids
                ):
                    raise ValueError(f"invalid one-token decision: {row['id']}")
                logprobs = result.outputs[0].logprobs[0]
                lp0, lp1 = [float(logprobs[t].logprob) for t in token_ids]
                record = {
                    **row,
                    "contract_sha256": contract_hash,
                    "logprob_0": lp0,
                    "logprob_1": lp1,
                    "score": binary_score(lp0, lp1),
                    "timestamp_unix": time.time(),
                    "prompt_tokens": len(result.prompt_token_ids),
                    "cached_tokens": result.num_cached_tokens,
                    "completion_tokens": len(result.outputs[0].token_ids),
                    "serving_prompt_sha256": hashlib.sha256(
                        prompt.encode()
                    ).hexdigest(),
                }
                stream.write(json.dumps(record) + "\n")
                completed.add(row["id"])
            stream.flush()
            os.fsync(stream.fileno())
            print(
                f"cache={len(completed)}/{len(references)} "
                f"elapsed={time.time() - started:.1f}",
                flush=True,
            )
    if completed != set(references):
        raise RuntimeError("incomplete prefix cache")
    (output / "complete.json").write_text(
        json.dumps(
            {
                "rows": len(completed),
                "contract_sha256": contract_hash,
                "elapsed_seconds_this_invocation": time.time() - started,
            },
            indent=2,
        )
        + "\n"
    )
