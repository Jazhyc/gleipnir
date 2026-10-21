"""Exploratory run sb03: reasoning before the binary decision (stock BF16 vLLM).

One offline vLLM engine per model serves the released merged BF16 checkpoint.
It scores three arms on the frozen manifests, greedily and once:

- ``immediate``: the sb02 ControlArena prompt, logits right after
  ``Prediction:``. This is the same-engine baseline.
- ``visible``: native thinking off; a concise analysis ending at
  ``\\nPrediction:`` (at most 4,096 tokens), then the logits.
- ``thinking``: native thinking on; thinking up to ``</think>`` (at most
  8,192 tokens), then ``\\n\\nPrediction:`` and the logits.

A capped or unterminated generation is closed at the decision boundary and
recorded as a recovery, as in the tool-trajectory reasoning harness. Each arm
writes ``results/sleightbench_eval/<name>_<arm>/<model>.jsonl`` plus a
completion receipt, so ``run.py report`` reads it unchanged.
"""

from __future__ import annotations

import argparse
import json
import math
from typing import Any

from experiments.release_serving.targets import TARGETS
from experiments.sleightbench_eval.run import CONFIG, RESULTS, frozen_prompts
from gleipnir.data.monitoring import write_json

ARMS = {
    "immediate": "control_arena",
    "visible": "control_arena_visible",
    "thinking": "control_arena_thinking",
}
DECISION = {"visible": "\nPrediction:", "thinking": "\n\nPrediction:"}


def close_generation(arm: str, text: str, finish_reason: str) -> tuple[str, str]:
    """Return (continuation ending at the decision boundary, recovery)."""
    stop = CONFIG["reasoning"][arm]["stop"]
    if arm == "visible":
        if text.endswith(stop) and text.count(stop) == 1:
            return text, "none"
        recovery = "length_cap" if finish_reason == "length" else "missing_boundary"
        return text.rstrip() + DECISION[arm], recovery
    if text.endswith(stop):
        return text + DECISION[arm], "none"
    recovery = "length_cap" if finish_reason == "length" else "missing_boundary"
    return text.rstrip() + "\n" + stop + DECISION[arm], recovery


def binary_score(output: Any, ids: list[int]) -> tuple[float, list[float]]:
    entries = output.outputs[0].logprobs[0]
    logprobs = [float(entries[i].logprob) for i in ids]
    if not all(math.isfinite(x) for x in logprobs):
        raise ValueError(f"nonfinite decision logprobs: {logprobs}")
    high = max(logprobs)
    weights = [math.exp(x - high) for x in logprobs]
    return weights[1] / sum(weights), logprobs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", choices=sorted(CONFIG["models"]))
    parser.add_argument("--name", default="sb03")
    args = parser.parse_args()

    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    spec = CONFIG["reasoning"]
    path = str(TARGETS[args.model].merged_path)
    tokenizer = AutoTokenizer.from_pretrained(path)
    ids = [tokenizer.convert_tokens_to_ids(t) for t in ("0", "1")]
    if ids != [15, 16]:
        raise ValueError(f"decision token ids changed: {ids}")
    outputs = {arm: RESULTS / f"{args.name}_{arm}" / args.model for arm in ARMS}
    for arm, out in outputs.items():
        if out.with_suffix(".complete.json").exists():
            raise ValueError(f"{arm} already scored for {args.model}; no reruns")
    engine = spec["engine"]
    llm = LLM(
        model=path,
        dtype=engine["dtype"],
        max_model_len=engine["max_model_len"],
        max_num_seqs=engine["max_num_seqs"],
        max_num_batched_tokens=engine["max_num_batched_tokens"],
        enable_prefix_caching=engine["enable_prefix_caching"],
        gpu_memory_utilization=engine["gpu_memory_utilization"],
        language_model_only=True,
        seed=spec["seed"],
        hf_overrides={"text_config": {"mamba_ssm_dtype": "float32"}},
    )
    margin = SamplingParams(
        max_tokens=1,
        temperature=0.0,
        logprobs=len(ids),
        logprob_token_ids=ids,
        allowed_token_ids=ids,
    )
    for arm, variant in ARMS.items():
        rows = frozen_prompts(args.model, variant)
        prompts = [r["prompt"] for r in rows]
        generations: list[dict[str, Any]] = [{} for _ in rows]
        if arm != "immediate":
            sampling = SamplingParams(
                max_tokens=spec[arm]["max_tokens"],
                temperature=spec["temperature"],
                seed=spec["seed"],
                stop=[spec[arm]["stop"]],
                include_stop_str_in_output=True,
            )
            for index, output in enumerate(llm.generate(prompts, sampling)):
                generated = output.outputs[0]
                continuation, recovery = close_generation(
                    arm, generated.text, str(generated.finish_reason)
                )
                generations[index] = {
                    "generation": generated.text,
                    "generated_tokens": len(generated.token_ids),
                    "finish_reason": str(generated.finish_reason),
                    "recovery": recovery,
                }
                prompts[index] += continuation
        scored = llm.generate(prompts, margin)
        out = outputs[arm]
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.with_suffix(".jsonl").open("w") as handle:
            for row, extra, output in zip(rows, generations, scored, strict=True):
                score, logprobs = binary_score(output, ids)
                handle.write(
                    json.dumps(
                        {
                            "id": row["id"],
                            "prompt_sha256": row["prompt_sha256"],
                            "prompt_tokens": row["prompt_tokens"],
                            "score": score,
                            "logprobs": logprobs,
                            **extra,
                        }
                    )
                    + "\n"
                )
        recoveries = [g.get("recovery", "none") for g in generations]
        write_json(
            out.with_suffix(".complete.json"),
            {
                "model": args.model,
                "arm": arm,
                "variant": variant,
                "rows": len(rows),
                "recoveries": {r: recoveries.count(r) for r in set(recoveries)},
                "engine": {"checkpoint": path, **engine},
            },
        )
        print(f"scored {args.model} {arm} rows={len(rows)}", flush=True)


if __name__ == "__main__":
    main()
