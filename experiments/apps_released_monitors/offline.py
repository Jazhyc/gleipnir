"""Stock BF16 vLLM scoring of the frozen APPS prompts (reference arms).

One offline engine per arm, unquantized BF16, logits for ``0``/``1`` right
after ``Prediction:`` as in SleightBench ``sb04``. ``base`` is the unadapted
model at its pinned revision; ``release`` is the released merged BF16
checkpoint, giving a same-engine control for the adapter.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from experiments.apps_released_monitors.run import CONFIG, RESULTS, frozen
from gleipnir.data.monitoring import write_json, write_rows

OFFLINE = json.loads(Path(__file__).with_name("offline.json").read_text())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("arm", choices=sorted(OFFLINE["arms"]))
    parser.add_argument("--name", required=True)
    args = parser.parse_args()

    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    spec = OFFLINE["arms"][args.arm]
    engine = OFFLINE["engine"]
    out = RESULTS / args.name / args.arm
    if (out / "complete.json").exists():
        raise ValueError(f"{args.arm} already scored in {args.name}; no reruns")
    if spec["checkpoint"] == "base":
        path, revision = spec["id"], spec["revision"]
    else:
        from experiments.release_serving.targets import TARGETS

        path, revision = str(TARGETS[spec["model"]].merged_path), None
    tokenizer = AutoTokenizer.from_pretrained(path, revision=revision)
    ids = [tokenizer.convert_tokens_to_ids(t) for t in ("0", "1")]
    if ids != [15, 16]:
        raise ValueError(f"decision token ids changed: {ids}")
    llm = LLM(
        model=path,
        revision=revision,
        tokenizer_revision=revision,
        language_model_only=True,
        seed=0,
        hf_overrides={"text_config": {"mamba_ssm_dtype": "float32"}},
        **engine,
    )
    sampling = SamplingParams(
        max_tokens=1,
        temperature=0.0,
        logprobs=len(ids),
        logprob_token_ids=ids,
        allowed_token_ids=ids,
    )
    out.mkdir(parents=True, exist_ok=True)
    files = {}
    for split in CONFIG["populations"]:
        rows = frozen(spec["model"], split)
        scored = []
        outputs = llm.generate([r["prompt"] for r in rows], sampling)
        for row, output in zip(rows, outputs, strict=True):
            entries = output.outputs[0].logprobs[0]
            logprobs = [float(entries[i].logprob) for i in ids]
            if not all(math.isfinite(x) for x in logprobs):
                raise ValueError(f"nonfinite decision logprobs: {row['id']}")
            high = max(logprobs)
            weights = [math.exp(x - high) for x in logprobs]
            scored.append(
                {
                    "id": row["id"],
                    "prompt_sha256": row["prompt_sha256"],
                    "prompt_tokens": row["prompt_tokens"],
                    "score": weights[1] / sum(weights),
                    "logprobs": logprobs,
                    **row["metadata"],
                }
            )
        write_rows(out / f"{split}.jsonl", scored)
        print(f"scored {args.arm} {split} {len(scored)}", flush=True)
        from gleipnir.data.monitoring import file_hash

        files[split] = file_hash(out / f"{split}.jsonl")
    write_json(
        out / "complete.json",
        {
            "arm": args.arm,
            "engine": {"checkpoint": path, "revision": revision, **engine},
            "files_sha256": files,
        },
    )


if __name__ == "__main__":
    main()
