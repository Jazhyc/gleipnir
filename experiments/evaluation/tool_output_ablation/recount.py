"""Rewrite ``prompt_tokens`` of the stripped workloads with the server encoder.

The release server tokenizes with Gigatoken and rejects a row whose count
differs from the client's. The standalone ``tokenizers`` count used by
``build.py`` undercounts 49 stripped prompts per model by one to four tokens
(same ``tokenizer.json``, SHA-256 ``5f9e4d49…``): Transformers replaces its
pre-tokenizer with one built from ``tokenizer_config.json``, which splits
some emoji and combining-mark sequences differently. Run this with the image's runtime
on the Pod after ``build.py``:

``/tmp/gleipnir-vllm031-runtime/bin/python -m
experiments.evaluation.tool_output_ablation.recount``
"""

from __future__ import annotations

import hashlib
import json

from experiments.evaluation.tool_output_ablation.count import CONFIG, ROOT

MERGED = "/tmp/gleipnir-merged/qad-v02-optimized01"


def main() -> None:
    import gigatoken
    from transformers import AutoTokenizer

    native = gigatoken.Tokenizer(AutoTokenizer.from_pretrained(MERGED))
    for model in CONFIG["models"]:
        path = ROOT / CONFIG["stripped_dir"] / f"{model}_ood.jsonl"
        rows = [json.loads(line) for line in path.open()]
        changed = 0
        for row in rows:
            count = len(native.encode(row["prompt"]))
            if count != row["prompt_tokens"]:
                changed += 1
                row["prompt_tokens"] = count
        with path.open("w") as handle:
            for row in rows:
                handle.write(json.dumps(row) + "\n")
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        tokens = sum(r["prompt_tokens"] for r in rows)
        print(model, "changed", changed, "tokens", tokens, "sha256", digest)


if __name__ == "__main__":
    main()
