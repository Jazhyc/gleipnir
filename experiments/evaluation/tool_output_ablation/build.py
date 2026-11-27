"""Write the stripped OOD workloads of both released models.

Each released model keeps the workload of its recorded OOD run (the 4B with
the compact instruction, the 27B with the full specification); only the
trajectory envelope changes. Prompt token counts come from the Qwen3.5
tokenizer, which can undercount the server by a few tokens on some emoji
and combining-mark sequences; ``recount.py`` fixes the counts on the Pod.

``PYTHONPATH=src:. python -m experiments.evaluation.tool_output_ablation.build``
"""

from __future__ import annotations

import hashlib
import json

from experiments.evaluation.tool_output_ablation.count import CONFIG, ROOT, tokenizer
from experiments.evaluation.tool_output_ablation.strip import strip_prompt


def main() -> None:
    tok = tokenizer()
    out_dir = ROOT / CONFIG["stripped_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = {}
    for model, spec in CONFIG["models"].items():
        source = ROOT / spec["workload"]
        rows = [json.loads(line) for line in source.open()]
        check = [
            len(e.ids)
            for e in tok.encode_batch(
                [r["prompt"] for r in rows[:200]], add_special_tokens=False
            )
        ]
        if check != [r["prompt_tokens"] for r in rows[:200]]:
            raise ValueError(f"{model}: tokenizer does not reproduce recorded counts")
        prompts = [strip_prompt(r["prompt"]) for r in rows]
        counts = [
            len(e.ids) for e in tok.encode_batch(prompts, add_special_tokens=False)
        ]
        path = out_dir / f"{model}_ood.jsonl"
        with path.open("w") as handle:
            for row, prompt, tokens in zip(rows, prompts, counts, strict=True):
                record = {
                    "id": row["id"],
                    "prompt": prompt,
                    "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
                    "prompt_tokens": tokens,
                }
                handle.write(json.dumps(record) + "\n")
        manifest[model] = {
            "source": spec["workload"],
            "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
            "path": str(path.relative_to(ROOT)),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "rows": len(rows),
            "prompt_tokens": sum(counts),
            "source_prompt_tokens": sum(r["prompt_tokens"] for r in rows),
        }
        print(model, json.dumps(manifest[model]))
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
