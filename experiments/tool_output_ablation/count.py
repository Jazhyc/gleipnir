"""Token counts of the OOD suite with and without tool outputs.

Counts the trajectory between the envelope tags with the Qwen3.5 tokenizer
(shared by both Gleipnir monitors), before and after ``strip_outputs``. The
instruction and chat template add a fixed amount per prompt and are left out.

``PYTHONPATH=src:. python -m experiments.tool_output_ablation.count`` writes
``results/tool_output_ablation/token_counts.json`` and prints a table.
"""

from __future__ import annotations

import json
import statistics
from collections import defaultdict
from pathlib import Path

from tokenizers import Tokenizer

from experiments.tool_output_ablation.strip import CLOSE, OPEN, strip_outputs

ROOT = Path(__file__).resolve().parents[2]
CONFIG = json.loads((Path(__file__).parent / "config.json").read_text())
OUTPUT = ROOT / "results/tool_output_ablation/token_counts.json"


def tokenizer() -> Tokenizer:
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(CONFIG["tokenizer"], "tokenizer.json")
    return Tokenizer.from_file(path)


def rows():
    with (ROOT / CONFIG["ood_prompts"]).open() as handle:
        for line in handle:
            row = json.loads(line)
            prompt = row["prompt"]
            body = prompt[prompt.index(OPEN) + len(OPEN) : prompt.index(CLOSE)]
            meta = row["metadata"]
            yield meta["source_dataset"], int(meta["ground_truth"]), body


def summarize(full: list[int], stripped: list[int]) -> dict:
    kept = [s / f for f, s in zip(full, stripped, strict=True) if f]
    return {
        "rows": len(full),
        "tokens_full": sum(full),
        "tokens_stripped": sum(stripped),
        "kept_fraction": sum(stripped) / sum(full),
        "kept_fraction_median_row": statistics.median(kept),
        "median_full": statistics.median(full),
        "median_stripped": statistics.median(stripped),
        "max_full": max(full),
        "max_stripped": max(stripped),
    }


def main() -> None:
    tok = tokenizer()
    data = list(rows())
    bodies = [b for _, _, b in data]
    full = [len(e.ids) for e in tok.encode_batch(bodies, add_special_tokens=False)]
    stripped_bodies = [strip_outputs(b) for b in bodies]
    encoded = tok.encode_batch(stripped_bodies, add_special_tokens=False)
    stripped = [len(e.ids) for e in encoded]
    groups: dict[str, tuple[list[int], list[int]]] = defaultdict(lambda: ([], []))
    for (source, label, _), f, s in zip(data, full, stripped, strict=True):
        for key in ("all", source, f"label_{label}"):
            groups[key][0].append(f)
            groups[key][1].append(s)
    out = {k: summarize(*v) for k, v in sorted(groups.items())}
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(out, indent=2) + "\n")
    for key, s in out.items():
        print(
            f"{key:18} n={s['rows']:5}  tokens {s['tokens_full'] / 1e6:6.2f}M ->"
            f" {s['tokens_stripped'] / 1e6:6.2f}M  kept {s['kept_fraction']:.2f}"
            f" (median row {s['kept_fraction_median_row']:.2f})"
            f"  median {s['median_full']:7,.0f} -> {s['median_stripped']:7,.0f}"
            f"  max {s['max_full']:7,} -> {s['max_stripped']:7,}"
        )
    print(f"wrote {OUTPUT}")


if __name__ == "__main__":
    main()
