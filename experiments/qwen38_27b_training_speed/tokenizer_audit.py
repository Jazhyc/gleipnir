"""Check that the 27B tokenizer reproduces the 4B campaign's token streams.

Encodes the frozen ID workload prompts and renders/encodes every training
student prompt (chat template, thinking on and off) with both tokenizers, and
reports any difference in rendered text or token ids.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from transformers import AutoTokenizer

MODELS = {
    "4b": ("Qwen/Qwen3.5-4B", "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"),
    "27b": ("Qwen/Qwen3.8-27B", "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0"),
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workload", type=Path, required=True)
    parser.add_argument("--training", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    tok = {
        k: AutoTokenizer.from_pretrained(m, revision=r) for k, (m, r) in MODELS.items()
    }
    report = {"workload": {"rows": 0, "id_mismatch": 0, "count_mismatch": 0}}
    for row in json.loads(args.workload.read_text()):
        ids = {
            k: t(row["prompt"], add_special_tokens=False)["input_ids"]
            for k, t in tok.items()
        }
        report["workload"]["rows"] += 1
        report["workload"]["id_mismatch"] += ids["4b"] != ids["27b"]
        report["workload"]["count_mismatch"] += len(ids["27b"]) != row["prompt_tokens"]
    report["training"] = {"rows": 0}
    for thinking in (False, True):
        key = f"thinking_{thinking}"
        report["training"][key] = {"text_mismatch": 0, "id_mismatch": 0}
    for line in args.training.read_text().splitlines():
        row = json.loads(line)
        report["training"]["rows"] += 1
        messages = [{"role": "user", "content": row["student_prompt"]}]
        for thinking in (False, True):
            texts = {
                k: t.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=thinking,
                )
                for k, t in tok.items()
            }
            entry = report["training"][f"thinking_{thinking}"]
            entry["text_mismatch"] += texts["4b"] != texts["27b"]
            entry["id_mismatch"] += (
                tok["4b"](
                    texts["4b"] + row["student_target"], add_special_tokens=False
                )["input_ids"]
                != tok["27b"](
                    texts["4b"] + row["student_target"], add_special_tokens=False
                )["input_ids"]
            )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
