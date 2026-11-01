"""Transformers + PEFT scoring of released monitors, one forward pass per row.

This is the bounded Transformers path from the inference guide: exact model
card arithmetic, gradient access for white-box attacks, and no serving layout.
Bulk evaluation can move to the cached vLLM path once scores agree.
"""

from __future__ import annotations

import importlib.metadata
import json
import time
from pathlib import Path
from typing import Any

from gleipnir.adversarial.contract import MAX_TOKENS, render
from gleipnir.adversarial.eval_sets import group, label
from gleipnir.adversarial.monitors import MONITORS
from gleipnir.evaluation.binary import binary_token_ids
from gleipnir.evaluation.probabilities import normalized_binary_probability

PACKAGES = (
    "torch",
    "transformers",
    "peft",
    "accelerate",
    "flash-linear-attention",
    "fla-core",
    "causal-conv1d",
)


class TooLong(ValueError):
    pass


class Scorer:
    def __init__(self, monitor: str, device: str = "cuda", dtype: str = "bfloat16"):
        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.monitor = monitor
        self.spec = MONITORS[monitor]
        self.device = device
        self.dtype = dtype
        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.spec.base, revision=self.spec.base_revision
        )
        base = AutoModelForCausalLM.from_pretrained(
            self.spec.base,
            revision=self.spec.base_revision,
            dtype=getattr(torch, dtype),
            device_map={"": device},
            attn_implementation="sdpa",
        )
        if self.spec.adapter is None:
            self.model = base.eval()
        else:
            self.model = PeftModel.from_pretrained(
                base, self.spec.adapter, revision=self.spec.adapter_revision
            ).eval()
        self.token_ids = binary_token_ids(self.tokenizer)

    def encode(self, user_prompt: str) -> list[int]:
        ids = self.tokenizer.encode(
            render(self.tokenizer, user_prompt), add_special_tokens=False
        )
        if len(ids) > MAX_TOKENS:
            raise TooLong(f"{len(ids)} tokens, limit {MAX_TOKENS}")
        return ids

    def score(self, user_prompt: str, adapter: bool = True) -> dict[str, Any]:
        """P("1") over the 0/1 logits at the decision position, plus raw logits.

        `adapter=False` scores the unadapted base, for the adapter-effect check.
        Monitors loaded without an adapter ignore it.
        """
        torch = self.torch
        ids = self.encode(user_prompt)
        with torch.inference_mode():
            inputs = torch.tensor([ids], device=self.device)
            if adapter or self.spec.adapter is None:
                logits = self.model(input_ids=inputs, logits_to_keep=1).logits
            else:
                with self.model.disable_adapter():
                    logits = self.model(input_ids=inputs, logits_to_keep=1).logits
            zero, one = logits[0, -1, self.token_ids].float().tolist()
        return {
            "score": normalized_binary_probability(zero, one),
            "logit_0": zero,
            "logit_1": one,
            "tokens": len(ids),
        }

    def identity(self) -> dict[str, Any]:
        from transformers.utils.import_utils import is_flash_linear_attention_available

        versions = {}
        for name in PACKAGES:
            try:
                versions[name] = importlib.metadata.version(name)
            except importlib.metadata.PackageNotFoundError:
                versions[name] = None
        return {
            "monitor": self.monitor,
            **self.spec._asdict(),
            "dtype": self.dtype,
            "device": self.torch.cuda.get_device_name()
            if self.device.startswith("cuda")
            else self.device,
            "packages": versions,
            "fla_available": is_flash_linear_attention_available(),
            "token_ids": list(self.token_ids),
        }


def row_record(row: dict[str, Any]) -> dict[str, Any]:
    metadata = row["metadata"]
    record = {"id": str(row["id"]), "group": group(row), "label": label(row)}
    for key in ("task_id", "parent_id"):
        if key in metadata:
            record[key] = metadata[key]
    return record


def score_rows(scorer: Scorer, rows: list[dict[str, Any]], path: Path) -> None:
    """Append one record per row to `path`, skipping ids already there."""
    done = set()
    if path.exists():
        done = {json.loads(line)["id"] for line in path.open()}
    todo = [row for row in rows if str(row["id"]) not in done]
    started = time.time()
    tokens = 0
    with path.open("a") as handle:
        for n, row in enumerate(todo, 1):
            record = row_record(row)
            try:
                record.update(scorer.score(row["prompt"]), status="ok")
                tokens += record["tokens"]
            except TooLong as error:
                record.update(status="too_long", error=str(error))
            handle.write(json.dumps(record) + "\n")
            handle.flush()
            if n % 50 == 0 or n == len(todo):
                elapsed = time.time() - started
                print(
                    f"{path.parent.name}: {n}/{len(todo)} rows, "
                    f"{n / elapsed:.2f} rows/s, {tokens / elapsed:,.0f} prompt tok/s",
                    flush=True,
                )
