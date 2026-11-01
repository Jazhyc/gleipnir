"""Append-only intent/result receipts; never silently repeat an uncertain call."""

from __future__ import annotations

import hashlib
import json
import math
import os
import time
from collections.abc import Callable
from pathlib import Path


class Journal:
    def __init__(self, path: Path, identity: dict, cap: int):
        self.path, self.cap = path, cap
        self.results: dict[str, dict] = {}
        self.intents: dict[str, dict] = {}
        path.parent.mkdir(parents=True, exist_ok=True)
        rows = (
            [json.loads(line) for line in path.read_text().splitlines()]
            if path.exists()
            else []
        )
        if rows:
            if rows[0] != {"type": "identity", "value": identity}:
                raise ValueError("run identity changed")
            for row in rows[1:]:
                if row["type"] not in ("intent", "result"):
                    raise ValueError("unknown journal record type")
                if row["type"] == "result" and row["id"] not in self.intents:
                    raise ValueError("result precedes its intent")
                target = self.intents if row["type"] == "intent" else self.results
                if row["id"] in target:
                    raise ValueError("duplicate journal record")
                target[row["id"]] = row
            if self.intents.keys() != self.results.keys():
                raise ValueError(
                    "unfinished call: preserve receipts and reconcile before resuming"
                )
        else:
            self.append({"type": "identity", "value": identity})

    def append(self, row: dict) -> None:
        with self.path.open("a") as handle:
            handle.write(json.dumps(row) + "\n")
            handle.flush()
            os.fsync(handle.fileno())

    def score(self, call_id: str, prompt: str, fn: Callable, metadata: dict) -> dict:
        prompt_hash = hashlib.sha256(prompt.encode()).hexdigest()
        if call_id in self.results:
            previous = self.intents[call_id]
            if (
                previous["prompt_sha256"] != prompt_hash
                or previous["metadata"] != metadata
            ):
                raise ValueError("cached call input changed")
            result = self.results[call_id]
            if result["status"] != "ok":
                raise RuntimeError("previous call failed; no automatic retries")
            return result["value"]
        if len(self.intents) >= self.cap:
            raise RuntimeError("total query cap exhausted")
        intent = {
            "type": "intent",
            "id": call_id,
            "prompt_sha256": prompt_hash,
            "metadata": metadata,
            "timestamp": time.time(),
        }
        self.append(intent)
        self.intents[call_id] = intent
        start = time.monotonic()
        try:
            value = fn(prompt)
            if not math.isfinite(value["score"]) or not 0 <= value["score"] <= 1:
                raise ValueError("invalid monitor score")
            row = {"type": "result", "id": call_id, "status": "ok", "value": value}
        except BaseException as error:
            row = {
                "type": "result",
                "id": call_id,
                "status": "error",
                "error_type": type(error).__name__,
            }
            self.append({**row, "seconds": time.monotonic() - start})
            self.results[call_id] = row
            raise
        row["seconds"] = time.monotonic() - start
        self.append(row)
        self.results[call_id] = row
        return value
