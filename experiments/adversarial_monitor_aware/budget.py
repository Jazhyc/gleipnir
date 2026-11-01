"""Tinker spend cap and progress ledger for the lean main run (Amendment 2).

The runner cannot see Tinker charges, so spend is estimated from token counts:
the list price of every token so far (prefill including cached, plus sampled),
scaled by the console charge observed for the smoke run over the smoke run's
list price. Each model call appends one line to `usage.jsonl` in the run
folder, so the estimate covers every pass of a resumed run. When the estimate
reaches the cap, the hook writes `CAP_REACHED` and interrupts the eval.

The hooks are off unless MA_RUN_DIR is set (only `run.py lean` sets it).
"""

from __future__ import annotations

import json
import os
import signal
import time
from pathlib import Path
from typing import Any

from inspect_ai.hooks import Hooks, ModelUsageData, SampleEnd, hooks

RUN_DIR_ENV = "MA_RUN_DIR"
LEDGER = "usage.jsonl"
PROGRESS = "progress.jsonl"
CAP_FILE = "CAP_REACHED"


def calibration(cap: dict[str, Any]) -> float:
    return cap["observed_usd"] / cap["smoke_list_usd_cache_billed"]


def estimate_usd(
    prefill: int, sampled: int, prices: dict[str, Any], cap: dict[str, Any]
) -> float:
    listed = prefill * prices["prefill"] / 1e6 + sampled * prices["sampled"] / 1e6
    return calibration(cap) * listed


def ledger_tokens(run_dir: Path) -> tuple[int, int]:
    """(prefill, sampled) summed over every logged model call."""
    path = run_dir / LEDGER
    prefill = sampled = 0
    if path.exists():
        for line in path.read_text().splitlines():
            if line.strip():
                row = json.loads(line)
                prefill += row["prefill"]
                sampled += row["sampled"]
    return prefill, sampled


def spent(run_dir: Path, config: dict[str, Any]) -> float:
    return estimate_usd(
        *ledger_tokens(run_dir),
        config["usage_prices_usd_per_million"],
        config["lean_main"]["tinker_cap"],
    )


@hooks(name="monitor_aware_tinker_cap", description="Tinker spend cap and progress.")
class TinkerCap(Hooks):
    def __init__(self) -> None:
        self.prefill: int | None = None
        self.sampled = 0
        self.stopped = False

    def enabled(self) -> bool:
        return bool(os.environ.get(RUN_DIR_ENV))

    async def on_model_usage(self, data: ModelUsageData) -> None:
        from experiments.adversarial_monitor_aware.run import CONFIG

        run_dir = Path(os.environ[RUN_DIR_ENV])
        if self.prefill is None:
            self.prefill, self.sampled = ledger_tokens(run_dir)
        u = data.usage
        prefill = (
            u.input_tokens
            + (u.input_tokens_cache_read or 0)
            + (u.input_tokens_cache_write or 0)
        )
        with (run_dir / LEDGER).open("a") as f:
            f.write(
                json.dumps(
                    {
                        "t": time.time(),
                        "prefill": prefill,
                        "prefill_cached": u.input_tokens_cache_read or 0,
                        "sampled": u.output_tokens,
                    }
                )
                + "\n"
            )
        self.prefill += prefill
        self.sampled += u.output_tokens
        cap = CONFIG["lean_main"]["tinker_cap"]
        usd = estimate_usd(
            self.prefill, self.sampled, CONFIG["usage_prices_usd_per_million"], cap
        )
        if usd >= cap["usd"] and not self.stopped:
            self.stopped = True
            (run_dir / CAP_FILE).write_text(f"{usd:.2f} of {cap['usd']}\n")
            os.kill(os.getpid(), signal.SIGINT)

    async def on_sample_end(self, data: SampleEnd) -> None:
        run_dir = Path(os.environ[RUN_DIR_ENV])
        error = data.sample.error.message if data.sample.error else None
        with (run_dir / PROGRESS).open("a") as f:
            f.write(
                json.dumps(
                    {
                        "t": time.time(),
                        "id": data.sample.id,
                        "error": error and error[:200],
                    }
                )
                + "\n"
            )
