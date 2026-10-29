"""The selected optimized worker, serving activation probes after block 15.

Three changes, all applied after the weights load (and after the stack's own
precision and monitor-head audits) but before profiling and compilation:

- **Truncation.** vLLM's Qwen3.5 forward iterates
  ``islice(layers, start_layer, end_layer)``; lowering ``end_layer`` to 16
  skips blocks 16-31 while every block stays loaded, so per-layer counts,
  receipts and kernel bindings are unchanged.
- **Identity final norm.** The decoder then returns ``hidden + residual``,
  the post-block-15 residual stream the probes were trained on, instead of
  its RMS-normalized form.
- **Probe pooler** (when ``GLEIPNIR_PROBE_BANK`` is set). vLLM's chunk-aware
  ALL pooling gathers each request's token states across chunked-prefill
  steps; when a request finishes, its states (attention sink dropped) are
  scored by the full01 probe banks on the GPU. The classify output is one
  flat vector: for each variant (``fp8`` = the study's blockwise FP8 round
  trip, ``raw``), for each checkpoint epoch, every probe's logit in bank
  order. The backbone runs compiled; the pooler runs eagerly after it.
  Prompted rows start their kept span after the instruction prefix: with
  ``GLEIPNIR_PROBE_KEEP``, the pooler looks each request's start up by the
  SHA-256 of its token ids. ``GLEIPNIR_PROBE_DUMP`` saves the kept states
  (BF16) of the requests whose keys its ``keys.json`` lists, for the
  activation diagnostic.

The 27B worker (``truncate_worker_27b.py``) reuses these pieces.
"""

from __future__ import annotations

import json
import os
from collections.abc import Set
from pathlib import Path

import torch
from torch import nn

from experiments.b200_gdn_fast.worker import FastGdnWorker
from experiments.probe_optimized_serving.keys import token_key

# A constant, not an option: this file's hash joins the serving compile
# identity (start.py), so truncated and full-depth graphs never share an AOT
# compile-cache entry. vLLM's own cache key ignores these module changes.
END_LAYER = 16


class ResidualStream(nn.Module):
    """Stands in for the final norm: return the un-normalized residual stream."""

    def forward(self, hidden_states, residual=None):
        if residual is not None:
            hidden_states = hidden_states + residual
        return hidden_states, hidden_states


def single_probe(full, specs: list, name: str, config: dict, width: int):
    """Slice one head-type probe (not mean_linear or AlphaEvolve) out of a
    fused bank into its own one-probe bank, and check it scores identically."""
    from gleipnir.probes.architectures import WIDTH
    from gleipnir.probes.training import make_sweep

    index = [s.name for s in specs].index(name)
    spec = specs[index]
    if index not in full.deep_index or spec.architecture == "mean_mlp":
        raise ValueError(f"{name} is not a head-type probe")
    p = full.deep_index.index(index)
    q = full.groups[spec.architecture].index(p)
    rows = slice(
        len(full.linear_index) + p * WIDTH, len(full.linear_index) + (p + 1) * WIDTH
    )
    one = make_sweep(
        [spec.architecture],
        [spec.objective],
        [spec.seed],
        width,
        lr=config["lr"],
        weight_decay=config["weight_decay"],
        device="cuda",
    ).bank
    key = spec.architecture.replace("-", "_")
    state = {
        "first_weight": full.first_weight[rows],
        "first_bias": full.first_bias[rows],
        "second_weight": full.second_weight[p : p + 1],
        "second_bias": full.second_bias[p : p + 1],
    }
    for suffix in ("value", "query", "bias"):
        name_ = f"heads_params.{key}_{suffix}"
        if name_ in one.state_dict():
            state[name_] = full.heads_params[f"{key}_{suffix}"][q : q + 1]
    one.load_state_dict(state)
    one.eval()
    # FP32 (TF32 off), not the serving autocast: BF16 rounding differs between
    # the fused and single-probe GEMM shapes (by ~1e-3 at the 27B's width).
    x = torch.randn(2, 300, width, device="cuda")
    mask = torch.ones(2, 300, dtype=torch.bool, device="cuda")
    mask[1, 200:] = False
    tf32 = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        with torch.inference_mode():
            expected = full(x, mask)[:, index]
            got = one(x, mask)[:, 0]
    finally:
        torch.backends.cuda.matmul.allow_tf32 = tf32
    if not torch.allclose(expected, got, rtol=1e-3, atol=1e-3):
        raise RuntimeError(f"single-probe slice of {name} differs: {expected} {got}")
    return one


def load_banks(
    directory: Path, epochs: list[int], select: list[str] | None = None
) -> tuple[list, list[str]]:
    """The full01 banks at ``epochs``; with ``select``, one one-probe bank per
    selected probe name (and a single epoch) instead."""
    from gleipnir.probes.training import make_sweep

    config = json.loads((directory / "config.json").read_text())
    if select and len(epochs) != 1:
        raise ValueError("probe selection takes exactly one epoch")
    banks, names = [], None
    for epoch in epochs:
        state = torch.load(directory / f"probes_e{epoch}.pt", map_location="cuda")
        width = state["bank"]["first_weight"].shape[1]
        sweep = make_sweep(
            config["architectures"],
            config["objectives"],
            config["seeds"],
            width,
            lr=config["lr"],
            weight_decay=config["weight_decay"],
            device="cuda",
        )
        if [tuple(s) for s in state["specs"]] != [
            (s.name, s.architecture) for s in sweep.specs
        ]:
            raise ValueError(f"probe order differs in probes_e{epoch}.pt")
        sweep.bank.load_state_dict(state["bank"])
        sweep.bank.eval()
        if select:
            return [
                single_probe(sweep.bank, sweep.specs, n, config, width) for n in select
            ], select
        banks.append(sweep.bank)
        names = [s.name for s in sweep.specs]
    return banks, names


class ProbePooler(nn.Module):
    def __init__(
        self,
        banks: list,
        variants: list[str],
        keep: dict[str, int] | None = None,
        dump: Path | None = None,
    ):
        super().__init__()
        from vllm.model_executor.layers.pooler.tokwise.methods import AllPool

        self.collect = AllPool()
        if not self.collect.enable_chunked_prefill or self.collect.clone_finished:
            raise ValueError("probe pooler expects chunked prefill without async")
        self.banks = nn.ModuleList(banks)
        self.variants = variants
        self.keep = keep
        self.dump = dump
        self.dump_keys = (
            set(json.loads((dump / "keys.json").read_text())) if dump else set()
        )

    def get_supported_tasks(self) -> Set[str]:
        return {"classify"}

    def get_pooling_updates(self, task):
        from vllm.model_executor.layers.pooler import PoolingParamsUpdate

        needs_ids = self.keep is not None or self.dump is not None
        return PoolingParamsUpdate(requires_token_ids=needs_ids)

    def start(
        self, pooling_metadata, index: int, length: int
    ) -> tuple[int | None, str]:
        """First kept position (None: not in the keep table) and token key of
        request ``index``."""
        if self.keep is None and self.dump is None:
            # NTP: drop the attention sink. vLLM's warmup runs use one-token
            # dummy sequences, which keep their only token.
            return (1 if length > 1 else 0), ""
        ids = pooling_metadata.prompt_token_ids_cpu
        if ids is None:
            ids = pooling_metadata.prompt_token_ids
        ids = ids[index, :length].tolist()
        key = token_key(ids)
        if self.keep is None:
            return (1 if length > 1 else 0), key
        # vLLM's warmup dummies are not in the table; their outputs are unused.
        return self.keep.get(key), key

    @torch.inference_mode()
    def forward(self, hidden_states, pooling_metadata):
        from gleipnir.probes import fp8

        collected = self.collect(hidden_states, pooling_metadata)
        outputs = []
        for index, states in enumerate(collected):
            if states is None:  # an unfinished chunked prefill
                outputs.append(None)
                continue
            begin, key = self.start(pooling_metadata, index, len(states))
            if begin is None:  # NaN logits: the client rejects nonfinite rows
                x = states
            else:
                x = states[begin:]
            if key in self.dump_keys:
                torch.save(x.to(torch.bfloat16).cpu(), self.dump / f"{key}.pt")
            logits = []
            for variant in self.variants:
                v = fp8.dequantize(*fp8.quantize(x)) if variant == "fp8" else x
                v = v.to(torch.bfloat16)[None]
                mask = torch.ones(v.shape[:2], dtype=torch.bool, device=v.device)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    logits.extend(bank(v, mask)[0].float() for bank in self.banks)
            out = torch.cat(logits)
            outputs.append(
                out if begin is not None else torch.full_like(out, torch.nan)
            )
        return outputs


def truncate(worker, blocks: int, end_layer: int) -> None:
    """Stop ``worker``'s ``blocks``-block decoder at ``end_layer`` and, when
    ``GLEIPNIR_PROBE_BANK`` is set, install the probe pooler."""
    model = worker.model_runner.model
    decoders = [
        m for m in model.modules() if hasattr(m, "end_layer") and hasattr(m, "layers")
    ]
    if len(decoders) != 1 or decoders[0].end_layer != blocks:
        raise RuntimeError(f"expected one {blocks}-block decoder, found {decoders}")
    decoder = decoders[0]
    decoder.end_layer = end_layer
    decoder.norm = ResidualStream()
    receipt = {"end_layer": end_layer, "final_norm": "identity residual stream"}
    bank_dir = os.environ.get("GLEIPNIR_PROBE_BANK")
    if bank_dir:
        epochs = [int(e) for e in os.environ["GLEIPNIR_PROBE_EPOCHS"].split(",")]
        variants = os.environ.get("GLEIPNIR_PROBE_VARIANTS", "fp8").split(",")
        select = os.environ.get("GLEIPNIR_PROBE_SELECT")
        banks, names = load_banks(
            Path(bank_dir), epochs, select.split(",") if select else None
        )
        keep_path = os.environ.get("GLEIPNIR_PROBE_KEEP")
        keep = json.loads(Path(keep_path).read_text()) if keep_path else None
        dump = os.environ.get("GLEIPNIR_PROBE_DUMP")
        if dump:
            Path(dump).mkdir(parents=True, exist_ok=True)
        from vllm.config import set_current_vllm_config

        with set_current_vllm_config(worker.vllm_config):
            model.pooler = ProbePooler(
                banks, variants, keep, Path(dump) if dump else None
            )
        receipt.update(
            bank=bank_dir,
            epochs=epochs,
            variants=variants,
            keep=keep_path,
            dump=dump,
            probes=names,
        )
    out = os.environ.get("GLEIPNIR_PROBE_RECEIPT")
    if out:
        Path(out).write_text(json.dumps(receipt, indent=2) + "\n")
    print(
        "gleipnir_probe_truncation",
        {k: v for k, v in receipt.items() if k != "probes"},
        flush=True,
    )


class TruncatedFastGdnWorker(FastGdnWorker):
    def load_model(self, *args, **kwargs):
        result = super().load_model(*args, **kwargs)
        truncate(self, 32, END_LAYER)
        return result
