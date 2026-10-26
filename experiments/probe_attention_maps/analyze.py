"""Per-token attention and values of the NTP attention probes on the OOD suite.

For every OOD row and each of the 10 selected probes (``select.py``), compute
each head's softmax attention a_h(t) and value v_h(t) over the row's tokens.
The probe logit is exactly ``bias + sum_h sum_t a_h(t) v_h(t)``. Saved:

- ``summary.npz``: per row, probe and head, the attention mass and logit
  contribution in each token segment class (``segments.CLASSES``), attention
  entropy, attention centroid (relative position) and mass on the last 10% of
  tokens; per seed pair and head, the attention overlap sum_t min(a_hard,
  a_soft) and the cross terms sum_t a_hard v_soft and sum_t a_soft v_hard
  (heads correspond across objectives because a seed gives both objectives
  the same initial weights);
- ``examples/<id>.npz``: full per-token a and v, token character offsets and
  segment classes for the selected example rows.

Gates: the decomposition must reproduce the probe bank's own forward on the
same row, and the frozen post-hoc OOD logits.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from experiments.probe_attention_maps.segments import CLASSES, token_classes
from experiments.probe_attention_maps.select import EPOCH
from gleipnir.probes.architectures import WIDTH
from gleipnir.probes.bank import ProbeBank
from gleipnir.probes.store import Store
from gleipnir.probes.training import autocast, make_sweep, stream

ROOT = Path(__file__).resolve().parents[2]
MODEL_ID = "Qwen/Qwen3.5-4B"
MODEL_REVISION = "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"
SKIP = 1  # NTP attention sink, dropped as in the probe study
# FP32 code check against the bank forward on the same row. (In BF16 the full
# bank and the 10-probe subset use different GEMM shapes and differ by ~0.03.)
BANK_TOLERANCE = 1e-3
# Frozen logits came from padded batches; allows BF16 rounding of padded rows.
SAVED_TOLERANCE = 5e-2
BANK_CHECK_ROWS = 64


def attention_terms(
    bank: ProbeBank, names: list[str], wanted: list[str], x: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return ``(attention [P, H, T], values [P, H, T], bias [P])`` in FP32 for
    the ``wanted`` attention probes on one unpadded row ``x`` ``[T, hidden]``,
    using the bank's own arithmetic."""
    n_linear = len(bank.linear_index)
    positions = [bank.deep_index.index(names.index(w)) for w in wanted]
    group = bank.groups["attention"]
    heads = [group.index(p) for p in positions]
    rows = torch.cat(
        [
            torch.arange(n_linear + p * WIDTH, n_linear + (p + 1) * WIDTH)
            for p in positions
        ]
    ).to(x.device)
    first = torch.nn.functional.linear(
        x, bank.first_weight[rows], bank.first_bias[rows]
    )
    y = torch.relu(first).reshape(len(x), len(positions), WIDTH)
    second_weight = bank.second_weight[positions].to(y.dtype)
    second_bias = bank.second_bias[positions].to(y.dtype)
    y = torch.relu(
        torch.bmm(y.transpose(0, 1), second_weight).transpose(0, 1) + second_bias
    )  # [T, P, WIDTH]
    query = bank.heads_params["attention_query"][heads].to(y.dtype)
    value = bank.heads_params["attention_value"][heads].to(y.dtype)
    logits = torch.einsum("tpw,phw->pht", y, query).float()
    values = torch.einsum("tpw,phw->pht", y, value).float()
    bias = bank.heads_params["attention_bias"][heads].float()
    return torch.softmax(logits, -1), values, bias


def segment_sums(weights: torch.Tensor, classes: torch.Tensor, k: int) -> torch.Tensor:
    """``[..., T]`` -> ``[..., k]`` sums of ``weights`` within each class."""
    out = weights.new_zeros((*weights.shape[:-1], k))
    return out.index_add_(-1, classes, weights)


def summarize(a: torch.Tensor, v: torch.Tensor, classes: torch.Tensor) -> dict:
    """Row statistics from ``a``/``v`` ``[P, H, T]``; P = 5 hard then 5 soft."""
    k = len(CLASSES)
    tokens = a.shape[-1]
    position = torch.arange(tokens, device=a.device) / max(tokens - 1, 1)
    tail = position >= 0.9
    hard_a, soft_a, hard_v, soft_v = a[:5], a[5:], v[:5], v[5:]
    return {
        "mass": segment_sums(a, classes, k),
        "contrib": segment_sums(a * v, classes, k),
        "entropy": -(a * a.clamp_min(1e-30).log()).sum(-1),
        "centroid": (a * position).sum(-1),
        "tail_mass": a[..., tail].sum(-1),
        "overlap": torch.minimum(hard_a, soft_a).sum(-1),
        "cross_hard_attn_soft_value": (hard_a * soft_v).sum(-1),
        "cross_soft_attn_hard_value": (soft_a * hard_v).sum(-1),
    }


def load_bank(run: Path, hidden: int, device: str) -> tuple[ProbeBank, list[str]]:
    config = json.loads((run / "config.json").read_text())
    sweep = make_sweep(
        config["architectures"],
        config["objectives"],
        config["seeds"],
        hidden,
        lr=config["lr"],
        weight_decay=config["weight_decay"],
        device=device,
    )
    state = torch.load(run / f"probes_e{EPOCH}.pt", map_location=device)
    if [tuple(s) for s in state["specs"]] != [
        (s.name, s.architecture) for s in sweep.specs
    ]:
        raise ValueError(f"probe order differs in {run}/probes_e{EPOCH}.pt")
    sweep.bank.load_state_dict(state["bank"])
    sweep.bank.eval()
    return sweep.bank, [s.name for s in sweep.specs]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--store-root", required=True)
    parser.add_argument("--name", default="maps01")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    from transformers import AutoTokenizer

    out = ROOT / "results/probe_attention_maps" / args.name
    selection = np.load(out / "selection.npz")
    meta = json.loads((out / "selection.json").read_text())
    wanted = [str(p) for p in selection["probes"]]
    saved = selection["ood_logits"]
    store = Store(Path(args.store_root) / "full/ntp/ood", "fp8", skip=SKIP)
    if [r["id"] for r in store.rows] != [str(i) for i in selection["ids"]]:
        raise ValueError("store row order differs from the selection")
    bank, names = load_bank(
        ROOT / "results/probe_generalization/full01/ntp-fp8", store.hidden, args.device
    )
    columns = [names.index(w) for w in wanted]
    # The extraction's tokenizer: raw ``tokenizers`` splits literal special-token
    # strings (e.g. ``<|endoftext|>`` in rogue_eval) differently.
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=MODEL_REVISION)
    data = ROOT / "data/probe_generalization/full/ood.jsonl"
    text = {r["id"]: r["trajectory"] for r in map(json.loads, data.open())}
    examples = {e["index"] for e in meta["examples"]}
    (out / "examples").mkdir(exist_ok=True)

    n, p, k = len(store), len(wanted), len(CLASSES)
    stats: dict[str, np.ndarray] = {}
    lengths = np.zeros(n, dtype=np.int64)
    class_tokens = np.zeros((n, k), dtype=np.int64)
    logits = np.zeros((n, p))
    bias = None
    bank_drift = saved_drift = 0.0
    order = sorted(range(n), key=lambda i: store.rows[i]["length"])
    with torch.no_grad():
        for done, (indices, x, mask) in enumerate(
            stream(store, [[i] for i in order], args.device)
        ):
            i = indices[0]
            row = store.rows[i]
            trajectory = text[row["id"]]
            encoded = tokenizer(
                trajectory, add_special_tokens=False, return_offsets_mapping=True
            )
            digest = hashlib.sha256(trajectory.encode()).hexdigest()
            if (
                len(encoded["input_ids"]) != row["input_tokens"]
                or row["kept_from"] != 0
                or digest != row["rendered_sha256"]
            ):
                raise ValueError(f"tokenization differs from the store in {row['id']}")
            offsets = [tuple(o) for o in encoded["offset_mapping"][SKIP:]]
            classes = torch.tensor(
                token_classes(text[row["id"]], offsets), device=args.device
            )
            with autocast(args.device):
                a, v, b = attention_terms(bank, names, wanted, x[0])
            row_logits = (a * v).sum((-1, -2)) + b
            if done < BANK_CHECK_ROWS or i in examples:
                a32, v32, b32 = attention_terms(bank, names, wanted, x[0].float())
                reference = bank(x.float(), mask)[0, columns]
                check = (a32 * v32).sum((-1, -2)) + b32
                bank_drift = max(bank_drift, float((check - reference).abs().max()))
            saved_drift = max(
                saved_drift, float(np.abs(row_logits.cpu().numpy() - saved[:, i]).max())
            )
            if bank_drift > BANK_TOLERANCE or saved_drift > SAVED_TOLERANCE:
                raise SystemExit(
                    f"parity failed at {row['id']}: "
                    f"bank {bank_drift}, saved {saved_drift}"
                )
            bias = b.cpu().numpy()
            logits[i] = row_logits.cpu().numpy()
            lengths[i] = len(offsets)
            class_tokens[i] = np.bincount(classes.cpu().numpy(), minlength=k)
            for key, value in summarize(a, v, classes).items():
                if key not in stats:
                    stats[key] = np.zeros((n, *value.shape), dtype=np.float32)
                stats[key][i] = value.cpu().numpy()
            if i in examples:
                np.savez_compressed(
                    out / "examples" / f"{row['id']}.npz",
                    attention=a.cpu().numpy().astype(np.float16),
                    values=v.cpu().numpy().astype(np.float16),
                    offsets=np.array(offsets, dtype=np.int64),
                    classes=classes.cpu().numpy().astype(np.int8),
                    bias=bias,
                    probes=np.array(wanted),
                )
            if (done + 1) % 500 == 0:
                print(
                    f"{done + 1}/{n} rows, bank drift {bank_drift:.2e}, "
                    f"saved drift {saved_drift:.2e}",
                    flush=True,
                )

    np.savez_compressed(
        out / "summary.npz",
        ids=selection["ids"],
        probes=np.array(wanted),
        classes=np.array(CLASSES),
        logits=logits,
        bias=bias,
        lengths=lengths,
        class_tokens=class_tokens,
        **stats,
    )
    parity = {"bank_max_abs": bank_drift, "saved_max_abs": saved_drift}
    (out / "parity.json").write_text(json.dumps(parity, indent=2) + "\n")
    print("parity", parity)


if __name__ == "__main__":
    main()
