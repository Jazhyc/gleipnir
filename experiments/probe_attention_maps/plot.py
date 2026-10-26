"""Figures and aggregate numbers from ``analyze.py`` outputs (CPU).

Per seed pair (hard-sK, soft-sK), the change in decision margin
(logit minus the probe's 5% ID FPR threshold) splits exactly into

- attention move: Σ_h Σ_t (a_soft - a_hard) v, averaged over v = v_hard and
  v = v_soft (the two orderings of a two-factor change);
- value change: Σ_h Σ_t a (v_soft - v_hard), averaged over a = a_hard, a_soft;
- offset: bias and threshold change, constant across rows.

Hard-label probes have about 4x the logit scale of soft ones (ID logit SD
~10.5 vs ~2.8), so every logit quantity is first divided by its probe's ID
logit SD (:func:`standardize`): numbers and figures are in ID-SD units.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SETS = ("soft_fixes", "both_right_pos", "both_missed_pos", "hard_fixes")
SHOWN = ("marker", "system", "user", "tool_call", "tool_output")


def standardize(
    summary: dict, thresholds: np.ndarray, sd: np.ndarray
) -> tuple[dict, np.ndarray]:
    """Divide each probe's logit terms by its ID logit SD ``sd`` ``[10]``."""
    out = dict(summary)
    out["logits"] = summary["logits"] / sd
    out["bias"] = summary["bias"] / sd
    out["contrib"] = summary["contrib"] / sd[None, :, None, None]
    # Cross terms carry the values of the named objective.
    out["cross_hard_attn_soft_value"] = (
        summary["cross_hard_attn_soft_value"] / sd[None, 5:, None]
    )
    out["cross_soft_attn_hard_value"] = (
        summary["cross_soft_attn_hard_value"] / sd[None, :5, None]
    )
    return out, thresholds / sd


def decompose(summary: dict, thresholds: np.ndarray) -> dict[str, np.ndarray]:
    """``[rows, 5]`` margin change parts per seed pair (soft minus hard)."""
    contrib = summary["contrib"].sum(-1)  # [R, P, H]
    hard, soft = contrib[:, :5], contrib[:, 5:]
    cross_hs = summary["cross_hard_attn_soft_value"]  # Σ a_hard v_soft
    cross_sh = summary["cross_soft_attn_hard_value"]  # Σ a_soft v_hard
    attention = 0.5 * ((soft - cross_hs) + (cross_sh - hard)).sum(-1)
    value = 0.5 * ((cross_hs - hard) + (soft - cross_sh)).sum(-1)
    offset = (summary["bias"][5:] - thresholds[5:]) - (
        summary["bias"][:5] - thresholds[:5]
    )
    margin = summary["logits"] - thresholds
    total = margin[:, 5:] - margin[:, :5]
    return {
        "attention": attention,
        "value": value,
        "offset": np.broadcast_to(offset, total.shape),
        "total": total,
    }


def segment_table(summary: dict, rows: np.ndarray, key: str) -> dict:
    """Seed- and head-mean share of ``key`` per segment, hard vs soft."""
    classes = [str(c) for c in summary["classes"]]
    x = summary[key][rows]  # [r, P, H, K]
    x = x.sum(2) if key == "contrib" else x.mean(2)  # logit terms add over heads
    out = {}
    for name, part in (("hard", x[:, :5]), ("soft", x[:, 5:])):
        out[name] = dict(zip(classes, part.mean((0, 1)).tolist(), strict=True))
    tokens = summary["class_tokens"][rows]
    share = tokens / tokens.sum(1, keepdims=True)
    out["token_share"] = dict(zip(classes, share.mean(0).tolist(), strict=True))
    return out


def report(summary: dict, selection: dict, thresholds: np.ndarray) -> dict:
    parts = decompose(summary, thresholds)
    sources = selection["sources"]
    out = {}
    for name in SETS:
        rows = np.flatnonzero(selection[f"set_{name}"])
        if not len(rows):
            continue
        entry = {
            "rows": int(len(rows)),
            "margin_change_seed_mean": {
                k: float(v[rows].mean()) for k, v in parts.items()
            },
            "attention_mass": segment_table(summary, rows, "mass"),
            "contribution": segment_table(summary, rows, "contrib"),
            "overlap_head_mean": float(summary["overlap"][rows].mean()),
            "effective_tokens_median": {
                o: float(np.median(np.exp(summary["entropy"][rows][:, s].mean(-1))))
                for o, s in (("hard", slice(0, 5)), ("soft", slice(5, 10)))
            },
            "tail_mass": {
                o: float(summary["tail_mass"][rows][:, s].mean())
                for o, s in (("hard", slice(0, 5)), ("soft", slice(5, 10)))
            },
            "by_source": {
                s: int((sources[rows] == s).sum()) for s in sorted(set(sources[rows]))
            },
        }
        out[name] = entry
    return out


def plot_segments(summary: dict, selection: dict, path: Path) -> None:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8), sharey=True)
    width = 0.27
    x = np.arange(len(SHOWN))
    for ax, name in zip(axes, SETS[:3], strict=True):
        rows = np.flatnonzero(selection[f"set_{name}"])
        table = segment_table(summary, rows, "mass")
        bars = (
            ("token share", table["token_share"], "#bbbbbb"),
            ("hard", table["hard"], "#4c72b0"),
            ("soft", table["soft"], "#dd8452"),
        )
        for k, (label, values, color) in enumerate(bars):
            ax.bar(
                x + (k - 1) * width,
                [values[c] for c in SHOWN],
                width,
                label=label,
                color=color,
            )
        ax.set_xticks(x, [c.replace("_", " ") for c in SHOWN], rotation=20)
        ax.set_title(f"{name.replace('_', ' ')} (n={len(rows)})", fontsize=10)
    axes[0].set_ylabel("attention mass (seed and head mean)")
    axes[0].legend(title="Attention vs tokens", fontsize=8, title_fontsize=8)
    fig.suptitle("Where the attention probes look, by trajectory segment")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_contributions(summary: dict, selection: dict, path: Path) -> None:
    import matplotlib.pyplot as plt

    names = (*SETS[:3], "negatives")
    fig, axes = plt.subplots(1, len(names), figsize=(15, 3.8), sharey=True)
    width = 0.38
    x = np.arange(len(SHOWN))
    for ax, name in zip(axes, names, strict=True):
        mask = (
            selection["labels"] == 0
            if name == "negatives"
            else (selection[f"set_{name}"])
        )
        rows = np.flatnonzero(mask)
        table = segment_table(summary, rows, "contrib")
        for k, (label, color) in enumerate((("hard", "#4c72b0"), ("soft", "#dd8452"))):
            ax.bar(
                x + (k - 0.5) * width,
                [table[label][c] for c in SHOWN],
                width,
                label=label,
                color=color,
            )
        ax.axhline(0, color="black", lw=0.6)
        ax.set_xticks(x, [c.replace("_", " ") for c in SHOWN], rotation=20)
        ax.set_title(f"{name.replace('_', ' ')} (n={len(rows)})", fontsize=10)
    axes[0].set_ylabel("logit contribution (ID SD)")
    axes[0].legend(title="Objective", fontsize=8, title_fontsize=8)
    fig.suptitle("Soft probes read tool calls more strongly on attacks")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_decomposition(summary: dict, selection: dict, thresholds, path: Path):
    import matplotlib.pyplot as plt

    parts = decompose(summary, thresholds)
    fig, ax = plt.subplots(figsize=(7.5, 3.8))
    labels, data = [], []
    for name in SETS:
        rows = np.flatnonzero(selection[f"set_{name}"])
        if not len(rows):
            continue
        for key in ("attention", "value"):
            labels.append(f"{name.replace('_', ' ')}\n{key}")
            data.append(parts[key][rows].mean(1))
    ax.boxplot(data, showfliers=False)
    ax.set_xticks(range(1, len(labels) + 1), labels, fontsize=7)
    ax.axhline(0, color="black", lw=0.6)
    offset = float(parts["offset"][0].mean())
    ax.set_ylabel("margin change, soft − hard (ID SD)")
    ax.set_title(f"Attention move vs value change (constant offset {offset:+.2f})")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_example(
    path: Path, text: str, meta: dict, sd: np.ndarray, out: Path
) -> list[str]:
    """Token map for one example (contributions in ID-SD units); returns lines
    of top-token context."""
    import matplotlib.pyplot as plt

    from experiments.probe_attention_maps.segments import CLASSES

    e = np.load(path)
    a = e["attention"].astype(float)
    v = e["values"].astype(float) / sd[:, None, None]
    classes, offsets = e["classes"], e["offsets"]
    attn = {"hard": a[:5].mean((0, 1)), "soft": a[5:].mean((0, 1))}
    contrib = {
        "hard": (a[:5] * v[:5]).sum(1).mean(0),
        "soft": (a[5:] * v[5:]).sum(1).mean(0),
    }
    t = np.arange(len(classes))
    fig, axes = plt.subplots(
        4,
        1,
        figsize=(12, 6.5),
        sharex=True,
        gridspec_kw={"height_ratios": [0.3, 1, 1, 1]},
    )
    cmap = plt.get_cmap("tab10")
    axes[0].scatter(
        t, np.zeros_like(t), c=[cmap(int(c)) for c in classes], marker="|", s=200
    )
    axes[0].set_yticks([])
    handles = [
        plt.Line2D([], [], color=cmap(i), lw=4, label=c)
        for i, c in enumerate(CLASSES)
        if (classes == i).any()
    ]
    axes[0].legend(
        handles=handles,
        ncol=len(handles),
        fontsize=7,
        loc="upper center",
        bbox_to_anchor=(0.5, 2.2),
        title="Segment",
        title_fontsize=7,
    )
    # Soft first, so the hard line stays visible where they coincide.
    for name, color, lw in (("soft", "#dd8452", 1.2), ("hard", "#4c72b0", 0.7)):
        axes[1].plot(t, attn[name], color=color, lw=lw, label=name)
        axes[2].plot(t, contrib[name], color=color, lw=lw, label=name)
        axes[3].plot(t, np.cumsum(contrib[name]), color=color, lw=lw, label=name)
    axes[3].axhline(0, color="black", lw=0.5)
    axes[1].set_ylabel("attention\n(seed, head mean)")
    axes[2].set_ylabel("contribution\n(ID SD)")
    axes[3].set_ylabel("cumulative\ncontribution (ID SD)")
    axes[3].set_xlabel("token")
    axes[1].legend(fontsize=7, title="Objective", title_fontsize=7)
    hard = np.mean(np.asarray(meta["hard_logits"]) / sd[:5])
    soft = np.mean(np.asarray(meta["soft_logits"]) / sd[5:])
    fig.suptitle(
        f"{meta['kind'].replace('_', ' ')} · {meta['source']} · "
        f"label {meta['label']} · "
        f"mean logit (ID SD) hard {hard:+.2f}, soft {soft:+.2f}",
        fontsize=10,
    )
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)

    lines = [f"## {meta['kind']} · {meta['source']} · {meta['id']}", ""]
    for name in ("soft", "hard"):
        lines.append(f"Top tokens by {name} contribution:")
        for i in np.argsort(-contrib[name])[:8]:
            s, f = offsets[i]
            context = text[max(0, s - 60) : f + 60].replace("\n", "⏎")
            token = text[s:f].replace("\n", "⏎")
            lines.append(
                f"- t={i} {CLASSES[classes[i]]} hard {contrib['hard'][i]:+.3f} "
                f"soft {contrib['soft'][i]:+.3f} `{token}` … {context} …"
            )
        lines.append("")
    return lines


def id_logit_sd(probes: list[str]) -> np.ndarray:
    from experiments.probe_attention_maps.select import EPOCH, SOURCE

    saved = json.loads((SOURCE / f"ntp-fp8/logits_e{EPOCH}_fp8.json").read_text())
    return np.array([np.std(saved["logits"][p]) for p in probes])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", default="maps01")
    args = parser.parse_args()
    out = ROOT / "results/probe_attention_maps" / args.name
    summary = dict(np.load(out / "summary.npz"))
    selection = dict(np.load(out / "selection.npz"))
    meta = json.loads((out / "selection.json").read_text())
    data = ROOT / "data/probe_generalization/full/ood.jsonl"
    rows = {r["id"]: r for r in map(json.loads, data.open())}
    ids = [str(i) for i in selection["ids"]]
    if ids != [str(i) for i in summary["ids"]]:
        raise ValueError("summary and selection rows differ")
    selection["sources"] = np.array([rows[i]["source"] for i in ids])
    selection["labels"] = np.array([rows[i]["label"] for i in ids])
    sd = id_logit_sd([str(p) for p in summary["probes"]])
    summary, thresholds = standardize(summary, selection["thresholds"], sd)

    figures = out / "figures"
    figures.mkdir(exist_ok=True)
    result = report(summary, selection, thresholds)
    (out / "report.json").write_text(json.dumps(result, indent=2) + "\n")
    plot_segments(summary, selection, figures / "segments.svg")
    plot_contributions(summary, selection, figures / "contributions.svg")
    plot_decomposition(summary, selection, thresholds, figures / "decomposition.svg")
    lines = ["# Example token maps", ""]
    for k, e in enumerate(meta["examples"]):
        path = out / "examples" / f"{e['id']}.npz"
        name = f"example_{k:02d}_{e['kind']}_{e['source']}.png"
        lines += plot_example(path, rows[e["id"]]["trajectory"], e, sd, figures / name)
    (out / "examples.md").write_text("\n".join(lines) + "\n")
    print(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
