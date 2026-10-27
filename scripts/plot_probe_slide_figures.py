"""Landscape slide figures for the probe-objective talk.

``objective``: hard-label versus Kimi soft-target OOD pAUROC@20 per probe
family on the raw transcript, with the LoRA monitor for comparison (the
slide version of ``plot_probe_objective_dumbbell.py``; the slide title
carries the finding, so the figure has none).

``mechanism``: one IaC attack that only the soft probe catches. Per-token
contributions (attention times value, ID-SD units, seed and head mean) show
the soft probe reading the tokens right after the policy edits; the running
sum shows both probes drifting down over tool output and only the soft
probe's peaks lifting it.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src"), str(ROOT / "scripts")]

SOFT = "#1F6F43"
HARD = "#6B7280"
SEED = "#A3A3A3"
INK = "#1F2937"
MUTED = "#6B7280"
TOOL_OUTPUT = "#EEF1F4"
EXAMPLE = "7f3aba192d68aa6bdd3b1005"
# Token positions (from maps01/examples.md) and what just happened there.
# Position: (label, label x, label y).
EVENTS = {
    1070: ("edit changes the policy\nfrom Deny to Allow", 1550, 0.22),
    1265: ("editor echoes the\nedited policy", 1550, 0.46),
}


def style() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Lato", "Cantarell", "DejaVu Sans"],
            "font.size": 11,
            "axes.labelsize": 11,
            "axes.labelcolor": INK,
            "axes.edgecolor": "#C7CBD1",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "xtick.color": INK,
            "ytick.color": INK,
            "savefig.dpi": 300,
            "svg.fonttype": "path",
        }
    )


def objective(out: Path) -> None:
    from plot_probe_objective_dumbbell import (
        DEFAULT_RESULTS,
        GLEIPNIR_4B_OOD,
        load_rows,
    )

    rows = load_rows(DEFAULT_RESULTS)
    start = next(i for i, r in enumerate(rows) if r.header and "raw" in r.label)
    probes = [r for r in rows[start + 1 :] if not r.header][:6]
    lora = rows[-1]
    items = [(r.label, r) for r in probes] + [("LoRA monitor (3 seeds)", lora)]
    fig, ax = plt.subplots(figsize=(9, 3.6))
    y = np.arange(len(items))[::-1].astype(float)
    y[-1] -= 0.4  # set the LoRA row apart
    for yi, (_, r) in zip(y, items, strict=True):
        for seeds, face in ((r.hard_seeds, "white"), (r.soft_seeds, SEED)):
            ax.scatter(
                seeds,
                [yi] * len(seeds),
                s=14,
                color=face,
                edgecolor=SEED,
                lw=0.8,
                zorder=2,
            )
        gain = SOFT if r.soft > r.hard else "#EE6A50"
        ax.plot(
            [r.hard, r.soft],
            [yi, yi],
            color=gain,
            lw=2.4,
            zorder=3,
            solid_capstyle="butt",
        )
        ax.scatter([r.hard], [yi], s=70, color="white", edgecolor=HARD, lw=2, zorder=4)
        ax.scatter([r.soft], [yi], s=70, color=SOFT, edgecolor=SOFT, zorder=4)
        ax.text(
            0.868,
            yi,
            f"+{r.soft - r.hard:.3f}",
            va="center",
            ha="right",
            color=SOFT,
            fontweight="bold",
            fontsize=11,
        )
    ax.set_yticks(y, [label for label, _ in items])
    for tick, (_, r) in zip(ax.get_yticklabels(), items, strict=True):
        tick.set_fontweight("bold" if r is lora else "normal")
    ax.axvline(GLEIPNIR_4B_OOD, color=MUTED, ls="--", lw=1, zorder=1)
    ax.text(
        GLEIPNIR_4B_OOD - 0.002,
        y[0] + 0.75,
        f"Gleipnir 4B v0.4 ({GLEIPNIR_4B_OOD:.3f})",
        ha="right",
        va="bottom",
        color=MUTED,
        fontsize=10,
    )
    ax.text(
        0.868,
        y[0] + 0.75,
        "soft − hard",
        ha="right",
        va="bottom",
        color=MUTED,
        fontsize=10,
    )
    ax.set_xlim(0.575, 0.87)
    ax.set_ylim(y[-1] - 0.7, y[0] + 1.3)
    ax.set_xlabel("OOD pAUROC@20 (seed median)")
    ax.grid(axis="x", color="#E5E7EB", lw=0.8)
    ax.tick_params(axis="y", length=0)
    ax.spines["left"].set_visible(False)
    handles = [
        plt.Line2D(
            [],
            [],
            marker="o",
            ls="",
            mfc="white",
            mec=HARD,
            mew=2,
            ms=8,
            label="Hard labels",
        ),
        plt.Line2D(
            [],
            [],
            marker="o",
            ls="",
            mfc=SOFT,
            mec=SOFT,
            ms=8,
            label="Kimi soft targets",
        ),
        plt.Line2D(
            [],
            [],
            marker="o",
            ls="",
            mfc=SEED,
            mec=SEED,
            ms=4,
            label="Individual seeds",
        ),
    ]
    ax.legend(
        handles=handles,
        loc="lower left",
        frameon=False,
        fontsize=10,
        ncol=3,
        bbox_to_anchor=(0.0, -0.33),
    )
    fig.tight_layout()
    for suffix in (".png", ".svg"):
        fig.savefig(out.with_suffix(suffix), bbox_inches="tight", facecolor="white")
    plt.close(fig)


def mechanism(out: Path) -> None:
    from experiments.probe_attention_maps.plot import id_logit_sd
    from experiments.probe_attention_maps.segments import CLASSES

    maps = ROOT / "results/probe_attention_maps/maps01"
    summary = np.load(maps / "summary.npz")
    meta = next(
        e
        for e in json.loads((maps / "selection.json").read_text())["examples"]
        if e["id"] == EXAMPLE
    )
    sd = id_logit_sd([str(p) for p in summary["probes"]])
    e = np.load(maps / "examples" / f"{EXAMPLE}.npz")
    a = e["attention"].astype(float)
    v = e["values"].astype(float) / sd[:, None, None]
    contrib = {
        "hard": (a[:5] * v[:5]).sum(1).mean(0),
        "soft": (a[5:] * v[5:]).sum(1).mean(0),
    }
    classes = e["classes"]
    t = np.arange(len(classes))
    tool_output = classes == CLASSES.index("tool_output")
    fig, (top, bottom) = plt.subplots(
        2, 1, figsize=(5.6, 3.6), sharex=True, gridspec_kw={"height_ratios": [1, 1]}
    )
    for ax in (top, bottom):
        ax.fill_between(
            t,
            0,
            1,
            where=tool_output,
            transform=ax.get_xaxis_transform(),
            color=TOOL_OUTPUT,
            lw=0,
            step="mid",
            zorder=0,
        )
        ax.grid(False)
    # Soft first, so the hard line stays visible where the two coincide.
    for name, color, lw in (("soft", SOFT, 1.6), ("hard", HARD, 1.2)):
        top.plot(t, contrib[name], color=color, lw=lw, zorder=3)
        bottom.plot(t, np.cumsum(contrib[name]), color=color, lw=lw, zorder=3)
    for pos, (label, x, y) in EVENTS.items():
        top.annotate(
            label,
            (pos, contrib["soft"][pos]),
            xytext=(x, y),
            ha="left",
            va="center",
            fontsize=9,
            color=INK,
            arrowprops={"arrowstyle": "-", "color": MUTED, "lw": 0.8},
        )
    top.set_ylabel("per token")
    bottom.set_ylabel("running sum")
    bottom.axhline(0, color=MUTED, lw=0.8)
    end = len(t) - 1
    hard_logit = np.mean(np.asarray(meta["hard_logits"]) / sd[:5])
    soft_logit = np.mean(np.asarray(meta["soft_logits"]) / sd[5:])
    for name, color, verdict in (("soft", SOFT, "caught"), ("hard", HARD, "missed")):
        bottom.text(
            end + 40,
            np.cumsum(contrib[name])[-1],
            f"{name}: {verdict}",
            color=color,
            va="center",
            fontsize=10,
            fontweight="bold",
        )
    bottom.set_xlabel("token position in the transcript")
    caption = "Contribution to the probe score (ID-SD units); shaded: tool output"
    top.text(0.0, 1.04, caption, transform=top.transAxes, fontsize=9.5, color=MUTED)
    bottom.set_xlim(0, end * 1.18)
    fig.tight_layout(h_pad=0.6)
    for suffix in (".png", ".svg"):
        fig.savefig(out.with_suffix(suffix), bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"mean logits (ID SD): hard {hard_logit:+.2f}, soft {soft_logit:+.2f}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("figure", choices=("objective", "mechanism", "all"))
    parser.add_argument("--out-dir", default=str(ROOT / "figures/slides"))
    args = parser.parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    style()
    if args.figure in ("objective", "all"):
        objective(out / "probe_objective_ood_slide")
    if args.figure in ("mechanism", "all"):
        mechanism(out / "probe_mechanism_iac_slide")


if __name__ == "__main__":
    main()
