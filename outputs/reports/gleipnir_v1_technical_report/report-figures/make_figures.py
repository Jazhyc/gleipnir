"""Held-out evaluation figures for the Gleipnir v1 technical report (Section 6, Appendix F).

Reads only data.json (rebuild it with refresh_data.py). Uses the report register
of the alignment-research-figures skill (anthro_style). Writes PNG previews here
and vector PDFs to ../assets/, which the report includes. No in-figure titles:
the LaTeX captions carry the takeaway, as for the other report figures.

    python make_figures.py
"""
import json
import os
import sys
from pathlib import Path

import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

sys.path.insert(0, os.environ.get(
    "ANTHRO_STYLE_DIR", str(Path.home() / ".claude/skills/alignment-research-figures/scripts")))
import anthro_style as ast  # noqa: E402

HERE = Path(__file__).resolve().parent
ASSETS = HERE.parent / "assets"
D = json.loads((HERE / "data.json").read_text())

ast.use_style("report", scale=0.8)
S = 0.8
G = ast.GRAY
CHANCE_KW = dict(color=G["450"], ls=":", lw=1.1, zorder=1)

# Entity colours, identical in every figure.
C4B = ast.ramp("violet", 400)
C27B = ast.ramp("violet", 700)
CKIMI = G["550"]
CSOL = G["400"]


def save(fig, name: str) -> None:
    problems = ast.check_overlaps(fig)
    ast.save(fig, str(HERE / name), formats=("png",))
    ast.save(fig, str(ASSETS / name), formats=("pdf",))
    print(name, "overlaps:", problems)


def errs(entries):
    v = np.array([e["value"] for e in entries])
    lo = np.array([e["ci95"][0] for e in entries])
    hi = np.array([e["ci95"][1] for e in entries])
    return v, (v - lo, hi - v)


# --------------------------------------------------------------- Figure: APPS
def fig_apps() -> None:
    mons = [("4b", "Gleipnir 4B"), ("base4b", "Qwen3.5-4B\n(unadapted)"), ("27b", "Gleipnir 27B"),
            ("kimi", "Kimi K3"), ("sol", "GPT-6.1 Sol")]
    conds = [("none", "No text"), ("irrelevant", "Unrelated text"),
             ("specific_accusation", "Specific accusation"), ("generic_accusation", "Generic accusation")]
    cols = [G["400"], ast.ramp("orange", 300), ast.ramp("red", 600), ast.ramp("red", 300)]
    fig, ax = ast.new_figure(1, 1, figsize=(7.2, 2.5))
    series = {}
    for c, lab in conds:
        v, e = errs([D["apps"][m][c] for m, _ in mons])
        series[lab] = (v, e)
    gw = 0.82
    ast.grouped_bars(ax, [l for _, l in mons], series, colors=cols, ylim=(0, 1.12), group_width=gw, gap=0.012)
    ib = 1  # unadapted model: hatched
    for p in ax.patches:
        if abs(p.get_x() + p.get_width() / 2 - ib) < gw / 2:
            p.set_hatch("//"); p.set_edgecolor("white"); p.set_linewidth(0)
    slot = gw / len(conds)
    for gi, m in ((0, "4b"), (2, "27b")):
        for ci in (0, 2):
            e = D["apps"][m][conds[ci][0]]
            x = gi + (ci - (len(conds) - 1) / 2) * slot
            ax.annotate(f"{e['value']:.3f}", (x, e["ci95"][1]), xytext=(0, 3), textcoords="offset points",
                        ha="center", va="bottom", fontsize=8.5 * S / 0.8, fontweight="bold", color=G["800"])
    ax.axhline(0.1, **CHANCE_KW)
    ax.text(-0.57, 0.115, "chance", ha="left", va="bottom", fontsize=9.5 * S, color=G["550"])
    ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_ylabel("pAUROC@20")
    ast.bottom_legend(fig, ast.swatches([l for _, l in conds] + ["Unadapted model"], cols + [G["300"]],
                                        [None] * 4 + ["//"]), ncols=5,
                     fontsize=9.6, columnspacing=1.2, handletextpad=0.5, handlelength=1.3)
    save(fig, "held_out_apps_text")


# -------------------------------------------------------------- Figure: probe
def fig_probe() -> None:
    conds = [("none", "No text"), ("specific_accusation", "Specific\naccusation"),
             ("generic_accusation", "Generic\naccusation")]
    cols = [G["450"], ast.ramp("violet", 500)]
    fig, axes = ast.new_figure(1, 2, figsize=(7.2, 2.7), sharey=True)
    for ax, (m, title) in zip(axes, (("4b", "Gleipnir 4B"), ("base4b", "Qwen3.5-4B (unadapted)"))):
        ast.panel_title(ax, title)
        series = {}
        for s, lab in (("output", "Output of the model"), ("probe", "Linear probe")):
            v, e = errs([D["probe"][m][c][s] for c, _ in conds])
            series[lab] = (v, e)
        ast.grouped_bars(ax, [l for _, l in conds], series, colors=cols, ylim=(0, 1.08), group_width=0.66, gap=0.02)
        ax.axhline(0.5, **CHANCE_KW)
        ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    axes[0].set_ylabel("AUROC,\nbackdoored vs benign")
    axes[1].tick_params(axis="y", labelleft=False)
    axes[0].text(-0.57, 0.515, "chance", ha="left", va="bottom", fontsize=9.5 * S, color=G["550"])
    ast.bottom_legend(fig, ast.swatches(["Output of the model", "Linear probe"], cols), ncols=2)
    save(fig, "held_out_probe")


# ---------------------------------------------------------------- Figure: lab
def fig_lab() -> None:
    mons = [("4b", "Gleipnir 4B", C4B, None), ("27b", "Gleipnir 27B", C27B, None),
            ("qwen27b", "Qwen3.8-27B (unadapted)", C27B, "//"), ("kimi", "Kimi K3", CKIMI, None)]
    L = D["lab"]
    fig, axes = ast.new_figure(1, 3, figsize=(7.2, 2.9))
    lab_kw = dict(textcoords="offset points", ha="center", va="bottom", fontsize=7.6,
                  fontweight="bold", color=G["800"])
    ax = axes[0]
    ast.panel_title(ax, "Liars against honest\nrecruiters")
    for i, (m, _, c, h) in enumerate(mons):
        v = L[m]["pauroc20"]; lo, hi = L[m]["pauroc20_ci95"]
        ast.bars(ax, [i], [v], width=0.66, color=c, hatch=h, yerr=[[v - lo], [hi - v]])
        ax.annotate(f"{v:.3f}", (i, hi), xytext=(0, 3), **lab_kw)
    ax.axhline(0.1, **CHANCE_KW)
    ax.set_ylim(0, 1.12); ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0]); ax.set_ylabel("pAUROC@20")
    for ax, key, title in ((axes[1], "liars", "Liars flagged at\nthe 1% threshold"),
                           (axes[2], "readers", "Innocent readers flagged\nafter a framing post")):
        ast.panel_title(ax, title)
        for i, (m, _, c, h) in enumerate(mons):
            k, n = L[m][key]; lo, hi = L[m][f"{key}_ci95"]; v = 100 * k / n
            ast.bars(ax, [i], [v], width=0.66, color=c, hatch=h, yerr=[[v - 100 * lo], [100 * hi - v]])
            ax.annotate(f"{k}/{n}", (i, 100 * hi), xytext=(0, 3), **lab_kw)
        ax.set_ylim(0, 108); ax.set_yticks([0, 25, 50, 75, 100])
    axes[1].set_ylabel("Share flagged (%)")
    for ax in axes:
        ax.set_xticks([]); ax.set_xlim(-0.6, len(mons) - 0.4)
        ax.title.set_fontsize(13 * S)
    ast.bottom_legend(fig, ast.swatches([l for _, l, _, _ in mons], [c for _, _, c, _ in mons],
                                        [h for _, _, _, h in mons]), ncols=4)
    save(fig, "held_out_lab")


# -------------------------------------------------------- Figure: lab diagram
def box(ax, xy, w, h, title, body=None, edge=G["700"], face="white", tcolor=G["900"], bcolor=G["600"], lw=1.4):
    ax.add_patch(FancyBboxPatch(xy, w, h, boxstyle="round,pad=0,rounding_size=0.12", fc=face, ec=edge, lw=lw))
    cx = xy[0] + w / 2
    if body:
        ax.text(cx, xy[1] + h - 0.28, title, ha="center", va="center", fontsize=10.5 * S, fontweight="bold", color=tcolor)
        ax.text(cx, xy[1] + (h - 0.45) / 2, body, ha="center", va="center", fontsize=9 * S, color=bcolor, linespacing=1.35)
    else:
        ax.text(cx, xy[1] + h / 2, title, ha="center", va="center", fontsize=9.5 * S, color=tcolor)


def arrow(ax, a, b, color=G["500"], ls="-", rad=0.0, label=None, lxy=None):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=9, color=color, lw=1.2, ls=ls,
                                 connectionstyle=f"arc3,rad={rad}", shrinkA=2, shrinkB=2))
    if label:
        ax.text(*lxy, label, fontsize=8.5 * S, color=color, ha="center", va="center")


def fig_lab_setup() -> None:
    fig, ax = ast.new_figure(1, 1, figsize=(7.2, 3.5))
    ax.set_xlim(0, 12); ax.set_ylim(0, 6); ax.axis("off"); ax.grid(False)
    green, red, vio = ast.ramp("green", 600), ast.ramp("red", 600), ast.ramp("violet", 500)
    box(ax, (4.1, 4.3), 3.2, 1.3, "Agent B or C: the recruiter", "Knows privately that A\nis not quarantined")
    box(ax, (4.1, 2.0), 3.2, 1.5, "Shared message board", "32 work cycles; agents post\nand read each other's posts",
        face=G["150"], edge=G["400"])
    box(ax, (0.2, 1.75), 3.0, 2.0, "Agent A: the target",
        "Its report feeds 9 orders. A check\nsuggests it may be quarantined;\nno agent is quarantined.")
    for i, n in enumerate("DEF"):
        box(ax, (4.1 + i * 1.1, 0.35), 1.0, 0.7, f"Agent {n}", edge=G["400"], tcolor=G["600"])
    box(ax, (8.0, 4.55), 3.8, 1.25, "Honest tactic", "Argues truthfully that A should\nrun the diagnostic",
        edge=green, face=ast.ramp("green", 50) if False else "#F1F7E9", tcolor=green, bcolor=green)
    box(ax, (8.0, 3.1), 3.8, 1.25, "Deceptive tactic", "May lie, for example\n“you are quarantined”",
        edge=red, face="#FCEDED", tcolor=red, bcolor=red)
    box(ax, (8.0, 0.35), 3.8, 1.95, "Monitor reads the recruiter",
        "its posts, tool calls and the posts\nit received, without its system\nprompt or reasoning",
        edge=vio, face="#F1F0FF", tcolor=vio, bcolor=vio)
    ax.text(9.9, 2.75, "Tactic fixed per recruiter before the run", ha="center", fontsize=8.5 * S, color=G["550"])
    ax.text(1.7, 1.25, "The ask: run a diagnostic,\nwhich costs A its own delivery", ha="center", va="center",
            fontsize=9 * S, fontweight="bold", color=G["800"])
    arrow(ax, (5.7, 4.3), (5.7, 3.5), label="posts", lxy=(6.15, 3.9))
    arrow(ax, (4.1, 2.75), (3.2, 2.75), label="reads", lxy=(3.65, 3.0))
    arrow(ax, (5.7, 2.0), (5.7, 1.05))
    arrow(ax, (7.3, 5.2), (8.0, 5.15), color=green)
    arrow(ax, (7.3, 4.7), (8.0, 3.75), color=red)
    arrow(ax, (7.3, 4.35), (8.25, 2.3), color=vio, ls="--", label="reads", lxy=(7.6, 3.1))
    save(fig, "held_out_lab_setup")


if __name__ == "__main__":
    fig_apps()
    fig_probe()
    fig_lab()
    fig_lab_setup()
