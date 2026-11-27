"""Export a paired same-adapter ID drift figure from completed observations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def plot(directory: Path) -> None:
    old = json.loads((directory / "reference.json").read_text())
    new = json.loads((directory / "repeat0.json").read_text())
    if [(r["id"], r["prompt_sha256"]) for r in old] != [
        (r["id"], r["prompt_sha256"]) for r in new
    ]:
        raise ValueError("plot prompt identity drift")
    a = np.array([r["score"] for r in old])
    b = np.array([r["score"] for r in new])
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    for source in sorted({r["dataset"] for r in old}):
        mask = np.array([r["dataset"] == source for r in old])
        axes[0].scatter(a[mask], b[mask], s=5, alpha=0.35, label=source)
        axes[1].hist(b[mask] - a[mask], bins=60, histtype="step", label=source)
    axes[0].plot([0, 1], [0, 1], color="black", linewidth=0.8)
    axes[0].set(xlabel="Frozen BF16 score", ylabel="Optimized stack score")
    axes[1].set(xlabel="Optimized minus frozen BF16 score", ylabel="Examples")
    axes[0].legend(fontsize=8)
    axes[1].legend(fontsize=8)
    fig.suptitle("Same trained adapter: one complete 3,012-example ID pass")
    for extension in ("png", "svg"):
        fig.savefig(directory / f"score_drift.{extension}", dpi=180)
    plt.close(fig)
    (directory / "plot_source.py").write_bytes(Path(__file__).read_bytes())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    plot(parser.parse_args().directory)


if __name__ == "__main__":
    main()
