"""Compare the trained 27B on the frozen OOD suite with the 4B seeds.

Uses the 4B seed-variation loader and metrics (source-macro pAUROC@20, the
paper Mean-OOD aggregate). The unadapted 27B (0.8901) and Kimi K3 (0.9084)
were scored under the teacher prompt and are cited from their records.
"""

from __future__ import annotations

import json

from experiments.qwen38_27b_campaign.ood import NAME
from experiments.teacher_seed_variation.compare import PREDICTIONS, load, summary
from experiments.teacher_seed_variation.prepare import ROOT

ARMS = {
    "27B trained, seed 0": f"results/{NAME}/evaluation/ood.jsonl",
    "4B seed 0": PREDICTIONS["sdpa_seed0_rescored"]["ood"],
    "4B seed 1": PREDICTIONS["sdpa_seed1"]["ood"],
    "4B seed 2": PREDICTIONS["sdpa_seed2"]["ood"],
}


def main() -> None:
    report = {name: summary(load(path, "ood")) for name, path in ARMS.items()}
    (ROOT / f"results/{NAME}/ood_comparison.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    for name, value in report.items():
        print(
            name,
            {k: round(v, 4) for k, v in value["macro"].items() if isinstance(v, float)},
            {k: round(v, 4) for k, v in value["by_source"].items()},
        )


if __name__ == "__main__":
    main()
