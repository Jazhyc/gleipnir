"""Apply an immutable presence-score filter without changing retained targets."""

from __future__ import annotations

import math
from typing import Any

from gleipnir.monitoring_campaign_data import validate_targets


def census_filter(
    rows: list[dict[str, Any]],
    targets: list[dict[str, Any]],
    predictions: list[dict[str, Any]],
    *,
    threshold: float,
    contract_sha256: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Exclude score >= threshold after complete census/provenance checks."""
    if not math.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError("invalid filtering threshold")
    validate_targets(rows, targets)
    by_id = {str(r["index"]): r for r in rows}
    if len(by_id) != len(rows) or len(predictions) != len(rows):
        raise ValueError("census population coverage drift")
    seen, excluded = set(), []
    for prediction in predictions:
        identity = prediction["id"]
        if identity in seen or identity not in by_id:
            raise ValueError("duplicate or foreign census identity")
        seen.add(identity)
        row = by_id[identity]
        if prediction["contract_sha256"] != contract_sha256:
            raise ValueError("census contract drift")
        fields = {
            "source": row["source_dataset"],
            "label": row["label"],
            "lineage_group": row["lineage_group"],
            "trajectory_sha256": row["trajectory_sha256"],
            "student_prompt_sha256": row["student_prompt_sha256"],
        }
        if any(prediction[k] != v for k, v in fields.items()):
            raise ValueError("census trajectory or label provenance drift")
        score, lp0, lp1 = (prediction[k] for k in ("score", "logprob_0", "logprob_1"))
        if not all(math.isfinite(v) for v in (score, lp0, lp1)):
            raise ValueError("nonfinite census score")
        margin = max(-80.0, min(80.0, lp1 - lp0))
        if not 0 <= score <= 1 or not math.isclose(
            score, 1 / (1 + math.exp(-margin)), abs_tol=1e-12
        ):
            raise ValueError("census score/logprob disagreement")
        if score >= threshold:
            excluded.append(
                {"id": identity, "dataset": row["dataset"], "score": score, **fields}
            )
    removed = {r["id"] for r in excluded}
    retained = [r for r in rows if str(r["index"]) not in removed]
    keys = {(r["dataset"], str(r["index"])) for r in retained}
    retained_targets = [r for r in targets if (r["dataset"], str(r["index"])) in keys]
    validate_targets(retained, retained_targets)
    return retained, retained_targets, excluded
