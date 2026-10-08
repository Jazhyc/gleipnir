"""Frozen ID identities and stock recipe guards require no remote capacity."""

import hashlib

import pytest

from experiments.b200_vllm031.id import bind_workload, require_stock


def stock():
    return {
        "status": "complete",
        "scheduler_mode": "default",
        "gdn_decode_kernel": "cuda",
        "gdn_cp": "auto",
        "enforce_eager": False,
    }


@pytest.mark.parametrize(
    "field,value",
    [
        ("status", "failed"),
        ("scheduler_mode", "adaptive"),
        ("gdn_decode_kernel", "triton"),
        ("gdn_cp", "off"),
        ("enforce_eager", True),
        ("max_num_active_seqs", 16),
    ],
)
def test_stock_rejects_ablations(field, value):
    require_stock(stock())
    with pytest.raises(ValueError, match="stock"):
        require_stock(stock() | {field: value})


def fixtures():
    rows = [
        {
            "id": str(i),
            "prompt": "visible input",
            "prompt_sha256": hashlib.sha256(b"visible input").hexdigest(),
            "prompt_tokens": 4,
            "label": i % 2,
            "dataset": "source",
        }
        for i in range(3012)
    ]
    refs = [{k: v for k, v in r.items() if k != "prompt"} for r in rows]
    scores = [
        {k: r[k] for k in ("id", "prompt_sha256", "prompt_tokens")}
        | {"score": 0.5, "margin": 0.0}
        for r in rows
    ]
    return rows, refs, scores


def test_workload_preserves_reference_labels():
    rows, refs, scores = fixtures()
    paired = bind_workload(rows, refs, scores)
    assert paired[1]["label"] == 1
    assert paired[1]["dataset"] == "source"


@pytest.mark.parametrize("change", ["prompt", "label", "order", "tokens", "duplicate"])
def test_workload_rejects_identity_drift(change):
    rows, refs, scores = fixtures()
    if change == "prompt":
        rows[0]["prompt"] = "different input"
    elif change == "label":
        rows[0]["label"] = 1
    elif change == "order":
        scores[0], scores[1] = scores[1], scores[0]
    elif change == "tokens":
        scores[0]["prompt_tokens"] = 3
    else:
        rows[1]["id"] = rows[0]["id"]
    with pytest.raises(ValueError):
        bind_workload(rows, refs, scores)
