import pytest

from gleipnir.prefix_audit import select_cache_audit, validate_fresh_audit
from gleipnir.prefix_cache import binary_score


def test_audit_selection_is_score_blind_and_order_independent():
    refs = [
        {
            "id": f"{source}:{parent}:{step}",
            "parent_prompt_id": f"{source}:{parent}",
            "source": source,
            "end_character": parent * 10 + step,
        }
        for source in ("stride", "bash_arena", "bash_bench", "gloom")
        for parent in range(20)
        for step in range(1, 4)
    ]
    selected = select_cache_audit(refs)
    assert len(selected) == len({r["parent_prompt_id"] for r in selected}) == 64
    assert selected == select_cache_audit(list(reversed(refs)))
    cached = {r["id"]: {"logprob_0": -1.0, "logprob_1": -1.0} for r in selected}
    rows = [
        {
            "id": r["id"],
            "logprob_0": -1.0,
            "logprob_1": -1.0,
            "score": 0.5,
            "cached_score": 0.5,
        }
        for r in selected
    ]
    audit = {
        "contract_sha256": "contract",
        "selection": "prefix-audit-v1",
        "rows": rows,
    }
    validate_fresh_audit(audit, refs, cached, "contract")
    with pytest.raises(ValueError, match="coverage"):
        validate_fresh_audit({**audit, "rows": rows[:-1]}, refs, cached, "contract")
    with pytest.raises(ValueError, match="provenance"):
        validate_fresh_audit(
            {**audit, "rows": [{**rows[0], "score": 0.9}, *rows[1:]]},
            refs,
            cached,
            "contract",
        )
    with pytest.raises(ValueError, match="provenance"):
        validate_fresh_audit(
            {**audit, "rows": [{**rows[0], "score": float("nan")}, *rows[1:]]},
            refs,
            cached,
            "contract",
        )
    changed = {**rows[0], "logprob_1": 0.0, "score": binary_score(-1.0, 0.0)}
    assert validate_fresh_audit(
        {**audit, "rows": [changed, *rows[1:]]}, refs, cached, "contract",
        allow_numerical_failure=True,
    ) is False
    with pytest.raises(ValueError, match="coverage"):
        validate_fresh_audit(
            {**audit, "rows": rows[:-1]}, refs, cached, "contract",
            allow_numerical_failure=True,
        )
    with pytest.raises(ValueError, match="numerical"):
        validate_fresh_audit(
            {**audit, "rows": [changed, *rows[1:]]}, refs, cached, "contract"
        )
    with pytest.raises(ValueError, match="four"):
        select_cache_audit([r for r in refs if r["source"] == "stride"])
