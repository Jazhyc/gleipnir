"""Check that reused serving parity fails closed on changed artifacts/runtimes."""

import json

import pytest

from gleipnir.monitoring_campaign_data import file_hash
from gleipnir.monitoring_campaign_evaluation import validate_parity_reuse


def fixture_receipt(tmp_path, *, mutation=None):
    limits = {
        "min_correlation": 0.99,
        "max_mean_absolute_difference": 0.02,
        "min_adapter_effect": 1e-6,
    }
    runtime = {
        "vllm": "0.24.0",
        "torch": "2.11.0+cu130",
        "gpu": "NVIDIA B200",
        "gdn_prefill_backend": "flashinfer",
    }
    reference = {"master_sha256": "master", "ids": ["a"], "prompt_sha256": ["p"]}
    parity = {
        "passed": True,
        "limits": limits,
        "gdn_prefill_backend": "flashinfer",
        "serving_sha256": "serving",
        "reference": reference,
        "comparisons": {
            k: {"correlation": 0.999, "mean_absolute_difference": 0.001}
            for k in ("base", "adapter")
        },
        "effects": {"eager": 0.3, "vllm": 0.3},
    }
    result = {"runtime": runtime, "model": {"id": "model"}, "rows": 3012}
    if mutation:
        mutation(parity, result)
    files = {
        "serving_parity.json": parity,
        "parity_reference.json": reference,
        "model/rebase_manifest.json": {
            "source_sha256": "master",
            "destination_sha256": "serving",
        },
        "id/result.json": result,
    }
    for name, value in files.items():
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value))
    receipt = {"files_sha256": {name: file_hash(tmp_path / name) for name in files}}
    return receipt, {"parity": limits, "models": {"4b": {"id": "model"}}}, dict(runtime)


def test_reuse_records_prior_pass_without_a_new_probe(tmp_path):
    receipt, config, runtime = fixture_receipt(tmp_path)
    reused = validate_parity_reuse(tmp_path, receipt, config, runtime, "flashinfer")
    assert reused["performed_this_run"] is False
    assert reused["prior_passed"] is True
    assert reused["reference_sha256"] == receipt["files_sha256"]["serving_parity.json"]


@pytest.mark.parametrize(
    "mutation",
    [
        lambda p, r: p.update(passed=False),
        lambda p, r: p.update(gdn_prefill_backend="triton"),
        lambda p, r: p["comparisons"]["adapter"].update(correlation=float("nan")),
        lambda p, r: p["effects"].update(vllm=0),
        lambda p, r: r.update(model={"id": "changed"}),
    ],
)
def test_reuse_rejects_failed_or_changed_receipts(tmp_path, mutation):
    receipt, config, runtime = fixture_receipt(tmp_path, mutation=mutation)
    with pytest.raises(ValueError):
        validate_parity_reuse(tmp_path, receipt, config, runtime, "flashinfer")


def test_reuse_rejects_current_runtime_or_frozen_file_drift(tmp_path):
    receipt, config, runtime = fixture_receipt(tmp_path)
    runtime["vllm"] = "changed"
    with pytest.raises(ValueError, match="backend/reference"):
        validate_parity_reuse(tmp_path, receipt, config, runtime, "flashinfer")
    (tmp_path / "serving_parity.json").write_text("{}")
    with pytest.raises(ValueError, match="artifact drift"):
        validate_parity_reuse(tmp_path, receipt, config, runtime, "flashinfer")
