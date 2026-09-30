"""Fail-closed coverage and preflight gates for the Runpod ID smoke campaign."""

import json

import pytest

from experiments.runpod_gleipnir4b_id import run as campaign


@pytest.mark.parametrize(
    "text",
    [
        "gdn_prefill_backend: flashinfer",
        "Using Triton/FLA GDN prefill kernel (requested=flashinfer)",
        "Using FLASHINFER attention backend",
    ],
)
def test_requested_backend_and_attention_backend_do_not_prove_gdn_activation(
    tmp_path, text: str
) -> None:
    log = tmp_path / "vllm.log"
    log.write_text(text)
    with pytest.raises(RuntimeError, match="did not activate"):
        campaign.require_flashinfer_prefill(log)


def test_active_flashinfer_gdn_backend_passes(tmp_path) -> None:
    log = tmp_path / "vllm.log"
    log.write_text("Using FlashInfer GDN prefill kernel (requested=flashinfer)")
    campaign.require_flashinfer_prefill(log)


@pytest.mark.parametrize(
    "predictions",
    [
        [],
        [{"id": "a"}, {"id": "a"}],
        [{"id": "unexpected"}],
    ],
)
def test_summary_refuses_missing_duplicate_or_foreign_ids(
    monkeypatch: pytest.MonkeyPatch, predictions: list[dict]
) -> None:
    monkeypatch.setattr(campaign, "load_json", lambda _: {})
    monkeypatch.setattr(campaign, "validate_inputs", lambda _: [{"id": "a"}])
    monkeypatch.setattr(campaign, "load_jsonl", lambda _: predictions)
    with pytest.raises(ValueError, match="coverage"):
        campaign.summarize()


def test_kernel_failure_stops_before_model_or_evaluation(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    monkeypatch.setattr(campaign, "OUTPUT", tmp_path)
    monkeypatch.setattr(campaign, "LOGS", tmp_path / "logs")
    monkeypatch.setattr(
        campaign,
        "load_json",
        lambda _: {"config_sha256": "frozen", "jobs_sha256": "frozen"},
    )
    monkeypatch.setattr(campaign, "sha256_file", lambda _: "frozen")
    monkeypatch.setattr(campaign, "validate_inputs", lambda _: [])
    monkeypatch.setattr(campaign, "validate_jobs", lambda *_: [{}])
    monkeypatch.setattr(campaign, "adapter_metadata", lambda _: {})
    monkeypatch.setenv("FLA_DISABLE_BACKEND_DISPATCH", "1")

    def kernel_failure(*_args, **_kwargs):
        raise RuntimeError("GPU kernel failed")

    def forbidden_model_call(*_args, **_kwargs):
        pytest.fail("Model execution must not occur after failed kernels")

    monkeypatch.setattr(
        campaign, "ensure_qwen35_long_trajectory_kernels", kernel_failure
    )
    monkeypatch.setattr(campaign.subprocess, "run", forbidden_model_call)
    with pytest.raises(RuntimeError, match="GPU kernel failed"):
        campaign.run()
    status = json.loads((tmp_path / "status.json").read_text())
    assert status["state"] == "failed"
    assert status["phase"] == "failed"
