import numpy as np
import pytest
import torch
from safetensors import safe_open
from safetensors.torch import save_file

from experiments.data_scaling.core import (
    fit_bounded_power_law,
    nested_stratified_selections,
    record_identity,
)
from experiments.data_scaling.evaluate import balanced_smoke_records
from experiments.data_scaling.rebase_adapters import (
    DESTINATION_PREFIX,
    SOURCE_PREFIX,
    rebase_adapter,
    rebase_key,
)
from experiments.data_scaling.run_lambda import balanced_lanes


def sample_records() -> list[dict[str, object]]:
    return [
        {"dataset": dataset, "index": index, "label": label}
        for dataset in ("a", "b")
        for label in (0, 1)
        for index in range(label * 100, label * 100 + 10)
    ]


def test_nested_selections_preserve_strata_and_are_nested() -> None:
    selections = nested_stratified_selections(
        sample_records(), [0.2, 0.5, 1.0], seed=7
    )
    identity_sets = [
        {record_identity(record) for record in selection}
        for selection in selections.values()
    ]
    assert len(selections[0.2]) == 8
    assert len(selections[0.5]) == 20
    assert len(selections[1.0]) == 40
    assert identity_sets[0] < identity_sets[1] < identity_sets[2]


def test_nested_selections_reject_duplicate_identities() -> None:
    record = {"dataset": "a", "index": 1, "label": 0}
    with pytest.raises(ValueError, match="duplicate"):
        nested_stratified_selections([record, dict(record)], [0.5], seed=0)


def test_bounded_power_law_recovers_synthetic_curve() -> None:
    sizes = np.asarray([100, 200, 400, 800, 1600, 3200], dtype=float)
    values = 0.97 - 1.5 * sizes ** (-0.6)
    fit = fit_bounded_power_law(sizes, values)
    assert fit["asymptote"] == pytest.approx(0.97, abs=0.002)
    assert fit["exponent"] == pytest.approx(0.6, abs=0.03)
    assert fit["rmse"] < 0.001


def test_lambda_lanes_balance_long_jobs_by_row_count() -> None:
    jobs = [
        {"job_name": str(index), "train_rows": rows}
        for index, rows in enumerate([100, 100, 50, 50, 25, 25])
    ]
    lanes = balanced_lanes(jobs, 2)
    loads = [sum(job["train_rows"] for _, job in lane) for lane in lanes]
    assert loads == [175, 175]
    assert sorted(index for lane in lanes for index, _ in lane) == list(range(6))


def test_rebase_key_targets_multimodal_language_model() -> None:
    suffix = "layers.3.self_attn.q_proj.lora_A.weight"
    assert rebase_key(SOURCE_PREFIX + suffix) == DESTINATION_PREFIX + suffix


def test_rebase_key_rejects_unknown_or_already_rebased_names() -> None:
    with pytest.raises(ValueError, match="unexpected causal adapter key"):
        rebase_key("base_model.model.visual.patch_embed.lora_A.weight")
    with pytest.raises(ValueError, match="already rebased"):
        rebase_key(DESTINATION_PREFIX + "layers.0.mlp.up_proj.lora_B.weight")


def test_rebase_adapter_preserves_values_and_records_checksums(tmp_path) -> None:
    source = tmp_path / "adapter"
    destination = tmp_path / "adapter_image_text_to_text"
    source.mkdir()
    (source / "adapter_config.json").write_text('{"r": 2}\n')
    key = SOURCE_PREFIX + "layers.0.mlp.down_proj.lora_A.weight"
    value = torch.arange(6, dtype=torch.float32).reshape(2, 3)
    save_file({key: value}, source / "adapter_model.safetensors")

    manifest = rebase_adapter(source, destination)

    with safe_open(
        destination / "adapter_model.safetensors", framework="pt"
    ) as handle:
        assert list(handle.keys()) == [
            DESTINATION_PREFIX + "layers.0.mlp.down_proj.lora_A.weight"
        ]
        assert torch.equal(handle.get_tensor(list(handle.keys())[0]), value)
    assert manifest["tensor_count"] == 1
    assert manifest["source_sha256"] != manifest["destination_sha256"]
    assert rebase_adapter(source, destination) == manifest


def test_balanced_smoke_records_uses_one_dataset_and_both_labels() -> None:
    records = [
        {"dataset": "one-sided", "label": 0, "index": index}
        for index in range(5)
    ] + [
        {"dataset": "balanced", "label": label, "index": index}
        for label in (0, 1)
        for index in range(5)
    ]

    selected = balanced_smoke_records(records, 8)

    assert len(selected) == 8
    assert {record["dataset"] for record in selected} == {"balanced"}
    assert [record["label"] for record in selected].count(0) == 4
    assert [record["label"] for record in selected].count(1) == 4
