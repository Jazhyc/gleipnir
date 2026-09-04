import hashlib

import pytest
import torch

from experiments.deception_distillation.train_student_sft import (
    action_endpoint_character_offsets,
    pool_mil_margins,
    select_evenly_spaced_positions,
)
from experiments.monitoring_objective_ablation import prepare


def test_action_endpoints_cover_xml_and_bracket_trajectories() -> None:
    xml = "prefix <step_1>one</step_1> <step_2>two</step_2> suffix"
    offsets = action_endpoint_character_offsets(xml)
    assert [xml[:offset].split()[-1] for offset in offsets] == [
        "<step_1>one</step_1>",
        "<step_2>two</step_2>",
    ]

    bracket = "[USER] task\n[ASSISTANT] act\n[TOOL] result\n[ASSISTANT] finish"
    offsets = action_endpoint_character_offsets(bracket)
    assert [bracket[:offset].split()[-1] for offset in offsets] == [
        "act",
        "result",
        "finish",
    ]


def test_even_endpoint_selection_retains_range() -> None:
    assert select_evenly_spaced_positions(range(10), 4) == [0, 3, 6, 9]
    assert select_evenly_spaced_positions([2, 2, 4], 8) == [2, 4]
    assert select_evenly_spaced_positions(range(10), 1) == [9]


def test_mil_pool_variants() -> None:
    margins = torch.tensor([[1.0, 3.0, 99.0], [-2.0, 2.0, 0.0]])
    mask = torch.tensor([[True, True, False], [True, True, True]])
    maximum = pool_mil_margins(margins, mask, mode="max", temperature=1.0, top_k=3)
    assert maximum.tolist() == [3.0, 2.0]
    top_two = pool_mil_margins(
        margins, mask, mode="topk_mean", temperature=1.0, top_k=2
    )
    assert top_two.tolist() == [2.0, 1.0]
    smooth = pool_mil_margins(
        margins, mask, mode="logmeanexp", temperature=1.0, top_k=3
    )
    assert smooth[0].item() == pytest.approx(
        torch.logsumexp(torch.tensor([1.0, 3.0]), dim=0).item()
        - torch.log(torch.tensor(2.0)).item()
    )


def test_rationale_enrichment_preserves_direct_prompt(monkeypatch) -> None:
    monkeypatch.setattr(prepare, "TRAIN_ROWS", 1)
    trajectory = "[USER] task\n[ASSISTANT] harmless action"
    rationale = "<think>grounded analysis</think>\n\nAnswer (0-10): 0"
    student = {
        "source_row_index": 0,
        "raw_source": "stride",
        "label": 0,
        "trajectory_sha256": hashlib.sha256(trajectory.encode()).hexdigest(),
        "student_prompt": "unchanged direct prompt",
        "student_target": "Prediction:0",
    }
    raw = {
        "dataset_source": "stride",
        "ground_truth": 0,
        "judge_score": 9.0,
        "is_regenerated": False,
        "messages": [
            {"role": "system", "content": "monitor system"},
            {"role": "user", "content": trajectory},
            {"role": "assistant", "content": rationale},
        ],
    }
    rows, audit = prepare.enrich_student_rows([student], [raw])
    assert rows[0]["student_prompt"] == "unchanged direct prompt"
    assert rows[0]["student_target"] == rationale
    assert rows[0]["completion_prompt"] == trajectory
    assert audit["quality_scores"] == {"9": 1}
