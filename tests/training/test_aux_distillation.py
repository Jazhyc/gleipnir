import math

import pytest
import torch

from gleipnir.training.aux_distillation import topk_tail_kl


def reference_kl(teacher: torch.Tensor, student_logits: torch.Tensor, ids: list[int]):
    """Exact KL over the {ids..., tail} partition from full distributions."""
    q = student_logits.softmax(-1)
    rest = [i for i in range(len(teacher)) if i not in ids]
    p_parts = [float(teacher[i]) for i in ids] + [float(teacher[rest].sum())]
    q_parts = [float(q[i]) for i in ids] + [float(q[rest].sum())]
    return sum(
        p * math.log(p / q) for p, q in zip(p_parts, q_parts, strict=True) if p > 0
    )


def test_matches_hand_computed_partition_kl() -> None:
    teacher = torch.tensor([0.5, 0.2, 0.15, 0.1, 0.05])
    ids = [0, 2]
    student = torch.tensor([[1.0, 0.0, 2.0, -1.0, 0.5]])
    result = topk_tail_kl(
        student,
        torch.tensor([ids]),
        teacher[ids].log()[None],
        torch.tensor([True]),
        [1],
    )
    assert result.shape == (1,)
    assert math.isclose(
        float(result[0]), reference_kl(teacher, student[0], ids), rel_tol=1e-5
    )


def test_zero_when_student_matches_teacher() -> None:
    logits = torch.randn(3, 7)
    lp = logits.log_softmax(-1)
    top = lp.topk(3, -1)
    result = topk_tail_kl(
        logits, top.indices, top.values, torch.ones(3, dtype=torch.bool), [3]
    )
    assert float(result.abs().max()) < 1e-6


def test_mask_and_rows_without_supervision() -> None:
    logits = torch.randn(4, 6)
    top = torch.randn(4, 6).log_softmax(-1).topk(2, -1)
    mask = torch.tensor([True, False, False, False])
    rows = topk_tail_kl(logits, top.indices, top.values, mask, [2, 2])
    alone = topk_tail_kl(logits[:1], top.indices[:1], top.values[:1], mask[:1], [1])
    assert torch.allclose(rows[0], alone[0])
    assert float(rows[1]) == 0.0


def test_teacher_mass_rounding_is_clamped_but_excess_raises() -> None:
    logits = torch.zeros(1, 4)
    ids = torch.tensor([[0, 1]])
    near_one = torch.tensor([[math.log(0.5002), math.log(0.5002)]])
    assert torch.isfinite(
        topk_tail_kl(logits, ids, near_one, torch.tensor([True]), [1])
    ).all()
    with pytest.raises(ValueError, match="exceeds 1"):
        topk_tail_kl(logits, ids, torch.tensor([[0.0, 0.0]]), torch.tensor([True]), [1])


def test_gradient_finite_when_student_tail_vanishes() -> None:
    logits = torch.tensor([[30.0, 30.0, -30.0, -30.0]], requires_grad=True)
    ids = torch.tensor([[0, 1]])
    teacher = torch.tensor([[math.log(0.45), math.log(0.45)]])
    loss = topk_tail_kl(logits, ids, teacher, torch.tensor([True]), [1]).sum()
    loss.backward()
    assert torch.isfinite(loss) and torch.isfinite(logits.grad).all()
