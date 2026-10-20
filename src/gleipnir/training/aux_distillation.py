"""Top-k-plus-tail KL distillation for auxiliary continuation tokens.

The teacher distribution at each continuation position is stored as its top-k
token log-probabilities; the remaining mass is one tail bucket. The student
is compared on the same k ids plus its own tail, so the KL is exact over the
k+1-way partition and never needs the teacher's full vocabulary.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch

# Stored top-k log-probabilities are rounded; mass slightly above 1 is noise.
TAIL_TOLERANCE = 1e-3
CHUNK = 256


def topk_tail_kl(
    logits: torch.Tensor,
    top_ids: torch.Tensor,
    top_logprobs: torch.Tensor,
    mask: torch.Tensor,
    aux_lengths: Sequence[int],
) -> torch.Tensor:
    """Per-row mean KL(teacher || student) over supervised continuation positions.

    ``logits`` is (M, V) at the M continuation positions of the packed rows, in
    row order; ``aux_lengths`` splits M by row. Rows without a supervised
    position contribute 0.
    """
    positions = logits.shape[0]
    if top_ids.shape != top_logprobs.shape or top_ids.shape[0] != positions:
        raise ValueError("auxiliary targets do not match the continuation positions")
    if mask.shape != (positions,) or sum(aux_lengths) != positions:
        raise ValueError("auxiliary mask or lengths do not match the positions")
    teacher_lp = top_logprobs.float()
    teacher_top = teacher_lp.exp()
    top_mass = teacher_top.sum(-1)
    if bool((top_mass > 1 + TAIL_TOLERANCE).any()):
        raise ValueError("teacher top-k mass exceeds 1")
    teacher_tail = (1 - top_mass).clamp_min(0)
    parts = []
    for start in range(0, positions, CHUNK):
        chunk = slice(start, start + CHUNK)
        student = logits[chunk].float()
        student_lp = student.gather(-1, top_ids[chunk]) - student.logsumexp(-1, True)
        covered = student_lp.logsumexp(-1)
        # log(1 - exp(covered)), stable as the student's top-k mass nears 1.
        student_tail = torch.log(-torch.expm1(covered.clamp_max(-1e-7)))
        tail = teacher_tail[chunk]
        kl = (teacher_top[chunk] * (teacher_lp[chunk] - student_lp)).sum(-1)
        kl = kl + torch.xlogy(tail, tail) - tail * student_tail
        parts.append(kl)
    per_position = torch.cat(parts) * mask.float()
    rows = torch.split(per_position, list(aux_lengths))
    counts = torch.split(mask.float(), list(aux_lengths))
    return torch.stack(
        [
            row.sum() / count.sum().clamp_min(1)
            for row, count in zip(rows, counts, strict=True)
        ]
    )
