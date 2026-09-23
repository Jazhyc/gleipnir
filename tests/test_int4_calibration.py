import torch

from experiments.int4_calibration.capture import select
from experiments.int4_calibration.screen import hadamard, quant, rotation


def test_rotation_preserves_product_with_signs():
    torch.manual_seed(1)
    x, w = torch.randn(5, 32), torch.randn(7, 32)
    h = hadamard(8, "cpu")
    signs = torch.tensor([1.0, -1.0, 1.0, -1.0, 1.0, 1.0, -1.0, -1.0])
    assert torch.allclose(
        rotation(x, h, signs) @ rotation(w, h, signs).t(), x @ w.t(), atol=1e-5
    )


def test_group_quantization_and_zero():
    x = torch.tensor([[0.0, 0.0, 7.0, -7.0, 70.0, -70.0, 0.0, 0.0]])
    assert torch.equal(quant(x, group=4), x)
    assert torch.equal(quant(torch.zeros(3, 16)), torch.zeros(3, 16))


def test_channel_rebalancing_preserves_product():
    torch.manual_seed(2)
    x, w = torch.randn(11, 16), torch.randn(7, 16)
    scale = x.abs().amax(0).sqrt() / w.abs().amax(0).sqrt()
    assert torch.allclose((x / scale) @ (w * scale).t(), x @ w.t(), atol=1e-5)


def test_fp8_and_clipping_are_finite_and_shape_preserving():
    x = torch.tensor([[0.0, 1.0, -1.0, 20.0], [0.0, 0.0, 0.0, 0.0]])
    for value in (quant(x, fp8=True), quant(x, clip=0.5)):
        assert value.shape == x.shape and torch.isfinite(value).all()
    assert quant(x, clip=0.5)[0, -1] == 10


def test_selection_excludes_ids_and_shared_trajectories():
    rows = [
        {
            "source": s,
            "label": label,
            "id": f"{s}{label}{i}",
            "tokens": i,
            "original_trajectory_sha256": f"{s}{label}{i}",
        }
        for s in ("a", "b")
        for label in (0, 1)
        for i in range(6)
    ]
    chosen = select(rows, [rows[0]])
    assert len(chosen) == 12
    assert rows[0]["id"] not in {r["id"] for r in chosen}
    assert sum(r["split"] == "heldout" for r in chosen) == 4
    assert len({r["original_trajectory_sha256"] for r in chosen}) == 12
