"""Bounded SM100 FP4 tile choices; preserve the validated projection graph."""

from __future__ import annotations

BASE_TILE = "CONFIG_sm100_128x256x128_128x256x32_cluster2x1_2ctamma"
CANDIDATES = (
    BASE_TILE,
    "CONFIG_sm100_128x256x128_128x256x32_cluster1x1_1ctamma",
    "CONFIG_sm100_128x128x128_128x128x32_cluster1x1_1ctamma",
    "CONFIG_sm100_128x128x128_128x128x32_cluster2x1_2ctamma",
    "CONFIG_sm100_256x256x128_128x256x32_cluster2x1_2ctamma",
    "CONFIG_sm100_128x256x128_128x256x32_cluster4x1_2ctamma",
)
SHAPES = {
    "mlp_gate_up": (2560, 18432),
    "gdn_input": (2560, 12288),
    "mlp_down": (9216, 2560),
    "gdn_output": (4096, 2560),
}


def shape_key(k: int, n: int) -> str:
    """Reject projections outside the measured four-shape envelope."""
    for name, dims in SHAPES.items():
        if dims == (k, n):
            return name
    raise ValueError(f"unmeasured FP4 projection: K{k}, N{n}")


def row_band(m: int) -> str:
    """Use a frozen small/large split within the validated serving envelope."""
    if not 1 <= m <= 32768:
        raise ValueError(f"FP4 tuning rows outside validated envelope: {m}")
    return "small" if m <= 4096 else "large"


def retile(plan, tile: str):
    """Compile the same BF16-rounding/descaling graph with a pinned NVIDIA tile."""
    import torch
    from cudnn.gemm.frost.compiler import jit_from_cudnn_graph
    from cudnn.gemm.frost.tile_config import by_name

    if tile not in CANDIDATES:
        raise ValueError(f"tile outside bounded tuning catalog: {tile}")
    plan.jit = jit_from_cudnn_graph(plan.graph, config=by_name(tile))
    if plan.jit.config.name != tile:
        raise ValueError("NVIDIA resolved a different tile")
    plan.workspace_bytes = int(getattr(plan.jit, "workspace_bytes", 0) or 0)
    plan.workspace = torch.empty(plan.workspace_bytes, device="cuda", dtype=torch.uint8)
    return plan


class ShapeTunedGemm:
    """Reuse precompiled symbolic-M plans without runtime compilation/fallback."""

    def __init__(self, plans: dict, selected: dict, audit=None) -> None:
        if set(selected) != {"small", "large"}:
            raise ValueError("both measured row bands are required")
        if any(tile not in plans for tile in selected.values()):
            raise ValueError("selected tile has no precompiled plan")
        self.plans, self.selected, self.audit = plans, selected, audit

    def __call__(self, a, b):
        m = a.codes.shape[0]
        tile = self.selected[row_band(m)]
        if self.audit is not None:
            self.audit(m, tile)
        return self.plans[tile](a, b)
