"""Forward-only adapters for measured GDN serving backend experiments."""

from typing import Any


def install_nvvm_compatibility(nvvm: Any) -> dict[str, str]:
    """Bridge two CuTe 4.8 NVVM renames without changing native operations."""
    patches = {}
    if not hasattr(nvvm, "Tcgen05GroupKind"):
        group = nvvm.CTAGroupKind
        if int(group.CTA_1) != 0 or int(group.CTA_2) != 1:
            raise ValueError("unexpected NVVM CTA group values")
        nvvm.Tcgen05GroupKind = group
        patches["Tcgen05GroupKind"] = "CTAGroupKind"
    if not hasattr(nvvm, "tcgen05_commit_arrive"):
        nvvm.tcgen05_commit_arrive = nvvm.tcgen05_commit
        patches["tcgen05_commit_arrive"] = "tcgen05_commit"
    return patches


def make_flashqla_prefill(function: Any, normalize: Any, *, auto_cp: bool) -> Any:
    """Preserve vLLM's log gates and V-first persistent state for FlashQLA."""

    def forward(
        q: Any,
        k: Any,
        v: Any,
        g: Any,
        beta: Any,
        initial_state: Any,
        output_final_state: bool,
        cu_seqlens: Any = None,
        use_qk_l2norm_in_kernel: bool = True,
    ) -> tuple[Any, Any]:
        if use_qk_l2norm_in_kernel:
            q, k = normalize(q.contiguous()), normalize(k.contiguous())
        _, _, output, _, state, _ = function(
            q=q.contiguous(),
            k=k.contiguous(),
            v=v.contiguous(),
            g=g.float().contiguous(),
            beta=beta.float().contiguous(),
            initial_state=initial_state.float().contiguous(),
            output_final_state=output_final_state,
            cu_seqlens=cu_seqlens,
            state_v_first=True,
            auto_cp=auto_cp,
            enable_fwd_cp_cache=False,
            output_h=False,
            is_train=False,
        )
        return output, state if output_final_state else None

    return forward
