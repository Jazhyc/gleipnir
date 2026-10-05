"""Fresh bounded model screen for the fused-preparation MXFP8 recipes."""

from pathlib import Path


def accept_native(receipt: dict, *, square: bool) -> dict:
    """Retain strict numerical failures while requiring execution correctness."""
    if (
        receipt.get("status") != "execution_complete"
        or receipt.get("square") is not square
    ):
        raise ValueError("native recipe execution incomplete or wrong precision mode")
    if receipt.get("finite") is not True or "exception" in receipt:
        raise ValueError("native recipe is nonfinite or failed")
    checks = receipt.get("quantizer_checks", [])
    if len(checks) != 2 or {x["heads"] for x in checks} != {16, 4}:
        raise ValueError("native producer checks are incomplete")
    key = "seven_reference_layouts_bitexact" if square else "seven_layouts_bitexact"
    if any(len(x.get(key, [])) != 7 or not all(x[key]) for x in checks):
        raise ValueError("native producer layout/reference check failed")
    if square and not all(x.get("shared_payload") is True for x in checks):
        raise ValueError("square payload was not shared")
    if not square:
        old = receipt["old_native_comparison"]
        if old["forward_relative_l2"] != 0 or old["gradient_relative_l2"] != [0, 0, 0]:
            raise ValueError("dual fusion changed original arithmetic")
    for gate in ("isolation", "graph_replay"):
        if receipt[gate]["passed"] is not True:
            raise ValueError(f"native {gate} failed")
    poison = receipt["poisoned_dead_storage"]
    if poison["forward_relative_l2"] != 0 or poison["gradient_relative_l2"] != [
        0,
        0,
        0,
    ]:
        raise ValueError("poisoned storage changes arithmetic")
    return {
        "execution_correct": True,
        "strict_parity_passed": receipt["fp32_comparison"]["strict_passed"],
        "fp32_comparison": receipt["fp32_comparison"],
    }


def main() -> None:
    from experiments.b200_nvidia_mxfp8_varlen.training_screen import main as shared_main

    shared_main(Path(__file__).with_name("training_config.yaml"))


if __name__ == "__main__":
    main()
