"""Calibrate every family without consulting held-out activation errors."""

import argparse
import itertools
import json
from pathlib import Path

import torch

from experiments.local_inference.core import write_json
from gleipnir.qwen35_adapter_rebase import sha256_file


def quant(
    x: torch.Tensor, clip: float = 1, group: int = 0, fp8: bool = False
) -> torch.Tensor:
    shape = x.shape
    z = x.float().reshape(-1, group or shape[-1])
    limit = 448 if fp8 else 7
    scale = (z.abs().amax(-1, keepdim=True) * clip / limit).clamp_min(1e-12)
    values = (z / scale).clamp(-limit, limit)
    values = values.to(torch.float8_e4m3fn).float() if fp8 else values.round()
    return (values * scale).reshape(shape)


def hadamard(block: int, device: str = "cuda") -> torch.Tensor:
    h = torch.ones((1, 1), device=device)
    while h.shape[0] < block:
        h = torch.cat((torch.cat((h, h), 1), torch.cat((h, -h), 1)), 0)
    return h / block**0.5


def rotation(x: torch.Tensor, h: torch.Tensor, signs: torch.Tensor) -> torch.Tensor:
    return ((x.float().reshape(-1, h.shape[0]) * signs) @ h).reshape(x.shape)


def error(y: torch.Tensor, ref: torch.Tensor) -> dict:
    assert torch.isfinite(y).all()
    diff = y.float() - ref
    return {
        "relative_l2": (diff.norm() / ref.norm()).item(),
        "max_abs_over_rms": (diff.abs().max() / ref.square().mean().sqrt()).item(),
    }


def recipes() -> list[dict]:
    result = [
        {"family": name}
        for name in ("naive", "weight_only", "activation_only", "bf16", "fp8")
    ]
    result += [
        {"family": "clip", "xclip": x, "wclip": w}
        for x, w in itertools.product((1, 0.95, 0.9, 0.8, 0.7, 0.6), repeat=2)
    ]
    result += [{"family": "smooth", "alpha": a} for a in (0.25, 0.5, 0.75)]
    result += [{"family": "rotate", "block": b} for b in (128, 512)]
    result += [{"family": "group", "group": g} for g in (64, 128)]
    return result


def predict(
    x: torch.Tensor, w: torch.Tensor, recipe: dict, calibration: torch.Tensor
) -> torch.Tensor:
    family = recipe["family"]
    if family == "bf16":
        return (x.bfloat16() @ w.bfloat16().t()).float()
    if family == "smooth":
        alpha = recipe["alpha"]
        scale = (
            calibration.abs().amax(0).clamp_min(1e-5) ** alpha
            / w.abs().amax(0).clamp_min(1e-5) ** (1 - alpha)
        ).clamp(0.001, 1000)
        x, w = x / scale, w * scale
    if family == "rotate":
        h = hadamard(recipe["block"])
        gen = torch.Generator(device="cpu").manual_seed(20260924)
        signs = (torch.randint(0, 2, (recipe["block"],), generator=gen) * 2 - 1).to(
            x.device
        )
        x, w = rotation(x, h, signs), rotation(w, h, signs)
    qx = (
        x
        if family == "weight_only"
        else quant(x, recipe.get("xclip", 1), recipe.get("group", 0), family == "fp8")
    )
    qw = (
        w
        if family == "activation_only"
        else quant(w, recipe.get("wclip", 1), recipe.get("group", 0), family == "fp8")
    )
    return qx @ qw.t()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    manifest = json.loads((args.capture / "manifest.json").read_text())
    assert manifest["state"] == "complete" and len(manifest["completed"]) == 12
    assert sha256_file(args.capture / "weights.pt") == manifest["weights_sha256"]
    for row in manifest["completed"]:
        assert sha256_file(args.capture / f"row_{row['row']}.pt") == row["sha256"]
    weights = torch.load(args.capture / "weights.pt", weights_only=True)
    captures = [
        torch.load(args.capture / f"row_{i}.pt", weights_only=True) for i in range(12)
    ]
    torch.backends.cuda.matmul.allow_tf32 = False
    result = {
        "capture_manifest_sha256": sha256_file(args.capture / "manifest.json"),
        "recipes": recipes(),
        "projections": [],
        "note": "Dequantized numerical simulation; not native timing or judge AUROC.",
    }
    write_json(args.output / "protocol.json", result)
    for key, weight in weights.items():
        w = weight.cuda().float()
        arrays = {
            split: torch.cat(
                [
                    c[key]
                    for c, r in zip(captures, manifest["rows"], strict=True)
                    if r["split"] == split
                ]
            )
            .cuda()
            .float()
            for split in ("calibration", "heldout")
        }
        cal = arrays["calibration"]
        reference = {s: x @ w.t() for s, x in arrays.items()}
        # Verify local transforms preserve the unquantized function.
        invariance = {}
        for block in (128, 512):
            h = hadamard(block)
            signs = torch.ones(block, device="cuda")
            inv = error(
                rotation(cal, h, signs) @ rotation(w, h, signs).t(),
                reference["calibration"],
            )
            assert inv["relative_l2"] < 1e-5
            invariance[f"rotate_{block}"] = inv
        scale = (
            cal.abs().amax(0).clamp_min(1e-5).sqrt()
            / w.abs().amax(0).clamp_min(1e-5).sqrt()
        )
        inv = error((cal / scale) @ (w * scale).t(), reference["calibration"])
        assert inv["relative_l2"] < 1e-5
        invariance["smooth"] = inv
        trials = []
        for recipe in recipes():
            metrics = error(predict(cal, w, recipe, cal), reference["calibration"])
            trials.append({"recipe": recipe, "calibration": metrics})
        selected = {}
        for trial in trials:
            family = trial["recipe"]["family"]
            if (
                family not in selected
                or trial["calibration"]["relative_l2"]
                < selected[family]["calibration"]["relative_l2"]
            ):
                selected[family] = dict(trial)
        # Persist frozen choices before reading any held-out prediction error.
        write_json(
            args.output / f"{key}_selection.json",
            {"trials": trials, "selected": selected},
        )
        for entry in selected.values():
            entry["heldout"] = error(
                predict(arrays["heldout"], w, entry["recipe"], cal),
                reference["heldout"],
            )
        row = {"projection": key, "selected": selected, "invariance": invariance}
        result["projections"].append(row)
        write_json(args.output / "result.json", result)
        print(
            key,
            {
                k: (v["recipe"], round(v["heldout"]["relative_l2"], 5))
                for k, v in selected.items()
            },
            flush=True,
        )
    sensitive = sorted(
        result["projections"],
        key=lambda r: r["selected"]["naive"]["calibration"]["relative_l2"],
        reverse=True,
    )[:3]
    result["fallback_projections"] = [r["projection"] for r in sensitive]
    result["macro_heldout_relative_l2"] = {
        family: sum(
            r["selected"][family]["heldout"]["relative_l2"]
            for r in result["projections"]
        )
        / 6
        for family in result["projections"][0]["selected"]
    }
    for fallback in ("bf16", "fp8"):
        result["macro_heldout_relative_l2"][f"selective_{fallback}"] = (
            sum(
                r["selected"][
                    fallback
                    if r["projection"] in result["fallback_projections"]
                    else "naive"
                ]["heldout"]["relative_l2"]
                for r in result["projections"]
            )
            / 6
        )
    write_json(args.output / "result.json", result)
    print("macro", result["macro_heldout_relative_l2"], flush=True)


if __name__ == "__main__":
    main()
