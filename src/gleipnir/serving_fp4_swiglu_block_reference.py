"""Independent mathematical reference for unit-global-scale NVFP4 output."""


def reference_decode(value):
    """Round E4M3 scales and E2M1 values mathematically, including even ties.

    Torch comparisons intentionally avoid native conversion/packing code.
    Input may be a fixed contiguous-column subset, in complete 16-value blocks.
    """
    import torch

    if value.ndim != 2 or value.shape[1] % 16:
        raise ValueError("FP4 reference needs complete 16-value blocks")
    v = value.float().reshape(value.shape[0], -1, 16)
    # The native conversion is satfinite: Torch's ordinary overflowing FP8
    # cast instead emits NaN, so explicitly implement finite clipping here.
    # A device denominator avoids Torch's CPU-scalar reciprocal rewrite, which
    # can move an exact scale midpoint before the E4M3 ties-to-even conversion.
    six = torch.tensor(6.0, device=v.device, dtype=torch.float32)
    sf = (
        (v.abs().amax(-1, keepdim=True) / six)
        .clamp(max=448)
        .to(torch.float8_e4m3fn)
        .float()
    )
    normalized = v / torch.where(sf > 0, sf, 1.0)
    boundaries = torch.tensor([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0], device=v.device)
    values = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], device=v.device)
    magnitude = normalized.abs().contiguous()
    index = torch.bucketize(magnitude, boundaries)
    tie = (index < 7) & (magnitude == boundaries[index.clamp(max=6)])
    index = index + (tie & (index % 2 == 1)).long()
    decoded = values[index] * normalized.sign() * sf
    return decoded.reshape(value.shape)
