"""Independent tensor definition of sequence-local 32x32 block scaling."""

import torch
import torch.nn.functional as F


def square_reference(source: torch.Tensor, lengths: tuple[int, ...], maximum: int):
    """Return FP8 payload and canonical/compact scales from block maxima."""
    heads, padded = source.shape[1], ((maximum + 127) // 128) * 128
    tiles = padded // 128
    payloads, rows, cols, packed_rows, packed_cols = [], [], [], [], []
    start = 0
    for length in lengths:
        x = source[start : start + length].transpose(0, 1).float()
        x = F.pad(x, (0, 0, 0, padded - length))
        blocks = x.reshape(heads, padded // 32, 32, 8, 32)
        amax = blocks.abs().amax(dim=(2, 4))
        ratio_bits = (amax * (1.0 / 448.0)).view(torch.int32)
        sf = ((ratio_bits >> 23) & 255) + ((ratio_bits & 0x7FFFFF) != 0)
        inverse = ((254 - sf) << 23).view(torch.float32)
        payload = (blocks * inverse[:, :, None, :, None]).reshape(heads, padded, 256)
        payloads.append(payload.to(torch.float8_e4m3fn)[:, :length].transpose(0, 1))
        row = (
            sf[:, :, None, :]
            .expand(heads, padded // 32, 32, 8)
            .reshape(heads, padded, 8)
        )
        col = (
            sf.transpose(1, 2)[:, :, None, :]
            .expand(heads, 8, 32, padded // 32)
            .reshape(heads, 256, padded // 32)
        )
        row = (
            row.reshape(heads, tiles, 4, 32, 2, 4)
            .permute(0, 1, 4, 3, 2, 5)
            .reshape(heads, tiles, 1024)
            .to(torch.uint8)
        )
        col = (
            col.reshape(heads, 2, 4, 32, tiles, 4)
            .permute(1, 0, 4, 3, 2, 5)
            .reshape(2, heads, tiles, 512)
            .to(torch.uint8)
        )
        rows.append(row)
        cols.append(col)
        live = (length + 127) // 128
        packed_rows.append(row[:, :live])
        packed_cols.append(
            col[:, :, :live].permute(1, 2, 0, 3).reshape(heads, live, 1024)
        )
        start += length
    if start != source.shape[0]:
        raise ValueError("lengths do not cover source")
    return (
        torch.cat(payloads).contiguous(),
        torch.stack(rows).flatten(),
        torch.stack(cols, dim=1).flatten(),
        torch.cat(packed_rows, dim=1),
        torch.cat(packed_cols, dim=1),
    )
