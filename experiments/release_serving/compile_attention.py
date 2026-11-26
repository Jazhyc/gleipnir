"""Compile a target's MXFP8 attention plan outside the server.

The image ships this kernel compiled for the B200. Another GPU needs its own
build, and compiling it inside the server fails with the CuTeDSL
``TYPE_UNSTABLE_JOIN`` error when the cuDNN-frontend cache is cold. Compiled
here, in a process of its own, it fills that cache and the server then loads
it. ``serve.py`` runs this with the server's environment before the first
start on a GPU other than the validated one.
"""

import argparse

import torch

RECORDED_LENGTH = 32768

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--heads",
        nargs=2,
        type=int,
        metavar=("QUERY", "KV"),
        help="the config-derived plan of the 27B stack; default: the 4B plan",
    )
    parser.add_argument(
        "--length",
        type=int,
        default=RECORDED_LENGTH,
        help="the server's context limit, which bounds the plan",
    )
    args = parser.parse_args()
    from gleipnir.serving import mxfp8

    if mxfp8.MAX_LENGTH != RECORDED_LENGTH:
        raise SystemExit("original MXFP8 envelope changed")
    length = max(args.length, RECORDED_LENGTH)
    if args.heads:
        from gleipnir.serving.mxfp8_geometry import forward_plan

        forward_plan(torch.device("cuda"), *args.heads, length)
    else:
        mxfp8.MAX_LENGTH = length
        mxfp8.forward_plan(torch.device("cuda"))
    print(f"gleipnir_attention_compiled length={length}", flush=True)
