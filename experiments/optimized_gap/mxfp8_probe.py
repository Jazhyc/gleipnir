"""Compile an MXFP8 attention plan in isolation (host/driver probe).

Without arguments: the 4B serving plan. With ``--geometry Q KV LENGTH``: the
config-derived plan the 27B quantized stack uses.
"""

import argparse

import torch

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--geometry", nargs=3, type=int)
    args = parser.parse_args()
    print("device", torch.cuda.get_device_name(), torch.version.cuda, flush=True)
    if args.geometry:
        from gleipnir.serving.mxfp8_geometry import forward_plan

        forward_plan(torch.device("cuda"), *args.geometry)
    else:
        from gleipnir.serving.mxfp8 import forward_plan

        forward_plan(torch.device("cuda"))
    print("mxfp8_plan_compiled", flush=True)
