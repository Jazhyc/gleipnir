"""Reuse the frozen resident Trainer with the direct FROST intervention."""

import os
from pathlib import Path

from experiments.b200_mlp_gemm.resident_launch import main

if __name__ == "__main__":
    os.environ["GLEIPNIR_FP4_RESIDENT_CANDIDATE"] = str(
        Path(__file__).with_name("candidate.py").resolve()
    )
    main()
