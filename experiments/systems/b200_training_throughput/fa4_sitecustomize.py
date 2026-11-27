"""Select isolated CuTe modules after the base environment's .pth processing."""

import sys
from pathlib import Path

sys.path.insert(
    0, str(Path(__file__).resolve().parent / "nvidia_cutlass_dsl" / "dsl_packages")
)
