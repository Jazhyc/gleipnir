"""Expose the already-pinned FlashQLA overlay to a serving subprocess."""

import runpy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / ".cache/kernels/flashqla-da06429"))

from gleipnir.vllm_frost_fp4 import FrostFp4Config  # noqa: E402,F401

if __name__ == "__main__":
    runpy.run_module("vllm.entrypoints.openai.api_server", run_name="__main__")
