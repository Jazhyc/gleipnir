"""Register training-forward FP4 before standard vLLM HTTP argument parsing."""

import runpy

from gleipnir.vllm_frost_fp4 import FrostFp4Config  # noqa: F401

if __name__ == "__main__":
    runpy.run_module("vllm.entrypoints.openai.api_server", run_name="__main__")
