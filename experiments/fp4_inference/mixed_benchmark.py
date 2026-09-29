"""Register the mixed FP8 recipe before the unchanged serving benchmark."""

import importlib

from experiments.local_inference.benchmark import main

importlib.import_module("gleipnir.vllm_mixed_fp8")

if __name__ == "__main__":
    main()
