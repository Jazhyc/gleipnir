"""Register the experimental online NVFP4 method before the standard benchmark."""

import importlib

from experiments.local_inference.benchmark import main

if __name__ == "__main__":
    importlib.import_module("gleipnir.vllm_nvfp4")
    main()
