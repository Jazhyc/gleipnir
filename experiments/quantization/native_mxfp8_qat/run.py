"""Reuse the archived-stage automation with a separate frozen attention arm."""

from experiments.quantization.native_fp8_qat.run import main

if __name__ == "__main__":
    main("experiments/quantization/native_mxfp8_qat/config.yaml")
