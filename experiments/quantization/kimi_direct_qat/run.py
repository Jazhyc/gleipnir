"""Automatic direct Kimi QAT, final merge/parity and frozen ID scoring."""

from experiments.quantization.native_fp8_qat.run import main

if __name__ == "__main__":
    main("experiments/quantization/kimi_direct_qat/config.yaml")
