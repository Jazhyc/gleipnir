"""Profiling preserves precision, schedule, and custom quantizer registration."""

from experiments.local_inference.profile import profiling_engine


def test_profile_preserves_configuration_without_mutation():
    config = {"engine": {"dtype": "bfloat16", "max_num_batched_tokens": 2048}}
    engine = profiling_engine(config, False)
    engine["dtype"] = "changed"
    assert config["engine"]["dtype"] == "bfloat16"
    assert profiling_engine(config, True)["worker_cls"].endswith("EarlyCuptiWorker")


def test_custom_profile_worker_keeps_fp4_registration():
    config = {"engine": {"quantization": "gleipnir_nvfp4", "worker_cls": "original"}}
    engine = profiling_engine(config, True)
    assert engine["worker_cls"].endswith("Fp4ProfileWorker")
    assert engine["quantization"] == "gleipnir_nvfp4"
    assert config["engine"]["worker_cls"] == "original"
