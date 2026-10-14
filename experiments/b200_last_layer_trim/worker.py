"""Long-context FP8 default with last-query attention in the final layer."""

import json
import os
import re
from pathlib import Path

from experiments.b200_attention_gdn_serving.worker import write
from experiments.b200_context_million.worker import LongContextWorker

ROOT = Path(__file__).resolve().parents[2]


def validate(receipt: dict) -> None:
    """Require complete bitwise kernel coverage from unchanged executed sources."""
    from experiments.b200_inference_benchmark.run import sha

    if not receipt.get("passed") or receipt.get("mismatched_cases"):
        raise ValueError("last-query kernel equivalence failed")
    if receipt.get("bitwise_cases", 0) != 6:
        raise ValueError("last-query kernel equivalence coverage incomplete")
    for path, expected in receipt["sources"].items():
        if sha(ROOT / path) != expected:
            raise ValueError(f"last-query source changed: {path}")


class LastLayerTrimWorker(LongContextWorker):
    def load_model(self, *, load_dummy_weights: bool = False) -> None:
        from vllm.model_executor.layers.attention import Attention
        from vllm.v1.attention.backends.flashinfer import TRTLLMPrefill

        from gleipnir.serving.last_query_attention import install

        receipt = json.loads(
            (ROOT / os.environ["GLEIPNIR_LAST_QUERY_VALIDATION"]).read_text()
        )
        validate(receipt)
        super().load_model(load_dummy_weights=load_dummy_weights)
        model_config = self.vllm_config.model_config
        pooler = model_config.pooler_config
        pooling = getattr(pooler, "seq_pooling_type", None) or getattr(
            pooler, "pooling_type", None
        )
        text = model_config.hf_text_config
        if pooling != "LAST" or not getattr(text, "is_causal", False):
            raise ValueError("last-query attention requires causal LAST pooling")
        layers = {}
        for name, module in self.model_runner.get_model().named_modules():
            match = re.search(r"layers\.(\d+)\.self_attn\.attn$", name)
            if match and isinstance(module, Attention):
                layers[int(match.group(1))] = (name, module)
        final = text.num_hidden_layers - 1
        if (
            max(layers) != final
            or text.layer_types[final] != "full_attention"
            or len(layers) != 8
        ):
            raise ValueError("final decoder layer is not the last full attention")
        name, layer = layers[final]
        self.last_query = {
            "passed": False,
            "layer": name,
            "worker_pid": os.getpid(),
            "validation_sha256": receipt["sha256"],
            "calls": 0,
            "trimmed_query_rows": 0,
            "attended_query_rows": 0,
            "max_history_tokens": 0,
        }

        def observe(call: dict) -> None:
            audit = self.last_query
            audit["calls"] += 1
            audit["trimmed_query_rows"] += call["tokens"]
            audit["attended_query_rows"] += call["decode_tokens"] + call["prefills"]
            audit["max_history_tokens"] = max(
                audit["max_history_tokens"], call["max_history_tokens"]
            )
            if audit["calls"] == 1:
                audit["passed"] = True
                write("last_query_attention.json", audit)

        install(layer, prefill_type=TRTLLMPrefill, observe=observe)
        self.precision["last_query_attention"] = {"layer": name}
        self.audit_serving_state()
        write("last_query_attention.json", self.last_query)
        print("last_query_attention_installed", name, flush=True)

    def last_query_audit(self) -> dict:
        return dict(self.last_query)
