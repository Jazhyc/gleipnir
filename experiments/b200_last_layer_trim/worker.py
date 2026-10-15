"""Long-context FP8 default with final-layer work trimmed to pooled rows."""

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
    from experiments.b200_last_layer_trim.native import CASES, SUBSETS

    if not receipt.get("passed") or receipt.get("mismatched_cases"):
        raise ValueError("last-query kernel equivalence failed")
    required = {name for name, *_ in CASES} | {f"{name}_subset" for name in SUBSETS}
    if (
        receipt.get("bitwise_cases") != len(required)
        or {c["name"] for c in receipt.get("cases", [])} != required
    ):
        raise ValueError("last-query kernel equivalence coverage incomplete")
    for path, expected in receipt["sources"].items():
        if sha(ROOT / path) != expected:
            raise ValueError(f"last-query source changed: {path}")


class LastLayerTrimWorker(LongContextWorker):
    """Last-query attention in the final layer (all prefill requests)."""

    finished_only = False
    trim_tail = False

    def load_model(self, *, load_dummy_weights: bool = False) -> None:
        from vllm.forward_context import get_forward_context
        from vllm.model_executor.layers.attention import Attention
        from vllm.v1.attention.backends.flashinfer import TRTLLMPrefill

        from gleipnir.serving import last_query_tail
        from gleipnir.serving.last_query_attention import LastQueryState, install

        receipt = json.loads(
            (ROOT / os.environ["GLEIPNIR_LAST_QUERY_VALIDATION"]).read_text()
        )
        validate(receipt)
        if self.trim_tail:
            last_query_tail.register()
        super().load_model(load_dummy_weights=load_dummy_weights)
        model_config = self.vllm_config.model_config
        pooler = model_config.pooler_config
        pooling = getattr(pooler, "seq_pooling_type", None) or getattr(
            pooler, "pooling_type", None
        )
        text = model_config.hf_text_config
        if pooling != "LAST" or not getattr(text, "is_causal", False):
            raise ValueError("last-query attention requires causal LAST pooling")
        if self.trim_tail and (
            last_query_tail.SPLITTING_OP
            not in self.vllm_config.compilation_config.splitting_ops
        ):
            raise ValueError("last-query tail is not a compilation splitting op")
        layers, decoders = {}, {}
        for name, module in self.model_runner.get_model().named_modules():
            match = re.search(r"layers\.(\d+)\.self_attn\.attn$", name)
            if match and isinstance(module, Attention):
                layers[int(match.group(1))] = (name, module)
            match = re.search(r"layers\.(\d+)$", name)
            if match:
                decoders[int(match.group(1))] = (name, module)
        final = text.num_hidden_layers - 1
        if (
            max(layers) != final
            or max(decoders) != final
            or text.layer_types[final] != "full_attention"
            or len(layers) != 8
        ):
            raise ValueError("final decoder layer is not the last full attention")
        name, layer = layers[final]
        state = LastQueryState()
        self.last_query = {
            "passed": False,
            "layer": name,
            "worker_pid": os.getpid(),
            "finished_only": self.finished_only,
            "trim_tail": self.trim_tail,
            "validation_sha256": receipt["sha256"],
            "calls": 0,
            "query_rows": 0,
            "attended_query_rows": 0,
            "prefills": 0,
            "attended_prefills": 0,
            "max_history_tokens": 0,
        }

        def observe(call: dict) -> None:
            audit = self.last_query
            audit["calls"] += 1
            audit["query_rows"] += call["tokens"]
            audit["attended_query_rows"] += (
                call["decode_tokens"] + call["attended_prefills"]
            )
            audit["prefills"] += call["prefills"]
            audit["attended_prefills"] += call["attended_prefills"]
            audit["max_history_tokens"] = max(
                audit["max_history_tokens"], call["max_history_tokens"]
            )
            if not audit["passed"]:
                audit["passed"] = True
                write("last_query_attention.json", audit)

        runner = self.model_runner

        def finished():
            batch = runner.input_batch
            count = batch.num_reqs
            return (
                batch.num_prompt_tokens[:count],
                runner.optimistic_seq_lens_cpu[:count].numpy(),
            )

        install(
            layer,
            prefill_type=TRTLLMPrefill,
            observe=observe,
            finished=finished if self.finished_only else None,
            state=state,
            current_context=get_forward_context,
        )
        if self.trim_tail:
            decoder_name, decoder = decoders[final]
            last_query_tail.install(decoder, decoder_name, state, get_forward_context)
            self.tail_name = decoder_name
        self.precision["last_query_attention"] = {
            "layer": name,
            "finished_only": self.finished_only,
            "trim_tail": self.trim_tail,
        }
        self.audit_serving_state()
        write("last_query_attention.json", self.last_query)
        print("last_query_attention_installed", name, self.last_query, flush=True)

    def last_query_audit(self) -> dict:
        audit = dict(self.last_query)
        if self.trim_tail:
            from gleipnir.serving import last_query_tail

            audit["tail"] = last_query_tail.audit(self.tail_name)
        return audit


class FinishedTrimWorker(LastLayerTrimWorker):
    """Final-layer attention only for requests finishing their prompt this step."""

    finished_only = True


class TailTrimWorker(FinishedTrimWorker):
    """Finished-only attention plus output projection/norm/MLP on pooled rows."""

    trim_tail = True
