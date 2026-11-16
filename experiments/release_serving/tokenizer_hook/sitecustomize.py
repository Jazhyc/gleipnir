"""Tokenize prompts with the native Gigatoken encoder on vLLM 0.31.

``gleipnir.serving.gigatoken`` builds the native encoder for every renderer
and replaces ``BaseRenderer._encode``. vLLM 0.31 no longer calls that method:
it tokenizes in ``BaseRenderer._tokenize_prompt``. ``serve.py`` puts this
directory on ``PYTHONPATH`` and sets ``GLEIPNIR_TOKENIZER_HOOK=1``; the hook
then patches ``_tokenize_prompt`` when vLLM imports its renderer module, to
use the encoder that is already installed. Requests that ask for character
offsets, and renderers without the encoder, keep the Transformers tokenizer.
The first native call and every 1,000th write the call counts to
``GLEIPNIR_TOKENIZER_RECEIPT``.
"""

import importlib.abc
import importlib.util
import json
import os
import sys

TARGET = "vllm.renderers.base"


def patch(module) -> None:
    renderer = module.BaseRenderer
    original = renderer._tokenize_prompt
    receipt = os.environ.get("GLEIPNIR_TOKENIZER_RECEIPT")
    calls = {"native": 0, "transformers": 0}

    def tokenize(self, prompt, params):
        encoder = getattr(self, "_gleipnir_encoder", None)
        if encoder is None or self._wants_offsets(prompt, params):
            calls["transformers"] += 1
            return original(self, prompt, params)
        ids = encoder.encode(prompt["prompt"], **params.get_encode_kwargs())
        calls["native"] += 1
        if receipt and calls["native"] % 1000 == 1:
            with open(receipt, "w") as handle:
                json.dump({"backend": "gigatoken_native", **calls}, handle)
        return self._build_tokens_prompt(ids, prompt)

    renderer._tokenize_prompt = tokenize


class Finder(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path, target=None):
        if name != TARGET:
            return None
        sys.meta_path.remove(self)
        spec = importlib.util.find_spec(name)
        execute = spec.loader.exec_module

        def exec_module(module):
            execute(module)
            patch(module)

        spec.loader.exec_module = exec_module
        return spec


if os.environ.get("GLEIPNIR_TOKENIZER_HOOK") == "1":
    sys.meta_path.insert(0, Finder())
