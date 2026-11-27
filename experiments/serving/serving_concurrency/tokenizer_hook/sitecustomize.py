"""Route vLLM 0.31 prompt tokenization through the native Gigatoken encoder.

``gleipnir.serving.gigatoken`` replaces ``BaseRenderer._encode``, which vLLM
0.31 no longer calls: it tokenizes in ``BaseRenderer._tokenize_prompt``. With
``GLEIPNIR_TOKENIZER_HOOK=1`` and this directory on ``PYTHONPATH``, that
method uses the installed native encoder whenever the file named by
``GLEIPNIR_TOKENIZER_MODE`` holds ``native``. The released image does not
contain it yet.
"""

import importlib.abc
import importlib.util
import json
import os
import sys
import time

TARGET = "vllm.renderers.base"


def patch(module) -> None:
    renderer = module.BaseRenderer
    original = renderer._tokenize_prompt
    mode_path = os.environ["GLEIPNIR_TOKENIZER_MODE"]
    stats_path = os.environ["GLEIPNIR_TOKENIZER_STATS"]
    stats = {"native": [0, 0.0], "hf": [0, 0.0]}

    def tokenize(self, prompt, params):
        with open(mode_path) as handle:
            mode = handle.read().strip()
        encoder = getattr(self, "_gleipnir_encoder", None)
        before = time.perf_counter()
        if mode != "native" or encoder is None or self._wants_offsets(prompt, params):
            mode = "hf"
            result = original(self, prompt, params)
        else:
            ids = encoder.encode(prompt["prompt"], **params.get_encode_kwargs())
            result = self._build_tokens_prompt(ids, prompt)
        entry = stats[mode]
        entry[0] += 1
        entry[1] += time.perf_counter() - before
        if sum(v[0] for v in stats.values()) % 64 == 0:
            with open(stats_path, "w") as handle:
                json.dump(stats, handle)
        return result

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
