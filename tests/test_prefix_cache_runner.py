import hashlib
import json
from types import SimpleNamespace

from experiments.monitoring_prefix_supervision.cache import run_cache


class Tokenizer:
    def apply_chat_template(self, messages, **kwargs):
        return messages[0]["content"] + "ASSISTANT"

    def encode(self, prompt, **kwargs):
        return list(range(len(prompt)))


class Engine:
    def __init__(self):
        self.batches = []

    def generate(self, prompts, sampling, **kwargs):
        self.batches.append(prompts)
        return [
            SimpleNamespace(
                outputs=[
                    SimpleNamespace(
                        logprobs=[
                            {
                                0: SimpleNamespace(logprob=-1.0),
                                1: SimpleNamespace(logprob=-1.0),
                            }
                        ],
                        token_ids=[1],
                    )
                ],
                prompt_token_ids=[1, 2],
                num_cached_tokens=1,
            )
            for _ in prompts
        ]


def test_full_cache_interleaves_and_resumes(tmp_path):
    source = tmp_path / "source.jsonl"
    grouped = {}
    records = []
    for parent in range(11):
        text = "first\nsecond\nfinal\n"
        records.append(
            {
                "prompt_id": parent,
                "student_prompt": f"<agent_trajectory>\n{text}</agent_trajectory>",
            }
        )
        grouped[parent] = []
        for end in (6, 13):
            user = f"rubric\n<agent_trajectory>\n{text[:end]}</agent_trajectory>\n"
            grouped[parent].append(
                {
                    "id": f"{parent}:{end}",
                    "parent_prompt_id": parent,
                    "end_character": end,
                    "prefix_sha256": hashlib.sha256(text[:end].encode()).hexdigest(),
                    "rendered_user_prompt_sha256": hashlib.sha256(
                        user.encode()
                    ).hexdigest(),
                }
            )
    source.write_text("".join(json.dumps(r) + "\n" for r in records))
    config = {"model": {"id": "mock"}, "prompt": {"assistant_suffix": "ASSISTANT"}}
    engine = Engine()
    args = (
        engine,
        Tokenizer(),
        None,
        [0, 1],
        grouped,
        source,
        "rubric",
        config,
        {"instruction_sha256": "test"},
        tmp_path / "output",
        {"passed": True},
    )
    run_cache(*args)
    assert len(engine.batches) == 4
    assert [len(batch) for batch in engine.batches] == [8, 8, 3, 3]
    output = tmp_path / "output/logits.jsonl"
    rows = [json.loads(line) for line in output.read_text().splitlines()]
    assert len(rows) == len({r["id"] for r in rows}) == 22
    run_cache(*args)
    assert len(engine.batches) == 4
    # A valid interrupted cache resumes exactly the missing rows.
    output.write_text("".join(json.dumps(r) + "\n" for r in rows[:9]))
    run_cache(*args)
    assert len(output.read_text().splitlines()) == 22
