"""Measure how well the 4B fits the auxiliary targets, before and after training.

The trainer logs only the combined loss. This scores a stratified sample of
training rows with the exact training tokenization and ``topk_tail_kl``,
under the initial and the trained adapter, and reports per section
(Position, Reason):

- the auxiliary KL(teacher || student), whose minimum is 0;
- the teacher's entropy over the same top-64-plus-tail partition, which is
  the minimum cross-entropy any student can reach;
- the agreement between the student's and the teacher's top tokens.

All rows were seen once in training, so this measures fit, not
generalization.
"""

from __future__ import annotations

import argparse
import random

import torch
import yaml

from experiments.aux_supervision.prepare import ARM, ARMS, ROOT, receipt
from gleipnir.data.monitoring import read_rows, write_json
from gleipnir.training.aux_distillation import topk_tail_kl

INITIAL_ADAPTER = "results/fp4_row_aot_training/initial_adapter"
MARKER = "\nReason:"


def sections(tokenizer, record: dict) -> list[str]:
    """Label each auxiliary target position forced, position or reason."""
    label = int(record["label"])
    forced = tokenizer.encode(f"{label}\nPosition:", add_special_tokens=False)
    generated = record["aux_input_ids"][len(forced) :]
    out = ["forced"] * (len(forced) - 1)
    # Generated step j predicts generated token j after seeing tokens < j.
    for j in range(len(record["aux_input_ids"]) - len(forced) + 1):
        seen = tokenizer.decode(generated[:j])
        out.append("reason" if MARKER in seen else "position")
    return out


def entropy(top_logprobs: torch.Tensor) -> torch.Tensor:
    p = top_logprobs.exp()
    tail = (1 - p.sum(-1)).clamp_min(0)
    return -(p * top_logprobs).sum(-1) - torch.xlogy(tail, tail)


def per_position_kl(logits, top_ids, top_lp) -> torch.Tensor:
    ones = torch.ones(len(top_ids), dtype=torch.bool, device=logits.device)
    return torch.stack(
        [
            topk_tail_kl(
                logits[i : i + 1],
                top_ids[i : i + 1],
                top_lp[i : i + 1],
                ones[i : i + 1],
                (1,),
            )[0]
            for i in range(len(top_ids))
        ]
    )


def main(count: int, seed: int, arm: str = ARM) -> None:
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from experiments.deception_distillation.train_student_sft import tokenize_record

    adapters = {
        "initial": INITIAL_ADAPTER,
        "trained": f"results/{arm}/4b/monitor/causal_adapter",
    }
    config = yaml.safe_load((ROOT / "data" / arm / "config.yaml").read_text())
    base = ROOT / config["base_model"]
    tokenizer = AutoTokenizer.from_pretrained(base, local_files_only=True)
    rows = read_rows(ROOT / "data" / arm / "training.jsonl")
    rng = random.Random(seed)
    sample = []
    for label in (0, 1):
        pool = [r for r in rows if int(r["label"]) == label]
        sample += rng.sample(pool, count // 2)
    features = []
    for record in sample:
        feature = tokenize_record(
            record,
            tokenizer,
            10**6,
            include_direct_target=True,
            include_completion_target=False,
            include_aux_target=True,
            dataset_id=0,
        )
        features.append((record, feature, sections(tokenizer, record)))
    report = {"rows": len(sample), "seed": seed, "adapters": adapters, "results": {}}
    for name, path in adapters.items():
        model = AutoModelForCausalLM.from_pretrained(
            base,
            dtype=torch.bfloat16,
            attn_implementation="sdpa",
            local_files_only=True,
        ).cuda()
        model = PeftModel.from_pretrained(model, ROOT / path).eval()
        stats = {
            s: {"kl": [], "entropy": [], "agree": []} for s in ("position", "reason")
        }
        row_kl = {0: [], 1: []}
        for record, feature, labels in features:
            ids = torch.tensor([feature["direct_input_ids"]], device="cuda")
            k = feature["aux_length"]
            with torch.no_grad():
                logits = model(input_ids=ids, logits_to_keep=k).logits[0].float()
            top_ids = torch.tensor(feature["aux_top_ids"], device="cuda")
            top_lp = torch.tensor(feature["aux_top_logprobs"], device="cuda")
            mask = torch.tensor(feature["aux_mask"], device="cuda")
            row_kl[int(record["label"])].append(
                topk_tail_kl(logits, top_ids, top_lp, mask, (k,)).item()
            )
            kl = per_position_kl(logits, top_ids, top_lp)
            ent = entropy(top_lp)
            agree = logits.argmax(-1) == top_ids[:, 0]
            for i, section in enumerate(labels):
                if section != "forced" and bool(mask[i]):
                    stats[section]["kl"].append(kl[i].item())
                    stats[section]["entropy"].append(ent[i].item())
                    stats[section]["agree"].append(float(agree[i]))
        mean = lambda v: sum(v) / max(len(v), 1)  # noqa: E731
        result = {
            "training_loss_term": {
                "row_mean_kl": mean(row_kl[0] + row_kl[1]),
                "weighted_0.1": 0.1 * mean(row_kl[0] + row_kl[1]),
                "by_label": {str(k): mean(v) for k, v in row_kl.items()},
            },
            "by_section": {
                s: {
                    "tokens": len(v["kl"]),
                    "kl": mean(v["kl"]),
                    "teacher_entropy": mean(v["entropy"]),
                    "cross_entropy": mean(v["kl"]) + mean(v["entropy"]),
                    "top1_agreement": mean(v["agree"]),
                }
                for s, v in stats.items()
            },
        }
        report["results"][name] = result
        print(name, result, flush=True)
        del model
        torch.cuda.empty_cache()
    write_json(receipt(arm, "kl_diagnostic.json"), report)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, default=256)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--arm", choices=ARMS, default=ARM)
    args = parser.parse_args()
    main(args.rows, args.seed, args.arm)
