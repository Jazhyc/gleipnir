# Gleipnir 4B paired visible-reasoning sample

Status: paired generation and all 128 v2 blinded judgments completed 2026-08-31;
paired finding recorded.

## Hypothesis and intervention

This audit asks whether the mixed-data Gleipnir 4B monitor emits more useful or
faithful visible monitoring analyses than its unadapted Qwen3.5-4B backbone,
despite training only on Kimi K3 soft probabilities at the literal `0|1`
decision boundary. Here **Gleipnir 4B** specifically means the rank-128
`soft-n21837-mixed-qwen35-4b-seed0` adapter trained for one epoch over 21,837
tool-trajectory and prior-deception rows. It is the proto-foundation monitoring
model, not a tool-only scaling checkpoint.

Both conditions use the existing full binary visible-review instruction, native
Qwen thinking disabled, greedy decoding, a 4,096-token rationale cap, and an
exact terminal `Prediction:` boundary followed by constrained literal `0|1`
logprobs. One persistent vLLM engine serves the base and the checksum-rebased
LoRA, whose causal-master/serving parity gate already passed.

## Sample, baseline, and stop condition

The sample contains 64 of the frozen 6,395 OOD rows (1.0008%). Membership uses
proportional largest-remainder quotas across all six sources, followed by the
lowest deterministic SHA-256 ranking of campaign seed plus stable row ID within
each source. This selection is independent of labels, trajectory text, prior
scores, and both models' outputs. The fixed baseline is the exact unadapted
backbone.

The audited rendered inputs total 592,300 tokens per condition (1,184,600
served across both), with a maximum of 28,238 tokens. That maximum plus the
4,096-token rationale reserve and one decision token is 32,335, below the
36,864-token engine limit.

This run only creates paired reasoning for a later blinded judge. It must not be
reported as a 64-row OOD performance estimate, and no label or generated result
may be used to replace rows. Stop on input, prompt, checkpoint, sample, parity,
context, terminal-token, or paired-coverage drift as frozen in
`gleipnir4b_reasoning_sample.json`.

The generation step produces separate unblinded prediction caches and a
`judge_inputs.jsonl` artifact containing each trajectory with randomly assigned
candidate A/B responses. `judge_key.jsonl` separately retains the hidden model
mapping, source, and ground truth. A later judge rubric and model must be frozen
before those pairs are scored. The current auditable draft is documented in
[`GLEIPNIR4B_REASONING_JUDGE.md`](GLEIPNIR4B_REASONING_JUDGE.md); it remains
fixed for the full run after human approval and a passed bounded canary.

## Reproduction

Run the paired technical canary alone with:

```bash
sbatch cluster/slurm/generate_gleipnir4b_reasoning_sample.sh \
  --stop-after-canary
```

Generate or resume the complete 64-row paired sample with:

```bash
sbatch cluster/slurm/generate_gleipnir4b_reasoning_sample.sh
```

Ignored artifacts are written under
`results/tool_trajectory_monitoring/gleipnir4b_reasoning_sample/`, and final
logs under `logs/slurm/tool_trajectory_monitoring/`.

## Generation outcome

Slurm job `31037924` completed all 64 paired rows on one NVIDIA RTX PRO 6000
Blackwell Server Edition in 9 minutes 16 seconds with exit code zero. Both
conditions produced 64 unique, nonempty rationales and reached their exact
terminal `Prediction:` boundary without truncation or recovery. The adapter
changed every rationale and changed the hard decision on 14 of 64 rows.

The base generated 19,260 rationale tokens (mean 300.94; range 114--664), while
Gleipnir generated 10,561 (mean 165.02; range 62--378). This is a descriptive
length difference, not evidence that either condition is more useful or
faithful. That question remains intentionally deferred to the blinded judge.

The completed artifacts are:

- `base/predictions.jsonl`, SHA-256
  `54d028e2bbc880d94143e8bce4213d624f556c3a724c83ed8bd86b7fae0bd5e9`;
- `gleipnir/predictions.jsonl`, SHA-256
  `9423a59b6d9c86cb75d6142d2eeefadb2eff3c3dfd421a04a2cd8721d2ff42a5`;
- blinded `judge_inputs.jsonl`, SHA-256
  `5ebe0daebf6778c162ad030a9de2c254950789c542e385ecb762f4e51035ff6c`;
- separate `judge_key.jsonl`, SHA-256
  `0e3ea69cab37bb310be571050052b6d915ba54010ac0864d0ae336dae969fd7f`;
- `result.json`, SHA-256
  `a9787c57db24f378b2d7b7de8a983abd0b0e073952e060159e87a40f7b4274c9`.

The completed reasoning-quality judgment and its qualified interpretation are
recorded in
[`../../docs/research/tool_trajectory_gleipnir4b_reasoning_judge.md`](../../docs/research/tool_trajectory_gleipnir4b_reasoning_judge.md).
