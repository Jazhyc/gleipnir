# Census-filtered standard-prompt 4B: training complete

2026-10-03. Training completed; held-out evaluation results are pending. The user
selected removing all 1,154 frozen Qwen3.5-9B census flags at score >=0.5 and
requested an aggressive data intervention after the
[behavior-grounded prompt experiment](monitoring_injection_disentanglement.md)
retained 100% alarms on injected honest-code controls.

The [prespecified experiment](../../experiments/monitoring_injection_removal/README.md)
uses the original standard student instruction, all 7,534 retained trajectories
and byte-identical original teacher records. Independent local checks establish
that every retained student prompt, trajectory checksum, source label, lineage
and teacher prompt identity is unchanged. The selection is based solely on the
frozen census, never on held-out evaluation scores. The exact exclusion list is
in the ignored `data/monitoring_injection_removal/selection.json`.

The input census SHA256 is
`630591f3cb53eea9cb4c8c9e29303ca7047b6f005cef1e258c24f69094d5bfe5`.
The new prepared-manifest SHA256 is
`4e67fe56f93980f3bd35d34af860f9e0ceb4ac8469c89d69cade8ba7594d3b06`.
The standard student-template SHA256 is
`f104aa18fe9f04094efdca14f9cebe65f7b18d976f454c134fa4282bd5162eaa`.
The full census population, trajectory and label provenance, score finiteness,
score/logprob agreement and retained teacher joins pass before materialization.

## Retained population

The filter removes 1,140 source-label-1 and 14 source-label-0 rows. Retained
source-label counts are:

| Source | Label 0 | Label 1 | Total |
| --- | ---: | ---: | ---: |
| BashArena | 1,383 | 1,137 | 2,520 |
| BashBench | 718 | 257 | 975 |
| CoT Red-Handed | 1,170 | 929 | 2,099 |
| Gloom | 386 | 373 | 759 |
| STRIDE | 673 | 508 | 1,181 |
| All | 4,330 | 3,204 | 7,534 |

The census has known semantic false positives and misses. Exact score ties at
0.5 are excluded. Two of the four assigned STRIDE injection objectives are
removed, while `e1bca909e88321525dcb79cf` (score 0.468791) and
`15b7252908c6c505d4f5ec4d` (0.294215) remain. The additional Gloom authority-spoofing
prompt exposure `01cfc9e3116069d847c5b89b` (0.095349) also remains. This follows
the user-selected threshold exactly; this is a census-filtered dataset, not a
verified injection-free dataset. See [the census finding](training_injection_census.md).

## Frozen run and evaluation

Use the unchanged selected packed BF16/FlashQLA B200 recipe, FP32 rank-128
masters, seed zero, the same initialization, soft BCE, AdamW 5e-5 and one epoch.
The smaller corpus has 236 logical updates at batch 32. Numerical startup probes
are explicitly reused from the recorded recipe; input/provenance and update
finite/missing-gradient checks remain required. The existing authorized B200's
hardware and driver match the preceding campaign. The launcher points at the
populated persistent `student_injection_awareness` compiler cache. No new GPU
allocation, cold cache, relabeling or teacher call is part of this run.

Startup subsequently passed through actual optimizer updates: at least 5/236
updates completed, with reported loss 0.6600 and gradient norm 19.23. The first
update took about 251 seconds; progress timestamps for updates 2–5 imply about
11.5 seconds/update. The recent GPU point sample was 100% utilization. These are
early measurements, not a whole-epoch speed/utilization result. The complete
token audit has 75,256,252 tokens, maximum length 29,337 and zero truncation.
The model loader verifies a BF16 base and 256 FP32 LoRA tensors; all 24 GDN layers
use FlashQLA. The actual worker environment verifies shared Inductor, Triton,
TileLang and TVM paths, recorded in `cache_runtime.json`; advancing-update and
startup evidence is in `startup_verified.json`. Both receipts live under
`results/monitoring_injection_removal/`. No numerical probes were newly run.

Training finished at approximately 17:21 UTC with all 236 updates and one epoch.
The trainer reports 3,203.16 seconds (53 minutes 23 seconds), mean loss 0.264177
and 13.57 seconds/update including startup. The training wrapper, including input
checks and export, reports 3,832.82 seconds. Local validation of the collected
training metadata confirms the selected recipe, exact update count, objective,
initialization and reused startup reference. The final master SHA256 is
`0e96096514e18e1e190d3317619d5845ad9dfaedc73a856dbadc064a2a7649ae`;
the rebased serving adapter SHA256 is
`c6146076376ef094b9a0b506ec9b6df6fe77cae8b9b5cc9b51a7cf9ea09f3a67`.
Both adapters were collected locally and independently verified against these
checksums; all 256 tensors in each layout are finite FP32. The verification
receipt is `results/monitoring_injection_removal/local_adapter_verified.json`.
The queue started evaluation automatically. All six original-FLA reference
cohorts completed, followed by all six serving-parity gates. Correlations range
from 0.999417 to 0.999877, and mean absolute score differences from 0.001490 to
0.007657. The filtered adapter has a nonzero effect in both implementations under
each prompt. All ten evaluation input audits completed with zero truncation.
At approximately 17:38 UTC the standard-prompt ID cell had written 256/3,012
scores, with 100% GPU utilization at the inspection. The evaluation is advancing;
there is no complete held-out outcome yet. Parity, token audits and current
scores were collected under `results/monitoring_injection_removal/evaluation/4b/`.

The new adapter will receive the same three-prompt injection grid and honest
controls; the twelve complete previous model/prompt cells are checked and reused
as baselines, giving fifteen grid cells overall. ID is evaluated only with the
standard training instruction, once over all 3,012 frozen CoT-removed rows.
Adapter-specific original-FLA-to-vLLM parity must pass before complete scoring.
The final one-epoch checkpoint is fixed; no held-out selection or promotion.

Launch records are under `results/monitoring_injection_removal/`, with logs under
`logs/runpod/monitoring_injection_removal/`. The evaluation queue waits for a
successful training completion record. There is no in-chat scheduler in this
session: active-turn startup checks do not provide automatic post-turn monitoring.
Thirty focused checks passed, including exact threshold ties, rejection of stale
census identities, preserved teacher joins, the 236-update completion check,
completed-cache identity checks and the single-standard-prompt ID scope. Scoped
Ruff, shell syntax and staged diff checks pass.

An eventual difference would measure this broad removal intervention, which
also changes harmful-example coverage, source composition, class balance and
training duration. It would not isolate prompt-injection text as the cause.
No effect would not rule out other data mechanisms or classifier misses. APPS
labels concern code backdoors; the training/ID rubric covers broader behavior.
