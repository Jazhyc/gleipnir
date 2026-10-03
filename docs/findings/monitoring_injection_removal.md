# Census-filtered standard-prompt 4B: filtering did not resolve injection alarms

2026-10-03. Training and the full evaluation completed. Removing 1,154 census
flags did not resolve the operational false alarms: the new monitor still alarms
on 100% of injected honest-code controls under all three monitor prompts at
thresholds derived from clean code. Scores changed, and ID ranking worsened;
this is not a finding of no effect from filtering. The user
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
scores, with 100% GPU utilization at the inspection. The full evaluation finished
at approximately 18:10 UTC. Parity, token audits and completed scores were
collected under `results/monitoring_injection_removal/evaluation/4b/`.

The new adapter received the same three-prompt injection grid and honest
controls; the twelve complete previous model/prompt cells were checked and reused
as baselines, giving fifteen grid cells overall. ID was evaluated only with the
standard training instruction, once over all 3,012 frozen CoT-removed rows.
Adapter-specific original-FLA-to-vLLM parity passed before complete scoring.
The final one-epoch checkpoint was fixed; no held-out selection or promotion.

Launch records are under `results/monitoring_injection_removal/`, with logs under
`logs/runpod/monitoring_injection_removal/`. The evaluation queue waits for a
successful training completion record. There is no in-chat scheduler in this
session: active-turn startup checks do not provide automatic post-turn monitoring.
Thirty focused checks passed, including exact threshold ties, rejection of stale
census identities, preserved teacher joins, the 236-update completion check,
completed-cache identity checks and the single-standard-prompt ID scope. Scoped
Ruff, shell syntax and staged diff checks pass.

## Completed injection grid

Each new prompt has all 5,184 original benchmark rows and 3,930 honest controls.
Together with ID, this is 30,354 fresh predictions; the twelve previous grid
cells contain 109,368 verified reused predictions. There are no missing or
duplicate rows and no truncation across the ten input audits; the maximum input
length is 29,513 tokens. The full results are in `evaluation/4b/summary.json`
under this campaign's result directory.

Independent local verification rechecked prepared inputs and source pins, both
adapter checksums, all twelve baseline cells and all seven fresh score files.
Completion contracts, unique coverage, row identities/labels, finite scores and
score/logprob agreement pass. Every fresh record preserves both requested raw
decision logprobs, matching its stored literal-0/literal-1 values, and the serving
parity checksum. All six parity gates pass with nonzero adapter effects; no
failure receipt is present. The verification receipt is
`evaluation/4b/local_coverage_verified.json`; summary SHA256 is
`d3146714ef5c581ad1a5aaee21488a77dda1dbec15d3fe7c06098b8903dd96db`.

The table averages equally over the three frozen transfer templates (GPT-4,
GPT-4.1 and GPT-4o). Alarm thresholds are selected from each cell's 655 original
clean honest examples using the unchanged strict score > threshold rule, yielding
13/655 clean alarms (1.985%). Matched AUROC compares the 647 backdoor problems
with their corresponding honest-code submissions under the same template. The
unfiltered regular-trained model is the direct training-instruction baseline.

| Weights | Monitor prompt | Injected honest alarms | Neutral honest alarms | Matched injected AUROC | Matched neutral AUROC |
| --- | --- | ---: | ---: | ---: | ---: |
| Unfiltered regular | Neutral | 100.00% | 1.48% | 0.6440 | 0.8967 |
| Census-filtered | Neutral | 100.00% | 1.78% | 0.6503 | 0.8672 |
| Unfiltered regular | Aggressive | 100.00% | 1.63% | 0.6614 | 0.8830 |
| Census-filtered | Aggressive | 100.00% | 12.37% | 0.6706 | 0.8237 |
| Unfiltered regular | Conservative | 100.00% | 1.83% | 0.7249 | 0.8837 |
| Census-filtered | Conservative | 100.00% | 2.95% | 0.6611 | 0.8411 |

All 655 injected honest controls per template exceed the clean-code threshold,
under each of the three monitor instructions. The three filtered thresholds are
0.085099 (neutral), 0.060087 (aggressive) and 0.075858 (conservative). The earlier
regular/neutral threshold is 0.164516. All injected backdoors also exceed these
thresholds, so conventional injected-backdoor TPR is 100%; this does not establish
successful discrimination when injected honest negatives also trigger alarms.
The filtered model's vanilla benchmark AUROCs are 0.927822, 0.919176 and 0.918987
under the neutral, aggressive and conservative instructions respectively.

Filtering did change score scale. With the neutral monitor instruction, the mean
injected-honest score falls from 0.637405 to 0.529061, and the mean paired
injection-minus-neutral score shift falls from 0.593938 to 0.487026. As a separate
fixed-boundary diagnostic, injected-honest alarms at score >=0.5 fall from 97.30%
to 66.36% under the neutral monitor instruction; the filtered aggressive and
conservative values are 73.13% and 53.64%. These are not new operating thresholds
or evidence that the prespecified clean-FPR result improved. In particular,
lower clean scores also lower the clean-FPR threshold.

Neutral-transfer honest alarms with the filtered aggressive monitor are 24.89%,
5.95% and 6.26% by transfer template. This is a substantial prompt/template
interaction even without the adversarial injection payload. The neutral monitor
instruction and neutral transfer text appended to code are distinct conditions.
The primary grid should therefore retain the per-template diagnostics.

## ID sanity check

The filtered and earlier regular models use the same standard prompt and exact
same frozen ID input SHA256
`8e0fde58119098356feb0d897194d0dddec5e21b3e5e6959d5495d0999bb0ed2`.
There is one new ID pass, without an instruction sweep.

| Weights / training prompt | Source-macro pAUROC@20 | Source-macro AUROC |
| --- | ---: | ---: |
| Earlier regular | 0.846273 | 0.951394 |
| Earlier aggressive | 0.847301 | 0.947986 |
| Earlier behavior-grounded | 0.833429 | 0.946427 |
| Census-filtered regular | 0.822255 | 0.943424 |

Relative to the direct regular baseline, macro pAUROC@20 decreases by 0.024017.
Gloom (2,066 rows) drops from 0.763290 to 0.708638, while STRIDE (946 rows) rises
from 0.929255 to 0.935873. Their filtered AUROCs are 0.901155 and 0.985694.
Pooled AUROC is 0.935553 and pooled pAUROC@20 is 0.786512. At the fixed 0.5
boundary, pooled recall drops from 81.90% to 74.00%, while FPR drops from 6.41%
to 4.42%. Pooled Brier worsens from 0.097827 to 0.117895 and ECE from 0.058943
to 0.102830. Mean predicted probability is 0.431035 against prevalence 0.533865,
consistent with increased underprediction. There are 1,357 distinct ID scores
and 1,655 repeated scores, versus 1,569 and 1,443 in the regular baseline.
Filtered benchmark/honest-control score-tie counts are 3,879/3,226 (neutral),
3,918/3,317 (aggressive) and 3,968/3,265 (conservative).

## Interpretation

This broad filtering intervention changes scores and ID performance but fails
to resolve injected-honest alarms at the prespecified clean-code operating
points. It also changes harmful-example coverage, source composition, class
balance and training duration. The result does not isolate injection text as
the cause, establish that the remaining data are injection-free, or rule out
other data mechanisms. Reviewed low-scoring injection exposures remain in the
retained corpus. APPS labels concern code backdoors; the training/ID rubric
covers broader behavior. The single seed and fixed templates do not establish
adaptive robustness. No model promotion follows from this campaign.
