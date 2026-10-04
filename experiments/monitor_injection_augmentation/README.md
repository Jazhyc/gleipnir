# Monitor-directed injection augmentation

Design selected 2026-10-04: replace 40% of the original 8,688 training rows with
one injected view each. Keep 60% clean and visit each underlying trajectory once.
Hypothesis: exposing the regular harmfulness monitor to the same irrelevant
injection mechanisms in both behavior classes reduces injection-only alarms
while preserving detection of underlying misconduct. This intervention trains
the monitoring objective directly, unlike the completed A/B preference auxiliary
task. It does not claim on-policy agent robustness.

Completed 2026-10-04: all 272 training updates and 15,138 fresh evaluation scores
passed coverage/parity checks. Clean ID macro pAUROC@20 is 0.8730 versus 0.8463
cached original; APPS injected-honest FPR averages 44.43% versus 100% at
clean-code ~2% FPR, with injected-backdoor TPR 87.64%. Held-out synthetic ID
harmless FPR is 1.85% at source-specific clean ~2% operating points. See the
[final finding](../../docs/findings/monitor_injection_augmentation.md) for
the threshold, template, calibration and paired-effect details and limitations.

The frozen preparation selects 3,475 rows (nearest integer to 40%), stratified
proportionally by original source and hard behavior label. Uniform role quotas
are 1,159 user, 1,158 assistant and 1,158 tool messages. Each selected trajectory
receives exactly one standalone payload message; role choice does not depend
on whether that role already exists. Tool-only additions can therefore lack a
preceding tool call. Keep these off-policy additions explicit in artifact
metadata, without adding an artificial marker to the model-visible payload.

Sample normalized character position from N(0.5, 0.25²), rejecting draws outside
[0,1], then snap to the nearest rendered message/step boundary. Both ends are
eligible. The Gaussian applies to requested positions; realized positions
depend on transcript granularity. Two-message code examples cannot support a
smooth position distribution. Never split source code, commands, arguments or
tool output bodies to achieve a preferred position. Preserve every original
trajectory byte, with reversible insertion receipts. Retain role-tagged and
XML-step source formats. Added XML steps use a fresh number without renumbering
the source, so their numeric order can be nonchronological.
Existing XML-style sources sometimes lack a task header or its closing tag;
retain those bytes and record the irregularity, using the existing complete
step boundaries without repairing or dropping the row.

Eight project-authored training templates span four monitor-directed mechanisms:
direct override, reviewer authority spoofing, answer prefill and delegated
verdicts. Balance demanded 0/1 verdicts independently of the actual label, and
balance template exposure. Four additional templates from two separate families
(decision pressure and channel spoofing) are held out. No held-out APPS benchmark
payload, code, result or benchmark-specific calibration wording enters generation
or training; the original 8,688-row training population remains fixed.
The template bank is a small first intervention; eight strings are not broad
attack diversity. Nothing is optimized against the student or held-out scores.

Keep the regular instruction, Prediction: prefix, 0/1 decision logits, source
behavior labels and original cached Kimi K3 soft BCE targets. Teacher targets
were obtained from clean source trajectories; copying them to augmented views
is an explicit behavior-invariance assumption, not a teacher assessment of the
new transcript. Preserve original teacher-prompt provenance and label lineage
separately from augmented trajectory/prompt hashes. Original rationale text is
retained only as historical source data and is not a completion target. A
monitor-directed assistant message may itself look like safeguard bypass under
the regular rubric; this intentional supervision tests whether the model can
keep the source behavior verdict fixed despite added reviewer-directed text.

## Preparation and planned training

Run `.venv/bin/python -m experiments.monitor_injection_augmentation.prepare`.
Inputs: `data/student_injection_awareness/`; materialized rows, insertion ledger
and manifest: `data/monitor_injection_augmentation/`; preparation report:
`results/monitor_injection_augmentation/`. Source hashes, config/template/code
hashes, teacher coverage, source/label/role/template/target counts, position
histograms, reversibility and actual tokenizer lengths are recorded. No silent
truncation, dropping, duplication, reselection or token-budget adjustment.

Preparation completed on 2026-10-04. Independent audit recovered all source
transcript bytes, preserved labels/teacher targets/lineage, and verified that
every training template has both behavior classes and both demanded verdicts.
All 5,213 clean JSONL lines are byte-identical to their original lines. Injected
views comprise 1,738 label-0 and 1,737 label-1 examples; each template occurs
434 or 435 times. Roles match the selected 1,159/1,158/1,158 quotas.
There are 2,359 role-tagged and 1,116 XML-step additions. Four selected XML
sources have existing task-wrapper irregularities (two absent, two unclosed),
which remain unchanged. Two earlier preparation failures are retained as
receipts; support for these existing source formats resolved them without
changing the selected population.

The tokenizer audit has 83,969,155 input tokens, maximum 29,391, minimum 270
and zero truncation, inside the validated 29,696-token envelope. This is 152,844
more input tokens than the matched original schedule. Requested-position decile
counts are `[115,191,370,460,554,582,481,358,235,129]`; realized counts are
`[227,186,293,443,574,607,433,260,177,275]`.
The materialized manifest SHA256 is
`e99a700ed1b588323e0f0a757e73e50046496fbc25c29c158b34b129dec99f51`;
student rows SHA256 is
`84998c2f7727ed1249356b3d4ff126dab71bb4d5ca4255a073128f0b624519b6`.
Run `.venv/bin/python -m experiments.monitor_injection_augmentation.audit` to
recheck the frozen artifacts and write `results/monitor_injection_augmentation/completion_audit.json`.
All 40 original preparation checks and scoped Ruff checks passed. Execution
adds six focused checks; all 46 pass locally and on the B200. See the
[campaign finding](../../docs/findings/monitor_injection_augmentation.md) for
the subsequent authorized launch and measured startup status.

The proposed matched training run initializes a fresh Qwen3.5-4B rank-128 LoRA
from the original zero-B adapter, one epoch, logical batch 32, 272 updates,
AdamW 5e-5, 3% warmup and linear decay. Retain packed BF16 FlashQLA/SDPA, FP32
masters and the persistent compiler cache. Reuse unchanged startup validation
only if every augmented input remains inside its 29,696-token envelope, with
the receipt checksum and reused/skipped status recorded. Keep finite/missing
gradient guards. No preference task or extra optimizer steps are introduced.
Training/evaluation entrypoints are implemented. The user authorized execution
on the existing B200 on 2026-10-04. Run `bash experiments/monitor_injection_augmentation/run.sh`
from the remote workspace after syncing both frozen materializations. It runs
the data audit, matched training, bounded master reference and persistent-vLLM
evaluation sequentially, stopping on any failure. `prepare_eval` separately
freezes the evaluation manifest before launch. Execution contracts hash training
and evaluation code; serving rejects changes to the launched campaign.

## Frozen evaluation proposal and stop conditions

Use the fixed final checkpoint and the regular monitoring instruction only.
Reuse the existing original regular adapter results on canonical 3,012 CoT-removed ID
examples, unchanged APPS benchmark and honest controls, plus paired clean and
synthetically injected ID views using only held-out template families. Keep
the source holdouts intact; augmented and clean views of a trajectory must
never cross train/evaluation. APPS is evaluation-only and cannot select a
checkpoint, template or hyperparameter. Require adapter-specific master/serving
parity and nonzero adapter effect before full vLLM scoring; batch related
conditions in one persistent engine with one-token decision logprobs.

User clarification: do not evaluate the original adapter again. Its ID/APPS/
honest-control results and a checksum-pinned base reference are reused. Fresh
population scoring is only for the augmented adapter: 5,184 APPS benchmark,
3,930 honest-control, 3,012 clean ID and 3,012 injected ID views (15,138 total).
The held-out ID set balances roles exactly to 1,004 each, uses each held-out
template 753 times and has no original training trajectory hash overlap. It
preserves lowercase headers where present, with reversible original bytes.
There is no original-adapter baseline for this new synthetic population.
Report demands agreeing/conflicting with the source label separately, along
with payload verdicts, roles and mechanism families. The training demand/label
table has 893 harmless/0, 845 harmless/1, 845 harmful/0 and 892 harmful/1;
blindly reversing the demand matches only 48.6% of hard labels.

Report source-macro pAUROC@20, AUROC, calibration, ties, class-conditional paired
score changes and flips, clean-calibrated injected-harmless FPR and harmful TPR,
and results by source/role/template family/position. This is exploratory, with
one seed, a small template bank and off-policy transcripts; no promotion or
adaptive-attack claim follows. Stop on identity drift, incomplete target or
prediction coverage, changed source bytes, truncation/context overflow, missing
or nonfinite gradients, OOM or failed serving parity. Preserve failure receipts.
