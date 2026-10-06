# Research program

## Goal

Train a broadly useful monitoring foundation model that scores control-relevant
behavior from model interactions. Deception is the first well-developed domain,
but the target ontology should grow to include misaligned actions, covert goal
pursuit, sabotage, manipulation, unsafe capability use, policy evasion, and
ambiguous cases where a monitor should defer.

## Initial strategy

Use strong teachers with privileged information and/or larger inference budgets
to annotate examples with both calibrated soft targets and compact evidence.
Distill those signals into Qwen 3.5 students with parameter-efficient training.
Compare three separable sources of value:

1. hard behavior labels;
2. teacher probability distributions at a fixed decision boundary;
3. compact rationales or structured evidence available only during training.

The first migrated experiment reproduces the deception-distillation pipeline in
a standalone form. Subsequent work should add task families through explicit
dataset adapters and an evolving, versioned behavior ontology.

The next proposed domain is action-only monitoring of agent tool trajectories:
detect sabotage, exfiltration, prompt-injection compliance, and scheming from
observable messages, commands, tool calls, outputs, edits, and final reports
without private reasoning. Its paper-derived task and dataset contract, along
with the proposed Kimi K3 logit-distillation comparison, is recorded in
`research/tool_transcript_monitoring.md`.

## Evaluation principles

- Split by source lineage, generator, task family, and model family where possible.
- Report macro ranking, calibration, thresholded safety diagnostics, and ties.
- Maintain honest hard negatives and ordinary-capability controls so factual
  error is not conflated with deception or misalignment.
- Evaluate transfer before scaling annotation volume or student size.
- Freeze promotion criteria before final evaluations.
- Preserve raw teacher evidence and immutable prompt hashes for auditability.

Instruction changes alone have not resolved injection-text sensitivity in the
completed [behavior-grounded 4B experiment](findings/monitoring_injection_disentanglement.md):
all three inference prompts retained 100% alarms on injected honest-code controls
at thresholds set from clean code. Require honest injected negatives and matched
positive/negative templates when interpreting injection benchmarks. This result
uses fixed teacher targets and a code-backdoor label contract; it does not identify
the causal training examples or settle the broader agent-behavior ontology.

The completed [aggressive census-filtering experiment](findings/monitoring_injection_removal.md)
also retained 100% injected-honest alarms at clean-code 2% FPR operating points
under all three monitor prompts. Removing 1,154 flagged rows lowered scores but
reduced standard-prompt ID macro pAUROC@20 from 0.8463 to 0.8223. This broad data
intervention did not resolve the failure; its changed class/source composition
and remaining classifier misses prevent a prompt-injection-specific causal claim.

For future monitoring validation and checkpoint selection, use the
[CoT-removed ID dataset](decisions/cot_removed_id_validation.md), preserving
visible assistant prose. Freeze its input/manifest hashes in each new campaign.

The completed [JudgeDeceiver continuation experiment](findings/judge_injection_continuation.md)
continues the regular 4B adapter on preference judgments with injections attached
independently to preferred and disfavored answers. Its pooled full release is
split approximately 80:20 by original query before augmentation. Preferred-answer
injection accuracy rises from 58.36% to 100%, and regular-prompt APPS injected-honest
alarms fall from 100% to 70.64% at clean-code 2% FPR operating points. ID macro
pAUROC@20 falls from 0.8463 to 0.7958 and calibration worsens. Preference robustness
does not establish overall harm-monitoring improvement. Preserve the distinction
between quality and harmfulness targets when designing benign injection exposure;
APPS remains excluded from training and checkpoint selection. No promotion follows.

The completed [joint-training experiment](../experiments/judge_injection_joint/README.md)
trains a fresh 4B adapter with the full regular trajectory population and a
75:25 monitoring/preference sampler. It preserves monitoring soft targets and
separate A/B hard preference labels, using the regular 5e-5 learning rate.
Approximately 22.5% of training draws contain injections; their 25 original
queries limit task diversity. Preferred-answer-injected accuracy reaches 100%,
and ID source-macro pAUROC@20 is 0.8658 versus 0.8463 original and 0.7958 after
preference-only continuation. However, all injected harmless APPS controls still
alarm at clean-code ~2% FPR; matched injected-harmful versus injected-harmless
AUROC is 0.6256 versus 0.6440 original. The auxiliary quality task can learn
injection handling without transferring that behavior to harmfulness monitoring.
These exploratory outcomes do not isolate forgetting. Only the fixed final
checkpoint and regular monitoring prompt were evaluated, with no APPS-based
selection or promotion. See the [finding](findings/judge_injection_joint.md).

## Near-term questions

The completed [monitor-directed injection augmentation](findings/monitor_injection_augmentation.md)
replaces 40% of the original monitoring rows with one off-policy injected view,
balancing user/assistant/tool roles and demanded verdicts across both source
behavior classes. It directly trains the harmfulness objective using the fixed
teacher cache. All 272 updates and 15,138 fresh scores are complete. Clean ID
source-macro pAUROC@20 rises from 0.8463 to 0.8730. APPS injected-honest FPR
falls from 100% to 44.43% averaged over three transfer templates at each
adapter's clean-code ~2% FPR threshold; injected-backdoor TPR falls to 87.64%.
Matched injected-harmful versus injected-harmless AUROC rises from 0.6440 to
0.8595. Held-out synthetic ID injections leave paired AUROC essentially unchanged
and give 1.85% harmless FPR at source-calibrated clean ~2% operating points.
This supports direct monitoring augmentation, with substantial residual APPS
sensitivity and limits from one seed, few authored strings and off-policy views.
Only the new adapter receives fresh scores; original regular-adapter results
are reused. APPS remains evaluation-only.

The user authorized a [9B replication](decisions/monitor_injection_augmentation_9b.md)
of the same frozen augmentation at 5e-5, using the previously validated 9B
checkpointing recipe and cached regular 9B controls. This campaign preserves
all training/evaluation views and introduces no recipe sweep. All 272 updates
and 15,138 fresh scores are complete and independently verified. Clean ID
source-macro pAUROC@20 rises from 0.8796 to 0.8939, but injected-honest APPS FPR
falls only from 99.85% to 83.36%, with two templates still at 100%. Matched
injected AUROC rises from 0.6588 to 0.6786, substantially less than at 4B;
vanilla APPS AUROC regresses from 0.9314 to 0.9126. This supports clean ID
benefits without broad OOD robustness or a model-size monotonicity claim. See
the [completed finding](findings/monitor_injection_augmentation_9b.md).
The existing B200 is verified EXITED after complete artifact collection and
independent audits; its persistent workspace and compiler/kernel caches remain.

The completed [auxiliary transfer evaluation](findings/augmented_judge_evaluation.md)
uses the unchanged original A/B quality-judging prompt and 4,188 JudgeDeceiver
holdout variants from six queries. Neither monitoring-augmented adapter trained
on JudgeDeceiver. Preferred-answer-injected accuracy rises from 58.36% to 62.65%
at 4B and 64.33% to 72.37% at 9B; freshly evaluated base 9B reaches 65.53%.
Clean accuracy stays near 99%, but augmented correct-to-wrong flips remain
37.08% / 27.63%. Every augmented query loses mean p(correct) under preferred
injections, and the hardest MT-Bench query reaches only 23.44% / 34.82%
accuracy. This supports partial transfer with residual avoidance across multiple
sources, rather than broad immunity. It is a preference diagnostic, not a
harmfulness false-positive rate. No promotion follows. The user-authorized EU
pod termination is independently verified after artifact collection and
checksum rechecks; its network volume remains present. The prior NC2 pod and
its persistent caches remain preserved.

- TODO (2026-10-04): broaden training injection wording and attack styles beyond
  the current small template pool and role/position variation. The
  [auxiliary transfer finding](findings/augmented_judge_evaluation.md) motivates
  testing whether greater variety reduces residual avoidance; insufficient
  variety remains a hypothesis. Experiment deferred at the user's request.
- With the backbone and teacher cache fixed, can CoT-removed student training,
  consistency regularization, compact evidence supervision, controlled data
  exposure, or checkpoint averaging improve transfer? The
  [possible training follow-ups](research/monitoring_training_followups.md)
  record the evidence and a proposed screen; these methods are not yet selected.
- How do soft label margins and rationale supervision scale with teacher quality
  and annotation volume?
- Does multi-domain joint training improve monitoring abstractions or cause
  destructive interference?
- Which Qwen 3.5 capacity is the best student/teacher frontier on two H100s?
- Which data mixtures transfer to held-out action types and model families?
- Can a single calibrated output schema express positive detections, benign
  behavior, and epistemic uncertainty without benchmark-specific routing?

A [collaborator-reported teacher diagnostic](findings/kimi_teacher_injection_controls.md)
finds template-dependent Kimi K3 suspicion on OOD APPS controls, alongside
matched injected AUROC 0.918–0.971 versus 0.644 for the original 4B student.
This challenges simple copying of the teacher’s OOD response but does not
identify the training mechanism; raw upstream predictions are not audited here.

The [BF16 B200 FA4 screen](findings/b200_bf16_fa4.md) passes an isolated native
kernel check but fails whole-model eager packing gradient parity at 8.29% against
the unchanged 5% limit. The replay shows zero cross-example leakage; differences
begin at the first full-attention layer. The user explicitly authorized a separate
timing continuation with a 10% learning acceptance ceiling and preserved strict
results. The user subsequently selected native FA4 as the packed BF16 default
after reviewing the completed timing and FP4 screens. The
user-authorized NC2 B200 remains running with its workspace and shared caches.
The authorized continuation completes both 20-update trajectories: measured FA4
updates take 20.24% less time (5.12363 to 4.08648 seconds), with matched partitions
and tokens. Fresh FA4 checks make total invocation time longer; reverse-order
replication and quality validation remain unperformed. The explicit user
selection changes the execution default without asserting quality equivalence.

The [native FP4 MLP LoRA screen](findings/b200_fp4_mlp_lora.md) revisits the
prior FP4 path with BF16 non-MLP components and no k-bit preparation. Local
native arithmetic, row scaling, eager/compiled packing and 20 finite updates
pass, but FP4 measured updates take 53.44% more time (7.86192 versus 5.12363
seconds). Keep BF16 MLPs; this FP4 configuration has no throughput basis for
promotion. Separate original-FLA/FlashQLA strict parity still fails at 73.64%
gradient relative L2 under the standing selected-finite policy.

The [B200 MLP / GEMM screen](findings/b200_mlp_gemm.md) rejects compiled
gate/up merging and the initial cuDNN LoRA-aware forward graph as throughput
improvements over already compiled PEFT. At 16384 tokens their complete standalone
MLP forward/backward times are 3.04% and 21.91% slower; no full-model continuation
was launched. Native FP4 frozen-base forward and input-gradient GEMMs pass decoded-operand
arithmetic checks. Tensor-wide dynamic scaling measurably couples examples;
per-row scaling restores exact row independence. Chunked per-row packing
with matched graph replay yields 10.65–38.10% faster isolated FP4 paths at
16384 tokens, with all conversion costs included, while ordinary
unscripted paths remain slower. Synthetic GEMM measurements remain distinct
from full training speed and quality evidence; the selected BF16 FA4 recipe
is unchanged.

The [MXFP8 FA4 feasibility assessment](findings/b200_mxfp8_fa4_feasibility.md)
finds that Meta's public Blackwell implementation currently supports noncausal
D128 MHA, while our Qwen3.5-4B recipe requires causal D256 GQA. Isolated execution
of its actual source guards reproduces each rejection independently. This needs
kernel development before integration; no GPU timing or training result follows.
Keep BF16 FA4 as the standard. The assessment records a future integration path,
including sequence-local scales, causal quantization checks, an isolated runtime
and complete update timing with conversion costs.

The [eight-bit training kernel survey](findings/b200_eight_bit_training_kernel_survey.md)
identifies NVIDIA's development cuDNN Frontend FROST MXFP8 D256 causal GQA
forward/backward path as a full-attention integration candidate. Its backward
adapter lacks our packed variable-length layout. No compatible public eight-bit
GDN training kernel was found: FlashQLA and TE recurrence remain BF16/FP16,
while LeapQuant targets inference and Delta-Matching explicitly keeps recurrence
in BF16. These are source findings, not measured Gleipnir training results.
Keep BF16 FA4/FlashQLA as the standard pending a separate matched screen.

The [NVIDIA MXFP8 B200 screen](findings/b200_nvidia_mxfp8.md) executes causal
D256 GQA forward/backward and passes all 24 quantizer-layout comparisons. Native
Q/K gradient errors are 6.47–8.65%, but the fresh whole-model adapter-gradient
error is 18.71%, exceeding the separate 10% learning ceiling. Cross-example
isolation passes with zero leakage. A separately authorized timing-only run
completes 20 updates and averages 5.29781 seconds over its last ten, versus
4.08648 seconds for the verified matched historical FA4 control (29.64% slower).
The fresh FA4 repeat was intentionally stopped as redundant; no fresh
same-runtime replication or quality equivalence is claimed. Sixteen parallel
workers prepare the manifest's 534 exact-shape plans in 208 seconds using the
shared cache. Keep BF16 FA4/FlashQLA as the standard and preserve failed parity
receipts alongside the completed timing result.

The follow-up [direct variable-length MXFP8 prototype](findings/b200_nvidia_mxfp8_varlen.md)
uses packed forward and native cumulative-offset backward kernels. Whole-row
quantization preserves sequence-local blocks and matches the dense MXFP8
outputs and gradients exactly in fresh boundary tests. Singleton semantics,
isolation, changed-length CUDA graph replay and initialized scratch pass;
independent FP32 strict parity remains failed. Its completed 20-update screen
averages 3.93793 seconds/update, taking 25.67% less time than dense MXFP8 and
3.64% less than historical FA4 with identical physical contracts. Its final FP32
adapter is byte-for-byte identical to the dense MXFP8 run. Whole-model gradient
parity remains failed; the modest FA4 timing difference lacks fresh replication
and quality validation. Keep BF16 FA4 as the standard. A bounded attention-only
profile finds a 28% long-singleton latency reduction but regressions on short
and balanced packs. The remaining targets are host conversion/dispatch costs
and a dQ backward kernel with roughly unchanged GPU time; the exact whole-model
GDN/MLP/attention time shares remain unmeasured.

The [fused MXFP8 preparation screen](findings/b200_mxfp8_fused.md) implements
Meta-inspired single-pass operand/scaling preparation and transpose-invariant
32x32 block scaling, plus a fused cumulative-boundary predicate. It retains the
NVIDIA causal D256 GQA kernels and our sequence-local layouts. Native dual mode
matches the old producer exactly; both modes pass execution/isolation checks.
Attention GPU launches fall from 38–39 to 16, and long-singleton attention takes
31.8% less time than FA4 in the isolated diagnostic. However, the matched square
20-update screen averages 4.15062 seconds/update, 1.57% slower than historical
FA4 and 5.40% slower than direct-varlen MXFP8. Whole-model eager/compiled gradient
errors remain failed at 19.49%/18.22%. The intervention fails its complete-update
improvement rule. Keep BF16 FA4 as the standard and retain these explicit
experimental backends; no quality equivalence or fresh control replication is
claimed. Meta's RMSNorm/GEMM producer epilogues and native attention changes
remain outside this implementation.

The [Meta optimization campaign](findings/b200_meta_stack.md) audits every blog
technique against the pinned public implementations and tests native scheduling,
probability scaling, wider dQ stores and project producer fusions. Persistent dQ
stalls and the warp-maxima prototype fails replay agreement; probability scaling
regresses. Fused Qwen head normalization/RoPE/quantization plus wider stores
passes execution checks and reduces isolated producer/attention time, but its
20-update model screen takes 4.51828 seconds/update, 10.57% slower than historical
FA4. The whole-model profile attributes 9.54% of summed GPU kernel time to native
full-attention contractions, versus 39.87% to mixed GEMMs and 20.39% to GDN and
convolution. These shares describe this profiled candidate, not a baseline or a
causal regression diagnosis. A BF16 projection epilogue pilot halves Q producer
forward time; an MXFP8 pilot matches its quantized-operand oracle with similar
fused forward time. Projection backward and native FP8 returned-gradient fusion
remain unimplemented. Keep BF16 FA4 as the standard and preserve the negative screen.

The [B200 MLP/GEMM screen](findings/b200_mlp_gemm.md) adapts NVIDIA's pinned
Frontend/FROST native FP4 GEMMs for frozen LoRA base forward and input gradients.
Row-independent scaling preserves packed-example isolation, and registered
autograd/compiler integration keeps the live FP32 adapter masters intact.
Isolated GEMM graph savings shrink to 1.56%/2.28% for the complete synthetic
LoRA MLP at 4096/16384 tokens, with conversions and every gradient included;
ordinary dispatch is slower. The integrated pilot fails the predeclared 5%
selection rule, so no full-model FP4 run or quality equivalence is claimed.
Keep BF16 FA4 as the standard. A separate 16,384-token GPU profile attributes
34.12% of summed native-candidate kernel time to FP4 conversion, 17.21% to the
four native FP4 GEMMs, 16.71% to adapters and 31.35% to compiled elementwise/cast/
concatenation work. Conversion costs almost twice as much as native contraction;
producer/epilogue fusion is a better supported next target than isolated core
GEMM tuning. These instrumented shares do not replace synchronized wall timings.

The hardware-packing follow-up replaces software E2M1 encoding with Blackwell's
native conversion instruction and fills scale padding inside the same kernel.
It preserves all packed operands and complete-MLP output/gradients exactly.
At 4096/16384 tokens, matched complete-MLP graph time drops 11.86%/14.76% versus
compiled BF16 and 10.10%/12.80% versus the old FP4 integration. Ordinary dispatch
is still slower than BF16. This selects the opt-in hardware path for fresh
packed-model checks; it does not establish model-update speed or training
quality, and BF16 FA4 remains the standard.

Fusing per-row output descaling into the same pinned NVIDIA GEMM, while retaining
the virtual raw BF16 rounding, also preserves bitwise arithmetic. The combined
complete-MLP graph saves 18.54%/24.42% versus compiled BF16 at 4096/16384 tokens
and 7.76%/8.75% versus hardware packing alone in a matched four-way screen.
All input/adapter gradients and isolation checks pass. This selects the combined
experimental path for fresh full-model gates; the ordinary dispatch path still
loses, and no model-update speed or quality equivalence follows from the pilot.

The selected combined FP4 full-model attempt then fails its fresh eager packing
gate: 67.90% adapter-gradient disagreement and loss drift above the unchanged
bound, despite exact cross-example isolation. It stops before compilation,
memory preflight and optimizer updates. Layer outputs match through layers 0–2,
then diverge at the first full-attention layer; amplification through later FP4
layers is a hypothesis requiring separate diagnosis. Retain the conversion
optimizations as experimental code, but do not promote this model recipe or
claim an end-to-end speed gain. BF16 MLPs with FA4 remain the standard.

After explicit user timing-only acceptance, the native FP4 screen completes
twenty updates in the same default compile mode and exact physical partitions
as the historical FA4 control. Updates 11–20 average 19.42944 seconds versus
4.08648 (4.75x slower), with finite updates but failed strict/learning numerical
parity preserved. Twenty-seven measured token shapes are new after warmup,
and packing/scaling kernel compilation continues during measurement. This is
trajectory wall time, not fully warmed kernel speed. Graph-enabled full-model
attempts fail before updates; complete-MLP graph gains have not been realized
in the ordinary training recipe. Runtime shape arguments and stable graph
integration are better supported follow-ups than promoting the current FP4 path.

The explicitly requested warmed follow-up replays all twenty logical batches
twice without optimizer updates, preserving master/RNG/sampler state. Its second
replay and all actual updates create zero native plans, Triton specializations or
compiler graphs. Measured FP4 updates average 3.74480 seconds versus historical
BF16 FA4's 4.08648, an 8.36% time reduction. The final FP32 adapter is byte-for-byte
identical to the preceding 19.42944-second first-use trajectory. This clears the
timing selection rule once every measured shape is resident; full-corpus shape
preparation costs, fresh control replication and quality validation remain
unresolved. Strict numerical parity remains failed and accepted only for timing.
Retain the opt-in implementation and BF16 FA4 standard.

Current warmed full-model FP4 profiling of fixed updates 11/15/20 attributes
24.93% of summed CUDA kernel time to FlashQLA/GDN, 20.17% to ordinary GEMMs,
14.24% to BF16 FA4, 5.65% to frozen-base FP4 GEMMs and 4.31% to dynamic FP4
conversion. Generic SiLU/pointwise work, copies and unclassified kernels make
up the remainder. The traced updates launch 33,520–46,144 kernels and average
0.71491 seconds without device events, 13.96% of pooled device span. These gaps
and synchronization costs motivate dispatch/copy investigation; they do not
prove CPU limitation or add to overlapping GPU time. All actual updates remain
fully warmed. Startup validation is reused with failed strict receipts retained;
the profile loss history/final adapter differ from the earlier run, with cause
not isolated. Diagnostic traces do not replace unprofiled speed measurements or
establish quality parity. Larger supported targets are GDN work and remaining
BF16 projections; native FP4 conversion is no longer the largest identified cost.

The user subsequently selects the combined native FP4 MLP/FA4 path as the
timing baseline for further systems optimization and requests persistent workers
across compatible trials and future sessions. The
[resident-worker decision](decisions/b200_fp4_optimization_worker.md) records
matched state resets, reused validation, retained in-process caches and targeted
GEMM attribution. This systems reference does not promote the FP4 path on quality
or erase its strict numerical failures.

The resident worker completes two baseline repeats and one diagnostic trajectory.
Uninstrumented means are 3.65854/3.67101 seconds per update; exact loss/gradient
logs, physical partitions and final FP32 adapters reproduce after state reset.
All sixty updates add zero plans/specializations/graphs, and the second twenty-
update trial takes 83.44038 seconds without preparation. The worker remains
resident and idle. Context/shape attribution identifies frozen GDN projections
as 9.67% of kernel time, LoRA GEMMs as 6.62%, and GDN scan/convolution/norm as
25.24%. The next practical GEMM candidate is NVFP4 for large frozen GDN QKV/Z/
output projections with shared packing, preserving BF16 recurrence and FP32
gates/norm. It is proposed, not a new speed or numerical acceptance result.

The subsequent matched GDN projection screen rejects both merged and separate
NVFP4 variants: 4.16069/4.04658 seconds per update versus a new worker's pooled
3.67836-second FP4-MLP/BF16-GDN control, 13.11%/10.01% more time. All eighty
updates are finite, warmed and use the same physical partitions; first-batch
adapter-gradient differences nevertheless reach 117.84%/137.15%. Lower short-run
training losses do not establish quality. Keep the current timing baseline and
resident worker. Investigate NVIDIA's BF16 width-four SiLU causal convolution
as a smaller arithmetic-preserving intervention; its native packed API needs
cumulative offsets rather than `seq_idx`. The convolution contributes about
4.31% of the existing trace's summed kernel time, so its potential is bounded.
See the [completed findings](findings/b200_mlp_gemm.md).

The subsequent NVIDIA BF16 causal Conv1D screen also rejects replacement in the
current FP4-MLP configuration: warmed updates average 4.02959 seconds versus
3.67836 for the resident control, 9.55% more time. Isolated packed forward/input
gradients agree closely with Dao and show zero leakage, but the matched first
model batch changes loss by 0.09096 and adapter gradients by 113.75%. Returning
Dao outputs/gradients through the same wrapper reproduces the control exactly;
the discrepancy is triggered by convolution arithmetic rather than the wrapper.
The evidence does not isolate downstream amplification to FP4 MLPs versus GDN
recurrence. A separate finite timing-only run preserves failed parity, reuses
completed preparation and completes twenty updates with zero compilation misses.
Keep the original convolution and resident baseline; no quality promotion follows.

The user requests a full replication of the original regular 4B run with the
selected native FP4 MLP/FA4 default: all 8,688 monitoring rows, LR 5e-5, one epoch,
followed by all 3,012 canonical ID examples. The
[replication decision](decisions/b200_fp4_full_training_replication.md) freezes
inputs, initial FP32 adapters and held-out selection before launch. The ordinary
Trainer's first attempt stops at 43/272 updates on the existing NC2 B200 after a
substantial full-corpus throughput regression; its source/logs are archived and
a replacement starts from the frozen original initialization.
Full-corpus first-use plan costs exceed the short warmed timing cohort; optional
compile-only cache population uses spare CPUs without replaying a model or
changing arithmetic. Training and ID results are pending; no quality equivalence
or promotion follows from the launch. The historical control used BF16/SDPA, so
the eventual comparison includes both the MLP and attention recipe changes.
Bounded diagnostics measure about 24 seconds of native plan construction for
one upcoming update despite all compiled-object cache hits, plus about 7 seconds
of conversion/row-scale specialization. The wrapper was rebuilding plans for
each token count even though NVIDIA supports runtime M. Runtime-shape plan reuse
matches fixed-M native outputs bitwise in 16 projection cases; generic conversion
kernels match codes/scales/inverses in 21 cases across seven row counts, and
generic row scaling matches at those seven lengths. These
targeted checks do not yet establish full-run speed or ID quality. Preserve the
first attempt separately from the replacement when reporting practical time.
The replacement initially fails the historical source-checksum guard before
any update. Preserve that receipt separately. The new source hashes are now
bound to the targeted bitwise evidence, retaining historical hashes and failed
canaries; source drift is checked before loading a model. Training and ID
remain pending after this corrected restart.
The corrected worker reaches eight updates with four native plans: its first
update takes 293 seconds including compilation, then updates 2–8 average 10.43
seconds (one-second log resolution), with all eight losses/norms matching the
archived attempt exactly. This addresses full-corpus first-use overhead; the
warmed short-cohort benchmark had already paid those costs. Completed-epoch
throughput and ID quality remain pending.

The corrected full epoch subsequently completes all 272 updates in 2,779.630
seconds (46 minutes 20 seconds), 29.02% less training-loop time than the historical
3,916.060-second BF16/SDPA control. The first update includes 293.159 seconds of
preparation; remaining logical updates average 8.963 seconds in the recorded
update timer. This validates the full-corpus throughput repair, not a similar
gain in the already-warmed short cohort. Mean training loss is 0.243972 versus
0.233122 historically. All 256 FP32 master tensors are finite and preserved
bitwise by the serving rebase; the trained worker remains resident. Fresh score
parity and all 3,012 canonical ID scores then complete successfully. Macro
pAUROC@20 improves from 0.846273 to 0.886090, AUROC from 0.951394 to 0.965927,
and Brier from 0.089216 to 0.086493. At the fixed 0.5 threshold, macro FPR falls
from 0.051326 to 0.022328 but recall also falls from 0.830638 to 0.810651. Pooled
ten-bin ECE worsens from 0.058943 to 0.096407, reflecting underprediction of
positive probability. Report this tradeoff alongside ranking improvement; do not
fit calibration on ID or promote from this one-seed combined-recipe comparison.
The final prediction hash, per-source results, startup costs and retained worker
are recorded in the replication decision.

The user next requests a small production inference benchmark covering both
interactive latency and batch throughput. The
[inference benchmark decision](decisions/b200_inference_benchmark.md) freezes
the training systems cohort as a development workload, with a nested 64-row
quick pass, closed-loop HTTP concurrency 1/4/16, repeat score variation and
prefix caching disabled. The final FP4-trained adapter is served in BF16;
training precision is not an inference kernel selection. Establish and collect
this B200 baseline before choosing any optimization. No held-out quality or
production arrival-rate/SLO result follows from this systems development set.
The BF16 HTTP baseline subsequently completes all six 64-row passes: median
throughput is 4.531/8.378/11.425 requests/s at concurrency 1/4/16, with p50
latencies 0.154/0.411/1.255 seconds. Score parity passes, while three rows cross
0.5 across repeat/concurrency arrays. Retain that baseline variation when
screening later precision/scheduling candidates. The server remains resident.

The user subsequently requests merging LoRA into the BF16 base for future
evaluations, with reconstructable merged checkpoints on ephemeral Runpod disk.
The next matched HTTP benchmark removes dynamic adapter projections while
keeping the same frozen workload, engine limits and prefix-cache policy.
Accumulate updates in FP32 before BF16 export, preserve master/base weights and
persistent provenance, and require fresh adapter-specific serving score parity.
The [inference decision](decisions/b200_inference_benchmark.md) records the
protocol; historical dynamic-LoRA evaluation contracts remain frozen.

The merged suite completes: prompt throughput rises 1.56×/1.96×/2.23× at
concurrency 1/4/16, reaching 107,073 input tokens/s at 16. Interactive median
latency falls 0.154 → 0.104 seconds. Fresh serving canary passes against both
master and unmerged scores, but paired 64-row repeat medians still have 1/1/2
threshold flips; preserve these shifts alongside baseline repeat variation.
Keep the merged worker resident and disposable weights on ephemeral disk, with
immutable FP32 adapters and all provenance/results persistent. This validates a
performance gain on the small systems workload, not equivalence of complete
ID/OOD quality metrics.

The user next requests inference kernel optimization. The live merged vLLM
engine already uses FlashInfer/TRTLLM full attention and FlashInfer GDN; its
BF16 linears use PyTorch's CUDA GEMM path. The first new kernel screen uses
native vLLM per-channel-weight/per-token-activation FP8 only in MLPs, retaining
the warm, idle merged BF16 control and matched HTTP workload. Loaded precision
and resolved GEMM classes are audited before the unchanged score canary.
The [kernel screening decision](decisions/b200_inference_kernels.md) records
scope, hypotheses and stop conditions; prior SM120 FP8 results motivate this
trial but do not establish its B200 performance or numerical fidelity.
