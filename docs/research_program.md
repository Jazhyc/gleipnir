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
