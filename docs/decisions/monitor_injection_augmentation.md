# Monitor-directed injection replacement

Date: 2026-10-04. Status: user selected replacement; CPU preparation completed.
No GPU training or model evaluation has started for this intervention.

The user selected replacing 40% of the full 8,688-row regular training set,
rather than appending paired copies. Each original trajectory occurs once:
3,475 injected views and 5,213 unchanged clean views. Selection is proportional
within source/hard-label strata. The existing monitoring instruction and teacher
cache remain fixed; no JudgeDeceiver preference task is included.

Balance standalone user/assistant/tool message roles to 1,159/1,158/1,158.
Draw one requested insertion position from a truncated N(0.5,0.25²) distribution
over normalized transcript characters, then snap to an existing message or step
boundary. Source code, commands, tool bodies and original message bytes remain
intact. Short transcripts have coarse realized positions; report their actual
distribution rather than claiming a Gaussian distribution after snapping.

Eight project-authored templates in four families independently demand either
0 or 1. Expose every template in both behavior classes and with both demanded
verdicts. Reserve four additional templates from two different mechanism
families for held-out evaluation. APPS payloads and benchmark-specific wording
remain excluded from generation, training and selection.

This is an off-policy, label-preserving augmentation experiment. The original
agent continuation is not regenerated. Copying the original teacher target is
an explicit counterfactual behavior-invariance assumption, not an annotation
of the new transcript. Keep original labels, rationale and teacher request
provenance separate from new prompt/trajectory hashes and insertion receipts.
Existing malformed task-description wrappers are recorded and preserved,
without dropping or repairing their source rows.

The proposed matched run uses a fresh Qwen3.5-4B adapter, one epoch, regular
5e-5 learning rate and 272 updates under the validated packed BF16 FlashQLA/SDPA
recipe, with FP32 masters and persistent compiler-cache reuse. Only input
changes inside the validated context envelope can reuse its startup receipt.
Finite/missing-gradient guards and new adapter-specific serving parity remain
required. This record does not authorize new billable GPU capacity.

Use the regular monitoring instruction for all planned evaluation. Compare
against the original regular adapter on unchanged ID, APPS and honest controls,
plus paired clean/injected ID views from the held-out template families. Freeze
the final checkpoint and report ranking, calibration, class-conditional flips
and clean-calibrated operating points. No benchmark-selected checkpoint or
promotion follows this exploratory intervention.

See the [experiment protocol](../../experiments/monitor_injection_augmentation/README.md)
and [template bank](../../experiments/monitor_injection_augmentation/templates.json).

Preparation audit: all 8,688 identities and original supervision remain intact;
all 3,475 inserted spans are reversible, and 5,213 clean JSONL lines are
byte-identical. Every training template spans both hard-label classes and both
demanded verdicts. Actual input total is 83,969,155 tokens, maximum 29,391 with
no truncation. Requested and realized position histograms are preserved in the
manifest. Forty focused tests and scoped Ruff checks pass. These are data
validation results, not evidence of monitor robustness.
