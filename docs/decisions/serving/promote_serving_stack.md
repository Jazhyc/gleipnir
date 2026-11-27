# Serve the optimized B200 stack from gleipnir.serving.stack

Status: current.
Date: 2026-11-27.

User decision, 2026-11-27: accept the promotion of the optimized serving
runtime from its experiments into
[`gleipnir.serving.stack`](../../../src/gleipnir/serving/stack/__init__.py), and
set its [manifest](../../../src/gleipnir/serving/stack/promotion.json) to
`accepted`.

**What it changes.** The accepted serving default and the released 4B and 27B
recipes are unchanged: their recorded commands, kernels and receipts stay as
they were. At launch, `gleipnir.serving.sources` verifies each recorded source
directly or through the manifest and serves the command on the promoted
modules. Old `experiments.*` module names keep resolving through `_compat`.

**Evidence** ([check](../../../experiments/serving/serving_stack_promotion/README.md)).
On one B200 from `gleipnir-serve:1.2`, the promoted code reproduced `main`'s
scores, raw logits and prompt token counts on all 248 synthetic identity
prompts for both Gleipnir 4B and 27B (0 rows changed), the frozen acceptance
rule. The ID pass and throughput timing were skipped by user decision: the
question was determinism.

**Rules.** Editing a promoted file, or a source a receipt binds, makes a new
candidate recipe: regenerate the manifest (`status: candidate`) and repeat the
identity check before serving it. A new `gleipnir-serve` image built from this
code is the next release step.
