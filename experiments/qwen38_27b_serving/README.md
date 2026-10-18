# Arithmetic-preserving serving optimizations for Qwen3.8-27B

Goal: speed up BF16 monitor serving of the trained Qwen3.8-27B
(`qwen38-27b-s0-01`, merged BF16) by porting the serving changes that do not
change precision. User scope, 2026-10-18: BF16-preserving changes only; the
FP8 attention / FP4 MLP and GDN projection recipe stays reserved for a future
QAD-adapted 27B ([serving decision](../../docs/decisions/b200_monitor_score_reference.md),
[transfer note](../../docs/research/qwen38_27b_serving_transfer.md)).

**Candidates**, each validated on the 4B as bitwise or exact-arithmetic:
1. last-query final-layer attention (layer 63 attends only from each
   request's pooled last token; every token still writes K/V);
2. the GDN context-parallel route gate (calls under 8,192 tokens take the
   non-CP route);
3. the lean GDN prefill host path (bitwise identical to upstream GDN).

**Control.** The campaign's BF16 server (`bf16_server`, `Bf16Worker`) on the
merged trained 27B, with the same scheduler, chunked prefill, prefix caching
off, 32,768 context and runner capacity 128. First the backlog's
batch/singleton consistency gate for BF16.

**Measurements** (per the inference guide): prompt tokens/s as the main
throughput comparison, plus requests/s and latency, on a fixed workload and
concurrency with all repeats; per-source and macro AUROC deviation and score
parity against the archived BF16 control predictions.

**Acceptance (frozen before measuring).** A candidate is kept when its scores
are bitwise equal to the control, or for the route gate within the control's
own repeat variation with no AUROC change beyond repeat noise; when every
startup check shows the layer actually ran; and when prompt-token throughput
improves by at least 2%. Adoption into a 27B serving default is the user's
decision.

**Stop** on missing or nonfinite scores, a failed startup check, failed
parity, or a mismatch in prompt identity or token counts.

## Run log
