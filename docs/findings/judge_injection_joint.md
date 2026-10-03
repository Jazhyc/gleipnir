# Joint monitoring and injected-preference training

Date: 2026-10-03. Status: training launched; first optimizer updates verified, evaluation pending.

User selected a fresh adapter with 75% monitoring and 25% preference examples,
using the regular 5e-5 learning rate. The hypothesis is that retaining trajectory
supervision throughout training limits the retention regression seen after
preference-only continuation. Forgetting and task interference are not isolated.

The sampler draws all 8,688 original regular-prompt trajectories once, plus
2,896 preference examples: 290 clean, 1,303 preferred-answer-injected and
1,303 disfavored-answer-injected. Each full logical batch has 24/8 examples.
Both A/B orders are balanced within each condition. There are 362 updates.
The 2,606 injected draws represent 22.497% of all 11,584 training draws.
The sampled preference population has 25 original queries, 940 answer pairs,
78 distinct payload IDs and 2,896 unique augmented examples. Variants are not
independent tasks; more exposure does not establish broader attack diversity.

The frozen schedule contains 84,842,449 input tokens: 83,816,311 monitoring,
93,414 clean preference, 464,934 disfavored-injected and 467,790 preferred-injected.
Auxiliary examples are 25% of example-level loss mass but only 1.21% of input
tokens. Maximum context is 29,337, inside the existing validated 29,696 envelope;
no truncation. The eligible preference pool remains the 16,760 variants from
25 query groups; the 4,188 variants from six held-out queries remain excluded.

Preserve soft teacher BCE and the regular 0/1 monitoring surface. Preference
uses its separate hard construction labels, quality rubric and A/B surface.
Each example contributes one loss; packed readouts gather its own token pair.
The verified initial adapter has all 128 B matrices zero, matching the original
regular run's initialization. It is not either trained checkpoint.

FP32 masters, frozen BF16 base, packed FlashQLA/SDPA, persistent compiler-cache
reuse, the checksum-pinned original startup-validation reference and mandatory
finite/missing-gradient guards are retained. Final checkpoint only, no sweeps
or benchmark-based checkpoint selection. New serving parity is mandatory.

Compare original, preference-only continuation and joint-training results on
identical preference, regular-prompt APPS/honest controls and CoT-removed ID.
APPS remains excluded from training, data generation and sampler construction.
Differences in optimizer updates, auxiliary exposure and learning rate limit
causal attribution; no model promotion follows this exploratory comparison.

Protocol: [experiment README](../../experiments/judge_injection_joint/README.md).

## Startup verification

The existing B200 launch succeeded (pod `3422tdq7tfog52`, launcher PID 11814).
The remote execution contract matches every locally reviewed training source
and the frozen manifest. The unchanged startup validation was reused; the log
records FlashQLA in all 24 GDN layers and strict parity as not repeated. The
shared cache resolves to the populated persistent
`/workspace/gleipnir/.cache/training/student_injection_awareness` directory.
The first optimizer update completed in 277.58 seconds during compilation;
the second took approximately 11 seconds. GPU utilization reached 100% with
143,742 MiB used. Per-update finite/missing-gradient guards passed. No quality
result is claimed yet; regular-only monitoring evaluation is queued after the
fixed final checkpoint.

Validation: 74 affected CPU tests passed, followed by seven focused experiment
tests including two additional tokenization/real-Trainer integration checks
(76 distinct checks in total). Ruff and staged diff checks passed. Expected
CPU-only NVML warnings are retained. In-chat heartbeat scheduling is unavailable;
startup was inspected during the active turn, without claiming future automatic
agent follow-ups.
