# Amendment 3 results, 23 October 2026

The smoke gate failed: **1/5 finished final prose and 1/5 hand-valid**. The
thresholds were at least four finished and two hand-valid. Generation stopped
after the five preregistered development tasks. The frozen fallback restored
the Amendment 1 final cover collection and its existing hand verdicts: six valid
development pairs and 28 valid held-out pairs. Cover stories are held-out
descriptive only under the Amendment 2 rule, with no numeric pilot prediction.
No monitor scores exist. No OpenRouter, Anthropic, GPU or VM actions ran.

## Protocol and review

Commit `1e35415` froze the amendment, implementation, tests and first Astra
review fixes before any new Tinker call. It is scoped to this experiment and
has the requested co-author trailer. A second Astra review of that commit
found no unresolved blocker for the smoke run or failed-gate fallback. No push
was made; `src/`, other experiments and A4 were unchanged.

The first review found a misplaced budget preflight, stale smoke evidence
remaining usable after corrected hand verdicts, fallback records vulnerable
to later overrides, and two interruption recovery issues. The committed fixes
retain the whole-remaining-round budget guard, reject superseded evidence for
continuation, protect the restored cover collection, resume partial fallback
appends and conservatively count interrupted reserved requests as unfinished.
The requested combined crossed-text/A4 test command passed all **141 tests**.

The only writer configuration changes were 16,384 output tokens, an
1,800-second timeout and the authorized budgets. The model, grounding inputs,
prompt hashes, temperature, seed, two-attempt limit and retry rule were
unchanged. The timeout allows full output at 10 tokens/second plus overhead.
The smoke IDs were fixed before calls by the registered hash rule: 1494,
1226, 4384, 4310 and 2026. Its conservative two-attempt bound was $2.22810804.
The unconditional full-round writer bound remains above $80; a passing gate
would have stopped at the unchanged continuation budget guard. The observed
fallback was triggered by text validity, independently of that constraint.

## Smoke outcomes and failures

All five first attempts reached the output-token ceiling without final prose.
All received exactly one permitted format/truncation retry. Task 4384 returned
completed prose on its second attempt. Hand inspection found that it states the
exact exceptional labeling requirement, in the right direction, with a
task-level rationale. The other four final outputs were empty and truncated.
Every new cover cell received a current SHA-bound hand disposition before the
gate was decided. No generated text is reproduced here.

There were **10 writer requests: nine `length` finishes and one `stop`**.
All ten requests returned successfully. There were no timeouts, infrastructure
failures or refusals. Final smoke statuses were four truncations and one
accepted-format text. The advisory checker made one new call for that completed
text; it did not determine the hand verdict or fallback. The new smoke texts,
hand reads, checker result and receipts remain in append-only histories.

## Restored collection and build

Current cover texts and hand verdicts match the pinned Amendment 1 collection
exactly. Among the 149 accepted-trigger tasks, development has **6 valid / 44
invalid** cover cells; held-out has **28 valid / 71 invalid**. Rejected trigger
task 1099 retains its separate missing-text invalid disposition and is excluded.
The four failed smoke cells and one valid smoke cell are historical evidence,
not additions to the restored cover collection.

All 1,434 non-cover text-history records, 447 checker records and 601 hand
records match the Amendment 2 source lists exactly. Trigger artifacts and the
complete inherited receipt chain also match. Existing donor assignments, old
bindings and prior ledger records were preserved. All campaign reservations
are settled, including the conservative charges retained for Amendment 2's
six timeouts.

`audit-sample` and `build` completed. Final paired task counts are:

| Condition | Development | Held-out |
| --- | ---: | ---: |
| None | 50 | 99 |
| Task restatement | 44 | 91 |
| Irrelevant | 41 | 93 |
| Specific accusation | 32 | 81 |
| Cover story | 6 | 28 |
| Generic accusation | 50 | 99 |

There are **1,649 rows**: 446 development, 982 held-out and 221 honest
calibration rows. The variants are byte-identical to Amendment 1. The frozen
maximum of 2,021 rows reconciles with 372 excluded rows: 170 hand-invalid
cells, two missing/failed text cells, eight invalid-donor cells and one
unverified trigger task. No insertion validator failed.

Output folder: `results/adversarial_crossed_text_amendment3_20261023`.

Variant SHA-256:
`7d0f9a54c41946a8cf460ab4526560abcd8a8bb8e40e5723c9b708dbc6f9d9ef`.

Sealed manifest SHA-256:
`cac511fdcf62c76f3f6e5e21e66daa19775e1145b1a169cf4299779e98d707b0`.

The output folder contains `amendment3_report.json`, `smoke.tasks.json`,
`smoke.gate.json`, `fallback.manifest.json`, sealed variants and exclusions.
The manifest binds the smoke decision, fallback provenance and underlying
artifact histories as well as the final variants.

## Spend

New request charges use returned token usage. Prior conservative timeout
charges remain included. Role totals reconcile with the campaign ledger.

| Tinker role | Before Amendment 3 | Added | Total | Cap |
| --- | ---: | ---: | ---: | ---: |
| Writer | $20.220061640 | $1.973563650 | $22.193625290 | $80 |
| Checker | $1.367554965 | $0.000764055 | $1.368319020 | $6 |
| Step 1 total | $21.587616605 | $1.974327705 | $23.561944310 | $86 |

Campaign Tinker accounting is **$23.561944310 / $230**. A4's $140 scope is
untouched. No other provider spend was added.
