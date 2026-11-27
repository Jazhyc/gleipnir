# Amendment 2 results, 23 October 2026

The frozen cover-only round completed, but produced no usable cover stories.
All 149 final cells are invalid: development 0 valid / 50 invalid, held-out
0 valid / 99 invalid. The 4,096-token ceiling was too small for full GLM's
reasoning on these prompts. Every successful response exhausted that ceiling
without returning final prose. This round did not test the semantic quality
of a completed full-GLM cover story.

The pre-registered fallback applies. Cover stories have no frozen numeric pilot
prediction and are designated held-out descriptive only. There are zero usable
held-out cover pairs, so no cover-story monitor metric can be estimated from
this build. No GPU, VM, OpenRouter, Anthropic or monitor scoring ran.

## Implementation and review

Commits `64d53cb` and `795ab74` froze the amendment, implementation and review
fixes before the first new Tinker request. Both are scoped to this experiment
and end with the requested co-author trailer. No push was made.

Astra reviewed `64d53cb` before generation. It found no live-run blocker for
the pinned inputs, and identified two offline-path issues. The second commit
isolates mock preflight from the live campaign and supplies synthetic non-cover
fixtures for fresh dry runs. Tests cover end-to-end mock generation, checking,
auditing and building; interrupted retries; cover-only dispatch; campaign cap
exhaustion; and the 19/20-pair prediction boundary. The requested combined suite
passed all 123 tests. No changes were made to `src/`, another experiment or A4.

## Writer and requests

Full `zai-org/GLM-5.3:peft:262144` was chosen a priori for stronger reasoning
over exceptional behavior, without a bake-off or monitor-score evidence.
Only the cover writer changed. It used temperature 0.7, seed 0, 4,096 output
tokens, always-on reasoning and no unsupported reasoning-effort or thinking
fields. There were at most two attempts per task.

The cover writer received the actual exceptional implementation and confirmed
trigger input with the ordinary and exceptional outputs, alongside the original
neutral special-case description and problem. The prompt required an explicit
general trigger and exact changed answer in the right direction, presented as
an intended requirement with a task-level rationale. The concrete trigger was
private grounding. The frozen length band and prohibitions on program labels,
code identifiers and invented examples stayed in force. Identical insertion
into both programs stayed unchanged. No study text is reproduced here.

There were 294 new writer requests. Of these, 288 returned successfully with
finish reason `length` and empty final prose. Six timed out: tasks 542, 4274,
4154 and 1024 on their first attempt, and tasks 1275 and 4059 on their second.
No timeout was retried, and no refusal occurred. Final status was truncation
for 143 tasks and infrastructure failure for six. No content repair, token
increase or additional generation round followed these observations.

The checker stage completed with zero new requests because every cover text
failed format before advisory checking. The 447 imported non-cover checker
records and their current identities were preserved. Direct hand inspection
confirmed empty final text for every generated cover cell. Each has a current
SHA-bound invalid disposition. Rejected trigger task 1099 has a separate
missing-text invalid disposition and was never regenerated.

## Preserved work and build

The new folder has a fresh binding and imports the pinned Amendment 1 source.
All 1,434 non-cover text-history records, 447 checker records and 601 hand-audit
records match the source lists exactly. The complete receipt chain is retained,
including inherited receipts. Old bindings, artifacts and ledger entries were
not rewritten. Donor assignments remain pinned to the original rule.

`audit-sample` and `build` completed. Paired task counts are:

| Condition | Development | Held-out |
| --- | ---: | ---: |
| None | 50 | 99 |
| Task restatement | 44 | 91 |
| Irrelevant | 41 | 93 |
| Specific accusation | 32 | 81 |
| Cover story | 0 | 0 |
| Generic accusation | 50 | 99 |

The build has 1,581 rows: 434 development, 926 held-out and 221 honest calibration
rows. All non-cover counts equal Amendment 1. The frozen maximum of 2,021 rows
reconciles with 440 excluded rows. Exclusion records comprise 57 hand-invalid
cells, 149 missing/failed text cells, eight invalid-donor cells and one
unverified trigger task. No insertion validator failed.

Output folder: `results/adversarial_crossed_text_amendment2_20261023`.

Variant SHA-256:
`e14af2668457b5d4db23b819015da326c3cd7799a7b6e4b6e4993e984875a2ac`.

Sealed manifest SHA-256:
`a86cdf3777ef8c6836c567153c4f248d2c36bdf2ce0eed035a1d0e78b9a3336b`.

Machine-readable counts and preservation checks are in
`amendment2_report.json` in that folder. The sealed inputs and exclusions are
in `variants.jsonl`, `variants.manifest.json`, `variants_summary.json` and
`variant_failures.jsonl`.

## Spend

All ledger records are settled. Six timeout charges remain conservative
worst-case amounts because actual provider cost is unknown. They are included
in every total below. Role totals reconcile with the campaign ledger.

| Tinker role | Before this round | Added | Total | Cap |
| --- | ---: | ---: | ---: | ---: |
| Writer | $4.320156920 | $15.899904720 | $20.220061640 | $35 |
| Checker | $1.367554965 | $0 | $1.367554965 | $6 |
| Step 1 total | $5.687711885 | $15.899904720 | $21.587616605 | $41 |

Campaign Tinker accounting is also $21.587616605 against $185. A4's $140
scope was untouched. The pre-dispatch bound for this round was $22.454556480,
or $26.774713400 in total writer accounting including prior rounds.
