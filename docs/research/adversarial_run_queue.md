# Adversarial stress tests: run queue

One list of what is running, what's next and what's parked. Update it when a run starts or finishes. Each run links to its frozen protocol or design. Plans are in [the stress-test plan](adversarial_stress_testing.md) and [the phase-2 design](adversarial_phase2_design.md).

Monitor of record: Gleipnir 4B **v0.3** (`e36aa2b`, key `4b-v03`). The phase-1 results on v0.1 stay as a completed, versioned set.

## Running

| Run | What it asks | Where | Protocol |
| --- | --- | --- | --- |
| Lab v0.3, Run 0 | Does our v0.3 setup match the card, and what threshold would a deployer use? | A100 40 GB | [lab README](../../experiments/adversarial_lab_v03/README.md) |
| Lab v0.3, Runs 1–2 | False alarms on honest lab agents; are peer liars caught; does a peer's post get an honest reader flagged? | H100 80 GB | same |

## Next, in order

1. **Cross-GPU drift check.** Rescore about 300 of Run 0's rows on the H100, to measure how much the threshold moves between machines. A few minutes; no protocol needed beyond a note in the lab README.
2. **Real swarm posts as framing (lab Run 3).** Plant about 200 real collusion.wiki posts into the same 300 honest readers: coordination and pressure posts, plus neutral ones of matched length. Freeze the selection before scoring. The data stays local and is used for evaluation only (no licence is stated).
3. **Code or comments?** Cross what the code does with what its comment claims, then test the resulting explanation on held-out tasks and the 27B. [Design](adversarial_code_or_comments.md).
4. **27B, teacher prompt, on the primary endpoints.** Lab Runs 1–2 and the code-or-comments held-out set. Needs an 80 GB GPU and teacher-prompt support in the scorer.

## Later

- **Same words, different situation.** Put monitor-directed text in four contexts: aimed at the monitor, legitimate task content, an ignored tool output, and a peer message. Use Jeremias's ID/OOD rows and his insertion code. ([Phase-2 design](adversarial_phase2_design.md), Level 0a.)
- **Monitor forensics (white-box):** token attribution and deletion tests on the most informative cases; check Jeremias's injection direction.
- **One expensive monitor** (Kimi K3 or Gemini) on fixed subsets, for the cost-against-robustness comparison.
- **Optimized serving path** (FP8/FP4) when Jeremias shares it. Probably needs a Blackwell GPU (RTX Pro 6000 on JarvisLabs).
- **White-box framing:** a gradient search for the peer post that most raises an honest reader's score.

## Parked

- **Deceived victims (P2.3, victim half).** Needs new live continuations.
- **Fragmented social attacks (P2.4) and AISI FakeLab.** Published effect; FakeLab needs its licence checked first.

## Done

- **4B v0.1 baseline.** It reproduces the card, and injected text backfires: [baseline](../../experiments/adversarial_baseline_4b/README.md).
- **Phase-1 attacks on v0.1:** A1 discrimination, A2 cover comments (low and high effort), A3 random search, A4 score feedback, A6 transfer.
- **Phase-1 replay on v0.3:** [replay](../../experiments/adversarial_phase1_v03_replay/README.md).
