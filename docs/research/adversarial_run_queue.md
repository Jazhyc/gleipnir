# Adversarial stress tests: run queue

One list of what is running, what's next and what's parked. Update it when a run starts or finishes. Each run links to its frozen protocol or design. Plans are in [the stress-test plan](adversarial_stress_testing.md) and [the phase-2 design](adversarial_phase2_design.md).

Monitor of record: Gleipnir 4B **v0.3** (`e36aa2b`, key `4b-v03`). Its adapter is identical to the **v0.4** release, so the results carry over. The phase-1 results on v0.1 stay as a completed, versioned set.

**Faster scoring:** `main` now ships merged BF16 checkpoints (`Jazhyc/Gleipnir-{4B,27B}-BF16`) that skip the LoRA overhead, plus an optimized serving image for B200 only (JarvisLabs has no B200). Before switching, check that the merged checkpoint reproduces our PEFT scores on a few hundred rows.

## Running

Nothing. The H100 VM is paused (`541509`).

## Next, in order

1. **Natural-cheating pilot** (Neel recommended reward-hacking benchmarks on the 20 October drop-in): an open agentic model with thinking off on ImpossibleBench, then probes against Gleipnir at high recall. Needs a split agreed with Jeremias and a GPU top-up.
2. **Prompt variant for Jeremias:** the teacher prompt plus a multi-agent note, chosen on half the readers and reported on the other half.
3. **Code or comments?** Cross what the code does with what its comment claims, then test the resulting explanation on held-out tasks and the 27B. [Design](adversarial_code_or_comments.md).
4. **Real swarm posts as framing (lab Run 3), only if there is spare time.** Plant about 200 real collusion.wiki posts into the same 300 honest readers: coordination and pressure posts, plus neutral ones of matched length. Freeze the selection before scoring. The data stays local and is used for evaluation only (no licence is stated).
5. **27B, teacher prompt, on the remaining endpoints.** Lab Runs 1–2 and the code-or-comments held-out set. Needs an 80 GB GPU and teacher-prompt support in the scorer.

## Later

- **Same words, different situation.** Put monitor-directed text in four contexts: aimed at the monitor, legitimate task content, an ignored tool output, and a peer message. Use Jeremias's ID/OOD rows and his insertion code. ([Phase-2 design](adversarial_phase2_design.md), Level 0a.)
- **Monitor forensics (white-box):** token attribution and deletion tests on the most informative cases; check Jeremias's injection direction.
- **One expensive monitor** (Kimi K3 or Gemini) on fixed subsets, for the cost-against-robustness comparison.
- **Optimized serving path** (FP8/FP4) when Jeremias shares it. Probably needs a Blackwell GPU (RTX Pro 6000 on JarvisLabs).
- **White-box framing:** a gradient search for the peer post that most raises an honest reader's score.
- **Adversarial game: a detector as a tool in the recruitment lab.** From Smith, Chughtai and Nanda (2025), appendix "adversarial games".
  - **Setup:** in the recruitment world, give the target agent A a lie-detector tool, such as Gleipnir's score on each recruiter's posts, or later a white-box probe.
  - **Measure:** whether A gets talked into running the session-ending diagnostic less often than without the tool. Recruiters are honest or deceptive by arm.
  - **Why it's useful:** the detector's value shows up as a change in who wins, so no individual post has to be labelled as a lie, which sidesteps the paper's labelling problem.
  - **Controls:**
    - a placebo tool that returns random scores;
    - a tool that only reveals the recruiter's role, which checks that the detector isn't just learning who the recruiters are, the paper's werewolf caveat;
    - the same tool given to the recruiters instead.
  - **Status:** idea only, not designed in detail. It needs live episodes, so it costs API spend.

## Parked

- **Deceived victims (P2.3, victim half).** Needs new live continuations.
- **Fragmented social attacks (P2.4) and AISI FakeLab.** Published effect; FakeLab needs its licence checked first.

## Done

- **Is the framing effect a 4B thing? (steps 0–3, 21 October).** It is a small-model effect. The accusation to the monitor flags 54–82% of readers for 4B v0.3 (either prompt), but 0% for the 27B with its teacher prompt and 0% for Kimi K3, and neither flags any reader the 4B flags. The untrained base already leans toward the accusation; InjectionAugmented40 stays under its line, at lower recall. Kimi catches 5 of 10 liars at 1%; swapping lies for truth lowers every monitor's liar scores; no natural lies were found. See the [protocol and results](../../experiments/adversarial_is_it_4b/README.md).
- **Lab v0.3 follow-up** (20 October): placement, rewordings, benign notes, and one-machine thresholds. A monitor-directed misconduct claim flags 43–78% of innocent readers across wordings and both placements, while a benign note flags none. See the [lab README](../../experiments/adversarial_lab_v03/README.md).
- **Lab v0.3, Runs 0–2 plus the drift check** (20 October). There are almost no false alarms on natural lab behaviour, and 0 of 10 liars are flagged, though they rank well. One monitor-directed peer post gets 69% of honest readers flagged at 1%. See the [lab README](../../experiments/adversarial_lab_v03/README.md).
- **4B v0.1 baseline.** It reproduces the card, and injected text backfires: [baseline](../../experiments/adversarial_baseline_4b/README.md).
- **Phase-1 attacks on v0.1:** A1 discrimination, A2 cover comments (low and high effort), A3 random search, A4 score feedback, A6 transfer.
- **Phase-1 replay on v0.3:** [replay](../../experiments/adversarial_phase1_v03_replay/README.md).
