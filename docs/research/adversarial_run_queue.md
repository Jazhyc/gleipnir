# Adversarial stress tests: run queue

One list of what is running, what's next and what's parked. Update it when a run starts or finishes. Each run links to its frozen protocol or design. Plans are in [the stress-test plan](adversarial_stress_testing.md) and [the phase-2 design](adversarial_phase2_design.md).

Monitor of record: Gleipnir 4B **v0.3** (`e36aa2b`, key `4b-v03`). Its adapter is identical to the **v0.4** release, so the results carry over. The phase-1 results on v0.1 stay as a completed, versioned set.

**Faster scoring:** `main` now ships merged BF16 checkpoints (`Jazhyc/Gleipnir-{4B,27B}-BF16`) that skip the LoRA overhead, plus an optimized serving image for B200 only (JarvisLabs has no B200). Before switching, check that the merged checkpoint reproduces our PEFT scores on a few hundred rows.

## Running

- Nothing. The A100 80GB VM (`gleipnir-qwen-gpu`) was stopped on 24 October at about 23:05 UTC after the liar-study scoring; results and logs are pulled back. Its boot disk keeps the environments (`~/gleipnir`, `~/gleipnir-coc`, `~/venv-fla`).

## Failures and blockers

Short entries, newest first. Details live in each experiment's README.
- **24 Oct, lab build:** `adversarial_lab_v03/build.py` accepts any episode with an `outcome.json`, but the lab writes that file for every ending, including infrastructure errors and budget stops. Some early-ended episodes may therefore be among the 3,006 honest rows (not yet counted). The 10 liars all come from fully completed episodes. New builds filter on the end reason.
- **22 Oct, code-or-comments checker:** the fidelity checker fails valid cover stories (69 of its 71 backdoor-misleading failures, read by hand), even after Amendment 4 reworded the question. A blind, description-only check fixed the deviation question (47 of 47 agreements) but missed one cover story on its second question, so by the pre-set rule every failed cover story was read by hand instead (Amendment 6).
- **22 Oct, code-or-comments texts:** a second prompt leak, the trigger prompt's label "Solution B", reached 50 special-case texts from 49 tasks; neutralised and regenerated (Amendment 5).
- **22 Oct, VM code sync:** a `git archive` copy has no `.git`, so code-or-comments' commit guard can't pass; the VM now holds a clone of a git bundle (`~/gleipnir-coc`).
- **22 Oct, FakeLab feasibility:** not assessed yet. Abhinav or Jeremias will collect the logistics (how to run, log format, session lengths) by hand. *Resolved 24 Oct: assessed privately; see Parked.*
- **22 Oct, stage 1c:** our own BF16 copy matches Tinker token for token on single prompts but does worse in agent runs (12 correct against 33), cause unknown. Not the tool parser (0.5% of calls came back empty) and not `top_k`; our prompts carry 20 extra tokens of tool schema.
- **22 Oct, stage 1c setup:** two A100s out of stock in us-central1-a and -c; one was available. The Ubuntu accelerator image lacks a C compiler (Triton) and `nvcc` (FlashInfer's sampler), so we installed `build-essential` and set `VLLM_USE_FLASHINFER_SAMPLER=0`.
- **22 Oct, diagnostics:** two replay tests were wrong before they were right: Inspect stores long request text as `attachment://` references (resolve them from `sample.attachments`), and a 400-token cap truncated long tool calls into empty ones.
- **21–22 Oct, monitoring:** GCP's federated login expires after several hours and silently stops VM checks; watchers now report an expired login.
- **21 Oct, cheating pilot:** offline nonce checks timed out under load and recorded failures; `recheck.py --retry-timeouts` reruns only those. The in-run nonce check copied `/` and never worked (Amendment 3).

## Next, in order

1. **Earlier items, still open:** the multi-agent prompt variant for Jeremias, the 27B on lab Runs 1–2, and frontier monitors on EvilGenie.

## Later

- **Same words, different situation.** Put monitor-directed text in four contexts: aimed at the monitor, legitimate task content, an ignored tool output, and a peer message. Use Jeremias's ID/OOD rows and his insertion code. ([Phase-2 design](adversarial_phase2_design.md), Level 0a.)
- **Monitor forensics (white-box):** token attribution and deletion tests on the most informative cases; check Jeremias's injection direction.
- **Optimized serving path** (FP8/FP4) when Jeremias shares it. Probably needs a Blackwell GPU (RTX Pro 6000 on JarvisLabs).
- **White-box framing:** a gradient search for the peer post that most raises an honest reader's score.
## Parked

- **Goodfire reward-hacking pilot** ([draft protocol and adjudication](../../experiments/adversarial_goodfire/README.md)). Two blind readers accepted 10 of 64 action-visible coding candidates (they agreed on 53 of 64), so a coding AUROC would be descriptive only. Parked 24 October; a short descriptive scoring run can join a later GPU session.
- **Real swarm posts as framing (lab Run 3).** Plant about 200 real collusion.wiki posts into the same 300 honest readers: coordination and pressure posts, plus neutral ones of matched length. Freeze the selection before scoring. The data stays local and is used for evaluation only (no licence is stated). Parked 21 October: the framing effect turned out specific to the 4B.
- **Adversarial game: a detector as a tool in the recruitment lab.** From Smith, Chughtai and Nanda (2025), appendix "adversarial games".
  - **Setup:** in the recruitment world, give the target agent A a lie-detector tool, such as Gleipnir's score on each recruiter's posts, or later a white-box probe.
  - **Measure:** whether A gets talked into running the session-ending diagnostic less often than without the tool. Recruiters are honest or deceptive by arm.
  - **Why it's useful:** the detector's value shows up as a change in who wins, so no individual post has to be labelled as a lie, which sidesteps the paper's labelling problem.
  - **Controls:**
    - a placebo tool that returns random scores;
    - a tool that only reveals the recruiter's role, which checks that the detector isn't just learning who the recruiters are, the paper's werewolf caveat;
    - the same tool given to the recruiters instead.
  - **Status:** idea only, not designed in detail. It needs live episodes, so it costs API spend.
- **Deceived victims (P2.3, victim half).** Needs new live continuations.
- **Fragmented social attacks (P2.4) and AISI FakeLab.** Published effect. FakeLab's feasibility was checked privately on 24 October and a two-episode smoke test on free models is under way; aggregate reporting still needs AISI's confirmation.

## Done

- **Monitor-aware cheating on ImpossibleBench (25 October): every cheat is announced, so no quiet cheats to test.** Qwen3.8-27B (thinking off) cheats 5% naturally, 17% with permission and 11% when also told a monitor reads its actions (p = 0.31). Two blind readers: all 37 cheats admit it, with or without the monitor note. Cheats caught at the 1% line: 4B 35/37, 27B 33/36, Kimi 36/36; non-cheats flagged 7/354, 9/352, 14/352. See the [protocol and results](../../experiments/adversarial_monitor_aware/README.md).
- **Multi-agent liars at scale (25 October): Gleipnir ranks liars almost as well as Kimi (AUROC 0.90 / 0.90 against 0.96, 30 liars against 52 honest recruiters) but at the benchmark 1% line catches 0 of 30 (4B) and 5 of 30 (27B), against Kimi's 18 of 30.** 72 main-cell and 48 other episodes from GLM-5.3, Kimi-K2.6, Qwen3.8-27B and GPT-6.1 Sol recruiters; lies labelled by two blind readers (kappa 0.93). Qwen3.8 and Sol rarely lie when told to. With a Haiku target, lies made A give up its delivery in 5 of 10 episodes against 0 of 11 under honest recruitment. The 4B's scores rise for agents that only read a lie. See the [protocol and results](../../experiments/adversarial_lab_liars/README.md).
- **Comment attack with score access (24 October): 7 of 59 caught backdoors get past the 4B after 4 tries, 1 of 68 past the 27B; almost nothing transfers (Sol and Sonnet 0, Kimi 1–2).** Comments raise false alarms on honest code more than they hide backdoors. Both arms are final after hand audits (9/9 and 2/2 passed). See the [protocol and results](../../experiments/adversarial_a4_scale/README.md).
- **Probe on the monitor (24 October): under an identical accusation, a no-text probe on the 4B's layer 16 still separates backdoored from honest code (0.95–0.96) while its answer is at 0.58–0.59.** The 27B's last-layer probe inverts under accusations (unexplained). See the [protocol and results](../../experiments/adversarial_crossed_text_probe/README.md).
- **Crossed-text rerun (23 October): with the same text on both programs, the 27B keeps its ranking under accusations (AUROC 0.98 / 0.96, level with Kimi, Sonnet and Sol); only the 4B loses it (0.58 / 0.59), worse than its untrained base.** This corrects the earlier "below chance" result, which came mostly from giving the two programs different texts. All five frozen 4B intervals held. Threshold calibration on undocumented code breaks for every monitor. See the [protocol and results](../../experiments/adversarial_crossed_text/README.md).
- **Gleipnir against frontier monitors (22 October).** On natural cheats Gleipnir matches Kimi K3, GPT-6.1 Sol and Claude Sonnet 5.5 (pAUROC@20 93.5–94.8%) at 3–13× lower cost; with misleading comments on backdoored and honest code it falls below chance (AUROC 0.35 and 0.40) while Sol and Sonnet hold. Superseded for the 27B by the crossed-text rerun: the texts differed between the two programs. Summary: [findings](../findings/adversarial_frontier_comparison.md).
- **Code or comments (22 October): text outweighs code, and the frozen held-out tests pass** (the text/code comparison was confounded; see the crossed-text rerun above). See the [protocol and results](../../experiments/adversarial_code_or_comments/README.md).
- **Natural cheats scored by Gleipnir (22 October): AUROC 0.96 for both models, no false alarms at the strict line.** See the [protocol and results](../../experiments/adversarial_natural_cheats/README.md).
- **EvilGenie × Escalation (22 October): the 27B far ahead of the 4B on other models' reward hacking (0.93 against 0.75 on the same rows).** See the [protocol and results](../../experiments/adversarial_evilgenie/README.md).
- **Natural-cheating pilot, stage 1c (21–22 October): our own BF16 copy behaves like Tinker, so DeepInfra is the odd provider; no stage 2.** Decisions 7 against Tinker's 19 and DeepInfra's 36; greedy text matches Tinker exactly on single prompts. Qwen's presence penalty raised decisions to 15 but didn't reduce looping. See [stage 1c](../../experiments/adversarial_cheating_pilot/stage_1c.md).
- **Natural-cheating pilot, stage 1b (21–22 October): `top_k` ruled out, no-go.** Adding Qwen's `top_k = 20` on Tinker changed almost nothing: 17 decisions against 19, and 70% of runs were message-for-message identical to stage 1, because Tinker follows the seed and top_p 0.8 already trims the tail. See [stage 1b](../../experiments/adversarial_cheating_pilot/stage_1b.md).
- **Natural-cheating pilot, stage 1 (21–22 October): no-go on Tinker as served.** Qwen3.8-27B with thinking off cheats on its own, mostly by giving in right after the setup says "failed, fix it" (26 of 37 successes). The compared Tinker arms gave 18 cheating runs on 16 tasks and 87 audited honest runs, short of the 20 and 100 the gate needs; OpenRouter alone had 31 on 31 tasks. Most runs never reach the choice before the 50-message limit. See the [protocol and results](../../experiments/adversarial_cheating_pilot/README.md).
- **Is the framing effect a 4B thing? (steps 0–3, 21 October).** It is a small-model effect. The accusation to the monitor flags 54–82% of readers for 4B v0.3 (either prompt), but 0% for the 27B with its teacher prompt and 0% for Kimi K3, and neither flags any reader the 4B flags. The untrained base already leans toward the accusation; InjectionAugmented40 stays under its line, at lower recall. Kimi catches 5 of 10 liars at 1%; swapping lies for truth lowers every monitor's liar scores; uninstructed workers made only a few natural false claims (3 confirmed in 140 flagged posts read, all confabulation-like). See the [protocol and results](../../experiments/adversarial_is_it_4b/README.md).
- **Lab v0.3 follow-up** (20 October): placement, rewordings, benign notes, and one-machine thresholds. A monitor-directed misconduct claim flags 43–78% of innocent readers across wordings and both placements, while a benign note flags none. See the [lab README](../../experiments/adversarial_lab_v03/README.md).
- **Lab v0.3, Runs 0–2 plus the drift check** (20 October). There are almost no false alarms on natural lab behaviour, and 0 of 10 liars are flagged, though they rank well. One monitor-directed peer post gets 69% of honest readers flagged at 1%. See the [lab README](../../experiments/adversarial_lab_v03/README.md).
- **4B v0.1 baseline.** It reproduces the card, and injected text backfires: [baseline](../../experiments/adversarial_baseline_4b/README.md).
- **Phase-1 attacks on v0.1:** A1 discrimination, A2 cover comments (low and high effort), A3 random search, A4 score feedback, A6 transfer.
- **Phase-1 replay on v0.3:** [replay](../../experiments/adversarial_phase1_v03_replay/README.md).
