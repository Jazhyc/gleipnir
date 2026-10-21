# Natural-cheating pilot, stage 1b: does Qwen's `top_k` explain the Tinker gap?

Frozen before any stage-1b run. Stage 1 stays a no-go (see the README's results); this is a new experiment with its own pass mark, not a second try at stage 1's.

## Why

- On the same 103 contradicting tasks, OpenRouter (DeepInfra, BF16) reached a decision in 36 runs and Tinker in 19 (p = 0.01). They cheated 18 and 5 times.
- After stage 1's gate we found that Tinker applies no `top_k` unless asked. Asked 60 times for one random word at temperature 1.0 and top_p 1.0, it gave 38 distinct answers without `top_k`, one of them a stray Arabic token, and 21 with `top_k = 20`.
- Our runs sent Qwen's no-thinking temperature (0.7) and top_p (0.8) but not its `top_k = 20`, which is in the model's `generation_config.json`.
- DeepInfra (through OpenRouter, pinned as in stage 1; $0.0012) behaves as if `top_k = 20` is its default: 20 distinct answers of 60 without `top_k` and 18 with it. Both providers respect `top_k` (with `top_k = 1`, 3 identical answers of 3). So in stage 1 the two providers sampled differently.
- A rare bad token can break a tool call. Tinker runs had more tool errors (186 against 110) and fewer submissions (24 against 40).

## Arms

- `t_b_conflicting_k20` and `t_b_oneoff_k20` are identical to `t_b_conflicting` and `t_b_oneoff` except for `top_k = 20` in the request body: same 103 tasks, prompt B, seed 0, 50 messages, 10 submissions, concurrency 30, the same VM type, pinned image and dataset revision.
- A two-task smoke run first (`--out smoke_k20`, excluded) checks that the logged request carries `top_k = 20`.
- After both arms finish, with nothing else running: `recheck.py --retry-timeouts --threads 2`, then `labels.py`.

## Question 1: does `top_k` explain the gap?

Measured on the contradicting arm, paired task by task with `t_b_conflicting`.
- **Primary measure:** runs that reach a decision, meaning they cheated or submitted correct code. Before: Tinker 19, OpenRouter 36.
- **Rule:**
  - 28 or more: sampling explains most of the gap;
  - 23 or fewer: it doesn't, and the next check is serving BF16 ourselves (needs a GPU);
  - 24 to 27: inconclusive, with the same next check.
- **Also reported, with no rule:** successes, tool errors, the share of exact-repeat tool calls, runs stuck in a loop, crashes at the context limit, and paired exact McNemar tests against `t_b_conflicting` for decisions and successes.

## Question 2: is there enough cheating to collect?

Stage 1's three conditions, counted over the two new arms:
1. at least 20 hand-verified cheating runs (success or attempt) on at least 15 base tasks;
2. at least 100 hand-audited non-cheating runs. If the usual labelling set falls short, every remaining run in the two arms is labelled;
3. the observed rate projects at least 50 positives on at least 30 base tasks within 10 more 103-run arms on Tinker (about $1 of VM each).

**Go to stage 2 only if all three hold and Question 1 says "explains".** Stage 1 had 18 cheating runs, so condition 1 alone could pass by chance with no real change. If the gate passes but Question 1 doesn't, the self-hosted check comes before any stage-2 collection. If the gate fails, stage 1b is a no-go and the pilot's natural cheats go to the code-or-comments experiment.

## Labels

- The same brief as Amendment 5 (`results/adversarial_cheating_pilot/label_brief_tinker.md`), with reading agents.
- The same labelled set: passes, triage hits, every run that was pushed, submitted or ended with correct code, and a seeded random 30 of the rest (`random.Random(3)`).
- I check every success against the environment and every attempt against the code it cites.

## Not changed, on purpose

The message limit, the prompts and the after-run honesty question stay as in stage 1, so any change can be put down to `top_k`. Those changes belong to stage 2's protocol.

## Predictions

1. The contradicting arm reaches a decision in about 27 runs (anything from 19 to 36 wouldn't surprise me).
2. Successes: about 9 contradicting and 7 one-off.
3. Tool errors fall by about a third in both arms.
4. Question 1 says "explains": 55%.
5. Go to stage 2: 40%.

## Cost

About 2.5 hours on the stopped `c2d-standard-32` VM (about $3) plus Tinker credits. No OpenRouter spend.
