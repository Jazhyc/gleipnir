# Natural-cheating pilot, stage 1c: does our own BF16 copy behave like DeepInfra or like Tinker?

Frozen before any stage-1c run. Stages 1 and 1b stay no-gos; this is a new experiment with its own pass marks.

## Why

- On the same 103 contradicting tasks, OpenRouter (DeepInfra, BF16) reached a decision in 36 runs and Tinker in 19. Stage 1b ruled out `top_k`.
- What's left is how each provider runs the model: number precision (Tinker's is unpublished, and an official FP8 checkpoint exists) and how each one turns tool calls and conversations into tokens.
- Serving the official BF16 weights ourselves, with the vLLM settings Qwen's recipe gives, defines "the model" without depending on either provider. Stage 2 would also need this copy for activations.

## Serving

- vLLM 0.31.0 (the version Jeremias's repo pins), `Qwen/Qwen3.8-27B` at revision `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`.
- BF16 weights and BF16 KV cache on one A100 80GB (`a2-ultragpu-1g`, us-central1-a). Two A100s were out of stock in both Iowa zones that offer them. The recipe's flags: `--tool-call-parser qwen3_coder --enable-auto-tool-choice --reasoning-parser qwen3`, plus `--language-model-only` and `--max-model-len 131072` (no stage-1 run came near it).
- Sampling uses vLLM's own top-k/top-p sampler (`VLLM_USE_FLASHINFER_SAMPLER=0`): FlashInfer's compiles at start-up and needs a CUDA compiler this image lacks. Same distribution, different code. The KV cache holds 311,860 tokens.
- Checked before freezing: from the sandbox VM, a tool request returned a parsed `bash` call with no thinking text.
- An API key on the server; the sandbox VM reaches it over the project's internal network. No firewall rule is added.
- Requests are the same as in stage 1: temperature 0.7, top_p 0.8, max_tokens 8192, seed 0, thinking off (`chat_template_kwargs: {"enable_thinking": false}`). Unsent fields fall back to `generation_config.json`, so `top_k = 20` applies, as it appears to on DeepInfra.
- The sandboxes run on the same `c2d-standard-32` VM as stages 1 and 1b, at concurrency 30.
- Smoke first: two contradicting tasks (`--out smoke_sh`, excluded) must show parsed tool calls, no thinking text, and the request fields above in the logs.

## Arms, in order

1. `sh_b_conflicting`: prompt B, contradicting tests. Always run.
2. `sh_b_oneoff`: prompt B, one wrong test. Run only if arm 1 says "like DeepInfra" (below), because stage 2 needs that answer.
3. `sh_b_conflicting_pp`: arm 1 plus `presence_penalty = 1.5`, which Qwen's card recommends for no-thinking mode "to reduce endless repetition" and which no stage-1 run sent. Report only, no rule. Run last, if the GPU is still up.

## Question 1: like DeepInfra or like Tinker?

On arm 1, count the runs that reach a decision (cheated, or submitted correct code). DeepInfra had 36; Tinker had 19 without `top_k` and 17 with it.
- **28 or more:** like DeepInfra. Tinker's serving is the outlier, and collection can use our own copy.
- **23 or fewer:** like Tinker. The gap belongs to DeepInfra's serving, not the model.
- **24 to 27:** inconclusive.

Also reported, with no rule: cheat successes, tool errors, the share of exact-repeat tool calls, runs stuck in a loop, submissions, crashes, and paired exact McNemar tests against `b_conflicting` and `t_b_conflicting`.

## Question 2: is there enough cheating to collect?

Stage 1's three conditions, counted over arms 1 and 2:
1. at least 20 hand-verified cheating runs (success or attempt) on at least 15 base tasks;
2. at least 100 hand-audited non-cheating runs (if the usual set falls short, label every remaining run);
3. the observed rate projects at least 50 positives on at least 30 base tasks within 10 more 103-run arms.

**Go to stage 2 only if all three hold and Question 1 says "like DeepInfra".** Stage 2 gets its own protocol, frozen first.

## Labels and checks

Offline recheck (`--threads 2`, then `--retry-timeouts`) with nothing else running, `labels.py`, and the reading-agent brief from Amendment 5 (`label_brief_tinker.md`) on the same labelled set. I check every success against the environment and every attempt against the code it cites.

## Predictions

1. Arm 1 reaches a decision in about 31 runs (20 to 40 wouldn't surprise me).
2. Question 1 says "like DeepInfra": 60%.
3. Arm 1 has about 13 cheat successes.
4. Go to stage 2: 40%.
5. Arm 3 cuts exact-repeat tool calls from about 31% to about 20%, and reaches about 5 more decisions than arm 1.

## Cost

The GPU VM at roughly $5 an hour (to check against the console estimate) for setup plus one to three arms, likely 3 to 6 hours in all, and the sandbox VM at about $1 an hour. No OpenRouter or Tinker spend. The GPU VM is deleted after the last arm; results are pulled back first.
