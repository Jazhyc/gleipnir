# Adversarial experiments

Stress tests of released monitors: attacks, cheating agents, reward hacking. Add new experiments to this table.

| Experiment | Question |
| --- | --- |
| [`adversarial_a1_discrimination`](adversarial_a1_discrimination/) | A1 follow-up: discrimination beneath injection alarms |
| [`adversarial_a2_cover_comments`](adversarial_a2_cover_comments/) | A2: natural cover comments |
| [`adversarial_a2_high_effort`](adversarial_a2_high_effort/) | A2 stronger-writer comparison |
| [`adversarial_a2_tinker_probe`](adversarial_a2_tinker_probe/) | Tinker connectivity probe |
| [`adversarial_a3_random_search`](adversarial_a3_random_search/) | A3: score-feedback random search pilot |
| [`adversarial_a4_feedback`](adversarial_a4_feedback/) | A4: LLM comments with score feedback |
| [`adversarial_a4_scale`](adversarial_a4_scale/) | A4 at scale |
| [`adversarial_a6_transfer`](adversarial_a6_transfer/) | A6 pilot: frozen A4 comments transferred from 4B to 9B |
| [`adversarial_baseline_4b`](adversarial_baseline_4b/) | Adversarial baseline: unattacked Gleipnir 4B scores |
| [`adversarial_cheating_pilot`](adversarial_cheating_pilot/) | Natural-cheating pilot, stage 1: does Qwen3.8-27B cheat with thinking off? |
| [`adversarial_code_or_comments`](adversarial_code_or_comments/) | Code or comments: does Gleipnir score what the code does, or what the comments claim? |
| [`adversarial_confession_strip`](adversarial_confession_strip/) | Confession strip: does Gleipnir catch the code or the confession? |
| [`adversarial_crossed_text`](adversarial_crossed_text/) | Identical text on honest and backdoored code |
| [`adversarial_crossed_text_probe`](adversarial_crossed_text_probe/) | Linear decoding of backdoor status under identical text |
| [`adversarial_evilgenie`](adversarial_evilgenie/) | Does Gleipnir catch other models' reward hacking? (EvilGenie × Escalation) |
| [`adversarial_goodfire`](adversarial_goodfire/) | Gleipnir on Goodfire reward-hacking rollouts |
| [`adversarial_is_it_4b`](adversarial_is_it_4b/) | Is the framing effect a 4B thing? (steps 0–3) |
| [`adversarial_lab_liars`](adversarial_lab_liars/) | Multi-agent liars against three monitors |
| [`adversarial_lab_liars_probe`](adversarial_lab_liars_probe/) | A linear probe for lab liars on Gleipnir's own activations |
| [`adversarial_lab_v03`](adversarial_lab_v03/) | Six-worker lab against Gleipnir 4B v0.3 |
| [`adversarial_monitor_aware`](adversarial_monitor_aware/) | Monitor-aware cheating on ImpossibleBench |
| [`adversarial_natural_cheats`](adversarial_natural_cheats/) | Does Gleipnir catch natural test cheating by an agent with reasoning off? |
| [`adversarial_peer_deception_prompt`](adversarial_peer_deception_prompt/) | Does explicitly naming peer deception change the 4B decision? |
| [`adversarial_phase1_v03_replay`](adversarial_phase1_v03_replay/) | Phase-1 fixed-attack replay on Gleipnir 4B v0.3 |
| [`adversarial_quiet_cheats`](adversarial_quiet_cheats/) | Quiet cheats: does Gleipnir catch the cheat, or the confession? |
| [`evilgenie_api_baselines`](evilgenie_api_baselines/) | API baselines on EvilGenie Escalation |
| [`evilgenie_assistant_messages`](evilgenie_assistant_messages/) | EvilGenie with the agent's assistant messages |
| [`evilgenie_released_stack`](evilgenie_released_stack/) | EvilGenie on the released serving stack |
