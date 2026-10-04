# Six-minute presentation script

Read the paragraphs aloud. Headings and bracketed cues are for you, not the audience.
The times are cumulative checkpoints, with room for pauses. Stop at the takeaways;
the remaining slides are backup for questions.

## Title — 0:00–0:10

Hi, we’re Jeremias Ferrao and Michael Kenney. We studied a failure mode we call emergent suspiciousness: trained monitors that become overly suspicious of prompt injections, even when the underlying behavior is harmless.

[Advance.]

## 1. Our approach — 0:10–0:40

We reuse Sinha and colleagues’ training-data framework, but change the training objective. Their monitors learn to produce rationales and scores through supervised training and reinforcement learning. Ours directly distill Kimi K3’s harmfulness probabilities into a binary decision.

As one illustrative comparison, our 4B monitor achieves a mean OOD partial AUROC of point seven seven six, versus point six eight four for their method. That strong performance motivated a closer robustness check.

[Advance.]

## 2. Can injections fool the monitor? — 0:40–1:10

Ordinary monitoring scores don’t tell us whether the monitor itself can be manipulated.

Terekhov and colleagues showed that injections embedded in code comments could fool GPT-4-era monitors. The example here claims benchmark authority and tells the monitor to output zero. In one reported setting, GPT-4.1 scored all 375 injected backdoors as benign.

We wanted to see how our trained monitors responded to these attacks.

[Advance.]

## 3. Initial results — 1:10–1:35

Initially, the results looked almost too good. Both our 4B and 9B adapters detected every injected backdoor, with AUROC around point nine nine nine.

The detection threshold was calibrated to about two percent false positives on clean harmless code. But that comparison had a crucial asymmetry: only the harmful examples carried injections. Were we detecting backdoors, or simply detecting injection text?

[Advance.]

## 4. Harmless-code controls — 1:35–2:00

So we added the same injections to harmless code, without adding a backdoor.

At the same thresholds, the 4B monitor flagged every example, and the 9B monitor flagged almost every example. Clean harmless code had only about two percent alarms.

The impressive detection results were hiding near-universal suspicion. Encountering an attack was being treated as evidence of harmful behavior.

[Advance.]

## 5. LLM-as-a-judge transfer — 2:00–2:30

We also tested a different task using JudgeDeceiver: choosing the better of two answers.

Without injections, both adapters were about 99 percent accurate. When we attached an injection to the better answer, accuracy fell to roughly 58 percent at 4B and 64 percent at 9B. Attaching it to the worse answer was much less disruptive.

This suggested avoidance of injected answers beyond the code-monitoring setting. These are quality judgments, rather than harmfulness labels.

[Advance.]

## 6. Diagnostic summary — 2:30–3:10

We investigated three possible explanations.

First, the training metadata identified only four explicitly tagged injection cases out of nearly nine thousand examples. We then removed over eleven hundred rows flagged by an LLM injection detector and retrained. The APPS false alarms persisted.

Second, the teacher was sensitive to some templates, but still separated injected backdoors from harmless code much better than the student.

Third, we retrained with an instruction explicitly separating injection presence from misconduct. That also failed to resolve the alarms.

These checks suggested that training may amplify a shortcut, although they did not establish the mechanism.

[Advance.]

## 7. Auxiliary-task training — 3:10–3:45

Our first intervention added JudgeDeceiver as an auxiliary task: 75 percent monitoring examples and 25 percent answer-quality judgments, including injections on both better and worse answers.

The model learned that task extremely well, reaching almost perfect held-out accuracy. But it still flagged every injected harmless APPS example.

So the robustness was context dependent. Learning to handle injections in answer judging did not generalize to harmfulness monitoring.

[Advance.]

## 8. Direct monitoring augmentation — 3:45–4:40

We then augmented the monitoring task itself. We replaced 40 percent of training rows with versions containing synthetic injections, across user, assistant and tool roles, at different positions. We included both harmless and harmful trajectories.

The recorded behavior and cached teacher targets stayed unchanged, so this required no new teacher queries. The idea was to encourage invariance to injection text when the underlying behavior stayed the same.

At 4B, matched injected AUROC rose from point six four four to point eight five nine, and harmless flags fell from 100 percent to about 44 percent. At 9B, the improvement was much smaller: harmless flags remained around 83 percent.

Backdoor detection also decreased, especially at 4B. This was a substantial but imperfect improvement, and its effect depended on the monitor.

[Advance.]

## 9. Broader transfer — 4:40–5:20

At 4B, augmentation also improved clean ID monitoring: mean partial AUROC rose from point eight four six to point eight seven three.

We then returned to JudgeDeceiver, which this augmented adapter had never trained on. There was some transfer, but accuracy on injections attached to better answers rose only from about 58 to 63 percent. Substantial avoidance remained.

Reduced shortcut learning is one possible explanation for the broader gains. We haven’t established that mechanism, and the improvements do not amount to general immunity.

[Advance.]

## 10. Takeaways — 5:20–6:00

Our main takeaway is that training can produce surprising generalization: broad suspicion emerged despite very few explicitly tagged injection examples.

Direct augmentation helped more than instruction changes or auxiliary-task training, but its benefits varied across monitors and contexts. We used only eight authored injection strings, so greater diversity is an obvious next experiment.

We also need mechanistic analysis, more extensive ablations, and replication beyond these two Qwen sizes.

Most importantly, evaluate harmless injection controls alongside harmful attacks. Otherwise, strong detection scores can hide a monitor that raises alarms on almost everything it finds suspicious.

Thank you.

[Stop here. Backup slides are for questions.]

## If you are running late

At slide 6, say only: “Filtering training data, checking teacher behavior, and changing instructions did not explain or resolve the failure. Training-amplified shortcuts remained our working hypothesis.”

At slide 9, say only: “Clean ID performance improved, but transfer to JudgeDeceiver was modest. Avoidance remained, and the mechanism is unresolved.”

Keep the harmless-code control, augmentation result and closing takeaway: they carry the central story.
