# Team handoff

**Next task:** Implement the two probability models and compare them on 2023 validation. The issue carries the claim, working branch and acceptance criteria as work progresses. [Pick up issue #1](https://github.com/Jerry-111/atp-match-prediction/issues/1)

## What can be reproduced on main

| Evidence | Where to look |
| --- | --- |
| The loader retains 26,372 eligible matches across 11 seasons. The higher-ranked player won **1,640 / 2,588** validation matches in 2023 (**63.37%**). | Run `python3 scripts/baseline.py` from the repo root. [Loader and ranking rule](scripts/baseline.py) |
| Training figures show how often the favorite wins as the rank gap grows, and how much prior surface history is available. With the adopted window, 77.5% of grass-court player appearances have fewer than five eligible prior matches, compared with 29.4% on clay and 13.8% on hard courts. | [Executed notebook, section 2](analysis.ipynb) |

The notebook currently measures history **counts**. The model task adds history **wins**, an outcome-independent A/B ordering, and the two fitted regressions. The effect of adding surface history is an open question. The test set has **5,088 eligible matches** across 2024–2025. Its final model scores, the result-based literature discussion, submission HTML and video are pending.

## What the next agent should deliver

Fit both models on the same 2016–2022 training matches. Evaluate both on the same 2023 validation matches. Return a table with the ranking rule, the rank-only logistic regression and the regression that adds same-surface win rate. Report accuracy and probability-model log loss. Walk through one match's A/B assignment, historical wins, match counts and cutoff date. Link the notebook, the rerun command and the frozen settings for the teammate doing final evaluation. [Task and acceptance criteria](https://github.com/Jerry-111/atp-match-prediction/issues/1)

## Research ideas and decisions

**Adopted question.** Does same-surface history improve prediction once world rankings are known? Use the same logistic regression and the same sample in the two fits so the extra feature is the change being measured. The current lag, smoothing and time splits are defined in the [research plan](PROJECT-PLAN.zh.md).

**Ideas to revisit.** These are candidates for a later decision. Each has a concrete first check.

| Idea | Why it might matter | Smallest useful check |
| --- | --- | --- |
| Look at close rankings | Rank alone gives less separation for players who are close together. | Compare the two frozen models on validation matches with a rank gap of at most 10. Include the match count. |
| Try a longer history on grass | The training figure shows many players have little grass-court history. | Compare coverage under 52 and 104 weeks on seasons with adequate warm-up. Check the change on validation before widening the main experiment. |
| Add recent overall form | A player's current level could explain part of what the surface feature captures. | Add one all-surface historical win-rate difference to a validation comparison, using the same cutoff. |
| Compare with Elo | Past opponents' strength can affect a player's raw win rate. | Use the cited papers to explain this limitation first. A separate validation benchmark would require a chronological rating history. |

**Decision history**

| Date | Change and reason | Consequence |
| --- | --- | --- |
| 2026-11-23 | The initial sketch included recent form, surface history and close-ranking subgroups. The team narrowed the main test to the incremental value of same-surface history so one feature comparison can answer the presentation's opening question. | Initial scope and ranking benchmark landed in [11d91ab](https://github.com/Jerry-111/atp-match-prediction/commit/11d91abad3d155f10cad6b7cadb14a65d5b5d2cf). The model comparison is the next handoff. |
