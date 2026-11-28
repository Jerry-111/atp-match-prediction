# Round 2: uncertainty-aware dynamic ratings

Recorded 2026-11-24, America/Los_Angeles. Owner: project user; implementation: Codex feature agent. This is an AI-assisted research proposal and implementation record, not a performance claim. The user requested another multi-agent optimization round after seeing Round 1. Its 2023 findings are already known, so subsequent 2023 comparisons are exploratory. No additional 2023 metrics were inspected to choose this feature design. The final 2024–2025 outcomes remain excluded.

## Independent proposal and alternatives

The independently proposed mechanism is **uncertainty-aware evolving strength**: equal rating gaps need not have equal reliability when one player is new or has been inactive. A fixed Elo update cannot directly represent this difference. The prediction is that dynamic uncertainty and confidence-adjusted skill can complement the existing ratings, form and point-statistic features in future-year log loss. This prediction is untested at implementation time.

Alternatives considered were an exponentially decayed opponent graph, a joint hierarchical global/surface state-space model, and learned per-player volatility. The graph would require repeated fitting and stronger computational choices; a joint state model would require covariance approximations; learned volatility introduces another small-sample optimization problem. The selected bounded implementation uses original Glicko updates with calendar-time uncertainty and a separately identified surface-pooling heuristic. It is not Glicko-2, and it is not a claim of novelty.

Criteria fixed before measurement: chronology and exchangeability are hard requirements; other criteria are complementary information, compute under 60 seconds, reproducible fixed settings and a falsifiable increment over the existing model. No arbitrary numerical creativity score was used.

## Located primary-method evidence

Glickman's original method represents skill with a rating and rating deviation (RD), uses opponents' RD in period updates, and increases uncertainty between periods. Updates within a period use the previous period's states. Its reference example gives approximately rating 1464 and RD 151.4 after three specified games; a unit test reproduces this calculation. [Glickman, *The Glicko system*, equations and example](https://www.glicko.net/glicko/glicko.pdf).

Glicko-2 additionally models volatility and requires an iterative volatility update. That mechanism is deliberately not implemented here. [Glickman, *Example of the Glicko-2 system*, 22 March 2022](https://www.glicko.net/glicko/glicko2.pdf).

Both author PDFs were inspected on 2026-11-24. This bounded methods check establishes formula provenance, not predictive superiority for tennis. Search scope was the two primary method documents; no new tennis-performance benchmark or exhaustive literature search was conducted for this component.

## Prespecified adaptations and assumptions

- Begin every overall and surface rating at 1500 with RD 350. Use original Glicko update equations, with no RD floor or estimated player volatility.
- Fix calendar diffusion at 12 rating points per square root week: before a later observation, replace RD with `min(350, sqrt(RD² + 12² × elapsed_days/7))`. This pragmatic value was chosen before Round 2 results, not optimized using 2023. A year of inactivity adds approximately 86.5 points of uncertainty in quadrature.
- Treat all eligible matches sharing a tournament start date as simultaneous. Sum their gradient and information contributions from unchanged pre-date states. Input row order cannot invent chronology within a tournament.
- Maintain separate overall and same-surface systems using the same completed, ranked Tour/major history since 2006. Their covariance is not modeled.
- For a target at T, select snapshots containing only results through T−21, inclusive, then propagate uncertainty to T using calendar time alone. Later results never enter that propagation.
- The symmetric confidence-adjusted skill difference uses both players' RD: multiply the rating gap by `g(sqrt(RD_A² + RD_B²))`. This is a forecast feature approximation; the actual original-Glicko update retains its opponent-specific attenuation.
- The surface blend weights surface precision above its initial prior against overall precision. With no recorded surface evidence, the blended rating equals the overall rating exactly. Overlapping histories make this a heuristic, not an independent-evidence Bayesian posterior. The confidence-adjusted blend uses conservative maximum RD, and no falsely precise fused posterior variance is exported.

## Features and interface

`scripts/dynamic_features.py` exposes `build_dynamic_features(targets, inventory)`. It returns 17 finite numeric columns in exactly the input target index and order, without using target outcomes. All column names ending `_diff` are odd under A/B exchange; all remaining columns are symmetric context.

The columns comprise overall and surface rating differences, confidence-adjusted differences, RD differences/sums and rating means; a global/surface blend; surface blending weights and specialization; and 90/364-day rating changes. The adjusted differences remain in rating units. A downstream raw probability can apply `sigmoid(log(10)/400 × adjusted_diff)`; any calibration or model combination must be selected within training-period chronological folds.

The builder filters history to Tour source and 2006–2023 seasons before converting any outcome. Development targets outside 2016–2023 are rejected. Duplicate historical IDs are rejected. Unsupported surfaces and nonbinary historical outcomes raise errors. All existing target rows are retained, including unseen players.

## Adversarial review and falsifiers

1. **Opponent-dependent uncertainty may be miscalibrated.** The binary approximation ignores dependence among matches and heterogeneous ability changes. An apparent gain could be training-fold noise. Require the parent experiment's chronological comparison; do not tune this design after inspecting 2023.
2. **Surface pooling can overcount evidence.** The overall system already sees surface matches. Exporting a pooled mean is a heuristic and should earn its place through ablation; it must not be sold as exact Bayesian inference.
3. **Independent surface scales can drift.** Glicko updates need not sum to zero, and surface player populations differ. The blend could be harmful for specialists or newcomers. Compare raw overall, same-surface and blended information on training folds; retain negative results.
4. **Dates are tournament-week proxies.** Same-date batching avoids fabricated match ordering but cannot recover exact result availability. Retain the established T−21 safeguard.
5. **Round 1 anchoring is unavoidable.** Existing validation results were visible before Round 2 authorization. Any later 2023 improvement is an exploratory check, not a pristine confirmatory test or proof of SOTA.

Decision: advance this single coherent 17-feature family to the parent's prespecified Round 2 comparison. Do not alter the diffusion, pooling or feature schema in response to 2023 scores. Revisit only with training-period evidence or after explicitly documenting another exploratory round.

## Verification

Run `.venv/bin/python -m unittest tests.test_dynamic_features -v`.

Six tests pass: the original author worked example; uncertainty growth during inactivity; exact simultaneous-period update and input order invariance; poisoning all results newer than T−21 and all target outcomes; ignored 2024–2025 sentinel outcomes; A/B sign symmetry and exact unseen-surface fallback; finite unseen-player predictions; duplicate rejection and final-holdout locking.

Full-table verification generates 18,774 × 17 finite features in **0.784 seconds**, including reading the two published CSVs in the local Python 3.12 environment. This is well below the 60-second budget. These checks validate implementation properties; they do not demonstrate predictive improvement.

The existing project's [brainstorming and evidence register](sota_evidence.md) documents the Scientific Agent Skills workflow attribution. This component uses its independent proposal, assumptions, explicit criteria and adversarial-review pattern.
