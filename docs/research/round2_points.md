# Round 2: opponent-adjusted point skills

Session date: 2026-11-24. Owner: project user. Contributor: Codex research agent, AI-assisted ideation and implementation. The scientific-brainstorming workflow structured the independent candidate round and critique; the [first-round evidence register](sota_evidence.md) contains the verified tennis references and workflow attribution.

## Scope and independent candidate round

Question: can the same observed point-winning rate be interpreted more usefully when the strength of the opposing server/returner and the age of the evidence are modeled explicitly?

The agent had seen the original feature design, the first-round handoff and the parent's suggestion of opponent adjustment. Thus this round was independent of other second-round agents, but was not unanchored or blind to the first-round 2023 results. This is exploratory development; no new parameter choice in this component used 2023 performance. The 2024–2025 final test remains sealed.

| Idea | Mechanism | Assumption / falsifier | Disposition |
| --- | --- | --- | --- |
| P1: dynamic opponent-adjusted serve/return skills | Attribute point results to separate server and returner abilities | Repeated opponents identify meaningful ability differences; reject if no temporal predictive increment | Implement |
| P2: exponential decay of pooled raw rates | Avoid discontinuities at fixed lookback windows | Recent information is more relevant; may only duplicate existing one-/two-year rates | Incorporate decay in P1, retain raw rates as comparators |
| P3: score-dominance rating | A dominant win conveys more than a binary win | Game/set outcomes add signal beyond point rates; exact scoring parser/format introduces additional dependency | Defer |
| P4: point-to-game scoring model | Convert serving ability into nonlinear game-winning probabilities | IID-point assumption is useful; reject if added transformation increases future loss | Add exact service-game summaries, not a full match simulator |

Criteria declared before measuring the new component: past-only data is mandatory; lower training-period forward log loss is the primary selection criterion; identical target coverage, reproducibility, and computational cost follow. No vote or creativity score selected a purported empirical winner.

The choice of P1 addresses a concrete weakness of the existing raw rates: a 65% serve-point win rate against strong returners need not imply the same serving skill as 65% against weak returners. This is a testable modeling assumption, not a causal claim about players.

## Implemented model

`build_adjusted_point_features(targets, inventory)` in `scripts/adjusted_point_features.py` returns 34 numeric columns on the original target index. Columns ending `_diff` change sign under A/B exchange; every other column is invariant. Prefixes are `adjpoints_h365_` and `adjpoints_h730_`.

For server i against returner j on surface c:

`p_serve(i,j,c) = sigmoid(b_c + serve_skill_i - return_skill_j)`.

Overall and separate surface skill banks are maintained. Both use pre-date surface baselines estimated from earlier point totals. Separate surface skills are pooled toward overall skills with weight `effective_points / (effective_points + 400)` for each skill. The pooling is an empirical safeguard, not a fitted hierarchical posterior.

Each history match contributes two fractional-binomial observations, one for each player's service points. To limit domination by extremely long matches and acknowledge within-match dependence, each observation has at most 100 effective points, preserving its observed success proportion. This is a fixed robustness choice; the filter's precision is not a calibrated uncertainty estimate.

For each skill, retain mean m, precision q, effective point exposure n, and update date. The prior is mean 0, precision 25. Before a new date, discount the old likelihood's natural parameters by `rho = 2^(-elapsed_days / half_life)`:

`q_new = 25 + rho * (q_old - 25)`;

`m_new = rho * q_old * m_old / q_new`;

`n_new = rho * n_old`.

This discounts information and shrinks poorly supported skills toward the neutral prior. High-confidence skills decay more slowly in mean than in information. Two fixed half-lives, 365 and 730 days, supply complementary representations; their values were declared before second-round validation.

All observations sharing the same tournament-start date use the same frozen pre-date skills and surface baseline. For server and returner, accumulate the signed score residual and diagonal binomial curvature. Then apply one damped update per skill:

`q_after = q_before + sum(n_eff * p * (1-p))`;

`m_after = m_before + 0.5 * sum(signed(y_eff - n_eff*p)) / q_after`.

Server gradient sign is positive, returner gradient sign negative. The diagonal approximation ignores server–returner covariance; damping limits the joint update step. This is an approximate recursive logistic filter, not exact Bayesian inference or a reproduction of a published named algorithm.

The surface baseline retains discounted successful and total effective service points, with a fixed 2,000-point prior at 65%. It is updated after skill gradients on each date. At prediction, all states are queried through T−21 inclusive and discounted to that cutoff. The target match's technical statistics and outcome do not enter its features.

Per half-life, outputs include overall and pooled-surface serve/return skill differences; hypothetical matchup serve-point probability differences/sums; exact IID service-game probability differences/sums; effective exposure difference/sum/minimum; unseen-player count; and surface baseline. For service-point probability p and q=1−p, the game probability is

`p^4(1+4q+10q^2) + 20p^3q^3 * p^2/(p^2+q^2)`.

These are game summaries. They are not calibrated match-win probabilities and do not assume a best-of-three or best-of-five match format.

## Data controls and executed checks

Only the published eligible Tour main-draw inventory and the checksum-pinned 2006–2023 annual files are read. `development_entries`, `_annual_rows`, and paired-valid `_counts` are reused from the first-round point-feature module. The join checks IDs, date, surface and season; missing technical statistics do not remove target rows. Test-season inventory rows are filtered before other fields are inspected. Production target seasons outside 2016–2023 raise an error.

Real-data feature construction completed in approximately **2.11 seconds** on the current host, producing **18,774 × 34** finite values. It matched **44,164** eligible history records, of which **44,085** had paired valid statistics and **79** were missing or invalid. No accuracy or loss was computed in this agent's component validation.

Seven tests pass via:

```sh
.venv/bin/python -m unittest tests.test_adjusted_point_features -v
```

The tests verify:

- Equal observed serve rates receive different adjustments against historically strong versus weak returners.
- Two matches on one date reproduce an independently calculated batch gradient; reversing input order preserves outputs.
- Altering statistics inside the excluded T−20-to-T interval leaves features unchanged; target labels are ignored, and poisoned test-season source fields are not read.
- Player exchange negates only `_diff` features; unseen players produce finite neutral contrasts and explicit missing-history indicators.
- Invalid paired counts, source identity mismatches and held-out target requests are handled explicitly.
- Prior/likelihood discounting matches a hand calculation and the game formula has the correct endpoints and complement symmetry.

## Adversarial critique and handoff

Self-critique by the originating agent is not independent review. The parent was asked to assign another agent a method/code review.

Potential alternative explanations for a gain are smoother history weighting, nonlinear probability transforms, or duplicated information rather than opponent adjustment itself. Compare this block added to the frozen first-round features; retain first-round models as comparators. A later mechanism-specific ablation could separate these effects, but must not silently use 2023 to choose new parameters.

Residual risks include correlated points, diagonal rather than joint posterior updates, approximate surface pooling, field-strength effects in the learned surface baseline, incomplete identifiability beyond the zero-centered priors, and source missingness. Fixed precision, damping, caps and priors are pragmatic choices. Good synthetic mechanics do not establish real-world calibration or superiority.

The next action is training-period temporal comparison using the parent experiment's fixed selection rules, followed by documented exploratory 2023 evaluation. Preserve all attempted results and the untouched final holdout. Recommend the block only if its measured contribution justifies its complexity; a null result is informative and should remain in the report.
