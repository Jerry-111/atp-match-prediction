# Round 5: compact technical composition and matchup features

Status: bounded, fixed feature implementation; no new 2023 model score was
observed to select its formulas or constants. F3 was independently proposed
alongside cross-level history (F1) and retirement/load (F2). The parent selected
F1 first and authorized this complementary branch before scoring. F2 remains
unimplemented; indoor/court-speed inference and external environment collection
are also deferred. See [the independent proposal record](round5_multilevel.md).

## Hypothesis and disconfirmation

Players can have similar overall service-point success but different ACE,
double-fault, first-serve and second-serve profiles. Those components may
interact with the opponent's return components, height or handedness. The prior
features contain total serve/return rates and opponent-adjusted total point
skills, but not these detailed opportunity rates. This is additional measured
information with a small set of fixed interactions, not an assumption that a
larger feature count improves predictions.

The direction is falsified as useful for this project if matched forward-fold
comparisons show no stable gain over the existing strength/point/age model,
especially if differences disappear across regularization settings or years.
The main rival explanation is that components simply recover already-known
overall skill; raw rates also reflect opponent composition. These component
rates are not themselves opponent-adjusted latent skills.

A bounded primary-source check found contextual evidence for handedness, but
also a failed surface-type hypothesis in [Breznik (2013)](https://www.jssm.org/jssm-12-346.xml-Fulltext).
That observational result does not establish incremental predictive value here.
The feature builder does not recreate the paper's model, assume universal
left-handed advantage, or force a coefficient's sign. The skill workflow's
independent generation/evidence separation is a process aid, not validation.

## Available information and temporal policy

`build_style_features(targets, inventory)` accepts nonempty 2009–2023 targets
and preserves their order/index. The existing source manifest is filtered to
2006–2023 before loading raw files. Only eligible Tour history, joined by full
match ID, season, date, surface and unordered pair of player IDs, contributes.
Inventory labels and target labels are never read by the calculation. The
winner/loser-named raw columns identify the owner of observed counts, not a
predictive label.

**Every technical aggregate uses T−364 through T−21 inclusive.** Raw counts
after the latest requested cutoff are never parsed. When earlier and later
targets are built together, historical cumulative sums are queried at each
target's own cutoff. This includes excluding an earlier target's own statistics
from its features, even though those statistics can legitimately become
history for a sufficiently later target.

Height and handedness also use only historical observations through T−21,
which is stricter than using attributes supplied in the current target row.
The latest unambiguous earlier value is retained. Same-date conflicting values
do not choose a winner-dependent row: the previous known value is preserved,
or the field remains missing. Height must be finite and between 130 and 230 cm;
only L/R are recognized hands. Missing height uses a fixed 185 cm reference;
missing hand is neither L nor R. Both have explicit missing indicators.

The full development build found no conflicting same-date height/hand fields.
Nevertheless these are provider snapshots, not verified measurements, and a
historical height can be stale for a growing player. Future snapshots never
repair earlier missingness. No same-tournament result, indoor field, target
technical statistic or final-test outcome is required.

## Seven opportunity-based rates

Each posterior rate is `(historical successes + 100*prior) /
(historical opportunities + 100)`. These are fixed pragmatic smoothing choices,
not optima estimated from validation outcomes.

| Rate | Numerator | Denominator | Fixed prior |
| --- | --- | --- | ---: |
| ACE | ACE count | service points | 0.06 |
| Double fault | double faults | service points minus first serves in | 0.08 |
| First serve in | first serves in | service points | 0.60 |
| First serve won | first-serve points won | first serves in | 0.70 |
| Second serve won | second-serve points won | service points minus first serves in | 0.50 |
| Return first won | opponent first serves in minus opponent first-serve wins | opponent first serves in | 0.30 |
| Return second won | opponent second-serve attempts minus opponent second-serve wins | opponent second-serve attempts | 0.50 |

Second-serve attempts include double faults: the returner wins those points.
Counts must be finite nonnegative integers, service points positive, successes
within opportunities, ACEs no more than total service wins, and double faults
plus second-serve wins no more than second-serve attempts. Both sides must pass;
otherwise the match contributes one missing-statistics observation per player
and no point counts. This prevents selectively retaining only one side's
statistics. Attributes remain independently usable from that historical match.

## Fixed thirty-column schema

The [builder](../../scripts/style_features.py) exports:

- Seven smoothed rates × A-minus-B difference and sum: 14 columns.
- Log point exposure difference/sum and log missing-statistics match-count
  difference/sum: 4 columns. Exposure is own service plus return opportunities.
- Height difference/mean and height-missing difference/sum: 4 columns.
- Left-hand difference, known opposite-hand matchup flag and hand-missing
  difference/sum: 4 columns.
- Four odd interactions: height-centered ACE profile difference, own-left-hand
  × opponent return profile contrast, and first-/second-serve × opponent
  return-failure contrasts: 4 columns.

The first-/second-serve interactions have the form
`serve_A*(1-return_B) - serve_B*(1-return_A)`. They are simple descriptive
products, not calibrated point probabilities or a physical independence model.
The left-hand interaction is `left_A*return_B - left_B*return_A`, where the
return profile equally averages the two smoothed component rates. The height
interaction centers on 185 cm and divides by 10 cm. All four are fixed before
model evaluation; their predictive direction belongs to the fitted model.

Every name ending `_diff` changes sign under player exchange; every other
numeric output is invariant. DataFrame attributes list both sets, all priors,
historical coverage, source manifests and temporal policy. No player ID is
encoded as a numeric feature. New players obtain finite neutral differences
and explicit missing/context values, so no target is dropped.

## Verification and limits

Run `.venv/bin/python -m unittest tests.test_style_features -v`.
Six tests cover hand-computed opportunities and smoothing; inclusive endpoints;
target/future statistics and attribute poisoning; earlier targets built beside
later targets; same-date order/conflict resolution; impossible counts and
paired rejection; A/B exchange; neutral unseen players; held-out sentinel
fields; target-season locking; identity mismatches and duplicates.

Full development benchmark: **36,421 × 30 finite features in 1.890 seconds**,
including CSV input, in the local Python 3.12 environment. Through the final
target cutoff, all **43,998** eligible Tour matches joined; **43,919** had valid
paired detailed counts and **79** were retained only as missing-statistics
observations. Same-date attribute conflicts: **0**. This is implementation
evidence, not accuracy or probability-quality evidence.

These are all-surface 364-day rates, intentionally keeping the branch compact.
They do not capture shot placement, spin, opponent-specific tactics, current
injury, match-day conditions or the three weeks hidden by the lag. Collinearity
with existing point features and the need to regularize small samples remain.
The parent's learner comparison must retain harmful and null results and the
same 2,588 validation rows. Neither this implementation nor reused 2023 scores
can establish 70% accuracy, SOTA, or final-test superiority.
