# Round 8: learn bounded corrections to a prematch market prior

Recorded2026-11-27 before any branch fit or new2023 score. This is P8-1, a
later-prematch information branch. The previous market branch fitted only a
small set of price/prior logits. Here a fixed price logit is an offset and the
learner estimates residual structure from individual historical features.
2023 remains exploratory; the complete2588 labels and threshold.5 are fixed.
2024–2025 remain sealed. This is not an improvement under pure T−21 inputs.

## Source and quote conditions

Reuse the independently audited R5 Valuebetennis market features and fixed
identity links. No new raw source or mapping is introduced. Compare opening
and latest-available-prematch conditions separately. Latest means closing
where valid, otherwise opening. When neither valid quote is available, return
exactly the frozen R4 probability. The opening condition cannot use closing
or movement inputs. The latest condition includes opening-to-closing movement
and quote-availability indicators. Provider-described prematch timing lacks
per-quote capture timestamps, as already documented in round5_market.md.

Post-fit interpretation clarification (no numerical or protocol-menu change):
"Opening" identifies the price field, not a verified prediction decision time.
For event-input variants, a player's earlier-round result is known before the
target match, but is not established to precede capture of that match's opening
quote. Consequently these are later-prematch predictions using opening prices,
not validated forecasts made at the instant those opening prices were posted.

In2021 every closing quote is missing. Consequently the latest-available
training prior in2021 is opening, whereas2022/2023 contain closing quotes;
this distribution shift is an explicit limitation, not evidence of a
constant quote-capture time. Both quote conditions keep all evaluation rows.

## Fixed settings and controls

For each of two quote conditions fit nine settings: three information groups
(price context only; plus326 old historical features and R4 OOF logit; plus
27 own-earlier-round fields) crossed with offset ridge logistic C=.003/.03
and offset CatBoost depth3/300 iterations, learning_rate.03, L2=50. Exactly
18 fitted settings overall. No extra depth/iteration/seed search is added.
The price-only group contains quote logit, overround, absolute quote logit,
surface and known best-of-five; latest additionally has movement/both/closing
indicators. No target ID, name, date, winner or score is a numeric predictor.

All learners use identical available-price training rows in the relevant
condition, uniform weights. Logistic uses only odd difference inputs, fitted
training RMS, no intercept, and minimizes sum Bernoulli negative log likelihood
with the price logit as fixed offset plus ||beta||²/(2C). CatBoost receives
mirrored inputs/labels/offsets with half weights per orientation, preserving
original total training weight. Its raw residual is explicitly antisymmetrized
before adding to the offset. Inference probabilities are expit(offset+alpha
residual), alpha fixed in{.25,.5,1}. Each condition has27 fixed output controls
plus its raw price prior. Different residual scales are not additional fits.

Earlier-round fields are computed exactly as in the audited R7 later-information
branch. They are legitimate only for that target's own strictly earlier
knockout rounds. Price-only and old-history groups do not use those fields.
Final missing-price probabilities, including deterministic residual controls,
return R4 exactly and never remove a match.

## Forward fitting and decisions

Use2021H2 then2022, with the same21-day purge as R5 and an additional check
that the linked actual market match UTC is before the fitting cutoff.
Fit only eligible2021–22 outcomes. Earlier R4 forecasts are stored true OOF
predictions; the previous R4 configuration itself has been chosen with reused
folds, so these are not newly independent tests. Final fits use both training
years before2023. Coefficients never update on2023 labels.

For each quote condition separately choose early accuracy, lower log loss,
then stable name, subject to raw-price loss+.005. Require a2022 improvement
of at least.005 accuracy and the same loss guard to adopt the residual method;
otherwise retain the raw prior in that condition. A pooled choice is saved
before2023 scoring and remains an exploratory comparator if the gate fails.
Do not pick across quote conditions using2023. Report all28 outputs/condition,
matched input contrasts, proper shared metrics and conditional paired intervals.
These practical gates have been reused and are not significance tests.

## Method evidence and checks

The approach is a regularized Bernoulli offset model and additive boosting
adaptation, not an attributed tennis paper's reproduced result. The official
CatBoost API supports initial RawFormulaVal baselines:
https://catboost.ai/docs/en/concepts/input-data_baseline and
https://catboost.ai/docs/en/concepts/python-reference_pool_set_baseline .
Primary market literature and its strong raw-market comparators motivate
retaining an unmodified quote reference, not expecting arbitrary correction
to outperform it. Three learner/group controls distinguish added information
from mere extra capacity, though they do not identify a causal effect.

Verify analytic offset-logistic gradients on synthetic inputs; independently
verify CatBoost training/prediction baseline semantics; A/B complementation,
ignored-column/current-result poisoning, missing-price exact fallback,
chronological selection and frozen serialization must pass before reporting.
