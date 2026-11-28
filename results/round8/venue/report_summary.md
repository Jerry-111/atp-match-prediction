# Round 8: historical tournament environment × style

The full 2,588-match 2023 cohort and threshold 0.5 are unchanged. This is
historical T−21 input, with no target-season statistics contributing to its
environment. All 2023 comparisons remain exploratory; 2024–2025 are sealed.

Branch-local frozen choice **previous_probability_recommendation**: 1,696/2,588 (65.5332%); log loss 0.609407.
The 2022 gate **retain_round4** had accuracy change
+0.3560%; required +0.5 percentage
points and log-loss increase ≤0.005. The pooled pre-2023 choice
**venue_strength_linear** scored 1,703/2,588 (65.8037%); log loss 0.607656.
Highest observed 2023 row **previous_round5_exploration**: 1,715/2,588 (66.2674%); log loss 0.609871. This observation does
not replace the frozen selection. Reaching 70% requires 1,812 correct.

Ten fixed fits compare base, surface context, raw venue context, adjusted
venue context, and a generic rank/Elo interaction control with the same linear
and spline learners. No mixture, threshold, prior, lookback or C search.
The original 326 fields are unchanged; 50 additional fields include 40 odd
interactions and ten even context/audit fields. Even audit fields are not
model inputs. Source rows have 41,498 valid serve
and 41,498 valid ace observations. In 2023,
256 rows lack prior venue serve
observations and 256 lack ace observations;
all remain in the cohort. Zero environment features are not an exact R4
probability fallback after refitting.

## Full-cohort results

| model | accuracy | mean_log_loss | recommended |
| --- | --- | --- | --- |
| previous_round5_exploration | 0.662674 | 0.609871 | False |
| venue_surface_spline | 0.659969 | 0.606981 | False |
| venue_adjusted_spline | 0.659969 | 0.607031 | False |
| venue_surface_linear | 0.659583 | 0.606928 | False |
| venue_base_spline | 0.659196 | 0.606453 | False |
| previous_round5_best_loss | 0.659196 | 0.606453 | False |
| venue_base_linear | 0.658810 | 0.606385 | False |
| venue_raw_spline | 0.658037 | 0.607600 | False |
| venue_strength_linear | 0.658037 | 0.607656 | False |
| venue_adjusted_linear | 0.657651 | 0.607041 | False |
| venue_strength_spline | 0.657651 | 0.607751 | False |
| venue_raw_linear | 0.656105 | 0.607527 | False |
| previous_probability_recommendation | 0.655332 | 0.609407 | True |
| rank_reference | 0.633694 | 0.629796 | False |

## Matched mechanism comparisons

Differences are candidate minus control; positive accuracy and negative loss
favor the candidate. Controls isolate surface versus tournament context and
opponent adjustment rather than crediting every change to added venue data.

| candidate | control | oof_accuracy_change | oof_loss_change | validation_accuracy_change | validation_loss_change |
| --- | --- | --- | --- | --- | --- |
| venue_surface_linear | venue_base_linear | 0.002121 | -0.000061 | 0.000773 | 0.000543 |
| venue_raw_linear | venue_surface_linear | -0.000163 | 0.000203 | -0.003478 | 0.000598 |
| venue_adjusted_linear | venue_surface_linear | -0.000653 | 0.000294 | -0.001932 | 0.000113 |
| venue_adjusted_linear | venue_raw_linear | -0.000489 | 0.000091 | 0.001546 | -0.000486 |
| venue_adjusted_linear | venue_strength_linear | -0.002121 | 0.000567 | -0.000386 | -0.000615 |
| venue_surface_spline | venue_base_spline | -0.002447 | -0.000015 | 0.000773 | 0.000528 |
| venue_raw_spline | venue_surface_spline | 0.004078 | 0.000182 | -0.001932 | 0.000619 |
| venue_adjusted_spline | venue_surface_spline | 0.001468 | 0.000316 | 0.000000 | 0.000051 |
| venue_adjusted_spline | venue_raw_spline | -0.002610 | 0.000134 | 0.001932 | -0.000569 |
| venue_adjusted_spline | venue_strength_spline | -0.001468 | 0.000586 | 0.002318 | -0.000719 |

Date-block paired intervals in summary.json are exploratory, without a
multiple-search correction. Repeated use of the 2022 gate is not independent
confirmation. ID/name/surface identifies a conservative tournament proxy,
not a verified physical venue: same-city moves, balls, weather, altitude and
player participation can still confound interpretation. Opponent adjustment
is an as-of adaptation of Sackmann's idea, not an exact paper replication.

[Registered protocol](../../../docs/research/round8_venue.md)
