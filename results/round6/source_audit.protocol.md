# Round 6 independent source-winner audit protocol

Recorded 2026-11-27 **before reading the external winner column for this audit**.
The source headers, manifest, previous outcome-blind linkage implementation,
and previously published model results were already visible. The 2023 ATP
labels and model scores were previously examined in earlier rounds; this is
an exploratory quality audit, not a new blinded validation experiment.

## Fixed question and scope

Do the already-linked external source records identify the same winner as the
unchanged ATP target labels? The audit tests source agreement and orientation;
it neither improves a predictive model nor proves either source correct.

- Inputs: pinned local Valuebetennis annual files for **2021, 2022, 2023 only**;
  the exact Round 5 `linked_matches.csv`; and existing expanded development
  targets. No downloads or access to 2024/2025 source files.
- Verify annual SHA-256 hashes and row counts against the market manifest.
  Verify linkage SHA-256 against the completed Round 5 market artifact audit.
  Preserve each target row, ID, season, existing label and match assignment.
- The permitted new field is `vainqueur_id`. Raw score, duration and odds are
  unnecessary and are not read by this audit. Winner information must never
  enter feature construction, source matching, model fitting or prediction.

## Rules fixed before the check

1. Process **2021, then 2022, then 2023**, using exactly the same function and
   rules. Finish each earlier year's summary before reading the next year's
   winner values. Do not change rules in response to either year's agreement.
2. Use only matches whose saved linkage status is `matched`; unmatched targets
   remain explicitly unaudited and retain their original rows. Do not run a
   new matcher, search for a winner-compatible match or use odds availability
   as a filter.
3. Check target/source IDs are unique, each accepted source match is used once,
   and the linked target pair equals the two source player IDs. Check source
   event ID and UTC date agree with the frozen metadata. Reject final-year or
   invalid dates rather than inferring a repair.
4. Interpret source winner only if it is exactly one of that source record's
   two players. Orient it using the saved source player-A ID: source A-win is
   one if and only if winner ID equals saved player-A ID. Compare that value
   with the unchanged binary target A-win label.
5. Report full target counts, matched/unmatched counts, valid winner counts,
   source player-one/player-two winner counts, missing/invalid winners,
   structural conflicts and winner disagreements, separately by season.
   Agreement percentages use valid, structurally consistent matched records;
   coverage always uses the full eligible target count.
6. Any disagreement or structural conflict becomes an explicit investigation
   row. Resolve a real disagreement only against a documented official event
   result, using the already-fixed identities and date. No majority vote,
   model probability or prediction-score improvement decides the label.
7. **Never modify target labels, linkage, model artifacts, cohort eligibility
   or reported model scores.** Even an official resolution is an audit finding
   for the coordinator, not automatic authorization to change training data.

## Checks and interpretation

Synthetic tests will check both A/B orientations, missing and unrelated winner
IDs, identity/date/season conflicts, duplicated source usage, unchanged row
order and the final-year guard. A winner change must alter only the audit
finding, never select a different source record.

If source agreement is nearly complete, stop the label-noise explanation for
the observed 70% gap rather than inventing a corrective feature. Agreement
cannot exclude a shared upstream error, and unmatched records remain
unverified. No statistical significance, SOTA, accuracy improvement or new
independent test claim follows from this audit.

Source provenance is inherited from [the market manifest](../../data/market-manifest.json)
and [outcome-blind matching record](../../docs/research/round5_market.md).
