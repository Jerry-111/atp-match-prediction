# Round 9: shared encoders of historical tournament sequences

Recorded before empirical fitting on 2026-11-27. This is the bounded sequence
branch of [the independent idea register](round9_ideas.md), proposed separately
by the features and research agents. Eight-round results and the 70% objective
were visible anchors. The scientific-brainstorming workflow distinguishes this
proposal from evidence of improvement. All 2023 measurements remain exploratory;
the unchanged 2,588 targets, threshold 0.5 and sealed 2024–2025 are mandatory.

## Mechanism, evidence and rival explanations

An encoder may distinguish recovery, persistent decline and abrupt changes
from histories with similar aggregate rates. It receives individual past event
summaries rather than only fixed-window aggregates. This does not create new
external information; it changes the representation and capacity applied to
already permitted results and technical counts.

[Bai, Kolter and Koltun (2018)](https://arxiv.org/abs/1803.01271) compare generic
sequence architectures and challenge the presumption that recurrence is always
best. [Zaheer et al., Deep Sets (2017)](https://arxiv.org/abs/1703.06114) motivate
a permutation-invariant pooling control. Neither demonstrates this tennis task.
The [Stanford CS230 tennis project](https://cs230.stanford.edu/projects_spring_2018/reports/8290687.pdf)
uses player histories, but excludes low-history players and is not a verified
full-cohort chronological comparator. We do not reproduce or claim its score.

Rivals: the neural head alone adds useful interactions; recency already explains
any benefit; sample noise and repeated player trajectories make the sequence
encoder overfit; or the most useful information occurs inside the unavailable
21-day interval. The same-input pooled encoder and identical-capacity shuffled
GRU address different alternatives. A pooled set still receives timestamps and
can theoretically infer order, so a GRU-versus-set gain is evidence about the
architecture's useful inductive bias, not proof that only GRUs can use chronology.

## Exact historical input

- Open only hash-pinned Tour annual files from 2006 through 2023, and join only
  the existing eligible completed Tour inventory. No new tournament population,
  low-level training labels, source aliases or raw outcome corrections are added.
  Existing metadata corrections apply. Validate player/date/surface identities
  and inventory-vs-raw outcome direction before consuming an eligible record.
- For each target date T and player, take the most recent **16 distinct
  historical tournament-start dates** in `[T−21−1460 days, T−21]`, inclusive.
  Aggregate every eligible match for that player on a tied date, including the
  exceptional case of more than one event ID. There is no fabricated within-week
  order, no round ordering and no source-row ordering. Tokens run oldest to
  newest, with left padding and a separate mask. The common four-year window
  bounds very old evidence without a new lookback grid.
- Each date token has 22 fields: smoothed event win rate; log match count;
  mean log own rank; mean log opponent rank; smoothed serve/return point rates;
  ace, double-fault, first-in, first-won, second-won, return-first-won and
  return-second-won rates; log serve/return point exposures; missing-statistic
  fraction; four surface fractions (Hard/Clay/Grass/Carpet); log gap since the
  previous eligible token/date; log age at the target's cutoff.
- All seven component rates use the same fixed 100-opportunity priors as
  Round5 style; total serve/return rates use 100 at .65/.35. Event win rate is
  `(wins+1)/(matches+2)`. Both players' technical records must pass the existing
  paired consistency checks; otherwise that match contributes outcome/ranks/
  surface and a missing observation, but no point counts. No target is removed.
  Ranks must be positive and finite, as required by the original inventory.
  A label-free pre-fit audit found all 44,164 permitted 2006–2023 Tour rows
  have finite positive `rank_a`/`rank_b` (range 1..2159), with no missing rank.
  Unexpected invalid ranks raise an explicit error rather than removing rows.
- Gap is measured from the previous retained date within the four-year window;
  the first retained date has gap zero. Date age and gaps are nonnegative.
  Pad values are zero with mask false. A zero-history player has an all-masked
  sequence and a zero encoded vector, retaining the same static predictors.
- Current targets supply only identities/date for querying; their own outcome,
  score and technical statistics are never read from the target frame by the
  feature builder. Past eligible inventory/raw outcomes are legitimate inputs;
  the target's own source row cannot enter its as-of history. Past
  2023 matches may enter later 2023 targets only after the same T−21 cutoff.
  Guard target seasons and actual dates; filter source years before parsing
  final-year fields. No final-year raw file is opened or downloaded.

The static input is exactly the **existing 326 Round5 features** for every
architecture. No feature subset, token count or window is selected on 2023.

## Three fixed architectures, two seeds

All use hidden dimension **16**, the same trainable token projection to 16
units with tanh, and the same pair head: a 32-unit tanh layer followed by a
scalar. The head receives both encoded player vectors and the 326 static
features. Its logit is `(head(A,B,x)−head(B,A,mirror(x)))/2`, enforcing exact
player-exchange complementarity independently of training augmentation.

| Architecture | History encoder | Purpose |
| --- | --- | --- |
| `sequence_mean` | Masked average of projected valid tokens | Same observations/timestamps without a recurrent state |
| `sequence_gru` | One shared single-layer GRU, 16 hidden units | Learns a chronological state path |
| `sequence_shuffled_gru` | The identical GRU with fixed performance/time reassignment | Matched-capacity negative control |

For the shuffled control, keep the final two temporal channels and valid-mask
positions fixed. Permute the other 20 channels together among valid positions
with the same fixed permutation for each history length, generated by seed
901+length. This breaks performance–time correspondence, not source eligibility.
It is independent of labels, identities, A/B orientation and model seed. No
shuffle is chosen by performance; lengths zero/one are unchanged. The shuffled
model receives this transformation in both training and inference.
Because the permutation is fixed for a given length, a sufficiently capable
model could learn its inverse. This control disrupts the natural presentation
and performance/time alignment; it does not information-theoretically delete
chronology. Interpret it as an inductive-bias control, with this limitation.

Seeds **158 and 2718** are fixed; average their probabilities. Never select or
report the better seed as the candidate. There are three selectable new
architectures, not six seed-selected candidates. At coordinator review, before
any empirical fit/score, each architecture additionally receives fixed R4
probability mixtures with new-model weights .25/.5/.75. Thus **3 raw + 9 fixed
mixtures + R4 = 13 eligible entries**. Mixtures add no fitted seed states;
there is no continuous weight search, calibration grid or label smoothing.
The pooling model has fewer recurrent parameters; report actual parameter
counts and use the shuffled GRU to separate recurrence capacity from timing.

## Training chronology and normalization

Forward outer folds are 2020/2021/2022. Eligible outer training rows are from
2009 onward, strictly earlier seasons and tournament dates no later than
the first outer target date minus 21 days. Final fitting uses all 33,833 eligible
2009–2022 rows; all 2,588 2023 rows remain evaluation targets.

For each outer fit and seed, use the latest eligible training calendar season
as an inner validation tail; inner training dates must be at least 21 days
before that tail's first target date. The tail never supplies gradients or
normalization statistics during epoch selection. Select the minimum inner
log-loss epoch, breaking ties by earliest epoch. Maximum 30 epochs, patience
5 after at least 8 epochs, improvement tolerance 1e-5. Then restart the same
seed from scratch and refit all outer training rows for exactly that many
epochs. The final stage repeats this procedure with a 2022 inner tail; no
2023 labels or scores determine epochs.

Static normalization fits only training rows, including mirrored counterparts:
odd columns have center zero, even columns their training mean, and scales
have a floor of 1e-6. Token normalization uses only valid tokens from both
players' training histories. Masks remain separate and transformed padding is
reset to zero. Both players use the identical frozen transform; no test-batch
normalization or BatchNorm is allowed. Normalization is refitted on the full
outer training set only during the final refit for that outer fold.

AdamW with learning rate .001, weight decay .001, batch size 512, gradient-norm
clip 1, no dropout, and BCE loss with fixed three-year recency weights.
Normalize each training population's recency weights to mean one. No learning
rate or optimizer search. Each batch evaluates both player orientations through
the antisymmetric head, without duplicating label mass.

Total: 3 architectures × 2 seeds × (3 outer folds + final) = **24 returned
seed-model fits**, each preceded by its own chronological inner epoch-selection
fit: **48 optimization stages**. Six final seed states are retained in the
bundle. Synthetic-only timing comes first; record a computational amendment
before real fitting if this budget is impractical. CPU threads ≤4. MPS may be
used only after coordinator permission to share the device; CPU inference and
serialization/reload must be checked regardless of training device.

## Selection, reporting and checks

Use guarded accuracy on 2020–2021 to select among the 13 entries,
with loss no more than R4+.005. The early choice must gain at least .005
accuracy on 2022 and satisfy that loss guard to pass this branch's gate.
Pooled 2020–2022 then selects the frozen exploratory candidate. Preserve the
gate recommendation separately from the pooled and posthoc best2023 rows.
R5 pooled/best-loss, R7 market-student and ranking references are report-only.
Freeze choices, epochs, weights, source hashes and normalizers before new 2023
scoring. The reused historical gate is not independent confirmation.

Required synthetic checks: same-date permutation and player swap; exact cutoff
and four-year boundary; poisoning target/future data; paired missing counts;
zero-history masks; train-only scaler isolation; same shuffle under exchange;
proper inner purge; padded-token invariance; known chronological pattern
recovery on synthetic data; seed-average determinism and saved-state reload.
Build all 36,421 sequences and audit dates/coverage before real fitting.
Export full OOF predictions **and all 18 outer-fold seed states**, each seed's
epochs and scores, seed-average metrics,
matched encoder contrasts, date-block descriptive intervals, model bundle,
source provenance and a concise Chinese report. An independent agent audits
features, selection and reload. A null outcome remains visible; no sample
restriction, threshold change or larger post-2023 grid can rescue this run.

Coordinator review is required before empirical fits. Implementation and
synthetic/data-construction checks are authorized by the existing task.
