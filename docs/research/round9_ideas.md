# Round 9 exploration queue — no new fits or choices yet

Recorded 2026-11-27 after Round8. The persistent70% goal is active. These are
candidate mechanisms for the next bounded experiments, **not a frozen training
protocol**, successful models, or permission to open2024–2025. Keep the same
2,588 targets and0.5 threshold. All future2023 scores remain exploratory.

The scientific-brainstorming workflow remains in use. Two AI agents proposed
directions separately before seeing each other's new proposals. Both knew the
eight-round evidence and70% goal. The research agent also received explicit
coordinator prompts about RealMLP, sequence models and residual distillation;
its ideas are not unprompted independent confirmation. No human expert panel
or subjective vote establishes validity. The coordinator prioritizes temporal
validity, a falsifiable mechanism and limited compute over paper headline scores.

| Candidate / origin | What would actually change | Rival explanation and minimum discriminating check | Disposition |
| --- | --- | --- | --- |
| R9-A / research, coordinator prompt | Full-data supervised tabular neural model; R7's foundation model only used a small context and was not fine-tuned | Same326 input columns and33,833 targets; compare a pre-tuned RealMLP variant against a matched-width plain MLP. Hold seeds/data fixed; no best-seed selection. A generic tabular benchmark does not establish temporal generalization | First feasibility check: synthetic runtime/memory, exact implementation/license, explicit chronological inner validation |
| R9-B / research and features separately | Shared player-history encoder uses order of the last16–32 eligible matches instead of only aggregate histories | Compare identical input histories with masked mean pooling and fixed shuffled-order control. If order adds nothing or only increases probability error, reject the sequence claim. Keep zero-history players with masks | Next information-representation experiment; must reconstruct per-player as-of records before modeling |
| F9-1 / features | Low-rank antisymmetric matchup residual models nontransitive style interactions | Existing sum/difference columns already recover individual style levels; this is a new interaction structure, not new information. Compare additive baseline, fixed low-rank residual and matched regularization without expanding a large grid | Small secondary alternative, downgraded by tennis-specific contrary evidence |
| R9-C / research, coordinator prompt | Historical market logit residual as privileged training supervision | R7 already tests the general soft-teacher idea andR8 uses historical market state. Honest prior-OOF probabilities are essential; fitted-on-own-label R4 scores cannot masquerade as OOF. Without new allowed pre2021 teacher coverage, this may repeat existing tests | Defer pending a distinct mechanism or source coverage |

## Primary evidence and contrary evidence

- Holzmüller, Grinsztajn and Steinwart, NeurIPS2024,
  [Better by default](https://proceedings.neurips.cc/paper_files/paper/2024/hash/2ee1c87245956e3eaa71aaba5f5753eb-Abstract.html).
  RealMLP's strong defaults motivate a small supervised-neural test; the paper's
  ordinary random splits do not answer our forward-time question. The
  [official standalone implementation](https://github.com/dholzmueller/realmlp-td-s_standalone)
  is specifically TD-S, not the fullTD system. If used, preserve this distinction
  and pin the source revision/license. Override random split defaults with
  chronological, purged inner validation. No benchmark claim implies70% tennis accuracy.
- Chen and Joachims, WSDM2016,
  [Modeling Intransitivity in Matchup and Comparison Data](https://www.cs.cornell.edu/~tj/publications/chen_joachims_16a.pdf),
  DOI10.1145/2835776.2835787. Their antisymmetric interaction construction supports
  a matchup mechanism. The ATP experiment gives only small likelihood benefit
  and no accuracy improvement; this is a reason to restrict, not enlarge, the budget.
- [Stanford CS2302018 project](https://cs230.stanford.edu/projects_spring_2018/reports/8290687.pdf)
  is a course report rather than peer-reviewed evidence. The research agent
  inspected its69.6% result and exclusion of players with fewer than25 matches;
  it does not establish an identical full-cohort chronological benchmark.
  Use only as a sequence-model idea, never as a target-performance guarantee.

## Required before the next run

1. Record exact finite structures, seeds, fixed mixtures and total training
   instances; do not quietly enlarge the menu after a disappointing2023 score.
2. Choose epochs/early stopping on a purged inner training-period tail, not
   outer2020/21/22 labels or2023. Freeze preprocessing on training rows; retain
   A/B complements and full population. Do not use arbitrary row-wise splits.
   For sequence histories, raw tournament-start dates do not identify within-week
   chronological order: aggregate tied event/date records or use an independently
   justified ordering. Never manufacture sequence order from source row order.
3. Use synthetic-only runtime checks to set feasible compute before new scores;
   separate neural dependencies from the pinned base environment.
4. Select on original forward folds and preserve the reused2022 gate caveat.
   Save weights, inference entry point, failures and matched controls.
5. Confirm the result against prior R4/R5/R7 roles; do not relabel a posthoc
   maximum as the chosen model or declare70% before audited evidence exists.
