# Round 8: active 70% goal and new structural experiments

2026-11-27. The user explicitly requested a persistent goal and continued
optimization until success. The goal is active; success requires at least
1,812/2,588 correct on the original full 2023 cohort, fixed 0.5 threshold,
legitimate prematch inputs, audited frozen artifacts and reproduction. A
reused 2023 score is exploratory even if it exceeds 70%; it is not an
independent SOTA claim. 2024–2025 remain sealed. No guarantee of attainment
is made and failure cannot be relabelled success. No task-completion tracking
automation was created; work continues under the active goal.

## Independent proposals before sharing

The scientific-brainstorming workflow and facilitation, evaluation and
responsible-AI references were applied again for this new exploration. Existing
skill attribution remains in sota_evidence.md. Three returning AI agents
submitted independent ideas before seeing each other's Round 8 suggestions;
prior seven-round outcomes were visible and are shared anchors. These are not
independent human experts. Criteria, in order, are legal/time/identity validity,
actual new information or mechanism, discriminating controls, and local compute
feasibility. No vote or subjective numeric score determines scientific truth.

| ID / origin | Proposal | Rival explanation or falsifier | Disposition |
| --- | --- | --- | --- |
| R8-L1 / research; A8-1 / audit; P8-2 / coordinator | Joint sparse binomial serve/return skill likelihood, with chronological latent-state updates | The old R2 already uses online diagonal Laplace; only joint fitting/time structure is new. Point forecasts can improve without match-win gains | Bounded joint MAP adaptation with matched controls; not a claim to replicate full Bayesian NUTS |
| F8-A / features; A8-2 / audit | Historical event serving environment interacts with player style | Player composition and venue moves can explain apparent speed; common intercept alone often preserves ordering | Conservative name/ID/surface histories, matched interaction controls; call it an event-environment proxy |
| F8-B / features | New multiple-bookmaker consensus | Quote timing, duplicate expert information, identity and provider permissions | Original tennis-data provider disallows automated/AI use; no download or mirror workaround. Defer this source |
| F8-C / features | Historical shot/charting composition | Manual annotations may postdate games; star/round coverage bias | Defer pending annotation-as-of-commit reconstruction; date alone insufficient |
| R8-L2 / research; P8-3 / coordinator | Shared sequence model / tabular neural model learns interactions and order | Added capacity or truncated-history selection can explain gains | Retain as next-stage alternative; no empirical claim |
| R8-L3 / research | Retirement/normal-completion multitask training | Additional labels may have little relevance to the unchanged complete-match target | Defer; never change evaluation population |
| A8-3 / audit | Market-minus-historical-OOF logit distillation | Previous student may merely smooth labels; data are limited | Retain alternative; different from coordinator's inference-time market residual |
| P8-1 / coordinator | Learn bounded residual departures from an actual prematch price prior using richer history/earlier-round inputs | Market is already strong; estimated residuals may add noise, especially at changed quote coverage | Separate later-information branch, matched input blocks and raw-price reference |
| P8-4 / coordinator, post-literature | Historical market logit graph estimates player strength for future pairs | Unknown players, graph sparsity, temporal shift; published comparisons discard some cases | Use allowed fixed Valuebetennis historical opening prices, full-cohort fallback and T−21 cutoff |

The coordinator recorded P8-1/2/3 before reading agent proposals; literature
then introduced P8-4. Proposals were subsequently shared and responsibilities
split. Joint estimation and event environment were not chosen because multiple
AI agents proposed them; their testability and existing inputs make bounded
experiments feasible. Negative/null outcomes remain useful.

## Evidence and boundaries

- Ingram (2019), DOI10.1515/jqas-2018-0008, author manuscript
  https://martiningram.github.io/papers/bayes_point_based.pdf. Binomial point
  likelihood and player/surface/time/tournament effects motivate the joint
  branch. Its reported point model trails its Elo comparator, so complexity
  is not assumed to help. Our approximation and full-cohort fallback differ.
- Penn, Michael and Bhatt, *Dynamic Graph-Based Forecasts of Bookmakers' Odds
  in Professional Tennis*, https://arxiv.org/html/2508.15956v1 and published
  DOI10.1098/rsos.252512. Past-market graph estimation motivates P8-4. The
  publisher reader returned403; author full text was retrieved. Its public
  aggregate results are not this project's held-out evaluation and do not
  justify dropping unknown-player targets.
- Kovalchik (2016), DOI10.1515/jqas-2015-0059, primary publisher description
  https://www.degruyterbrill.com/document/doi/10.1515/jqas-2015-0059/html,
  discusses bookmaker consensus; this does not override current data-source
  access/use constraints or prove benefit from multiple correlated quotes.
- Holzmüller, Grinsztajn and Steinwart, NeurIPS2024, *Better by Default*,
  https://proceedings.neurips.cc/paper_files/paper/2024/file/2ee1c87245956e3eaa71aaba5f5753eb-Paper-Conference.pdf.
  This motivates a deferred full-data neural alternative; its generic
  tabular evidence is not tennis performance evidence.

Queries included joint point-based Bayesian tennis model, tabular RealMLP
strong defaults, historical bookmaker graph forecasting, bookmaker consensus,
and serving/court environment. This is targeted source checking, not a
systematic review. Individual branch protocols specify the exact finite
menus, temporal selection, fallbacks and primary method sources before fits.
