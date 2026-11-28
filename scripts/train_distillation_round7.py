"""Privileged opening-market supervision; historical-only T-21 inference.

Only 2021/2022 opening quotes are loaded. No 2023 quote, external winner or
closing price enters this experiment. See the pre-fit research protocol.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn.linear_model import LogisticRegression
from threadpoolctl import threadpool_limits

from scripts.market_data import de_vig
from scripts.metrics import score_probabilities
from scripts.round2_learning import AntisymmetricSplineClassifier, candidate_specs as spline_specs
from scripts.train_models import ROOT, SEED, mean_loss, sha256, cluster_interval
from scripts.train_round5 import (REFERENCE, ACCURACY_GATE, LOSS_GUARD, accuracy,
    accuracy_interval, predict_round5, selection_probability)

R5_COMPARATOR = 'previous_round5_exploration'
R5_LOSS = 'previous_round5_best_loss'
HARD = 'hard_labels'
TEACHER_YEARS = (2021, 2022)
LINK_COLUMNS = ['match_id', 'season', 'status', 'target_event_id', 'market_match_id',
    'market_event_id', 'market_match_utc', 'market_player_a_id', 'market_player_b_id']
SOURCE_COLUMNS = ['match_id', 'date', 'tournoi_id', 'joueur1_id', 'joueur2_id',
    'cote1_ouverture', 'cote2_ouverture']


def validate_rows(rows, *, fitting=False):
    if rows.empty or rows.match_id.duplicated().any():
        raise ValueError('Nonempty unique target IDs required')
    seasons = pd.to_numeric(rows.season, errors='coerce')
    upper = 2022 if fitting else 2023
    dates = pd.to_datetime(rows.tourney_date, errors='coerce')
    compatible = dates.dt.year.eq(seasons) | (
        dates.dt.year.eq(seasons-1) & dates.dt.month.eq(12) & dates.dt.day.ge(25))
    if (not seasons.between(2009, upper).all() or not np.equal(seasons, np.floor(seasons)).all()
            or dates.isna().any() or not compatible.all() or dates.dt.year.gt(upper).any()):
        raise ValueError('Development date/season mismatch or sealed fitting year')
    return dates


def _id(value):
    number = pd.to_numeric(value, errors='coerce')
    if not np.isfinite(number) or number <= 0 or number != int(number):
        raise ValueError('Invalid source identifier')
    return int(number)


def build_teacher(rows, linked, source):
    """Use fixed links and opening prices only; source outcomes are irrelevant.

    All input target rows must be pre-2023. Filter link/source seasons before
    accessing identities or prices so future-year poison cannot be consumed.
    """
    validate_rows(rows, fitting=True)
    links = linked.loc[pd.to_numeric(linked.season).isin(TEACHER_YEARS), LINK_COLUMNS].copy()
    source = source.loc[pd.to_numeric(source.source_season).isin(TEACHER_YEARS),
                        SOURCE_COLUMNS + ['source_season']].copy()
    if links.match_id.duplicated().any() or source.match_id.duplicated().any():
        raise ValueError('Duplicate fixed links or raw source IDs')
    targets = rows[rows.season.isin(TEACHER_YEARS)]
    links = links[links.match_id.isin(targets.match_id)].set_index('match_id')
    if set(links.index) != set(targets.match_id):
        raise ValueError('Frozen links omit market-era training targets')
    source.index = source.match_id.map(_id)
    accepted = links[links.status.eq('matched')]
    if accepted.market_match_id.map(_id).duplicated().any():
        raise ValueError('A fixed source match is reused')
    result = pd.DataFrame({'teacher_probability': np.nan, 'teacher_available': 0,
        'teacher_match_utc': pd.NaT, 'source_match_id': np.nan}, index=rows.index)
    result['teacher_match_utc'] = pd.Series(pd.NaT, index=rows.index, dtype='datetime64[ns, UTC]')
    for index, row in targets.iterrows():
        link = links.loc[row.match_id]
        if int(link.season) != int(row.season) or str(link.target_event_id) != str(row.tourney_id):
            raise ValueError('Fixed target metadata changed')
        if link.status != 'matched':
            continue
        identifier = _id(link.market_match_id)
        if identifier not in source.index:
            raise ValueError('Fixed source match missing from permitted raw years')
        raw = source.loc[identifier]
        a, b = _id(link.market_player_a_id), _id(link.market_player_b_id)
        first, second = _id(raw.joueur1_id), _id(raw.joueur2_id)
        timestamp = pd.to_datetime(raw.date, utc=True, errors='coerce')
        declared = int(raw.source_season)
        valid_date = pd.notna(timestamp) and (timestamp.year == declared or (
            timestamp.year == declared-1 and timestamp.month == 12 and timestamp.day >= 25))
        if (declared != int(row.season) or not valid_date or timestamp.year > 2022
                or timestamp != pd.to_datetime(link.market_match_utc, utc=True)
                or _id(raw.tournoi_id) != _id(link.market_event_id)
                or a == b or first == second or {a, b} != {first, second}):
            raise ValueError('Fixed source identity, event or timestamp changed')
        odds = (raw.cote1_ouverture, raw.cote2_ouverture)
        oa, ob = odds if a == first else odds[::-1]
        log_odds, _, available = de_vig(oa, ob)
        if available:
            result.loc[index, 'teacher_probability'] = expit(log_odds)
            result.loc[index, 'teacher_available'] = 1
            result.loc[index, 'teacher_match_utc'] = timestamp
            result.loc[index, 'source_match_id'] = identifier
    return result


def load_teacher(rows, *, cache=ROOT / 'data/raw/market',
                 manifest_path=ROOT / 'data/market-manifest.json',
                 linkage_path=ROOT / 'results/round5/market/linked_matches.csv',
                 audit_path=ROOT / 'results/round5/market/artifact_audit.json'):
    validate_rows(rows, fitting=True)
    audit = json.loads(Path(audit_path).read_text())
    if sha256(linkage_path) != audit['artifact_sha256']['linked_matches.csv']:
        raise ValueError('Audited outcome-blind links changed')
    if sha256(manifest_path) != audit['source_manifest_sha256']:
        raise ValueError('Pinned market manifest changed')
    manifest = json.loads(Path(manifest_path).read_text())
    entries = {int(e['season']): e for e in manifest['files']}
    # The linkage reader has no odds/results columns; 2023 identities are then
    # discarded before their source IDs are accessed. Never load 2023 raw data.
    linked = pd.read_csv(linkage_path, usecols=LINK_COLUMNS)
    linked = linked[linked.season.isin(TEACHER_YEARS)].copy()
    frames = []
    for year in TEACHER_YEARS:
        path = Path(cache) / f'{year}.csv'
        if sha256(path) != entries[year]['sha256']:
            raise ValueError(f'Pinned opening source changed: {year}')
        source = pd.read_csv(path, sep=';', usecols=SOURCE_COLUMNS)
        if len(source) != entries[year]['rows']:
            raise ValueError('Pinned raw source row count changed')
        frames.append(source.assign(source_season=year))
    return build_teacher(rows, linked, pd.concat(frames, ignore_index=True))


def candidate_specs(columns):
    base = next(s for s in spline_specs(columns)
                if s['params']['C'] == .01 and s['params']['half_life'] == 3.)
    specs = [dict(deepcopy(base), name=HARD, teacher='hard', lambda_=0.)]
    for teacher in ('market', 'uniform'):
        for weight in (.25, .5, 1.):
            specs.append(dict(deepcopy(base), name=f'{teacher}_lambda{weight:g}',
                              teacher=teacher, lambda_=weight))
    return specs


def soft_targets(y, teacher, kind, weight, cutoff):
    y = np.asarray(y, dtype=float)
    if y.ndim != 1 or len(y) != len(teacher) or not np.isin(y, [0, 1]).all():
        raise ValueError('Aligned original binary labels required')
    if kind not in ('hard', 'market', 'uniform') or not np.isfinite(weight) or not 0 <= weight <= 1:
        raise ValueError('Unknown teacher or invalid lambda')
    if not teacher.teacher_available.isin([0, 1]).all():
        raise ValueError('Binary teacher availability required')
    times = pd.to_datetime(teacher.teacher_match_utc, utc=True, errors='coerce')
    declared = teacher.teacher_available.eq(1).to_numpy()
    if times[declared].isna().any() or times[declared].dt.year.gt(2022).any():
        raise ValueError('Teacher timestamp missing or outside training years')
    eligible = declared & times.le(pd.to_datetime(cutoff, utc=True)).to_numpy()
    target = y.copy()
    if kind != 'hard':
        q = teacher.teacher_probability.to_numpy(dtype=float)[eligible] if kind == 'market' else np.full(eligible.sum(), .5)
        if not np.isfinite(q).all() or np.any((q <= 0) | (q >= 1)):
            raise ValueError('Available market teacher must be a finite interior probability')
        target[eligible] = (1-weight) * y[eligible] + weight * q
    return target, eligible


class DistilledSpline:
    """Same historical design, with an optional soft-target logistic head."""
    def __init__(self, spec):
        self.spec = spec

    def fit(self, x, rows, teacher, *, cutoff):
        dates = validate_rows(rows, fitting=True)
        cutoff = pd.Timestamp(cutoff)
        if cutoff.tzinfo is not None:
            cutoff = cutoff.tz_convert(None)
        if pd.isna(cutoff) or cutoff >= pd.Timestamp('2023-01-01') or dates.gt(cutoff).any():
            raise ValueError('Training observations exceed the frozen chronological cutoff')
        if not x.index.equals(rows.index) or not teacher.index.equals(rows.index):
            raise ValueError('Features, targets and teacher must share exact row indices')
        y = rows.a_won.to_numpy()
        target, eligible = soft_targets(y, teacher, self.spec['teacher'], self.spec['lambda_'], cutoff)
        seasons = rows.season.to_numpy(dtype=float)
        self.model_ = AntisymmetricSplineClassifier(self.spec).fit(x, y, seasons=seasons)
        # Existing fit obtains the unchanged train-only basis. Hard control
        # remains bit-for-bit the original learner; soft variants replace only
        # its head, preserving the total sample-weight mass and regularization.
        if self.spec['teacher'] != 'hard':
            design = self.model_._design(self.model_._input(x))
            weights = 2. ** (-(seasons.max()-seasons)/self.spec['params']['half_life'])
            labels = np.r_[np.zeros(len(y), dtype=int), np.ones(len(y), dtype=int)]
            soft_weights = np.r_[weights*(1-target), weights*target]
            self.model_.estimator_ = LogisticRegression(C=self.spec['params']['C'],
                fit_intercept=False, max_iter=2000, random_state=SEED)
            self.model_.estimator_.fit(np.vstack([design, design]), labels, sample_weight=soft_weights)
        self.training_rows_ = len(y)
        self.teacher_rows_ = int(eligible.sum())
        self.cutoff_ = str(cutoff.date())
        self.training_end_season_ = int(seasons.max())
        return self

    def predict_proba(self, x):
        return self.model_.predict_proba(x)


def fold_masks(rows, fold):
    dates = validate_rows(rows)
    if fold == '2021H2':
        valid = rows.season.eq(2021) & dates.ge('2021-07-01')
        prior = rows.season.le(2021)
    elif fold in ('2022', '2023'):
        valid = rows.season.eq(int(fold))
        prior = rows.season.lt(int(fold))
    else:
        raise ValueError('Only 2021H2, 2022 and final development 2023 folds exist')
    if not valid.any():
        raise ValueError('Empty validation fold')
    cutoff = dates[valid].min()-pd.Timedelta(days=21)
    train = prior & dates.le(cutoff) & ~valid
    if not train.any() or (train & valid).any():
        raise ValueError('Empty or overlapping temporal training fold')
    return train, valid, cutoff


def select_candidate(y, predictions, candidates, mask):
    y, mask = np.asarray(y), np.asarray(mask, dtype=bool)
    names = [s['name'] for s in candidates] + [REFERENCE]
    if y.shape != mask.shape or not mask.any() or len(set(names)) != len(names):
        raise ValueError('Nonempty aligned selection and unique candidates required')
    for name in names:
        p = np.asarray(predictions[name])
        if p.shape != y.shape or not np.isfinite(p).all() or np.any((p < 0) | (p > 1)):
            raise ValueError('Invalid historical selection probability')
    limit = mean_loss(y[mask], predictions[REFERENCE][mask])+LOSS_GUARD
    eligible = [n for n in names if mean_loss(y[mask], predictions[n][mask]) <= limit+1e-12]
    key = lambda n: (-accuracy(y[mask], predictions[n][mask]), mean_loss(y[mask], predictions[n][mask]), n)
    choice = min(eligible, key=key)
    return dict(choice=choice, accuracy=accuracy(y[mask], predictions[choice][mask]),
        mean_log_loss=mean_loss(y[mask], predictions[choice][mask]),
        selection_rows=int(mask.sum()), eligible_models=len(eligible), loss_limit=limit)


def choose_procedure(y, predictions, specs, folds):
    folds, y = np.asarray(folds), np.asarray(y)
    if not np.isin(folds, ['2021H2', '2022']).all() or set(folds) != {'2021H2', '2022'}:
        raise ValueError('Only complete 2021H2 selection and 2022 gate are permitted')
    early = select_candidate(y, predictions, specs, folds == '2021H2')
    gate = folds == '2022'
    ref, candidate = predictions[REFERENCE][gate], predictions[early['choice']][gate]
    gain = accuracy(y[gate], candidate)-accuracy(y[gate], ref)
    loss_change = mean_loss(y[gate], candidate)-mean_loss(y[gate], ref)
    passed = gain >= ACCURACY_GATE-1e-12 and loss_change <= LOSS_GUARD+1e-12
    return dict(procedure='adopt_distillation' if passed else 'retain_round4', early_selection=early,
        accuracy_improvement=gain, loss_increase=loss_change,
        required_accuracy_improvement=ACCURACY_GATE, allowed_loss_increase=LOSS_GUARD,
        reference_2022_accuracy=accuracy(y[gate], ref), candidate_2022_accuracy=accuracy(y[gate], candidate),
        reference_2022_loss=mean_loss(y[gate], ref), candidate_2022_loss=mean_loss(y[gate], candidate),
        selection_fold='2021H2', gate_fold='2022')


def predict_distillation(bundle, features):
    required = bundle['feature_names']
    if features.empty or features.columns.duplicated().any() or not set(required).issubset(features):
        raise ValueError('Missing, duplicate or empty frozen historical features')
    x = features[required].astype(float)
    if not np.isfinite(x.to_numpy()).all():
        raise ValueError('Frozen historical features must be finite')
    predictions = {name: model.predict_proba(x) for name, model in bundle['models'].items()}
    old = predict_round5(bundle['previous_bundle'], x)
    predictions.update({alias: old[name].to_numpy() for alias, name in bundle['previous_aliases'].items()})
    if any(np.asarray(p).shape != (len(x),) or not np.isfinite(p).all()
           or np.any((p < 0) | (p > 1)) for p in predictions.values()):
        raise ValueError('Invalid frozen probability')
    return pd.DataFrame(predictions, index=features.index)


def write_report(out, summary, frozen, table, matched):
    chosen = frozen['pooled_selection']['choice']
    lines = ['# Round 7: privileged opening-market distillation', '',
        'Inference uses only the existing T−21 historical features. Training additionally uses '
        '2021–2022 opening-market supervision. No 2023 market prices are loaded. All numbers '
        'below describe the unchanged 2,588-match 2023 cohort and are exploratory; 2024–2025 remain sealed.', '',
        f"This market-supervised branch's recommendation: **{frozen['recommendation']}**. Early selection: "
        f"`{frozen['procedure_gate']['early_selection']['choice']}`. The 2022 gate changes accuracy by "
        f"{100*frozen['procedure_gate']['accuracy_improvement']:+.3f} percentage points "
        f"(required +0.5), loss by {frozen['procedure_gate']['loss_increase']:+.6f}.", '',
        f"Pooled H2+2022 choice, frozen before new 2023 scoring: **{chosen}**.", '',
        '| Model | Correct / 2,588 | Accuracy | Log loss |',
        '| --- | ---: | ---: | ---: |']
    for row in table.itertuples():
        lines.append(f'| {row.model} | {row.correct} | {100*row.accuracy:.3f}% | {row.mean_log_loss:.6f} |')
    best = table.iloc[0]
    lines += ['', f"70% requires **{summary['correct_required_for_70']} correct**. The best observed row above "
        f"is `{best.model}` with {int(best.correct)} correct, still "
        f"{max(0, summary['correct_required_for_70']-int(best.correct))} short. This observed ordering does not alter selection.", '',
        f"All seven students fit {summary['training_rows']:,} historical targets. "
        f"{summary['final_teacher_rows']:,} have eligible teacher probabilities; missing-price rows keep hard labels. "
        'Market and uniform teachers modify exactly the same rows. The uniform teacher is a required rival explanation, '
        'not a separate information improvement. The purely historical-label R4 recommendation remains a distinct condition.', '',
        '| Lambda | Pooled CV market − hard accuracy (pp) | Market − uniform accuracy (pp) | Market − uniform loss |',
        '| ---: | ---: | ---: | ---: |']
    for row in matched[matched.fold.eq('pooled')].itertuples():
        lines.append(f'| {row.lambda_} | {100*row.accuracy_vs_hard:+.3f} | '
                     f'{100*row.accuracy_vs_uniform:+.3f} | {row.loss_vs_uniform:+.6f} |')
    comparison = summary['comparisons'][HARD][chosen]
    delta, interval = comparison['log_loss']['mean_log_loss_reduction'], comparison['log_loss']['ci95']
    lines += ['', f"Against the identical hard-label learner, the frozen choice changes correct decisions by "
        f"{comparison['accuracy']['net_correct']:+d} and reduces log loss by {delta:.6f}; "
        f"its conditional 95% loss-reduction interval is [{interval[0]:.6f}, {interval[1]:.6f}]. "
        'This matched comparison is more informative about supervision than comparison with a different R4 architecture. '
        'An interval spanning zero does not establish a reproducible improvement.', '',
        'The highest observed accuracy belongs to an unselected row and must be assessed together with its loss. '
        'In particular, heavy uniform smoothing can raise accuracy while degrading the probability estimates.']
    lines += ['', 'The full matched table includes both forward folds and exploratory 2023. '
        'Positive accuracy differences favor market supervision; negative loss differences favor it. '
        'Paired date-block intervals in summary.json are conditional descriptive intervals and do not account '
        'for the full adaptive search, repeated 2022 gate, or all player dependence.', '',
        'Quotes are described as prematch by the provider, but capture timestamps are absent. '
        'Their recorded match times bound training eligibility under that source claim. The teacher may '
        'contain late fitness/news information unavailable to the student and impossible to reconstruct from '
        'T−21 history. This experiment cannot establish SOTA or validate a 70% claim.', '',
        'Reproduce: `.venv/bin/python -m scripts.train_distillation_round7`. '
        'Reload without fitting: append `--predict-features results/round5/development_features.csv`. '
        'Only use trusted local joblib artifacts.', '']
    (out / 'report_summary.md').write_text('\n'.join(lines))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT / 'results/round7/distillation')
    parser.add_argument('--predict-features', type=Path)
    args = parser.parse_args()
    out, previous = args.output, ROOT / 'results/round5'
    out.mkdir(parents=True, exist_ok=True)
    if args.predict_features:
        frame = pd.read_csv(args.predict_features)
        p = predict_distillation(joblib.load(out / 'frozen_models.joblib'), frame)
        if 'match_id' in frame:
            p.insert(0, 'match_id', frame.match_id.to_numpy())
        p.to_csv(out / 'inference_predictions.csv', index=False)
        return
    protocol = ROOT / 'docs/research/round7_distillation.md'
    if not protocol.is_file():
        raise ValueError('Recorded distillation protocol required before fitting')
    audit = json.loads((previous / 'artifact_audit.json').read_text())
    for name in ('development_features.csv', 'frozen_models.joblib', 'oof_predictions.csv', 'search_registry.json', 'frozen_selection.json'):
        if sha256(previous / name) != audit['artifact_sha256'][name]:
            raise ValueError(f'Previous audited artifact changed: {name}')
    old_registry = json.loads((previous / 'search_registry.json').read_text())
    inventory_file = ROOT / 'data/processed/mens_singles_2006_2025.csv'
    if sha256(inventory_file) != old_registry['inventory_sha256']:
        raise ValueError('Pinned inventory changed')
    rows = pd.read_csv(ROOT / 'results/round3/development_targets.csv')
    validate_rows(rows)
    inventory = pd.read_csv(inventory_file, dtype={'a_won': 'string'})
    inventory = inventory[inventory.season.le(2023)].copy()
    from scripts.train_transfer_round6 import verify_tour_targets
    verify_tour_targets(rows, inventory)
    x = pd.read_csv(previous / 'development_features.csv')
    if len(x) != len(rows) or x.columns.duplicated().any() or not np.isfinite(x.to_numpy()).all():
        raise ValueError('Historical feature matrix changed')
    specs = candidate_specs(x.columns)
    teacher = load_teacher(rows[rows.season.le(2022)])
    teacher.to_csv(out / 'training_teacher.csv', index_label='target_index')
    old_frozen = json.loads((previous / 'frozen_selection.json').read_text())
    old_oof = pd.read_csv(previous / 'oof_predictions.csv')
    identity = ['match_id', 'season', 'a_won']
    if not rows[rows.season.isin([2020, 2021, 2022])][identity].reset_index(drop=True).equals(old_oof[identity]):
        raise ValueError('Old OOF identity mismatch')
    fold_plan = {fold: fold_masks(rows, fold) for fold in ('2021H2', '2022')}
    union = np.logical_or.reduce([v.to_numpy() for _, v, _ in fold_plan.values()])
    orows = rows[union]
    labels = np.where(orows.season.eq(2021), '2021H2', '2022')
    y = orows.a_won.to_numpy()
    old_oof = old_oof.set_index('match_id').loc[orows.match_id]
    predictions = {REFERENCE: old_oof[REFERENCE].to_numpy(),
        R5_COMPARATOR: selection_probability(old_frozen['pooled_selection'],
            {c: old_oof[c].to_numpy() for c in old_frozen['pooled_selection']['components']}),
        R5_LOSS: old_oof['all_inputs_spline_C0.01'].to_numpy()}
    registry = dict(round=7, experiment='privileged_market_teacher', candidates=specs,
        feature_names=list(x), training_information='2021/2022 opening quote teacher when available',
        inference_information='T-21 historical features only', source_years_read=list(TEACHER_YEARS),
        protocol_sha256=sha256(protocol), source_code_sha256=sha256(Path(__file__)),
        previous_artifact_sha256=audit['artifact_sha256'],
        target_sha256=sha256(ROOT / 'results/round3/development_targets.csv'),
        inventory_sha256=sha256(inventory_file), market_manifest_sha256=sha256(ROOT / 'data/market-manifest.json'),
        frozen_links_sha256=sha256(ROOT / 'results/round5/market/linked_matches.csv'),
        training_teacher_sha256=sha256(out / 'training_teacher.csv'),
        folds={f: dict(train=int(t.sum()), valid=int(v.sum()), cutoff=str(c.date()))
               for f, (t, v, c) in fold_plan.items()},
        old_r5_comparisons_selection_eligible=False, decision_threshold=.5,
        exploratory_after_2023=True, final_test_evaluated=False)
    (out / 'search_registry.json').write_text(json.dumps(registry, indent=2)+'\n')
    cv, fit_counts = [], []
    for i, spec in enumerate(specs, 1):
        p = pd.Series(index=orows.index, dtype=float)
        for fold, (train, valid, cutoff) in fold_plan.items():
            model = DistilledSpline(spec).fit(x[train], rows[train], teacher.loc[rows.index[train]], cutoff=cutoff)
            p.loc[rows.index[valid]] = model.predict_proba(x[valid])
            fit_counts.append(dict(model=spec['name'], fold=fold, rows=model.training_rows_, teacher_rows=model.teacher_rows_))
        predictions[spec['name']] = p.to_numpy()
        print(f'[{i}/{len(specs)}] {spec["name"]} CV accuracy={accuracy(y,p.to_numpy()):.6f} loss={mean_loss(y,p.to_numpy()):.6f}', flush=True)
    for name, p in predictions.items():
        for fold in ('2021H2', '2022', 'pooled'):
            mask = np.ones(len(y), dtype=bool) if fold == 'pooled' else labels == fold
            cv.append(dict(model=name, fold=fold, n=int(mask.sum()), accuracy=accuracy(y[mask], p[mask]),
                           mean_log_loss=mean_loss(y[mask], p[mask])))
    pd.DataFrame(cv).to_csv(out / 'temporal_cv.csv', index=False)
    gate = choose_procedure(y, predictions, specs, labels)
    pooled = select_candidate(y, predictions, specs, np.ones(len(y), dtype=bool))
    choice = pooled['choice'] if gate['procedure'] == 'adopt_distillation' else REFERENCE
    frozen = dict(recommendation=choice, pooled_selection=pooled, procedure_gate=gate,
        history_lag_days=21, decision_threshold=.5, fit_end_season=2022,
        training_information=registry['training_information'], inference_information=registry['inference_information'],
        exploratory_after_2023=True, final_test_evaluated=False)
    (out / 'frozen_selection.json').write_text(json.dumps(frozen, indent=2)+'\n')
    print('Frozen selection:', json.dumps(frozen), flush=True)
    train, valid, cutoff = fold_masks(rows, '2023')
    models = {}
    for spec in specs:
        model = DistilledSpline(spec).fit(x[train], rows[train], teacher.loc[rows.index[train]], cutoff=cutoff)
        models[spec['name']] = model
        fit_counts.append(dict(model=spec['name'], fold='final_fit', rows=model.training_rows_, teacher_rows=model.teacher_rows_))
    aliases = {REFERENCE: REFERENCE, R5_COMPARATOR: old_frozen['pooled_selection']['choice'], R5_LOSS: 'all_inputs_spline_C0.01'}
    bundle = dict(models=models, feature_names=list(x), frozen=frozen, previous_aliases=aliases,
                  previous_bundle=joblib.load(previous / 'frozen_models.joblib'))
    p, vr = predict_distillation(bundle, x[valid]), rows[valid]
    scores = []
    for name in p:
        score = score_probabilities(zip(vr.match_id, vr.a_won), zip(vr.match_id, p[name]))
        scores.append(dict(model=name, recommended=name == choice, **score,
                          brier=float(np.mean((vr.a_won.to_numpy()-p[name].to_numpy())**2))))
    table = pd.DataFrame(scores).sort_values(['accuracy', 'mean_log_loss'], ascending=[False, True])
    table.to_csv(out / 'validation_comparison.csv', index=False)
    pd.concat([vr.reset_index(drop=True), p.reset_index(drop=True)], axis=1).to_csv(out / 'validation_predictions.csv', index=False)
    pd.concat([orows[identity].reset_index(drop=True).assign(fold=labels), pd.DataFrame(predictions)], axis=1).to_csv(out / 'oof_predictions.csv', index=False)
    pd.DataFrame(fit_counts).to_csv(out / 'fit_populations.csv', index=False)
    matched = []
    for weight in (.25, .5, 1.):
        for fold in ('2021H2', '2022', 'pooled', '2023_exploratory'):
            if fold == '2023_exploratory':
                target, values, mask = vr.a_won.to_numpy(), p, np.ones(len(vr), dtype=bool)
            else:
                target, values = y, predictions
                mask = np.ones(len(y), dtype=bool) if fold == 'pooled' else labels == fold
            market, uniform, hard = [np.asarray(values[n])[mask] for n in (f'market_lambda{weight:g}', f'uniform_lambda{weight:g}', HARD)]
            target = target[mask]
            matched.append(dict(lambda_=weight, fold=fold,
                accuracy_vs_hard=accuracy(target,market)-accuracy(target,hard),
                loss_vs_hard=mean_loss(target,market)-mean_loss(target,hard),
                accuracy_vs_uniform=accuracy(target,market)-accuracy(target,uniform),
                loss_vs_uniform=mean_loss(target,market)-mean_loss(target,uniform)))
    matched = pd.DataFrame(matched)
    matched.to_csv(out / 'matched_controls.csv', index=False)
    comparisons = {ref: {n: dict(accuracy=accuracy_interval(vr,p[ref].to_numpy(),p[n].to_numpy()),
        log_loss=cluster_interval(vr,p[ref].to_numpy(),p[n].to_numpy())) for n in p}
        for ref in (REFERENCE, R5_COMPARATOR, HARD)}
    teacher_comparisons = {str(w): dict(
        accuracy=accuracy_interval(vr,p[f'uniform_lambda{w:g}'].to_numpy(),p[f'market_lambda{w:g}'].to_numpy()),
        log_loss=cluster_interval(vr,p[f'uniform_lambda{w:g}'].to_numpy(),p[f'market_lambda{w:g}'].to_numpy()))
        for w in (.25,.5,1.)}
    summary = dict(recommendation=choice, new_candidates=len(specs), historical_features=x.shape[1],
        training_rows=int(train.sum()), validation_matches=len(vr), correct_required_for_70=int(np.ceil(.7*len(vr))),
        final_teacher_rows=models[HARD].teacher_rows_, comparisons=comparisons,
        matched_teacher_comparisons=teacher_comparisons,
        hard_control_vs_old_same_model_max_error=float(np.max(np.abs(p[HARD]-p[R5_LOSS]))),
        source_years_read=list(TEACHER_YEARS), inference_requires_market=False,
        exploratory_after_2023=True, final_test_evaluated=False)
    (out / 'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    joblib.dump(bundle, out / 'frozen_models.joblib')
    write_report(out, summary, frozen, table, matched)
    print(table.to_string(index=False), flush=True)


if __name__ == '__main__':
    from scripts.train_distillation_round7 import main as canonical_main
    with threadpool_limits(limits=4):
        canonical_main()
