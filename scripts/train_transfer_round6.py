"""Frozen, compact supervised Challenger/qualifying-to-Tour transfer experiment.

Only earlier-season supervision is fitted; each feature uses T-21 history.
See docs/research/round6_transfer.md for the pre-fit design and limitations.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import SplineTransformer
from threadpoolctl import threadpool_limits

from scripts.advanced_features import _date
from scripts.metrics import score_probabilities
from scripts.multilevel_features import SOURCES, _season_date, build_multilevel_features
from scripts.train_models import FOLDS, ROOT, mean_loss, sha256
from scripts.train_round5 import accuracy, accuracy_interval, predict_round5, selection_probability

REFERENCE = 'round4_reference'
R5_COMPARATOR = 'round5_pooled_comparator'
LOSS_GUARD = .005
ACCURACY_GATE = .005
SPLINE_CORE = ('transfer_log_rank_diff', 'ml_common_overall_diff',
               'ml_common_blend_diff', 'ml_level_overall_diff')


def verify_tour_targets(tour, inventory):
    """Audit cached target metadata against the independently pinned inventory."""
    eligible = inventory[inventory.source.eq('tour') & inventory.season.isin(range(2009, 2024))].copy()
    eligible.index = eligible.match_id.str.removeprefix('tour:')
    if eligible.index.duplicated().any() or tour.match_id.duplicated().any() or set(tour.match_id) != set(eligible.index):
        raise ValueError('Prior Tour target membership differs from inventory')
    for row in tour.itertuples():
        source = eligible.loc[row.match_id]
        if (any(str(getattr(row, c)) != str(source[c]) for c in ('player_a_id', 'player_b_id', 'surface'))
                or any(float(getattr(row, c)) != float(source[c]) for c in ('season', 'rank_a', 'rank_b', 'a_won'))
                or _date(row.tourney_date) != _date(source.tourney_date)):
            raise ValueError('Prior Tour target metadata differs from inventory')


def build_transfer_targets(tour, inventory):
    """Keep every Tour development row; append only 2009-22 auxiliary rows."""
    original = tour.copy()
    if original.empty or not original.season.isin(range(2009, 2024)).all():
        raise ValueError('Tour targets must be nonempty development seasons 2009-2023')
    original['source'] = 'tour'
    seasons = pd.to_numeric(inventory.season, errors='coerce')
    mask = inventory.source.isin(('challenger', 'qualifying')) & seasons.isin(range(2009, 2023))
    auxiliary = inventory.loc[mask].copy()
    auxiliary['tourney_date'] = auxiliary.tourney_date.map(lambda d: _date(d).isoformat())
    auxiliary = auxiliary.sort_values(['season', 'tourney_date', 'match_id'])
    frame = pd.concat([original, auxiliary], ignore_index=True)
    if frame.match_id.duplicated().any():
        raise ValueError('Duplicate target match IDs')
    for column in ('rank_a', 'rank_b', 'a_won', 'season'):
        frame[column] = pd.to_numeric(frame[column], errors='raise')
    if (not np.isfinite(frame[['rank_a', 'rank_b']].to_numpy()).all()
            or (frame[['rank_a', 'rank_b']] <= 0).any().any()
            or not frame.a_won.isin((0, 1)).all()):
        raise ValueError('Targets require positive ranks and binary outcomes')
    for row in frame.itertuples():
        _season_date(_date(row.tourney_date), int(row.season))
        if str(row.player_a_id) >= str(row.player_b_id) or row.surface not in ('Hard', 'Clay', 'Grass'):
            raise ValueError('Targets require canonical distinct players and a supported surface')
    frame['a_won'] = frame.a_won.astype(int)
    frame.attrs.update(tour_rows=len(original), auxiliary_rows=len(auxiliary), final_test_evaluated=False)
    return frame


def build_transfer_features(targets, inventory):
    """48 finite fields; odd advantages have *_diff names, others are even."""
    frame = build_multilevel_features(targets, inventory)
    ranks = targets[['rank_a', 'rank_b']].astype(float).to_numpy()
    if not np.isfinite(ranks).all() or (ranks <= 0).any():
        raise ValueError('Positive finite target ranks required')
    logs = np.log(ranks)
    frame['transfer_log_rank_diff'] = logs[:, 1] - logs[:, 0]
    frame['transfer_reciprocal_rank_diff'] = 1 / ranks[:, 0] - 1 / ranks[:, 1]
    frame['transfer_log_rank_mean'] = logs.mean(axis=1)
    frame['transfer_log_rank_abs_gap'] = np.abs(logs[:, 1] - logs[:, 0])
    for surface in ('Hard', 'Clay', 'Grass'):
        flag = targets.surface.eq(surface).to_numpy(dtype=float)
        frame[f'transfer_surface_is_{surface.lower()}'] = flag
        if surface != 'Hard':
            frame[f'transfer_log_rank_{surface.lower()}_diff'] = frame.transfer_log_rank_diff * flag
            frame[f'transfer_ml_{surface.lower()}_diff'] = frame.ml_common_overall_diff * flag
    if frame.shape[1] != 48 or not np.isfinite(frame.to_numpy()).all():
        raise ValueError('Unexpected transfer feature matrix')
    return frame


def candidate_specs(columns):
    columns = list(columns)
    odd = [c for c in columns if c.endswith('_diff')]
    if not odd or not set(SPLINE_CORE).issubset(columns):
        raise ValueError('Compact transfer feature schema is incomplete')
    return [dict(name=f'transfer_{kind}_C{c:g}_aux{weight:g}', kind=kind,
                 C=c, auxiliary_weight=weight, half_life=3., columns=columns,
                 odd_columns=odd, spline_columns=list(SPLINE_CORE) if kind == 'spline' else [])
            for kind, c in (('linear', .01), ('linear', .1), ('spline', .01))
            for weight in (0., .1, .3, 1.)]


def transfer_weights(rows, auxiliary_weight):
    if not np.isfinite(auxiliary_weight) or auxiliary_weight < 0:
        raise ValueError('Nonnegative finite auxiliary weight required')
    seasons = rows.season.to_numpy(dtype=float)
    if (not len(rows) or not np.isin(seasons, range(2009, 2023)).all()
            or not rows.source.isin(SOURCES).all()):
        raise ValueError('Fit supervision must be development-only 2009-2022')
    dates = [_date(d) for d in rows.tourney_date]
    for day, season in zip(dates, seasons):
        _season_date(day, int(season))
        if day.year > 2022:
            raise ValueError('Fit dates must precede 2023')
    tour = rows.source.eq('tour').to_numpy()
    if not tour.any():
        raise ValueError('Tour training targets are required')
    weights = 2. ** (-(seasons.max() - seasons) / 3.) * np.where(tour, 1., auxiliary_weight)
    return weights * tour.sum() / weights.sum()


def weighted_rms(values, weights):
    scale = np.sqrt(np.average(values * values, axis=0, weights=weights))
    return np.where(scale > 1e-10, scale, 1.)


class TransferClassifier:
    """Weighted antisymmetric logistic head with optional fixed odd splines."""
    def __init__(self, spec):
        self.spec = spec

    def _input(self, x):
        if x.columns.duplicated().any() or not set(self.spec['columns']).issubset(x):
            raise ValueError('Missing or duplicate transfer features')
        values = x[self.spec['columns']].astype(float)
        if not np.isfinite(values.to_numpy()).all():
            raise ValueError('Transfer features must be finite')
        return values

    def _shape(self, odd):
        return np.column_stack([.5 * (s.transform(odd[:, [i]]) - s.transform(-odd[:, [i]]))
                                for i, s in self.splines_]) if self.splines_ else np.empty((len(odd), 0))

    def _design(self, x):
        odd = np.clip(x[self.spec['odd_columns']].to_numpy() / self.odd_scale_, -8., 8.)
        shape = self._shape(odd)
        return np.column_stack([odd, .3 * shape / self.shape_scale_]) if shape.shape[1] else odd

    def fit(self, x, rows):
        if len(x) != len(rows) or not x.index.equals(rows.index):
            raise ValueError('Training features and target index must align')
        weights = transfer_weights(rows, self.spec['auxiliary_weight'])
        positive = weights > 0
        # A weight-zero auxiliary feature cannot influence scaling, even if invalid.
        x = self._input(x.loc[positive])
        y = rows.loc[positive, 'a_won'].to_numpy(dtype=float)
        weights = weights[positive]
        if not np.isin(y, (0, 1)).all() or len(np.unique(y)) != 2:
            raise ValueError('Training requires both binary classes')
        odd_names = self.spec['odd_columns']
        if any(not name.endswith('_diff') for name in odd_names):
            raise ValueError('Only odd features may enter the no-intercept head')
        self.odd_scale_ = weighted_rms(x[odd_names].to_numpy(), weights)
        odd = np.clip(x[odd_names].to_numpy() / self.odd_scale_, -8., 8.)
        self.splines_ = []
        for name in self.spec['spline_columns']:
            index = odd_names.index(name)
            transformer = SplineTransformer(degree=2, knots=np.linspace(-3, 3, 5)[:, None],
                                            include_bias=True, extrapolation='constant')
            transformer.fit(np.array([[-3.], [3.]]))  # Fixed knots, no query-dependent basis.
            self.splines_.append((index, transformer))
        shape = self._shape(odd)
        self.shape_scale_ = weighted_rms(shape, weights) if shape.shape[1] else np.array([])
        self.estimator_ = LogisticRegression(C=self.spec['C'], fit_intercept=False, max_iter=2000,
                                             random_state=158)
        self.estimator_.fit(self._design(x), y, sample_weight=weights)
        self.training_metadata_ = dict(positive_rows=len(x), tour_rows=int(rows.source.eq('tour').sum()),
                                      weight_sum=float(weights.sum()), last_season=int(rows.season.max()))
        return self

    def predict_proba(self, x):
        return np.clip(self.estimator_.predict_proba(self._design(self._input(x)))[:, 1], 1e-8, 1 - 1e-8)


def fold_masks(rows, year):
    if year not in (2020, 2021, 2022, 2023):
        raise ValueError('Only frozen development folds are allowed')
    valid = rows.source.eq('tour') & rows.season.eq(year)
    dates = pd.to_datetime(rows.tourney_date)
    if not valid.any():
        raise ValueError('Tour validation targets required')
    cutoff = dates[valid].min() - pd.Timedelta(days=21)
    train = rows.season.lt(year) & dates.le(cutoff)
    return train, valid


def blend_registry(specs):
    return {f"{s['name']}_r4blend{w:g}": dict(model=s['name'], model_weight=w)
            for s in specs for w in (.25, .5, .75)}


def add_blends(predictions, blends):
    result = {n: np.asarray(p, dtype=float) for n, p in predictions.items()}
    for name, setting in blends.items():
        weight = setting['model_weight']
        if not np.isfinite(weight) or not 0 <= weight <= 1:
            raise ValueError('Blend must be convex')
        result[name] = weight * result[setting['model']] + (1 - weight) * result[REFERENCE]
    return result


def select_candidate(y, predictions, mask):
    y, mask = np.asarray(y), np.asarray(mask, dtype=bool)
    if y.shape != mask.shape or not mask.any() or REFERENCE not in predictions:
        raise ValueError('Aligned nonempty historical selection required')
    limit = mean_loss(y[mask], predictions[REFERENCE][mask]) + LOSS_GUARD
    allowed = [name for name, p in predictions.items() if name != R5_COMPARATOR
               and mean_loss(y[mask], p[mask]) <= limit + 1e-12]
    choice = min(allowed, key=lambda name: (-accuracy(y[mask], predictions[name][mask]),
                                           mean_loss(y[mask], predictions[name][mask]), name))
    return dict(choice=choice, accuracy=accuracy(y[mask], predictions[choice][mask]),
                mean_log_loss=mean_loss(y[mask], predictions[choice][mask]),
                eligible_models=len(allowed), selection_rows=int(mask.sum()), loss_limit=limit)


def choose_procedure(y, predictions, seasons):
    seasons, y = np.asarray(seasons), np.asarray(y)
    if not np.isin(seasons, FOLDS).all() or not set(FOLDS).issubset(seasons):
        raise ValueError('Only 2020-2022 OOF decisions are allowed')
    early = select_candidate(y, predictions, np.isin(seasons, (2020, 2021)))
    gate = seasons == 2022
    reference, candidate = predictions[REFERENCE][gate], predictions[early['choice']][gate]
    gain = accuracy(y[gate], candidate) - accuracy(y[gate], reference)
    loss_increase = mean_loss(y[gate], candidate) - mean_loss(y[gate], reference)
    passed = gain >= ACCURACY_GATE - 1e-12 and loss_increase <= LOSS_GUARD + 1e-12
    return dict(procedure='adopt_transfer' if passed else 'retain_round4', early_selection=early,
                reference=REFERENCE, accuracy_improvement=gain, loss_increase=loss_increase,
                required_accuracy_improvement=ACCURACY_GATE, allowed_loss_increase=LOSS_GUARD,
                selection_seasons=[2020, 2021], gate_season=2022,
                reference_2022_accuracy=accuracy(y[gate], reference),
                candidate_2022_accuracy=accuracy(y[gate], candidate),
                reference_2022_loss=mean_loss(y[gate], reference),
                candidate_2022_loss=mean_loss(y[gate], candidate))


def predict_transfer(bundle, features):
    required = bundle['feature_names']
    if features.empty or features.columns.duplicated().any() or not set(required).issubset(features):
        raise ValueError('Frozen inference features are missing, duplicate or empty')
    x = features[required].astype(float)
    if not np.isfinite(x.to_numpy()).all():
        raise ValueError('Frozen inference features must be finite')
    old = predict_round5(bundle['previous_bundle'], x)
    p = {REFERENCE: old['previous_probability_recommendation'].to_numpy(),
         R5_COMPARATOR: old[bundle['previous_pooled_choice']].to_numpy()}
    p.update({name: model.predict_proba(x) for name, model in bundle['models'].items()})
    p = add_blends(p, bundle['blends'])
    if any(v.shape != (len(x),) or not np.isfinite(v).all() or np.any((v < 0) | (v > 1)) for v in p.values()):
        raise ValueError('Invalid frozen transfer probability')
    return pd.DataFrame(p, index=features.index)


def score_table(rows, predictions):
    scores = []
    for name, values in predictions.items():
        scores.append(dict(model=name, **score_probabilities(zip(rows.match_id, rows.a_won),
                          zip(rows.match_id, values)),
                          brier=float(np.mean((rows.a_won.to_numpy() - values) ** 2))))
    return pd.DataFrame(scores).sort_values(['accuracy', 'mean_log_loss'], ascending=[False, True])


def write_report(out, summary, frozen, table, matched):
    index = table.set_index('model')
    chosen = frozen['pooled_selection']['choice']
    lines = ['# Round 6: supervised lower-level transfer', '',
        'All results use the same 2,588 Tour matches in 2023 with T−21 historical information. '
        'They are exploratory after earlier inspection of 2023; 2024–2025 remain sealed.', '',
        f"Formal recommendation: **{frozen['recommendation']}**. Early-selected 2022 gain "
        f"{100*frozen['procedure_gate']['accuracy_improvement']:.3f} percentage points "
        f"(required 0.5), loss change {frozen['procedure_gate']['loss_increase']:+.6f}.", '',
        '| Frozen or matched comparator | Correct / 2,588 | Accuracy | Log loss |',
        '| --- | ---: | ---: | ---: |']
    names = list(dict.fromkeys([REFERENCE, R5_COMPARATOR, chosen] + summary['matched_selected_controls']))
    for name in names:
        row = index.loc[name]
        lines.append(f"| {name} | {int(row.correct)} | {100*row.accuracy:.3f}% | {row.mean_log_loss:.6f} |")
    lines += ['', f"Pooled 2020–2022 selection was frozen as `{chosen}` before scoring new 2023 predictions. "
              'Every raw and blended candidate is reported in validation_comparison.csv; observed 2023 ordering does not alter selection.', '',
              f"{summary['auxiliary_rows']} additional supervision targets are available. Positive-auxiliary-weight models use "
              f"{summary['training_rows']} eligible fitting rows, including {summary['training_tour_rows']} Tour rows. "
              f"The frozen pooled choice has auxiliary weight zero and only {summary['training_tour_rows']} active Tour training rows. "
              'Auxiliary weight zero removes auxiliary rows before scaling. All weight sums normalize to the Tour fit count.', '',
              'The matched CV table isolates auxiliary supervision with the same compact inputs and regularization. '
              f"All {len(matched[matched.fold.astype(str).eq('pooled')])} positive-auxiliary configurations worsen pooled CV log loss "
              'relative to their matched Tour-only control; only the linear C=0.1, auxiliary-weight=0.1 setting improves pooled accuracy. '
              'The rich Round 4/Round 5 references also contain technical/player information absent from this compact experiment. '
              'A null transfer result cannot establish that all future transfer approaches fail.', '',
              'The 2022 gate and historical reference have been reused across rounds; this is a practical exploration rule, '
              'not independent confirmation. Challenger and qualifying labels may follow different conditional relationships. '
              'No SOTA or 70% claim is established by this experiment.', '',
              'Reproduce: `.venv/bin/python -m scripts.train_transfer_round6`. Reload without fitting: '
              '`.venv/bin/python -m scripts.train_transfer_round6 --predict-features results/round6/transfer/inference_features.csv`.', '']
    (out / 'report_summary.md').write_text('\n'.join(lines))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT / 'results/round6/transfer')
    parser.add_argument('--predict-features', type=Path)
    args = parser.parse_args()
    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    if args.predict_features:
        frame = pd.read_csv(args.predict_features)
        result = predict_transfer(joblib.load(out / 'frozen_models.joblib'), frame)
        if 'match_id' in frame:
            result.insert(0, 'match_id', frame.match_id.to_numpy())
        result.to_csv(out / 'inference_predictions.csv', index=False)
        return
    previous = ROOT / 'results/round5'
    audit = json.loads((previous / 'artifact_audit.json').read_text())
    for name in ('development_features.csv', 'frozen_models.joblib', 'oof_predictions.csv'):
        if sha256(previous / name) != audit['artifact_sha256'][name]:
            raise ValueError(f'Previous audited artifact changed: {name}')
    old_registry = json.loads((previous / 'search_registry.json').read_text())
    inventory_file = ROOT / 'data/processed/mens_singles_2006_2025.csv'
    original_file = ROOT / 'data/processed/model_features.csv'
    if sha256(inventory_file) != old_registry['inventory_sha256'] or sha256(original_file) != old_registry['target_sha256']:
        raise ValueError('Pinned source changed')
    tour = pd.read_csv(ROOT / 'results/round3/development_targets.csv')
    inventory = pd.read_csv(inventory_file, dtype={'a_won': 'string'})
    inventory = inventory[inventory.season.le(2023)].copy()
    # Verify original target identity against the pinned inventory without final-test labels.
    verify_tour_targets(tour, inventory)
    rows = build_transfer_targets(tour, inventory)
    print(f'Building {len(rows)} compact target rows', flush=True)
    x = build_transfer_features(rows, inventory)
    old_x = pd.read_csv(previous / 'development_features.csv')
    if len(old_x) != len(tour):
        raise ValueError('Prior feature cohort size mismatch')
    full = old_x.copy()
    for column in x:
        values = x.loc[:len(tour)-1, column].to_numpy()
        if column in full:
            if not np.allclose(full[column], values, rtol=0, atol=1e-10):
                raise ValueError(f'Prior overlapping features changed: {column}')
        else:
            full[column] = values
    specs, blends = candidate_specs(x.columns), blend_registry(candidate_specs(x.columns))
    old_oof = pd.read_csv(previous / 'oof_predictions.csv')
    orows = rows[rows.source.eq('tour') & rows.season.isin(FOLDS)]
    if not orows[['match_id', 'season', 'a_won']].reset_index(drop=True).equals(old_oof[['match_id', 'season', 'a_won']]):
        raise ValueError('Prior OOF target alignment changed')
    old_frozen = json.loads((previous / 'frozen_selection.json').read_text())
    oof = {REFERENCE: old_oof.previous_probability_recommendation.to_numpy(),
           R5_COMPARATOR: selection_probability(old_frozen['pooled_selection'], {n: old_oof[n].to_numpy() for n in old_frozen['pooled_selection']['components']})}
    fold_counts = {}
    for year in (*FOLDS, 2023):
        train, valid = fold_masks(rows, year)
        fold_counts[str(year)] = dict(training_rows=int(train.sum()), validation_rows=int(valid.sum()),
            source_training_counts={s: int((train & rows.source.eq(s)).sum()) for s in SOURCES},
            latest_fit_date=rows.loc[train, 'tourney_date'].max(), earliest_validation_date=rows.loc[valid, 'tourney_date'].min())
    registry = dict(round=6, branch='supervised_transfer', candidates=specs, blends=blends,
        fit_candidates=12, selection_entries=49, feature_names=list(x), inference_feature_names=list(full),
        target_sha256=sha256(original_file), inventory_sha256=sha256(inventory_file),
        prior_target_sha256=sha256(ROOT / 'results/round3/development_targets.csv'),
        protocol_sha256=sha256(ROOT / 'docs/research/round6_transfer.md'),
        source_code_sha256=sha256(Path(__file__)), folds=fold_counts,
        feature_metadata=x.attrs, final_test_evaluated=False, auxiliary_fit_last_season=2022,
        threshold=.5, weight_normalization='sum equals Tour training target count')
    (out / 'search_registry.json').write_text(json.dumps(registry, indent=2) + '\n')
    rows.to_csv(out / 'development_targets.csv', index=False)
    x.to_csv(out / 'development_features.csv', index=False)
    full.to_csv(out / 'inference_features.csv', index=False)
    y = orows.a_won.to_numpy()
    for i, spec in enumerate(specs, 1):
        p = pd.Series(index=orows.index, dtype=float)
        for year in FOLDS:
            train, valid = fold_masks(rows, year)
            model = TransferClassifier(spec).fit(x[train], rows[train])
            p.loc[rows.index[valid]] = model.predict_proba(x[valid])
        oof[spec['name']] = p.to_numpy()
        print(f"[{i}/12] {spec['name']}: CV accuracy={accuracy(y, p.to_numpy()):.5f}, loss={mean_loss(y, p.to_numpy()):.6f}", flush=True)
    oof = add_blends(oof, blends)
    gate = choose_procedure(y, oof, orows.season)
    pooled = select_candidate(y, oof, np.ones(len(y), dtype=bool))
    recommendation = pooled['choice'] if gate['procedure'] == 'adopt_transfer' else REFERENCE
    frozen = dict(round=6, branch='supervised_transfer', recommendation=recommendation,
        procedure_gate=gate, pooled_selection=pooled, decision_threshold=.5, fit_end_season=2022,
        history_lag_days=21, exploratory_after_2023=True, final_test_evaluated=False)
    (out / 'frozen_selection.json').write_text(json.dumps(frozen, indent=2) + '\n')
    print('Frozen selection:', json.dumps(frozen), flush=True)
    cv = score_table(orows, oof)
    for year in FOLDS:
        mask = orows.season.eq(year).to_numpy()
        cv[f'{year}_accuracy'] = cv.model.map({n: accuracy(y[mask], p[mask]) for n, p in oof.items()})
        cv[f'{year}_log_loss'] = cv.model.map({n: mean_loss(y[mask], p[mask]) for n, p in oof.items()})
    cv.to_csv(out / 'temporal_cv.csv', index=False)
    pd.concat([orows[['match_id', 'season', 'a_won']].reset_index(drop=True), pd.DataFrame(oof)], axis=1).to_csv(out / 'oof_predictions.csv', index=False)
    matched = []
    for spec in specs:
        if not spec['auxiliary_weight']:
            continue
        base = next(s['name'] for s in specs if s['kind'] == spec['kind'] and s['C'] == spec['C'] and not s['auxiliary_weight'])
        for year in (2020, 2021, 2022, 'pooled'):
            mask = np.ones(len(y), dtype=bool) if year == 'pooled' else orows.season.eq(year).to_numpy()
            matched.append(dict(model=spec['name'], matched_tour_only=base, fold=year,
                accuracy_improvement=accuracy(y[mask], oof[spec['name']][mask]) - accuracy(y[mask], oof[base][mask]),
                log_loss_improvement=mean_loss(y[mask], oof[base][mask]) - mean_loss(y[mask], oof[spec['name']][mask])))
    matched = pd.DataFrame(matched)
    matched.to_csv(out / 'matched_transfer_controls.csv', index=False)
    train, valid = fold_masks(rows, 2023)
    models = {s['name']: TransferClassifier(s).fit(x[train], rows[train]) for s in specs}
    bundle = dict(models=models, feature_names=list(full), compact_feature_names=list(x), frozen=frozen,
                  blends=blends, previous_bundle=joblib.load(previous / 'frozen_models.joblib'),
                  previous_pooled_choice=old_frozen['pooled_selection']['choice'])
    joblib.dump(bundle, out / 'frozen_models.joblib')
    vr = rows[valid]
    p = predict_transfer(bundle, full.loc[vr.index])
    table = score_table(vr, {n: p[n].to_numpy() for n in p})
    table['recommended'] = table.model.eq(recommendation)
    table.to_csv(out / 'validation_comparison.csv', index=False)
    pd.concat([vr.reset_index(drop=True), p.reset_index(drop=True)], axis=1).to_csv(out / 'validation_predictions.csv', index=False)
    controls = [s['name'] for s in specs if not s['auxiliary_weight']]
    controls += [min((s['name'] for s in specs if s['kind'] == kind and s['C'] == c),
                    key=lambda n: (-accuracy(y, oof[n]), mean_loss(y, oof[n]), n))
                 for kind, c in (('linear', .01), ('linear', .1), ('spline', .01))]
    summary = dict(round=6, branch='supervised_transfer', recommendation=recommendation,
        pooled_choice=pooled['choice'], fit_candidates=12, selection_entries=49,
        compact_features=x.shape[1], inference_features=full.shape[1], total_target_rows=len(rows),
        auxiliary_rows=int(rows.source.ne('tour').sum()), training_rows=int(train.sum()),
        training_tour_rows=int((train & rows.source.eq('tour')).sum()), validation_matches=len(vr),
        correct_required_for_70=int(np.ceil(.7*len(vr))), matched_selected_controls=list(dict.fromkeys(controls)),
        pooled_comparison_vs_round4=accuracy_interval(vr, p[REFERENCE].to_numpy(), p[pooled['choice']].to_numpy()),
        exploratory_after_2023=True, final_test_evaluated=False)
    (out / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    write_report(out, summary, frozen, table, matched)
    print(table.to_string(index=False), flush=True)


if __name__ == '__main__':
    from scripts.train_transfer_round6 import main as canonical_main
    with threadpool_limits(limits=4):
        canonical_main()
