"""Accuracy-oriented temporal experiments on the unchanged full ATP cohort.

The user authorized accuracy as the round-five objective. The .5 decision
threshold is fixed; log loss remains a selection guard. No final test is read.
"""
from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from scripts.metrics import score_probabilities
from scripts.round2_learning import candidate_specs as spline_specs
from scripts.round2_learning import make_model as make_spline
from scripts.round5_learning import candidate_specs as margin_specs
from scripts.round5_learning import make_model as make_margin
from scripts.train_models import (FOLDS, ROOT, SEED, SymmetricClassifier,
                                  cluster_interval, mean_loss, sha256, temporal_masks)
from scripts.train_round4 import predict_round4

LOSS_GUARD = .005
ACCURACY_GATE = .005
REFERENCE = 'previous_probability_recommendation'


def accuracy(y, p):
    return float(np.mean((np.asarray(p) >= .5) == np.asarray(y)))


def candidate_specs(x, groups):
    """75 settings, fixed before fitting; includes matched information controls."""
    result = []
    core = groups['core_age']
    sets = {'core': core, 'multilevel': core + groups['multilevel'],
            'style': core + groups['style'], 'both': core + groups['multilevel'] + groups['style']}
    for group, columns in sets.items():
        odd = [c for c in columns if c.endswith('_diff')]
        for c, half in itertools.product((.001, .01, .1), (None, 3.)):
            result.append(dict(name=f'{group}_linear_C{c:g}_half{half}', family=f'{group}_linear',
                method='linear', kind='logistic', columns=odd, params={'C': c}, half_life=half))
        for c in (.003, .01, .03):
            spec = next(s for s in spline_specs(columns) if s['params']['C'] == .01 and s['params']['half_life'] == 3.)
            result.append(dict(spec, name=f'{group}_spline_C{c:g}_half3', family=f'{group}_spline',
                method='spline', params={**spec['params'], 'C': c}))
    columns = sets['both']
    base = next(s for s in spline_specs(columns) if s['params']['C'] == .01 and s['params']['half_life'] == 3.)
    for half in (None, 1., 5.):
        result.append(dict(base, name=f'both_spline_C0.01_half{half}', family='both_spline', method='spline',
                           params={**base['params'], 'half_life': half}))
    for half in (1., 5.):
        result.append(dict(name=f'both_linear_C0.01_half{half}', family='both_linear', method='linear',
            kind='logistic', columns=[c for c in columns if c.endswith('_diff')], params={'C': .01}, half_life=half))
    for spec in margin_specs(columns):
        result.append(dict(spec, method=spec['family']))
    for depth, iterations in itertools.product((4, 6), (600, 1200)):
        result.append(dict(name=f'both_cat_d{depth}_it{iterations}', family='new_catboost', method='catboost',
            kind='catboost', columns=columns, half_life=3.,
            params=dict(depth=depth, iterations=iterations, learning_rate=.03, l2_leaf_reg=20)))
    for leaves in (7, 15):
        result.append(dict(name=f'both_hist_l{leaves}', family='new_histboost', method='histboost',
            kind='histboost', columns=columns, half_life=3.,
            params=dict(max_iter=500, max_leaf_nodes=leaves, learning_rate=.03,
                        l2_regularization=50, min_samples_leaf=100)))
    all_columns = list(x)
    result.append(dict(name='all_inputs_linear_C0.01', family='all_inputs', method='linear', kind='logistic',
        columns=[c for c in all_columns if c.endswith('_diff')], params={'C': .01}, half_life=3.))
    base = next(s for s in spline_specs(all_columns) if s['params']['C'] == .01 and s['params']['half_life'] == 3.)
    result.append(dict(base, name='all_inputs_spline_C0.01', family='all_inputs', method='spline'))
    if len({s['name'] for s in result}) != len(result):
        raise ValueError('Duplicate configuration names')
    return result


def make_model(spec):
    if spec['kind'].startswith('r5_'):
        return make_margin(spec)
    return make_spline(spec) if spec['kind'] == 'odd_spline' else SymmetricClassifier(spec)


def fit_model(spec, x, rows):
    model = make_model(spec)
    args = (x, rows.a_won.to_numpy(), rows.season.to_numpy())
    if spec['kind'].startswith('r5_'):
        return model.fit(*args, dates=rows.tourney_date.to_numpy())
    return model.fit(*args)


def simplex_weights(n, units=10):
    """All nonnegative grid weights summing to one, including single models."""
    if n < 1 or units < 1:
        raise ValueError('Positive simplex dimension and grid size required')
    for cuts in itertools.combinations(range(units + n - 1), n - 1):
        boundaries = (-1,) + cuts + (units + n - 1,)
        yield np.array([boundaries[i+1] - boundaries[i] - 1 for i in range(n)], dtype=float) / units


def select_accuracy(y, oof, specs, mask, reference=REFERENCE):
    """Choose single/family/grid blend using only the supplied historical mask."""
    y = np.asarray(y)
    mask = np.asarray(mask, dtype=bool)
    if mask.shape != y.shape or not mask.any() or reference not in oof:
        raise ValueError('Aligned selection rows and reference probabilities required')
    names = [s['name'] for s in specs]
    if len(names) != len(set(names)) or set(names) != set(oof):
        raise ValueError('OOF and candidate names must agree exactly')
    limit = mean_loss(y[mask], oof[reference][mask]) + LOSS_GUARD
    def key(name):
        return -accuracy(y[mask], oof[name][mask]), mean_loss(y[mask], oof[name][mask]), name
    allowed = [n for n in names if mean_loss(y[mask], oof[n][mask]) <= limit]
    best = min(allowed, key=key)
    selected = {}
    for family in sorted({s['family'] for s in specs}):
        group = [s['name'] for s in specs if s['family'] == family and s['name'] in allowed]
        if group:
            selected[family] = min(group, key=key)
    # Distinct learner families provide a bounded candidate mixture. There is
    # no outcome-dependent routing or arbitrary threshold optimization.
    method_winners = []
    for method in sorted({s['method'] for s in specs}):
        group = [s['name'] for s in specs if s['method'] == method and s['name'] in allowed]
        if group:
            method_winners.append(min(group, key=key))
    components = sorted(set(method_winners), key=key)[:4]
    matrix = np.column_stack([oof[n][mask] for n in components])
    best_weights = None
    best_key = (float('inf'), float('inf'), '')
    for weights in simplex_weights(len(components)):
        p = matrix @ weights
        loss = mean_loss(y[mask], p)
        if loss > limit:
            continue
        candidate_key = (-accuracy(y[mask], p), loss, tuple(weights))
        if candidate_key < best_key:
            best_key, best_weights = candidate_key, weights
    if best_weights is None:
        raise ValueError('No admissible blend despite eligible single models')
    used = [i for i, w in enumerate(best_weights) if w > 0]
    components = [components[i] for i in used]
    weights = best_weights[used]
    blend = sum(w * oof[n] for n, w in zip(components, weights))
    choice = 'accuracy_grid_blend' if len(components) > 1 else components[0]
    return dict(choice=choice, best_single=best, selected_by_family=selected,
        components=components, weights=weights.tolist(), eligible_models=len(allowed),
        selected_accuracy=accuracy(y[mask], blend[mask]), selected_log_loss=mean_loss(y[mask], blend[mask]),
        loss_limit=limit, selection_rows=int(mask.sum()), grid_units=10)


def selection_probability(selection, predictions):
    names = selection['components']
    weights = np.asarray(selection['weights'], dtype=float)
    if (not names or len(names) != len(set(names)) or weights.shape != (len(names),)
            or not np.isfinite(weights).all() or np.any(weights < 0)
            or not np.isclose(weights.sum(), 1., rtol=0, atol=1e-12)
            or any(name not in predictions for name in names)):
        raise ValueError('Frozen blend needs unique components and nonnegative unit-sum weights')
    values = [np.asarray(predictions[name], dtype=float) for name in names]
    shape = values[0].shape
    if (len(shape) != 1 or not shape[0] or any(v.shape != shape for v in values)
            or any(not np.isfinite(v).all() or np.any((v < 0) | (v > 1)) for v in values)):
        raise ValueError('Frozen blend probabilities must be aligned, finite and in [0,1]')
    return np.clip(np.column_stack(values) @ weights, 0., 1.)


def choose_procedure(y, oof, specs, seasons):
    seasons = np.asarray(seasons)
    early_mask, gate_mask = np.isin(seasons, (2020, 2021)), seasons == 2022
    if not early_mask.any() or not gate_mask.any():
        raise ValueError('Need early folds and 2022 gate')
    early = select_accuracy(y, oof, specs, early_mask)
    candidate = selection_probability(early, oof)
    ref = oof[REFERENCE]
    before, after = accuracy(y[gate_mask], ref[gate_mask]), accuracy(y[gate_mask], candidate[gate_mask])
    ref_loss, new_loss = mean_loss(y[gate_mask], ref[gate_mask]), mean_loss(y[gate_mask], candidate[gate_mask])
    passed = after - before >= ACCURACY_GATE - 1e-12 and new_loss <= ref_loss + LOSS_GUARD + 1e-12
    return dict(procedure='adopt_accuracy_model' if passed else 'retain_round4', early_selection=early,
        reference=REFERENCE, reference_2022_accuracy=before, candidate_2022_accuracy=after,
        accuracy_improvement=after-before, required_accuracy_improvement=ACCURACY_GATE,
        reference_2022_loss=ref_loss, candidate_2022_loss=new_loss, allowed_loss_increase=LOSS_GUARD,
        selection_seasons=[2020, 2021], gate_season=2022)


def accuracy_interval(rows, reference, candidate, repeats=2000):
    y = rows.a_won.to_numpy()
    gain = ((candidate >= .5) == y).astype(int) - ((reference >= .5) == y).astype(int)
    frame = pd.DataFrame({'date': rows.tourney_date.to_numpy(), 'gain': gain})
    groups = frame.groupby('date').gain.agg(['sum', 'count']).to_numpy()
    rng = np.random.default_rng(SEED)
    samples = groups[rng.integers(0, len(groups), size=(repeats, len(groups)))]
    distribution = samples[:, :, 0].sum(axis=1) / samples[:, :, 1].sum(axis=1)
    return dict(accuracy_improvement=float(gain.mean()), net_correct=int(gain.sum()),
        corrected=int((gain == 1).sum()), newly_wrong=int((gain == -1).sum()),
        ci95=np.quantile(distribution, [.025, .975]).tolist(), date_blocks=len(groups), repetitions=repeats,
        interpretation='Exploratory paired date-block interval; no correction for repeated search.')


def predict_round5(bundle, features):
    required = bundle['feature_names']
    if features.empty or features.columns.duplicated().any() or not set(required).issubset(features):
        raise ValueError('Missing, duplicate or empty frozen features')
    x = features[required].astype(float)
    if not np.isfinite(x.to_numpy()).all():
        raise ValueError('Frozen features must be finite')
    p = {n: model.predict_proba(x) for n, model in bundle['models'].items()}
    old = predict_round4(bundle['previous_bundle'], x)
    p.update({alias: old[name].to_numpy() for alias, name in bundle['previous_aliases'].items()})
    selection = bundle['frozen']['pooled_selection']
    selected_probability = selection_probability(selection, p)
    if len(selection['components']) > 1:
        p['accuracy_grid_blend'] = selected_probability
    for name, values in p.items():
        if np.asarray(values).shape != (len(x),) or not np.isfinite(values).all() or np.any((values < 0) | (values > 1)):
            raise ValueError(f'Invalid probabilities: {name}')
    return pd.DataFrame(p, index=features.index)


def load_previous(rows, directory):
    frozen = json.loads((directory / 'frozen_selection.json').read_text())
    older = json.loads((ROOT / 'results/round3/frozen_selection.json').read_text())
    base = pd.read_csv(directory / 'oof_predictions.csv')
    third = pd.read_csv(ROOT / 'results/round3/oof_predictions.csv')
    identity = rows.loc[rows.season.isin(FOLDS), ['match_id', 'season', 'a_won']].reset_index(drop=True)
    for table in (base, third):
        if not identity.equals(table[identity.columns]):
            raise ValueError('Previous OOF identity mismatch')
    probabilities = {
        REFERENCE: base[frozen['recommendation']].to_numpy(),
        'previous_accuracy_comparator': third[older['selected_by_family']['history2009_linear_format1']].to_numpy(),
        'previous_exploratory_best_loss': base['without_graph_spline_C0.01'].to_numpy(),
    }
    aliases = {REFERENCE: frozen['recommendation'], 'previous_accuracy_comparator': 'round3_linear',
               'previous_exploratory_best_loss': 'without_graph_spline_C0.01',
               'rank_reference': 'rank_reference', 'round1_best_single': 'round1_best_single'}
    specs = [dict(name=name, family=name, method='previous', kind='previous') for name in probabilities]
    return probabilities, aliases, specs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT / 'results/round5')
    parser.add_argument('--previous', type=Path, default=ROOT / 'results/round4')
    parser.add_argument('--predict-features', type=Path)
    args = parser.parse_args()
    out, previous = args.output, args.previous
    out.mkdir(parents=True, exist_ok=True)
    if args.predict_features is not None:
        features = pd.read_csv(args.predict_features)
        prediction = predict_round5(joblib.load(out / 'frozen_models.joblib'), features)
        if 'match_id' in features:
            prediction.insert(0, 'match_id', features.match_id.to_numpy())
        prediction.to_csv(out / 'inference_predictions.csv', index=False)
        return
    audit = json.loads((previous / 'artifact_audit.json').read_text())
    for name in ('development_features.csv', 'frozen_models.joblib', 'oof_predictions.csv'):
        if sha256(previous / name) != audit['artifact_sha256'][name]:
            raise ValueError(f'Previous audited artifact changed: {name}')
    old_registry = json.loads((previous / 'search_registry.json').read_text())
    source = ROOT / 'data/processed/model_features.csv'
    inventory_file = ROOT / 'data/processed/mens_singles_2006_2025.csv'
    if sha256(source) != old_registry['target_sha256'] or sha256(inventory_file) != old_registry['inventory_sha256']:
        raise ValueError('Pinned source hash mismatch')
    rows = pd.read_csv(ROOT / 'results/round3/development_targets.csv')
    from scripts.expanded_training import build_expanded_targets
    inventory = pd.read_csv(inventory_file, dtype={'a_won': 'string'})
    inventory = inventory[inventory.season.le(2023)].copy()
    fresh = build_expanded_targets(pd.read_csv(source), inventory)
    if not fresh[['match_id', 'season', 'a_won']].equals(rows[['match_id', 'season', 'a_won']]):
        raise ValueError('Expanded targets changed')
    base = pd.read_csv(previous / 'development_features.csv')
    if len(base) != len(rows):
        raise ValueError('Feature and target rows mismatch')
    print('Building cross-level and technical style information', flush=True)
    from scripts.multilevel_features import build_multilevel_features
    from scripts.style_features import build_style_features
    blocks = {'multilevel': build_multilevel_features(rows, inventory), 'style': build_style_features(rows, inventory)}
    for block in blocks.values():
        if not block.index.equals(rows.index):
            raise ValueError('New feature rows mismatch')
    x = pd.concat([base, *blocks.values()], axis=1)
    if x.columns.duplicated().any() or not np.isfinite(x.to_numpy()).all():
        raise ValueError('Invalid extended features')
    groups = dict(core_age=old_registry['feature_groups']['base'] +
                  [c for c in old_registry['feature_groups']['profile'] if c.startswith('profile_age')],
                  **{name: list(block) for name, block in blocks.items()})
    specs = candidate_specs(x, groups)
    old_oof, aliases, old_specs = load_previous(rows, previous)
    all_specs = specs + old_specs
    registry = dict(round=5, seed=SEED, folds=list(FOLDS), features=list(x), feature_groups=groups,
        feature_metadata={name: block.attrs for name, block in blocks.items()}, candidates=all_specs,
        target_sha256=sha256(source), inventory_sha256=sha256(inventory_file),
        previous_features_sha256=sha256(previous / 'development_features.csv'),
        objective='Accuracy first with .005 log-loss guard; fixed .5 threshold and whole cohort.',
        accuracy_gate=ACCURACY_GATE, loss_guard=LOSS_GUARD, exploratory_after_2023=True, final_test_evaluated=False)
    (out / 'search_registry.json').write_text(json.dumps(registry, indent=2) + '\n')
    x.to_csv(out / 'development_features.csv', index=False)
    orows = rows[rows.season.isin(FOLDS)]
    y = orows.a_won.to_numpy()
    oof = dict(old_oof)
    cv = []
    for i, spec in enumerate(specs, 1):
        p = pd.Series(index=orows.index, dtype=float)
        fold_scores = {}
        for year in FOLDS:
            train, valid = temporal_masks(rows, year)
            model = fit_model(spec, x[train], rows[train])
            prediction = model.predict_proba(x[valid])
            p.loc[rows.index[valid]] = prediction
            fold_scores[f'{year}_accuracy'] = accuracy(rows.a_won[valid], prediction)
            fold_scores[f'{year}_log_loss'] = mean_loss(rows.a_won[valid].to_numpy(), prediction)
        oof[spec['name']] = p.to_numpy()
        cv.append(dict(name=spec['name'], family=spec['family'], accuracy=accuracy(y, p.to_numpy()),
                       mean_log_loss=mean_loss(y, p.to_numpy()), **fold_scores))
        pd.DataFrame(cv).to_csv(out / 'temporal_cv.csv', index=False)
        print(f"[{i}/{len(specs)}] {spec['name']}: acc={cv[-1]['accuracy']:.5f} loss={cv[-1]['mean_log_loss']:.6f}", flush=True)
    for spec in old_specs:
        p = oof[spec['name']]
        fold_scores = {}
        for year in FOLDS:
            mask = orows.season.eq(year).to_numpy()
            fold_scores[f'{year}_accuracy'] = accuracy(y[mask], p[mask])
            fold_scores[f'{year}_log_loss'] = mean_loss(y[mask], p[mask])
        cv.append(dict(name=spec['name'], family=spec['family'], accuracy=accuracy(y, p), mean_log_loss=mean_loss(y, p), **fold_scores))
    pd.DataFrame(cv).to_csv(out / 'temporal_cv.csv', index=False)
    gate = choose_procedure(y, oof, all_specs, orows.season)
    pooled = select_accuracy(y, oof, all_specs, np.ones(len(y), dtype=bool))
    choice = pooled['choice'] if gate['procedure'] == 'adopt_accuracy_model' else REFERENCE
    frozen = dict(round=5, recommendation=choice, probability_reference=REFERENCE, procedure_gate=gate,
        pooled_selection=pooled, fit_end_season=2022, history_lag_days=21, decision_threshold=.5,
        exploratory_after_2023=True, final_test_evaluated=False)
    (out / 'frozen_selection.json').write_text(json.dumps(frozen, indent=2) + '\n')
    print('Frozen accuracy procedure:', choice, gate, flush=True)
    needed = set(pooled['selected_by_family'].values()) | set(pooled['components']) | {pooled['best_single']}
    models = {}
    by_name = {s['name']: s for s in specs}
    train, valid = temporal_masks(rows, 2023)
    for name in sorted(needed & set(by_name)):
        models[name] = fit_model(by_name[name], x[train], rows[train])
    bundle = dict(models=models, feature_names=list(x), frozen=frozen, previous_aliases=aliases,
                  previous_bundle=joblib.load(previous / 'frozen_models.joblib'))
    p = predict_round5(bundle, x[valid])
    vr = rows[valid]
    scores = []
    for name in p:
        score = score_probabilities(zip(vr.match_id, vr.a_won), zip(vr.match_id, p[name]))
        scores.append(dict(model=name, recommended=name == choice, **score,
            brier=float(np.mean((vr.a_won.to_numpy() - p[name].to_numpy()) ** 2))))
    table = pd.DataFrame(scores).sort_values(['accuracy', 'mean_log_loss'], ascending=[False, True])
    table.to_csv(out / 'validation_comparison.csv', index=False)
    pd.concat([vr.reset_index(drop=True), p.reset_index(drop=True)], axis=1).to_csv(out / 'validation_predictions.csv', index=False)
    pd.concat([orows[['match_id', 'season', 'a_won']].reset_index(drop=True), pd.DataFrame(oof)], axis=1).to_csv(out / 'oof_predictions.csv', index=False)
    reference = p[REFERENCE].to_numpy()
    intervals = {n: dict(accuracy=accuracy_interval(vr, reference, p[n].to_numpy()),
                        log_loss=cluster_interval(vr, reference, p[n].to_numpy())) for n in p}
    summary = dict(round=5, recommendation=choice, new_candidates=len(specs), reused_candidates=len(old_specs),
        features=x.shape[1], training_rows=int(train.sum()), validation_matches=len(vr),
        correct_required_for_70=int(np.ceil(.7 * len(vr))), comparisons_vs_round4=intervals,
        exploratory_after_2023=True, final_test_evaluated=False)
    (out / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    joblib.dump(bundle, out / 'frozen_models.joblib')
    print(table.to_string(index=False), flush=True)


if __name__ == '__main__':
    from scripts.train_round5 import main as canonical_main
    with threadpool_limits(limits=4):
        canonical_main()
