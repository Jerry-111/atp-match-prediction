"""Round-six matched information experiments, with unchanged T-21 targets."""
import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from scripts.metrics import score_probabilities
from scripts.train_models import ROOT, FOLDS, SEED, mean_loss, temporal_masks
from scripts.train_round4 import sha256, cluster_interval
from scripts.round2_learning import candidate_specs as spline_specs
from scripts.train_round5 import (REFERENCE, ACCURACY_GATE, LOSS_GUARD, accuracy,
    fit_model, choose_procedure, select_accuracy, selection_probability,
    accuracy_interval, predict_round5)


def candidate_specs(groups):
    base = groups['base']
    extra = groups['extended'] + groups['trajectory'] + groups['breakpoint']
    variants = {'base': base, 'extended': base + groups['extended'],
        'trajectory': base + groups['trajectory'], 'breakpoint': base + groups['breakpoint'],
        'combined': base + extra}
    specs = []
    for group, columns in variants.items():
        for c in (.003, .01, .03):
            specs.append(dict(name=f'r6_{group}_linear_C{c:g}', kind='logistic',
                method='linear', family=f'{group}_linear', columns=[n for n in columns if n.endswith('_diff')],
                params={'C': c}, half_life=3.))
            spline = next(s for s in spline_specs(columns)
                if s['params']['C'] == .01 and s['params']['half_life'] == 3.)
            specs.append(dict(spline, name=f'r6_{group}_spline_C{c:g}', method='spline',
                family=f'{group}_spline', params={**spline['params'], 'C': c}))
    columns = groups['all_previous'] + extra
    specs.append(dict(name='r6_all_linear_C0.01', kind='logistic', method='linear', family='all_linear',
        columns=[n for n in columns if n.endswith('_diff')], params={'C': .01}, half_life=3.))
    spline = next(s for s in spline_specs(columns)
        if s['params']['C'] == .01 and s['params']['half_life'] == 3.)
    specs.append(dict(spline, name='r6_all_spline_C0.01', method='spline', family='all_spline'))
    for depth in (4, 6):
        specs.append(dict(name=f'r6_combined_cat_d{depth}', kind='catboost', method='catboost',
            family='combined_catboost', columns=variants['combined'], half_life=3.,
            params=dict(depth=depth, iterations=600, learning_rate=.03, l2_leaf_reg=20)))
    if len(specs) != 34 or len({s['name'] for s in specs}) != len(specs):
        raise ValueError('Expected34 distinct configurations')
    return specs


def predict_round6(bundle, features):
    required = bundle['feature_names']
    if features.empty or features.columns.duplicated().any() or not set(required).issubset(features):
        raise ValueError('Missing, duplicate or empty frozen features')
    x = features[required].astype(float)
    if not np.isfinite(x.to_numpy()).all():
        raise ValueError('Frozen features must be finite')
    values = {n: m.predict_proba(x) for n, m in bundle['models'].items()}
    old = predict_round5(bundle['previous_bundle'], x)
    values.update({alias: old[name].to_numpy() for alias, name in bundle['previous_aliases'].items()})
    selection = bundle['frozen']['pooled_selection']
    blended = selection_probability(selection, values)
    if len(selection['components']) > 1:
        values['accuracy_grid_blend'] = blended
    for name, p in values.items():
        if np.asarray(p).shape != (len(x),) or not np.isfinite(p).all() or np.any((p < 0) | (p > 1)):
            raise ValueError(f'Invalid probabilities: {name}')
    return pd.DataFrame(values, index=features.index)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT / 'results/round6')
    parser.add_argument('--predict-features', type=Path)
    args = parser.parse_args()
    out, previous = args.output, ROOT / 'results/round5'
    out.mkdir(parents=True, exist_ok=True)
    if args.predict_features:
        features = pd.read_csv(args.predict_features)
        p = predict_round6(joblib.load(out / 'frozen_models.joblib'), features)
        if 'match_id' in features:
            p.insert(0, 'match_id', features.match_id.to_numpy())
        p.to_csv(out / 'inference_predictions.csv', index=False)
        return
    audit = json.loads((previous / 'artifact_audit.json').read_text())
    for name in ('development_features.csv', 'frozen_models.joblib', 'oof_predictions.csv', 'search_registry.json'):
        if sha256(previous / name) != audit['artifact_sha256'][name]:
            raise ValueError(f'Previous audited artifact changed: {name}')
    old_registry = json.loads((previous / 'search_registry.json').read_text())
    inventory_file = ROOT / 'data/processed/mens_singles_2006_2025.csv'
    target_file = ROOT / 'data/processed/model_features.csv'
    if sha256(inventory_file) != old_registry['inventory_sha256'] or sha256(target_file) != old_registry['target_sha256']:
        raise ValueError('Pinned source hashes changed')
    rows = pd.read_csv(ROOT / 'results/round3/development_targets.csv')
    inventory = pd.read_csv(inventory_file, dtype={'a_won': 'string'})
    inventory = inventory[inventory.season.le(2023)].copy()
    from scripts.expanded_training import build_expanded_targets
    fresh = build_expanded_targets(pd.read_csv(target_file), inventory)
    identity = ['match_id', 'season', 'a_won']
    if not fresh[identity].equals(rows[identity]):
        raise ValueError('Expanded targets changed')
    base = pd.read_csv(previous / 'development_features.csv')
    if len(base) != len(rows):
        raise ValueError('Feature rows mismatch')
    from scripts.extended_history_features import build_extended_history_features
    from scripts.breakpoint_features import build_breakpoint_features
    from scripts.rank_trajectory_features import build_rank_trajectory_features
    print('Building extended histories, ranking trajectories and breakpoint summaries', flush=True)
    blocks = {'extended': build_extended_history_features(rows, inventory),
        'trajectory': build_rank_trajectory_features(rows, inventory),
        'breakpoint': build_breakpoint_features(rows, inventory)}
    if any(not b.index.equals(rows.index) for b in blocks.values()):
        raise ValueError('New feature identity mismatch')
    x = pd.concat([base, *blocks.values()], axis=1)
    if x.columns.duplicated().any() or not np.isfinite(x.to_numpy()).all():
        raise ValueError('Invalid extended feature matrix')
    groups = dict(base=old_registry['feature_groups']['core_age'] + old_registry['feature_groups']['multilevel'] +
        old_registry['feature_groups']['style'], all_previous=list(base), **{n: list(b) for n, b in blocks.items()})
    specs = candidate_specs(groups)
    old = pd.read_csv(previous / 'oof_predictions.csv')
    orows = rows[rows.season.isin(FOLDS)]
    if not orows[identity].reset_index(drop=True).equals(old[identity]):
        raise ValueError('Previous OOF identities changed')
    oof = {REFERENCE: old[REFERENCE].to_numpy()}
    reference_spec = dict(name=REFERENCE, family='previous_reference', method='previous', kind='previous')
    all_specs = specs + [reference_spec]
    registry = dict(round=6, seed=SEED, folds=list(FOLDS), features=list(x), feature_groups=groups,
        feature_metadata={n: b.attrs for n, b in blocks.items()}, candidates=all_specs,
        inventory_sha256=sha256(inventory_file), target_sha256=sha256(target_file),
        previous_artifact_sha256=audit['artifact_sha256'], accuracy_gate=ACCURACY_GATE, loss_guard=LOSS_GUARD,
        prior_round5_models_selection_eligible=False, final_test_evaluated=False, exploratory_after_2023=True)
    (out / 'search_registry.json').write_text(json.dumps(registry, indent=2) + '\n')
    x.to_csv(out / 'development_features.csv', index=False)
    cv, y = [], orows.a_won.to_numpy()
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
    ref = oof[REFERENCE]
    fs = {}
    for year in FOLDS:
        mask = orows.season.eq(year).to_numpy()
        fs[f'{year}_accuracy'], fs[f'{year}_log_loss'] = accuracy(y[mask], ref[mask]), mean_loss(y[mask], ref[mask])
    cv.append(dict(name=REFERENCE, family='previous_reference', accuracy=accuracy(y, ref), mean_log_loss=mean_loss(y, ref), **fs))
    pd.DataFrame(cv).to_csv(out / 'temporal_cv.csv', index=False)
    gate = choose_procedure(y, oof, all_specs, orows.season)
    pooled = select_accuracy(y, oof, all_specs, np.ones(len(y), dtype=bool))
    choice = pooled['choice'] if gate['procedure'] == 'adopt_accuracy_model' else REFERENCE
    frozen = dict(round=6, recommendation=choice, procedure_gate=gate, pooled_selection=pooled,
        history_lag_days=21, decision_threshold=.5, fit_end_season=2022,
        exploratory_after_2023=True, final_test_evaluated=False)
    (out / 'frozen_selection.json').write_text(json.dumps(frozen, indent=2) + '\n')
    print('Frozen procedure:', json.dumps(gate), flush=True)
    needed = set(pooled['selected_by_family'].values()) | set(pooled['components']) | {pooled['best_single']}
    by_name = {s['name']: s for s in specs}
    train, valid = temporal_masks(rows, 2023)
    models = {name: fit_model(by_name[name], x[train], rows[train]) for name in sorted(needed & set(by_name))}
    old_frozen = json.loads((previous / 'frozen_selection.json').read_text())
    aliases = {REFERENCE: REFERENCE, 'previous_round5_exploration': old_frozen['pooled_selection']['choice'],
        'previous_round5_best_loss': 'all_inputs_spline_C0.01', 'rank_reference': 'rank_reference'}
    bundle = dict(models=models, feature_names=list(x), frozen=frozen, previous_aliases=aliases,
        previous_bundle=joblib.load(previous / 'frozen_models.joblib'))
    p, vr = predict_round6(bundle, x[valid]), rows[valid]
    scores = []
    for name in p:
        score = score_probabilities(zip(vr.match_id, vr.a_won), zip(vr.match_id, p[name]))
        scores.append(dict(model=name, recommended=name == choice, **score,
            brier=float(np.mean((vr.a_won.to_numpy()-p[name].to_numpy())**2))))
    table = pd.DataFrame(scores).sort_values(['accuracy', 'mean_log_loss'], ascending=[False, True])
    table.to_csv(out / 'validation_comparison.csv', index=False)
    pd.concat([vr.reset_index(drop=True), p.reset_index(drop=True)], axis=1).to_csv(out / 'validation_predictions.csv', index=False)
    pd.concat([orows[identity].reset_index(drop=True), pd.DataFrame(oof)], axis=1).to_csv(out / 'oof_predictions.csv', index=False)
    references = [REFERENCE, 'previous_round5_exploration']
    intervals = {ref: {n: dict(accuracy=accuracy_interval(vr, p[ref].to_numpy(), p[n].to_numpy()),
        log_loss=cluster_interval(vr, p[ref].to_numpy(), p[n].to_numpy())) for n in p} for ref in references}
    summary = dict(round=6, recommendation=choice, new_candidates=len(specs), features=x.shape[1],
        training_rows=int(train.sum()), validation_matches=len(vr), comparisons=intervals,
        correct_required_for_70=int(np.ceil(.7*len(vr))), exploratory_after_2023=True, final_test_evaluated=False)
    (out / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    joblib.dump(bundle, out / 'frozen_models.joblib')
    print(table.to_string(index=False), flush=True)


if __name__ == '__main__':
    from scripts.train_round6 import main as canonical_main
    with threadpool_limits(limits=4):
        canonical_main()
