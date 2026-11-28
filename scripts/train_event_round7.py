"""Own earlier knockout-round information, a separate actual-prematch branch."""
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
    variants = {'base': base, 'context': base + groups['context'],
        'load': base + groups['context'] + groups['load'],
        'performance': base + groups['context'] + groups['performance'],
        'combined': base + groups['context'] + groups['load'] + groups['performance']}
    specs = []
    for group, columns in variants.items():
        specs.append(dict(name=f'event_{group}_linear', kind='logistic',
            method='linear', family=f'{group}_linear', columns=[n for n in columns if n.endswith('_diff')],
            params={'C': .01}, half_life=3.))
        spline = next(s for s in spline_specs(columns)
            if s['params']['C'] == .01 and s['params']['half_life'] == 3.)
        specs.append(dict(spline, name=f'event_{group}_spline', method='spline', family=f'{group}_spline'))
    columns = groups['all_previous'] + groups['context'] + groups['load'] + groups['performance']
    for depth in (4, 6):
        specs.append(dict(name=f'event_all_cat_d{depth}', kind='catboost', method='catboost',
            family='event_catboost', columns=columns, half_life=3.,
            params=dict(depth=depth, iterations=600, learning_rate=.03, l2_leaf_reg=20)))
    if len(specs) != 12 or any(len(s['columns']) != len(set(s['columns'])) for s in specs):
        raise ValueError('Expected 12 fixed configurations with distinct columns')
    return specs


def event_fallback(x, prediction, reference):
    available = x.event_any_prior.to_numpy()
    p, ref = np.asarray(prediction), np.asarray(reference)
    if not np.isin(available, (0, 1)).all() or p.shape != (len(x),) or ref.shape != p.shape:
        raise ValueError('Aligned probabilities and binary own-history indicator required')
    return np.where(available == 1, p, ref)


def predict_event(bundle, features):
    required = bundle['feature_names']
    if features.empty or features.columns.duplicated().any() or not set(required).issubset(features):
        raise ValueError('Missing, duplicate or empty frozen features')
    x = features[required].astype(float)
    if not np.isfinite(x.to_numpy()).all():
        raise ValueError('Frozen features must be finite')
    old = predict_round5(bundle['previous_bundle'], x)
    values = {n: event_fallback(x, m.predict_proba(x), old[REFERENCE].to_numpy())
        for n, m in bundle['models'].items()}
    values.update({alias: old[name].to_numpy() for alias, name in bundle['previous_aliases'].items()})
    selection = bundle['frozen']['pooled_selection']
    blended = event_fallback(x, selection_probability(selection, values), old[REFERENCE].to_numpy())
    if len(selection['components']) > 1:
        values['accuracy_grid_blend'] = blended
    for name, p in values.items():
        if np.asarray(p).shape != (len(x),) or not np.isfinite(p).all() or np.any((p < 0) | (p > 1)):
            raise ValueError(f'Invalid probabilities: {name}')
    return pd.DataFrame(values, index=features.index)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT / 'results/round7/event')
    parser.add_argument('--predict-features', type=Path)
    args = parser.parse_args()
    out, previous = args.output, ROOT / 'results/round5'
    out.mkdir(parents=True, exist_ok=True)
    if args.predict_features:
        features = pd.read_csv(args.predict_features)
        p = predict_event(joblib.load(out / 'frozen_models.joblib'), features)
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
    from scripts.event_state_features import build_event_state_features
    print('Building verified own earlier-round features', flush=True)
    event, lineage = build_event_state_features(rows)
    metadata = dict(event.attrs)
    event['event_round_rank_diff'] = event.event_round_order * base.log_rank_diff
    event['event_round_elo_diff'] = event.event_round_order * base.blend_elo_k32_diff
    x = pd.concat([base, event], axis=1)
    if not event.index.equals(rows.index) or x.columns.duplicated().any() or not np.isfinite(x.to_numpy()).all():
        raise ValueError('Invalid event feature matrix')
    context = ['event_'+n for n in ('prior_matches_diff', 'completed_matches_diff', 'noncomplete_matches_diff',
        'walkovers_diff', 'round_order', 'prior_matches_sum', 'completed_matches_sum', 'any_prior',
        'round_rank_diff', 'round_elo_diff')]
    workload = ['event_'+n for n in ('sets_diff', 'sets_lost_diff', 'last_sets_diff', 'last_games_diff',
        'log_games_diff', 'log_minutes_diff', 'missing_minutes_diff', 'missing_minutes_sum')]
    performance = [n for n in event if n not in context+workload]
    if (len(event.columns), len(performance)) != (27, 9):
        raise ValueError('Expected 27 fixed event fields')
    groups = dict(base=old_registry['feature_groups']['core_age'] + old_registry['feature_groups']['multilevel'] +
        old_registry['feature_groups']['style'], all_previous=list(base), context=context,
        load=workload, performance=performance)
    lineage.to_csv(out / 'feature_lineage.csv', index=False)
    specs = candidate_specs(groups)
    old = pd.read_csv(previous / 'oof_predictions.csv')
    orows = rows[rows.season.isin(FOLDS)]
    if not orows[identity].reset_index(drop=True).equals(old[identity]):
        raise ValueError('Previous OOF identities changed')
    oof = {REFERENCE: old[REFERENCE].to_numpy()}
    reference_spec = dict(name=REFERENCE, family='previous_reference', method='previous', kind='previous')
    all_specs = specs + [reference_spec]
    registry = dict(round=7, branch='own_earlier_rounds', seed=SEED, folds=list(FOLDS), features=list(x), feature_groups=groups,
        feature_metadata=metadata, candidates=all_specs,
        protocol_sha256=sha256(ROOT / 'docs/research/round7_event.md'),
        source_audit_sha256=sha256(ROOT / 'docs/research/round7_event_source_audit.md'),
        feature_code_sha256=sha256(ROOT / 'scripts/event_state_features.py'),
        runner_sha256=sha256(Path(__file__)),
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
            prediction = event_fallback(x[valid], model.predict_proba(x[valid]),
                old.loc[old.season.eq(year), REFERENCE].to_numpy())
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
    frozen = dict(round=7, branch='own_earlier_rounds', recommendation=choice, procedure_gate=gate, pooled_selection=pooled,
        historical_input_lag_days=21, event_input='Own completed earlier knockout rounds',
        original_T21_recommendation_unchanged=True, decision_threshold=.5, fit_end_season=2022,
        exploratory_after_2023=True, final_test_evaluated=False)
    (out / 'frozen_selection.json').write_text(json.dumps(frozen, indent=2) + '\n')
    print('Frozen procedure:', json.dumps(gate), flush=True)
    needed = {s['name'] for s in specs}  # Report all 12 predeclared matched controls, not just winners.
    by_name = {s['name']: s for s in specs}
    train, valid = temporal_masks(rows, 2023)
    models = {name: fit_model(by_name[name], x[train], rows[train]) for name in sorted(needed & set(by_name))}
    old_frozen = json.loads((previous / 'frozen_selection.json').read_text())
    aliases = {REFERENCE: REFERENCE, 'previous_round5_exploration': old_frozen['pooled_selection']['choice'],
        'previous_round5_best_loss': 'all_inputs_spline_C0.01', 'rank_reference': 'rank_reference'}
    bundle = dict(models=models, feature_names=list(x), frozen=frozen, previous_aliases=aliases,
        previous_bundle=joblib.load(previous / 'frozen_models.joblib'))
    p, vr = predict_event(bundle, x[valid]), rows[valid]
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
    summary = dict(round=7, branch='own_earlier_rounds', recommendation=choice, pooled_choice=pooled['choice'], new_candidates=len(specs), features=x.shape[1],
        original_T21_recommendation_unchanged=True,
        validation_with_prior=int(x.loc[valid, 'event_any_prior'].sum()),
        validation_fallback=int(x.loc[valid, 'event_any_prior'].eq(0).sum()),
        training_rows=int(train.sum()), validation_matches=len(vr), comparisons=intervals,
        correct_required_for_70=int(np.ceil(.7*len(vr))), exploratory_after_2023=True, final_test_evaluated=False)
    (out / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    joblib.dump(bundle, out / 'frozen_models.joblib')
    print(table.to_string(index=False), flush=True)


if __name__ == '__main__':
    from scripts.train_event_round7 import main as canonical_main
    with threadpool_limits(limits=4):
        canonical_main()
