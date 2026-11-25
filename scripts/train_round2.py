"""Round-two exploratory search; preserves round one and the final holdout.

Requires the reproducible round-one OOF artifacts. New candidate selection is
restricted to training-period folds; 2023 is explicitly reused exploration.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.special import expit
from threadpoolctl import threadpool_limits

from scripts.train_models import (BASE, FOLDS, ROOT, SEED, SymmetricClassifier,
    cluster_interval, fit_blend, mean_loss, sha256, temporal_masks)
from scripts.metrics import score_probabilities

GATE_IMPROVEMENT = .001


def candidate_specs(x, groups):
    specs = []
    old_odd = [c for c in groups['round1'] if c.endswith('_diff')]
    dynamic = [c for c in groups['dynamic'] if c.endswith('_diff')]
    adjusted = [c for c in groups['adjusted_points'] if c.endswith('_diff')]
    all_odd = [c for c in x if c.endswith('_diff')]
    compact = [c for c in groups['round1'] if c in BASE or
        c in ('elo_k32_diff', 'blend_elo_k32_diff', 'elo_k32_trend90_diff', 'overall_inactivity_log_diff') or
        ('points_' in c and 'rate_diff' in c) or
        ('form_90_' in c and 'diff' in c)] + dynamic + adjusted
    linear_groups = {'dynamic_linear': old_odd+dynamic,
        'adjusted_linear': old_odd+adjusted, 'fusion_linear': all_odd,
        'compact_linear': list(dict.fromkeys(compact))}
    def add(name, kind, cols, params, family, **extra):
        specs.append(dict(name=name, kind=kind, columns=list(cols), params=params, family=family, **extra))
    for family, columns in linear_groups.items():
        for c in [.003, .01, .03, .1]:
            add(f'{family}_C{c:g}', 'logistic', columns, {'C': c}, family)
    for half in (3., 5.):
        for c in (.01, .03):
            add(f'fusion_recent_C{c:g}_h{half:g}', 'logistic', all_odd, {'C': c},
                'fusion_linear', half_life=half)
    compact_tree = compact + [c for c in x if c.startswith('surface_is_') or c=='log_rank_mean']
    compact_tree = list(dict.fromkeys(compact_tree))
    for label, columns in [('compact', compact_tree), ('full', list(x))]:
        for depth, iterations, l2 in [(3, 600, 10), (4, 700, 20)]:
            add(f'cat2_{label}_d{depth}', 'catboost', columns,
                dict(depth=depth, iterations=iterations, learning_rate=.025, l2_leaf_reg=l2), 'fusion_catboost')
        for leaves, l2 in [(7, 20.), (15, 50.)]:
            add(f'hist2_{label}_l{leaves}', 'histboost', columns,
                dict(max_iter=300, max_leaf_nodes=leaves, learning_rate=.03,
                     l2_regularization=l2, min_samples_leaf=100), 'fusion_histboost')
    from scripts.round2_learning import candidate_specs as spline_specs
    specs.extend(spline_specs(list(x)))
    for col in ('glicko_rating_diff','glicko_adjusted_diff','pooled_glicko_adjusted_diff'):
        if col in x:
            add('raw_'+col, 'raw', [col], {}, 'dynamic_raw')
    return specs


def make_model(spec):
    if spec['kind'] in ('logistic', 'catboost', 'histboost'):
        return SymmetricClassifier(spec)
    from scripts.round2_learning import make_model as make_spline
    return make_spline(spec)


def select_families(specs, oof, y, mask):
    families = sorted(set(s['family'] for s in specs))
    return {family: min((s['name'] for s in specs if s['family']==family),
        key=lambda n: (mean_loss(y[mask], oof[n][mask]), n)) for family in families}


def choose_procedure(y, oof, specs, seasons, old_gate):
    """Use 2020--21 for hyperparameters/weights and 2022 for the upgrade gate."""
    early = np.asarray(seasons) < 2022
    late = np.asarray(seasons) == 2022
    selected = select_families(specs, oof, y, early)
    names = list(selected.values())
    best = min(names, key=lambda n: mean_loss(y[early], oof[n][early]))
    weights = fit_blend(y[early], [oof[n][early] for n in names])
    blend = np.column_stack([oof[n][late] for n in names]) @ weights
    loss_old = mean_loss(y[late], old_gate[late])
    loss_single = mean_loss(y[late], oof[best][late])
    loss_blend = mean_loss(y[late], blend)
    candidate = 'new_blend' if loss_blend < loss_single else 'new_single'
    adopted = candidate if min(loss_single, loss_blend) <= loss_old-GATE_IMPROVEMENT else 'retain_round1'
    return dict(procedure=adopted, threshold=GATE_IMPROVEMENT,
        reference_2022_loss=loss_old, new_single_2022_loss=loss_single, new_blend_2022_loss=loss_blend,
        early_best=best, early_selected=selected, early_weights=weights.tolist(),
        selection_period='2020--2021', gate_period=2022)


def predict_round2(bundle, features):
    """Predict all frozen comparisons from a prepared numeric feature matrix."""
    required = bundle['feature_names']
    if features.empty or features.columns.duplicated().any():
        raise ValueError('Prediction matrix must be nonempty with unique columns')
    missing = set(required) - set(features)
    if missing:
        raise ValueError(f'Missing frozen features: {sorted(missing)}')
    x = features[required].astype(float)
    if not np.isfinite(x.to_numpy()).all():
        raise ValueError('Prediction features must be finite')
    probabilities = {n: m.predict_proba(x) for n, m in bundle['models'].items()}
    for n, column in bundle['raw_columns'].items():
        probabilities[n] = expit(np.log(10)/400*x[column].to_numpy())
    for name, meta in bundle['blends'].items():
        weights = np.asarray(meta['weights'])
        if np.any(weights < 0) or not np.isclose(weights.sum(), 1):
            raise ValueError('Invalid convex weights')
        probabilities[name] = np.column_stack([probabilities[n] for n in meta['components']]) @ weights
    return pd.DataFrame(probabilities, index=features.index)


def load_round1(rows, directory):
    registry = json.loads((directory/'search_registry.json').read_text())
    if registry['target_sha256'] != sha256(ROOT/'data/processed/model_features.csv'):
        raise ValueError('Round-one target hash does not match')
    if registry['inventory_sha256'] != sha256(ROOT/'data/processed/mens_singles_2006_2025.csv'):
        raise ValueError('Round-one inventory hash does not match')
    if registry['folds'] != list(FOLDS):
        raise ValueError('Round-one folds do not match')
    cached = pd.read_csv(directory/'oof_predictions.csv')
    expected = rows.loc[rows.season.isin(FOLDS), ['match_id','season','a_won']].reset_index(drop=True)
    if not cached[['match_id','season','a_won']].equals(expected):
        raise ValueError('Round-one OOF row identity does not match')
    relevant = {'advanced_linear', 'catboost', 'histboost'}
    specs = [dict(s, name='r1__'+s['name'], family='r1__'+s['family'])
             for s in registry['candidates'] if s['family'] in relevant]
    specs += [dict(name=f'r1__raw_elo_k{k}_diff', kind='raw', columns=[f'elo_k{k}_diff'],
                   params={}, family='r1__raw_elo') for k in (16,32,64)]
    oof = {s['name']: cached[s['name'][4:]].to_numpy() for s in specs}
    return specs, oof, json.loads((directory/'frozen_selection.json').read_text())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT/'results/round2')
    parser.add_argument('--round1', type=Path, default=ROOT/'results/modeling')
    parser.add_argument('--predict-features', type=Path,
                        help='Predict from a prepared feature CSV with the trusted frozen bundle; do not fit')
    args = parser.parse_args()
    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    if args.predict_features is not None:
        bundle = joblib.load(out/'frozen_models.joblib')
        features = pd.read_csv(args.predict_features)
        predictions = predict_round2(bundle,features)
        if 'match_id' in features:
            predictions.insert(0,'match_id',features.match_id.to_numpy())
        predictions.to_csv(out/'inference_predictions.csv',index=False)
        print(f'Predicted {len(predictions)} rows without fitting; wrote {out}/inference_predictions.csv')
        return
    rows = pd.read_csv(ROOT/'data/processed/model_features.csv')
    if not rows.season.between(2016,2023).all() or rows.match_id.duplicated().any():
        raise ValueError('Expected unique development targets only')
    old_specs, old_oof, old_frozen = load_round1(rows, args.round1)
    inventory = pd.read_csv(ROOT/'data/processed/mens_singles_2006_2025.csv', dtype={'a_won':'string'})
    history = inventory[inventory.season.le(2023)].copy()
    del inventory
    from scripts.advanced_features import build_advanced_features
    from scripts.serve_features import build_serve_features
    from scripts.dynamic_features import build_dynamic_features
    from scripts.adjusted_point_features import build_adjusted_point_features
    print('Building round-two past-only feature matrix', flush=True)
    old_x = pd.concat([rows[BASE], build_advanced_features(rows,history), build_serve_features(rows,history)],axis=1)
    dynamic = build_dynamic_features(rows,history)
    adjusted = build_adjusted_point_features(rows,history)
    x = pd.concat([old_x,dynamic,adjusted],axis=1)
    if x.columns.duplicated().any() or not np.isfinite(x.to_numpy()).all():
        raise ValueError('Numeric features must be finite and unique')
    groups = dict(round1=list(old_x), dynamic=list(dynamic), adjusted_points=list(adjusted))
    specs = candidate_specs(x,groups)
    registry = dict(round=2, exploratory_after_2023=True, seed=SEED, folds=list(FOLDS),
        gate_improvement=GATE_IMPROVEMENT, feature_groups=groups, candidates=specs,
        reused_candidates=old_specs, dynamic_metadata=dynamic.attrs, adjusted_metadata=adjusted.attrs,
        input_hashes={p.name:sha256(p) for p in [ROOT/'data/processed/model_features.csv', ROOT/'data/processed/mens_singles_2006_2025.csv']},
        final_test_evaluated=False)
    (out/'search_registry.json').write_text(json.dumps(registry,indent=2)+'\n')
    x.to_csv(out/'development_features.csv',index=False)
    oof_mask = rows.season.isin(FOLDS)
    oof_rows = rows[oof_mask]
    y = oof_rows.a_won.to_numpy()
    oof, cv = dict(old_oof), []
    for spec in old_specs:
        p = oof[spec['name']]
        cv.append(dict(name=spec['name'],family=spec['family'],reused=True,
            mean_log_loss=mean_loss(y,p), **{str(yr):mean_loss(y[oof_rows.season.eq(yr)],p[oof_rows.season.eq(yr)]) for yr in FOLDS}))
    for number,spec in enumerate(specs):
        p = pd.Series(index=oof_rows.index,dtype=float)
        folds = {}
        for year in FOLDS:
            train,valid = temporal_masks(rows,year)
            if spec['kind']=='raw':
                p.loc[rows.index[valid]] = expit(np.log(10)/400*x.loc[valid,spec['columns'][0]].to_numpy())
            else:
                model = make_model(spec).fit(x[train], rows.a_won[train], rows.season[train])
                p.loc[rows.index[valid]] = model.predict_proba(x[valid])
            folds[str(year)] = mean_loss(rows.a_won[valid].to_numpy(),p.loc[rows.index[valid]].to_numpy())
        values = p.to_numpy()
        oof[spec['name']] = values
        cv.append(dict(name=spec['name'],family=spec['family'],reused=False,
                       mean_log_loss=mean_loss(y,values),**folds))
        print(f"[{number+1}/{len(specs)}] {spec['name']} {cv[-1]['mean_log_loss']:.6f}",flush=True)
        pd.DataFrame(cv).to_csv(out/'temporal_cv.csv',index=False)
    all_specs = old_specs+specs
    # First-round gate predictions use only early-selected components/weights.
    old_gate = np.column_stack([oof['r1__'+n] for n in old_frozen['gate_components']]) @ old_frozen['gate_weights']
    decision = choose_procedure(y,oof,all_specs,oof_rows.season,old_gate)
    selected = select_families(all_specs,oof,y,np.ones(len(y),dtype=bool))
    components = list(selected.values())
    weights = fit_blend(y,[oof[n] for n in components])
    best = min(components,key=lambda n:mean_loss(y,oof[n]))
    choice = {'new_blend':'round2_ensemble','new_single':best,'retain_round1':'round1_ensemble'}[decision['procedure']]
    frozen = dict(round=2,recommendation=choice,procedure_gate=decision,
        selected_by_family=selected,best_single=best,ensemble_components=components,
        ensemble_weights=weights.tolist(),exploratory_after_2023=True,final_test_evaluated=False,
        fit_end_season=2022,history_lag_days=21)
    (out/'frozen_selection.json').write_text(json.dumps(frozen,indent=2)+'\n')
    print('Frozen follow-up procedure:',choice,decision,flush=True)
    # Only now score new 2023 predictions. Prior 2023 exposure is explicitly logged.
    train,valid=temporal_masks(rows,2023)
    by_name={s['name']:s for s in all_specs}
    names=set(selected.values())|{'r1__'+n for n in old_frozen['ensemble_components']}
    names.add('r1__'+old_frozen['best_single'])
    original = joblib.load(args.round1/'frozen_models.joblib')
    models,raw_columns={},{}
    for name in sorted(names):
        spec=by_name[name]
        if spec['kind']=='raw':
            raw_columns[name]=spec['columns'][0]
        elif name.startswith('r1__') and name[4:] in original['models']:
            models[name]=original['models'][name[4:]]
        else:
            models[name]=make_model(spec).fit(x[train],rows.a_won[train],rows.season[train])
    # Fixed ranking reference from the first run.
    models['rank_reference']=original['models'][old_frozen['selected_by_family']['lr_000']]
    bundle=dict(models=models,raw_columns=raw_columns,feature_names=list(x),frozen=frozen,
        blends={'round1_ensemble':dict(components=['r1__'+n for n in old_frozen['ensemble_components']],weights=old_frozen['ensemble_weights']),
                'round2_ensemble':dict(components=components,weights=weights.tolist())})
    p=predict_round2(bundle,x[valid])
    vr=rows[valid]
    table=[]
    for name in p:
        scores=score_probabilities(zip(vr.match_id,vr.a_won),zip(vr.match_id,p[name]))
        table.append(dict(model=name,recommended=name==choice,**scores,
                          brier=float(np.mean((vr.a_won.to_numpy()-p[name].to_numpy())**2))))
    result=pd.DataFrame(table).sort_values('mean_log_loss')
    result.to_csv(out/'validation_comparison.csv',index=False)
    pd.concat([vr.reset_index(drop=True),p.reset_index(drop=True)],axis=1).to_csv(out/'validation_predictions.csv',index=False)
    pd.concat([oof_rows[['match_id','season','a_won']].reset_index(drop=True),pd.DataFrame(oof)],axis=1).to_csv(out/'oof_predictions.csv',index=False)
    intervals={n:cluster_interval(vr,p['round1_ensemble'].to_numpy(),p[n].to_numpy()) for n in p}
    summary=dict(recommendation=choice,comparisons_vs_round1=intervals,final_test_evaluated=False,
                 exploratory_after_2023=True,validation_matches=len(vr),new_candidates=len(specs),reused_candidates=len(old_specs))
    (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    joblib.dump(bundle,out/'frozen_models.joblib')
    print(result.to_string(index=False),flush=True)


if __name__=='__main__':
    from scripts.train_round2 import main as canonical_main
    with threadpool_limits(limits=4):
        canonical_main()
