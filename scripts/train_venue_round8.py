"""Ten preregistered tournament-environment controls; no mixture search."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from scripts.metrics import score_probabilities
from scripts.round2_learning import candidate_specs as spline_specs
from scripts.train_distillation_round7 import validate_rows
from scripts.train_models import ROOT, FOLDS, SEED, sha256, temporal_masks, mean_loss, cluster_interval, mirror_features
from scripts.train_round5 import REFERENCE, ACCURACY_GATE, LOSS_GUARD, accuracy, accuracy_interval, fit_model, predict_round5
from scripts.train_transfer_round6 import verify_tour_targets
from scripts.venue_features import build_venue_features


def candidate_specs(groups):
    base, surface = groups['base'], groups['surface']
    variants = dict(base=base, surface=base+surface,
        raw=base+surface+groups['raw'], adjusted=base+surface+groups['adjusted'],
        strength=base+surface+groups['strength'])
    specs = []
    for group, columns in variants.items():
        if len(columns) != len(set(columns)):
            raise ValueError('Repeated feature in matched control')
        specs.append(dict(name=f'venue_{group}_linear', kind='logistic', method='linear',
            family=group, columns=[c for c in columns if c.endswith('_diff')], params={'C': .01}, half_life=3.))
        spline = next(s for s in spline_specs(columns) if s['params']['C'] == .01 and s['params']['half_life'] == 3.)
        specs.append(dict(spline, name=f'venue_{group}_spline', method='spline', family=group))
    return specs


def select_single(y, predictions, mask):
    y, mask = np.asarray(y), np.asarray(mask, dtype=bool)
    if (y.ndim != 1 or not np.isin(y, (0, 1)).all() or mask.shape != y.shape or not mask.any()
            or REFERENCE not in predictions):
        raise ValueError('Aligned binary selection labels, mask and reference required')
    for name, p in predictions.items():
        p = np.asarray(p)
        if p.shape != y.shape or not np.isfinite(p).all() or np.any((p < 0) | (p > 1)):
            raise ValueError(f'Invalid selection probability: {name}')
    limit = mean_loss(y[mask], predictions[REFERENCE][mask]) + LOSS_GUARD
    eligible = [n for n,p in predictions.items() if mean_loss(y[mask], p[mask]) <= limit]
    best = min(eligible, key=lambda n: (-accuracy(y[mask], predictions[n][mask]),
               mean_loss(y[mask], predictions[n][mask]), n))
    return dict(choice=best, selected_accuracy=accuracy(y[mask], predictions[best][mask]),
        selected_log_loss=mean_loss(y[mask], predictions[best][mask]), loss_limit=limit,
        eligible_models=eligible, selection_rows=int(mask.sum()), mixture_search=False)


def choose_procedure(y, predictions, seasons):
    seasons = np.asarray(seasons)
    if seasons.shape != np.asarray(y).shape or not np.isin(seasons, FOLDS).all():
        raise ValueError('Only 2020..2022 OOF predictions may select models')
    early, held = np.isin(seasons, (2020, 2021)), seasons == 2022
    if not early.any() or not held.any():
        raise ValueError('Need early folds and separate 2022 gate')
    selection = select_single(y, predictions, early)
    y = np.asarray(y)
    ref, candidate = predictions[REFERENCE][held], predictions[selection['choice']][held]
    before, after = accuracy(y[held], ref), accuracy(y[held], candidate)
    ref_loss, loss = mean_loss(y[held], ref), mean_loss(y[held], candidate)
    passed = after-before >= ACCURACY_GATE-1e-12 and loss <= ref_loss+LOSS_GUARD+1e-12
    return dict(procedure='adopt_accuracy_model' if passed else 'retain_round4', early_selection=selection,
        reference=REFERENCE, reference_2022_accuracy=before, candidate_2022_accuracy=after,
        accuracy_improvement=after-before, required_accuracy_improvement=ACCURACY_GATE,
        reference_2022_loss=ref_loss, candidate_2022_loss=loss, allowed_loss_increase=LOSS_GUARD,
        selection_seasons=[2020, 2021], gate_season=2022)


def predict_venue(bundle, features):
    required = bundle['feature_names']
    if features.empty or features.columns.duplicated().any() or not set(required).issubset(features):
        raise ValueError('Missing, duplicate or empty frozen features')
    x = features[required].astype(float)
    if not np.isfinite(x.to_numpy()).all():
        raise ValueError('Frozen features must be finite')
    old = predict_round5(bundle['previous_bundle'], x)
    values = {name: model.predict_proba(x) for name,model in bundle['models'].items()}
    if set(values) & set(bundle['previous_aliases']):
        raise ValueError('New models may not overwrite old references')
    values.update({alias: old[name].to_numpy() for alias,name in bundle['previous_aliases'].items()})
    for name, p in values.items():
        if np.asarray(p).shape != (len(x),) or not np.isfinite(p).all() or np.any((p < 0) | (p > 1)):
            raise ValueError(f'Invalid prediction: {name}')
    return pd.DataFrame(values, index=features.index)


def write_report(out, summary, frozen, comparison, matched):
    def markdown(frame):
        def show(v):
            return f'{v:.6f}' if isinstance(v, (float,np.floating)) else str(v)
        return '\n'.join(['| '+' | '.join(map(str,frame.columns))+' |',
            '| '+' | '.join(['---']*len(frame.columns))+' |']+
            ['| '+' | '.join(show(v) for v in row)+' |' for row in frame.itertuples(index=False,name=None)])
    choice = comparison.set_index('model').loc[frozen['recommendation']]
    pooled = comparison.set_index('model').loc[frozen['pooled_selection']['choice']]
    best = comparison.iloc[0]
    def line(row):
        return f"{round(row.accuracy*2588):,}/2,588 ({row.accuracy:.4%}); log loss {row.mean_log_loss:.6f}"
    text = f'''# Round 8: historical tournament environment × style

The full 2,588-match 2023 cohort and threshold 0.5 are unchanged. This is
historical T−21 input, with no target-season statistics contributing to its
environment. All 2023 comparisons remain exploratory; 2024–2025 are sealed.

Branch-local frozen choice **{frozen['recommendation']}**: {line(choice)}.
The 2022 gate **{frozen['procedure_gate']['procedure']}** had accuracy change
{frozen['procedure_gate']['accuracy_improvement']:+.4%}; required +0.5 percentage
points and log-loss increase ≤0.005. The pooled pre-2023 choice
**{frozen['pooled_selection']['choice']}** scored {line(pooled)}.
Highest observed 2023 row **{best.model}**: {line(best)}. This observation does
not replace the frozen selection. Reaching 70% requires 1,812 correct.

Ten fixed fits compare base, surface context, raw venue context, adjusted
venue context, and a generic rank/Elo interaction control with the same linear
and spline learners. No mixture, threshold, prior, lookback or C search.
The original 326 fields are unchanged; 50 additional fields include 40 odd
interactions and ten even context/audit fields. Even audit fields are not
model inputs. Source rows have {summary['valid_serve_matches']:,} valid serve
and {summary['valid_ace_matches']:,} valid ace observations. In 2023,
{summary['validation_venue_serve_missing']:,} rows lack prior venue serve
observations and {summary['validation_venue_ace_missing']:,} lack ace observations;
all remain in the cohort. Zero environment features are not an exact R4
probability fallback after refitting.

## Full-cohort results

{markdown(comparison[['model','accuracy','mean_log_loss','recommended']])}

## Matched mechanism comparisons

Differences are candidate minus control; positive accuracy and negative loss
favor the candidate. Controls isolate surface versus tournament context and
opponent adjustment rather than crediting every change to added venue data.

{markdown(matched)}

Date-block paired intervals in summary.json are exploratory, without a
multiple-search correction. Repeated use of the 2022 gate is not independent
confirmation. ID/name/surface identifies a conservative tournament proxy,
not a verified physical venue: same-city moves, balls, weather, altitude and
player participation can still confound interpretation. Opponent adjustment
is an as-of adaptation of Sackmann's idea, not an exact paper replication.

[Registered protocol](../../../docs/research/round8_venue.md)
'''
    (out/'report_summary.md').write_text(text)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT/'results/round8/venue')
    parser.add_argument('--predict-features', type=Path)
    args = parser.parse_args()
    out, previous = args.output, ROOT/'results/round5'
    out.mkdir(parents=True, exist_ok=True)
    if args.predict_features:
        x = pd.read_csv(args.predict_features)
        result = predict_venue(joblib.load(out/'frozen_models.joblib'), x)
        if 'match_id' in x:
            result.insert(0, 'match_id', x.match_id.to_numpy())
        result.to_csv(out/'inference_predictions.csv', index=False)
        return
    audit = json.loads((previous/'artifact_audit.json').read_text())
    for name in ('development_features.csv','frozen_models.joblib','oof_predictions.csv','search_registry.json'):
        if sha256(previous/name) != audit['artifact_sha256'][name]:
            raise ValueError(f'Previous audited artifact changed: {name}')
    old_registry = json.loads((previous/'search_registry.json').read_text())
    inventory_file = ROOT/'data/processed/mens_singles_2006_2025.csv'
    target_file = ROOT/'data/processed/model_features.csv'
    if sha256(inventory_file) != old_registry['inventory_sha256'] or sha256(target_file) != old_registry['target_sha256']:
        raise ValueError('Pinned source hashes changed')
    rows = pd.read_csv(ROOT/'results/round3/development_targets.csv')
    validate_rows(rows)
    inventory = pd.read_csv(inventory_file, dtype={'a_won':'string'})
    inventory = inventory[inventory.season.le(2023)].copy()
    verify_tour_targets(rows, inventory)
    base = pd.read_csv(previous/'development_features.csv')
    if base.shape != (len(rows),326):
        raise ValueError('Expected unchanged prior 326-column matrix')
    print('Building past-season venue contexts', flush=True)
    venue = build_venue_features(rows, inventory, base)
    metadata = dict(venue.attrs)
    x = pd.concat([base,venue],axis=1)
    if not venue.index.equals(rows.index) or x.columns.duplicated().any() or not np.isfinite(x.to_numpy()).all():
        raise ValueError('Invalid venue matrix')
    groups = dict(base=list(base),**metadata['groups'])
    specs = candidate_specs(groups)
    old = pd.read_csv(previous/'oof_predictions.csv')
    orows = rows[rows.season.isin(FOLDS)]
    identity = ['match_id','season','a_won']
    if not orows[identity].reset_index(drop=True).equals(old[identity]):
        raise ValueError('Previous OOF identities changed')
    oof = {REFERENCE:old[REFERENCE].to_numpy()}
    registry = dict(round=8,branch='historical_venue_style',seed=SEED,folds=list(FOLDS),
        features=list(x),feature_groups=groups,feature_metadata=metadata,candidates=specs,
        selection_eligible=[s['name'] for s in specs]+[REFERENCE],mixture_search=False,
        protocol_sha256=sha256(ROOT/'docs/research/round8_venue.md'),
        feature_code_sha256=sha256(ROOT/'scripts/venue_features.py'),runner_sha256=sha256(Path(__file__)),
        inventory_sha256=sha256(inventory_file),target_sha256=sha256(target_file),
        expanded_target_sha256=sha256(ROOT/'results/round3/development_targets.csv'),
        previous_artifact_sha256=audit['artifact_sha256'],accuracy_gate=ACCURACY_GATE,loss_guard=LOSS_GUARD,
        prior_round5_models_selection_eligible=False,final_test_evaluated=False,exploratory_after_2023=True)
    (out/'search_registry.json').write_text(json.dumps(registry,indent=2)+'\n')
    x.to_csv(out/'development_features.csv',index=False)
    y, cv = orows.a_won.to_numpy(), []
    for i,spec in enumerate(specs,1):
        p = pd.Series(index=orows.index,dtype=float)
        scores = {}
        for year in FOLDS:
            train,valid = temporal_masks(rows,year)
            validate_rows(rows[train],fitting=True)
            model = fit_model(spec,x[train],rows[train])
            prediction = model.predict_proba(x[valid])
            p.loc[rows.index[valid]] = prediction
            scores[f'{year}_accuracy'] = accuracy(rows.a_won[valid],prediction)
            scores[f'{year}_log_loss'] = mean_loss(rows.a_won[valid].to_numpy(),prediction)
        oof[spec['name']] = p.to_numpy()
        cv.append(dict(name=spec['name'],family=spec['family'],method=spec['method'],
            accuracy=accuracy(y,p),mean_log_loss=mean_loss(y,p),**scores))
        pd.DataFrame(cv).to_csv(out/'temporal_cv.csv',index=False)
        print(f"[{i}/{len(specs)}] {spec['name']}: acc={cv[-1]['accuracy']:.5f} loss={cv[-1]['mean_log_loss']:.6f}",flush=True)
    ref = oof[REFERENCE]
    scores = {}
    for year in FOLDS:
        mask = orows.season.eq(year).to_numpy()
        scores[f'{year}_accuracy'],scores[f'{year}_log_loss'] = accuracy(y[mask],ref[mask]),mean_loss(y[mask],ref[mask])
    cv.append(dict(name=REFERENCE,family='previous_reference',method='previous',accuracy=accuracy(y,ref),
        mean_log_loss=mean_loss(y,ref),**scores))
    pd.DataFrame(cv).to_csv(out/'temporal_cv.csv',index=False)
    gate = choose_procedure(y,oof,orows.season)
    pooled = select_single(y,oof,np.ones(len(y),dtype=bool))
    choice = pooled['choice'] if gate['procedure']=='adopt_accuracy_model' else REFERENCE
    frozen = dict(round=8,branch='historical_venue_style',recommendation=choice,procedure_gate=gate,
        pooled_selection=pooled,frozen_at_utc=datetime.now(timezone.utc).isoformat(),decision_threshold=.5,
        historical_lag_days=21,environment_prior_seasons_only=True,fit_end_season=2022,
        exploratory_after_2023=True,final_test_evaluated=False)
    (out/'frozen_selection.json').write_text(json.dumps(frozen,indent=2)+'\n')
    print('Frozen:',json.dumps(frozen),flush=True)
    train,valid = temporal_masks(rows,2023)
    validate_rows(rows[train],fitting=True)
    models = {s['name']:fit_model(s,x[train],rows[train]) for s in specs}
    old_frozen = json.loads((previous/'frozen_selection.json').read_text())
    aliases = {REFERENCE:REFERENCE,'previous_round5_exploration':old_frozen['pooled_selection']['choice'],
        'previous_round5_best_loss':'all_inputs_spline_C0.01','rank_reference':'rank_reference'}
    bundle = dict(models=models,feature_names=list(x),frozen=frozen,previous_aliases=aliases,
        previous_bundle=joblib.load(previous/'frozen_models.joblib'))
    p,vr = predict_venue(bundle,x[valid]),rows[valid]
    scores = []
    for name in p:
        score = score_probabilities(zip(vr.match_id,vr.a_won),zip(vr.match_id,p[name]))
        scores.append(dict(model=name,recommended=name==choice,**score,
            brier=float(np.mean((vr.a_won.to_numpy()-p[name].to_numpy())**2))))
    comparison = pd.DataFrame(scores).sort_values(['accuracy','mean_log_loss'],ascending=[False,True])
    comparison.to_csv(out/'validation_comparison.csv',index=False)
    pd.concat([vr.reset_index(drop=True),p.reset_index(drop=True)],axis=1).to_csv(out/'validation_predictions.csv',index=False)
    pd.concat([orows[identity].reset_index(drop=True),pd.DataFrame(oof)],axis=1).to_csv(out/'oof_predictions.csv',index=False)
    matched = []
    for method in ('linear','spline'):
        for candidate,control in (('surface','base'),('raw','surface'),('adjusted','surface'),('adjusted','raw'),('adjusted','strength')):
            cn,bn = f'venue_{candidate}_{method}',f'venue_{control}_{method}'
            matched.append(dict(candidate=cn,control=bn,
                oof_accuracy_change=accuracy(y,oof[cn])-accuracy(y,oof[bn]),
                oof_loss_change=mean_loss(y,oof[cn])-mean_loss(y,oof[bn]),
                validation_accuracy_change=accuracy(vr.a_won,p[cn])-accuracy(vr.a_won,p[bn]),
                validation_loss_change=mean_loss(vr.a_won.to_numpy(),p[cn])-mean_loss(vr.a_won.to_numpy(),p[bn])))
    matched = pd.DataFrame(matched)
    matched.to_csv(out/'matched_controls.csv',index=False)
    intervals = {r:{n:dict(accuracy=accuracy_interval(vr,p[r].to_numpy(),p[n].to_numpy()),
        log_loss=cluster_interval(vr,p[r].to_numpy(),p[n].to_numpy())) for n in p}
        for r in (REFERENCE,'previous_round5_exploration')}
    coverage = venue[['venue_missing_serve','venue_missing_ace','venue_exposure_serve','venue_exposure_ace']].groupby(rows.season).agg(['mean','sum'])
    coverage.to_csv(out/'coverage_by_season.csv')
    summary = dict(round=8,branch='historical_venue_style',recommendation=choice,pooled_choice=pooled['choice'],
        new_candidates=len(specs),features=x.shape[1],training_rows=int(train.sum()),validation_matches=len(vr),
        validation_venue_serve_missing=int(venue.loc[valid,'venue_missing_serve'].sum()),
        validation_venue_ace_missing=int(venue.loc[valid,'venue_missing_ace'].sum()),
        valid_serve_matches=metadata['valid_serve_matches'],valid_ace_matches=metadata['valid_ace_matches'],
        comparisons=intervals,correct_required_for_70=1812,exploratory_after_2023=True,final_test_evaluated=False)
    (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    joblib.dump(bundle,out/'frozen_models.joblib')
    reloaded = predict_venue(joblib.load(out/'frozen_models.joblib'),x[valid])
    swapped = predict_venue(bundle,mirror_features(x[valid]))
    check = dict(reload_max_error=float(np.max(np.abs(p.to_numpy()-reloaded.to_numpy()))),
        swap_max_error=float(np.max(np.abs(p.to_numpy()+swapped.to_numpy()-1))),
        base_spline_prior_max_error=float(np.max(np.abs(p.venue_base_spline-p.previous_round5_best_loss))),
        old_features_max_error=float(np.max(np.abs(x[list(base)].to_numpy()-base.to_numpy()))),
        feature_sha256=sha256(out/'development_features.csv'),validation_scoring_only_after_freeze=True)
    if check['reload_max_error'] > 1e-12 or check['swap_max_error'] > 1e-12 or check['base_spline_prior_max_error'] > 1e-10:
        raise ValueError(f'Frozen prediction self-check failed: {check}')
    (out/'self_check.json').write_text(json.dumps(check,indent=2)+'\n')
    write_report(out,summary,frozen,comparison,matched)
    print(comparison.to_string(index=False),flush=True)


if __name__ == '__main__':
    from scripts.train_venue_round8 import main as canonical_main
    with threadpool_limits(limits=4):
        canonical_main()
