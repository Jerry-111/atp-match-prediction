"""Five jointly optimized point models, with strictly chronological calibration."""
from __future__ import annotations

import argparse
from datetime import timedelta
import json
from pathlib import Path
import time

import joblib
import numpy as np
import pandas as pd
from scipy.special import expit, logit
from threadpoolctl import threadpool_limits

from scripts.advanced_features import _date
from scripts.expanded_training import build_expanded_targets
from scripts.joint_point_model import (LAG, attach_metadata, candidate_specs,
    generate_raw_predictions, load_point_history)
from scripts.metrics import score_probabilities
from scripts.multilevel_features import _season_date
from scripts.round5_learning import positive_calibration_slope
from scripts.train_models import ROOT, FOLDS, cluster_interval, mean_loss, sha256
from scripts.train_round5 import accuracy, accuracy_interval
from scripts.train_transfer_round6 import verify_tour_targets
from scripts.train_margin_round7 import choose_procedure as _choose_procedure
from scripts.train_margin_round7 import select_candidate as _select_candidate

REFERENCE = 'round4_reference'
R5_ACCURACY = 'round5_pooled_comparator'
R5_LOSS = 'round5_probability_comparator'
R2_COMPARATOR = 'round2_adjusted_linear_comparator'
MIXTURES = (.25, .5, .75)
PAIRS = (('serve_only_recent','joint_recent','opponent_adjustment'),
         ('joint_static','joint_recent','recency'),
         ('joint_recent','joint_surface','surface_preference'),
         ('joint_surface','joint_surface_event','event_intercept'))


def blend_registry(specs):
    return {f"{spec['name']}_r4blend{w:g}": dict(model=spec['name'], model_weight=w)
            for spec in specs for w in MIXTURES}


def add_blends(predictions, specs):
    result = {name: np.asarray(value, dtype=float) for name, value in predictions.items()}
    for name, mixture in blend_registry(specs).items():
        weight = mixture['model_weight']
        result[name] = weight*result[mixture['model']] + (1-weight)*result[REFERENCE]
    return result


def selectable(predictions, specs):
    names = [REFERENCE] + [s['name'] for s in specs] + list(blend_registry(specs))
    return {name: predictions[name] for name in names}


def select_candidate(y, predictions, mask, specs):
    return _select_candidate(y, selectable(predictions, specs), mask)


def choose_procedure(y, predictions, seasons, specs):
    gate = _choose_procedure(y, selectable(predictions, specs), seasons)
    if gate['procedure'] == 'adopt_margin':
        gate['procedure'] = 'adopt_joint_points'
    return gate


def calibrate(raw, rows, year):
    if year not in (*FOLDS, 2023) or not raw.index.equals(rows.index):
        raise ValueError('Aligned raw as-of probabilities and development fold required')
    for day, season in zip(rows.tourney_date,rows.season):
        if int(season) not in range(2009,2024):
            raise ValueError('Calibration data must be development-only')
        _season_date(_date(day),int(season))
    valid = rows.season.eq(year)
    if not valid.any():
        raise ValueError('No outer fold targets')
    cutoff = min(rows.loc[valid,'tourney_date'].map(_date))-timedelta(days=LAG)
    train = rows.season.between(2017, year-1) & rows.tourney_date.map(_date).le(cutoff)
    if not train.any() or rows.loc[train,'season'].max() > 2022:
        raise ValueError('Calibration is earlier-season development-only')
    slopes = {name: positive_calibration_slope(logit(np.clip(raw.loc[train,name],1e-8,1-1e-8)),
                rows.loc[train,'a_won'].to_numpy()) for name in raw}
    metadata = dict(year=year, rows=int(train.sum()), ids=rows.loc[train,'match_id'].tolist(),
        latest_calibration_date=max(rows.loc[train,'tourney_date'].map(_date)).isoformat(),
        first_target_date=min(rows.loc[valid,'tourney_date'].map(_date)).isoformat(),
        purge_cutoff=cutoff.isoformat(), slopes=slopes, calibrator='positive no-intercept slope on strictly past as-of predictions')
    return slopes, metadata


def calibrated_predictions(raw, slopes):
    return {name: expit(slope*logit(np.clip(raw[name].to_numpy(),1e-8,1-1e-8))) for name, slope in slopes.items()}


def score_table(rows, predictions, *, recommendation=None):
    output = []
    for name, p in predictions.items():
        p = np.asarray(p)
        score = score_probabilities(zip(rows.match_id,rows.a_won),zip(rows.match_id,p))
        output.append(dict(model=name,recommended=name==recommendation,**score,
            brier=float(np.mean((rows.a_won.to_numpy()-p)**2))))
    return pd.DataFrame(output).sort_values(['accuracy','mean_log_loss'],ascending=[False,True])


def matched_controls(rows, predictions, period):
    output = []
    for baseline, candidate, contrast in PAIRS:
        a,b = np.asarray(predictions[baseline]),np.asarray(predictions[candidate])
        output.append(dict(contrast=contrast,baseline=baseline,candidate=candidate,period=period,
            matches=len(rows),accuracy_gain=accuracy(rows.a_won,b)-accuracy(rows.a_won,a),
            log_loss_reduction=mean_loss(rows.a_won.to_numpy(),a)-mean_loss(rows.a_won.to_numpy(),b)))
    return output


def predict_joint_points(bundle, targets, references):
    if targets.empty or targets.match_id.duplicated().any() or references.match_id.duplicated().any():
        raise ValueError('Unique nonempty inference targets and references required')
    refs = references.set_index('match_id')
    if not set(targets.match_id).issubset(refs.index):
        raise ValueError('Missing prior-model probabilities for requested targets')
    predictions = {name: refs.loc[targets.match_id,name].to_numpy() for name in bundle['reference_names']}
    raw = pd.DataFrame(index=targets.index,columns=[s['name'] for s in bundle['specs']],dtype=float)
    cutoffs = targets.tourney_date.map(lambda d: (_date(d)-timedelta(days=LAG)).isoformat())
    for spec in bundle['specs']:
        name = spec['name']
        for cutoff in cutoffs.unique():
            if cutoff not in bundle['states'][name]:
                raise ValueError('This cutoff has no frozen state; rebuild the as-of state using the frozen protocol before inference')
            index = targets.index[cutoffs.eq(cutoff)]
            raw.loc[index,name] = bundle['states'][name][cutoff].predict_proba(targets.loc[index])
    predictions.update(calibrated_predictions(raw,bundle['slopes']))
    predictions = add_blends(predictions,bundle['specs'])
    predictions.update({f'raw_{name}':raw[name].to_numpy() for name in raw})
    if any(not np.isfinite(p).all() or np.any((p<0)|(p>1)) for p in predictions.values()):
        raise ValueError('Invalid inference probability')
    return pd.DataFrame(predictions,index=targets.index)


def _verify_identity(expected, actual):
    if actual.match_id.duplicated().any():
        raise ValueError('Duplicate prior target identity')
    aa = actual.set_index('match_id')
    bb = expected.set_index('match_id')
    if set(aa.index) != set(bb.index) or not aa.loc[bb.index,'a_won'].equals(bb.a_won):
        raise ValueError('Previous probability cohort/labels changed')
    return aa.loc[bb.index]


def load_references(rows, final=False):
    directory = ROOT/'results/round7/foundation'
    filename = 'validation_predictions.csv' if final else 'oof_predictions.csv'
    audit = json.loads((directory/'artifact_audit.json').read_text())
    if sha256(directory/filename) != audit['artifact_sha256'][filename]:
        raise ValueError('Previously audited probability file changed')
    old = _verify_identity(rows,pd.read_csv(directory/filename))
    refs = old[[REFERENCE,R5_ACCURACY,R5_LOSS]].copy()
    older = _verify_identity(rows,pd.read_csv(ROOT/'results/round2'/filename))
    refs[R2_COMPARATOR] = older['adjusted_linear_C0.003'].to_numpy()
    refs.index = rows.index
    return refs


def write_report(out):
    frozen = json.loads((out/'frozen_selection.json').read_text())
    summary = json.loads((out/'summary.json').read_text())
    table = pd.read_csv(out/'validation_comparison.csv').set_index('model')
    selected = table.loc[frozen['pooled_selection']['choice']]
    gate = frozen['procedure_gate']
    matched = pd.read_csv(out/'matched_controls.csv')
    lines = ['# Round 8: jointly fitted point skills','',
        'The same 2,588 matches, original labels and 0.5 threshold are retained. All 2023 results are exploratory; 2024–2025 remain sealed.', '',
        f"Branch recommendation: **{frozen['recommendation']}**. Early-selected 2022 accuracy gain: "
        f"**{100*gate['accuracy_improvement']:+.3f} percentage points** (requires +0.5), "
        f"loss change {gate['loss_increase']:+.6f} (allows at most +0.005).", '',
        f"Pooled-CV frozen comparison `{frozen['pooled_selection']['choice']}`: **{int(selected.correct)}/2,588 = "
        f"{selected.accuracy:.3%}**, log loss **{selected.mean_log_loss:.6f}**. "
        f"It needs {1812-int(selected.correct)} additional correct predictions to reach 70%.", '',
        'Five point-model structures jointly optimize all eligible past service observations at each T−21 cutoff. '
        'Fifteen fixed R4 mixtures add no separately fitted point models. Prior scales, history horizon and recency were fixed before fitting. '
        'Only earlier-season as-of predictions fit each positive calibration slope. Raw probabilities below are diagnostic, excluded from selection.', '',
        '| Model | Correct / 2,588 | Accuracy | Log loss | Role |', '| --- | ---: | ---: | ---: | --- |']
    for name,r in table.iterrows():
        role = 'raw diagnostic only' if name.startswith('raw_') else ('branch choice' if name==frozen['recommendation']
            else 'pooled frozen comparison' if name==frozen['pooled_selection']['choice'] else 'fixed control / prior comparator')
        lines.append(f'| `{name}` | {int(r.correct)} | {r.accuracy:.3%} | {r.mean_log_loss:.6f} | {role} |')
    lines += ['', '## Matched structural controls', '',
        '| Contrast | Pooled CV accuracy change / pp | Pooled CV loss reduction | 2023 accuracy change / pp | 2023 loss reduction |',
        '| --- | ---: | ---: | ---: | ---: |']
    for contrast in matched.contrast.unique():
        aa = matched[(matched.contrast==contrast)&(matched.period=='pooled')].iloc[0]
        bb = matched[(matched.contrast==contrast)&(matched.period=='2023')].iloc[0]
        lines.append(f'| {contrast} | {100*aa.accuracy_gain:+.3f} | {aa.log_loss_reduction:+.6f} | {100*bb.accuracy_gain:+.3f} | {bb.log_loss_reduction:+.6f} |')
    lines += ['', 'The old R2 adjusted-linear comparator has a different history/classifier/input system, so it is not a same-input method control. '
        'The nested structural comparisons above keep the history population and priors fixed except for the explicitly named term/weight. '
        'IID points, MAP plug-in uncertainty, standardized seven-point deciding-set tiebreaks, missing point records and reused validation remain limitations.', '',
        f"Source: {summary['source']['valid_history_matches']:,} valid historical matches / "
        f"{summary['source']['eligible_history_matches']:,} eligible inventory matches. Missing counts remove likelihood observations only, never targets. "
        'Each cutoff uses a four-year observation pool. Earlier 2023 points can update latent skills only once T−21 eligible; '
        '2023 never estimates calibration or chooses settings. Event identity includes source code, normalized name and surface to avoid known reused-code collisions.', '',
        'This is a sparse penalized-likelihood adaptation of Ingram (2019), not a claimed reproduction of the paper’s NUTS random-walk posterior. '
        'The published model itself was slightly worse than its Elo benchmark; no literature result proves 70% here.', '',
        'Reproduce: `.venv/bin/python -m scripts.train_joint_points_round8`. '
        'Reload: `.venv/bin/python -m scripts.train_joint_points_round8 --predict-targets results/round8/joint_points/inference_targets.csv`. '
        'Report-only: `.venv/bin/python -m scripts.train_joint_points_round8 --report-only`.', '',
        '[Protocol and primary evidence](../../../docs/research/round8_joint_points.md) · '
        '[Frozen choice](frozen_selection.json) · [Matched controls](matched_controls.csv) · [Synthetic timing](synthetic_smoke.json)', '']
    (out/'report_summary.md').write_text('\n'.join(lines))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=ROOT/'results/round8/joint_points')
    parser.add_argument('--predict-targets',type=Path)
    parser.add_argument('--reference-predictions',type=Path)
    parser.add_argument('--report-only',action='store_true')
    args = parser.parse_args()
    out = args.output
    out.mkdir(parents=True,exist_ok=True)
    if args.report_only:
        write_report(out); return
    if args.predict_targets:
        bundle = joblib.load(out/'frozen_models.joblib')
        targets = pd.read_csv(args.predict_targets)
        references = pd.read_csv(args.reference_predictions or out/'reference_predictions.csv')
        p = predict_joint_points(bundle,targets,references)
        pd.concat([targets[['match_id']],p],axis=1).to_csv(out/'inference_predictions.csv',index=False)
        return
    smoke = json.loads((out/'synthetic_smoke.json').read_text())
    if not smoke['synthetic_only'] or smoke['outcome_scores_computed'] or smoke['gradient_check_error'] > 2e-5:
        raise ValueError('Synthetic gradient/timing check must precede empirical fitting')
    inventory_path = ROOT/'data/processed/mens_singles_2006_2025.csv'
    previous_registry = json.loads((ROOT/'results/round5/search_registry.json').read_text())
    if sha256(inventory_path) != previous_registry['inventory_sha256']:
        raise ValueError('Pinned inventory changed')
    inventory = pd.read_csv(inventory_path,dtype={'a_won':'string'})
    inventory = inventory[inventory.season.le(2023)].copy()
    target_path = ROOT/'results/round3/development_targets.csv'
    rows = pd.read_csv(target_path)
    verify_tour_targets(rows,inventory)
    # Independently reconstruct preknown target metadata, not just its current hash.
    fresh = build_expanded_targets(pd.read_csv(ROOT/'data/processed/model_features.csv'),inventory)
    for key in ('match_id','season','player_a_id','player_b_id','surface','rank_a','rank_b','a_won'):
        if not np.array_equal(rows[key].to_numpy(),fresh[key].to_numpy()):
            raise ValueError(f'Cached target changed: {key}')
    if list(rows.tourney_date.map(_date)) != list(fresh.tourney_date.map(_date)):
        raise ValueError('Cached target dates changed')
    history,metadata,source = load_point_history(inventory)
    rows = attach_metadata(rows,metadata)
    rows = rows[rows.season.ge(2017)].copy()
    specs = candidate_specs()
    registry = dict(round=8,branch='joint_points',fit_candidates=5,selection_entries=21,
        specs=specs,blends=blend_registry(specs),history_lag_days=LAG,
        calibration_first_season=2017,decision_threshold=.5,source=source,
        inventory_sha256=sha256(inventory_path),target_sha256=sha256(target_path),
        protocol_sha256=sha256(ROOT/'docs/research/round8_joint_points.md'),
        source_code_sha256=sha256(Path(__file__)),engine_sha256=sha256(ROOT/'scripts/joint_point_model.py'),
        smoke_sha256=sha256(out/'synthetic_smoke.json'),final_test_evaluated=False)
    (out/'search_registry.json').write_text(json.dumps(registry,indent=2)+'\n')
    started = time.monotonic()
    def progress(index,total,cutoff,diagnostics):
        if index==1 or index % 10 == 0 or index==total:
            print(f'As-of {index}/{total}: {cutoff}, last-fit seconds={sum(d["fit_seconds"] for d in diagnostics):.3f}, elapsed={time.monotonic()-started:.1f}s',flush=True)
    development = rows[rows.season.le(2022)]
    raw,diagnostics,_ = generate_raw_predictions(development,history,specs,progress=progress)
    pd.concat([development[['match_id','season','a_won','tourney_date']],raw],axis=1).to_csv(out/'raw_development_predictions.csv',index=False)
    diagnostics.to_csv(out/'development_fit_diagnostics.csv',index=False)
    orows = rows[rows.season.isin(FOLDS)]
    refs = load_references(orows)
    oof = {name:refs[name].to_numpy() for name in refs}
    calibrated = pd.DataFrame(index=orows.index,columns=[s['name'] for s in specs],dtype=float)
    calibrations = {}
    for year in FOLDS:
        slopes,calibration = calibrate(raw,development,year)
        calibrations[str(year)] = calibration
        valid = orows.season.eq(year)
        prediction = calibrated_predictions(raw.loc[orows.index[valid]],slopes)
        for name,p in prediction.items(): calibrated.loc[valid,name] = p
    oof.update({name:calibrated[name].to_numpy() for name in calibrated})
    oof = add_blends(oof,specs)
    oof.update({f'raw_{name}':raw.loc[orows.index,name].to_numpy() for name in raw})
    gate = choose_procedure(orows.a_won.to_numpy(),oof,orows.season.to_numpy(),specs)
    pooled = select_candidate(orows.a_won.to_numpy(),oof,np.ones(len(orows),dtype=bool),specs)
    recommendation = pooled['choice'] if gate['procedure']=='adopt_joint_points' else REFERENCE
    frozen = dict(round=8,branch='joint_points',recommendation=recommendation,procedure_gate=gate,
        pooled_selection=pooled,history_lag_days=LAG,threshold=.5,fit_end_season=2022,
        latent_state_updates='Past observations up to query T-21; structural choices and calibration frozen',
        exploratory_after_2023=True,final_test_evaluated=False)
    (out/'frozen_selection.json').write_text(json.dumps(frozen,indent=2)+'\n')
    print('Frozen selection:',json.dumps(frozen),flush=True)
    cv = score_table(orows,oof)
    for year in FOLDS:
        mask = orows.season.eq(year).to_numpy()
        for index,row in cv.iterrows():
            cv.loc[index,f'{year}_accuracy'] = accuracy(orows.a_won.to_numpy()[mask],oof[row.model][mask])
            cv.loc[index,f'{year}_log_loss'] = mean_loss(orows.a_won.to_numpy()[mask],oof[row.model][mask])
    cv.to_csv(out/'temporal_cv.csv',index=False)
    pd.concat([orows[['match_id','season','a_won']].reset_index(drop=True),pd.DataFrame(oof)],axis=1).to_csv(out/'oof_predictions.csv',index=False)
    # The final calibrator is learned before computing any new 2023 probability.
    combined_raw = raw.reindex(rows.index)
    final_slopes,calibrations['2023'] = calibrate(combined_raw,rows,2023)
    (out/'calibration_metadata.json').write_text(json.dumps(calibrations,indent=2)+'\n')
    validation = rows[rows.season.eq(2023)]
    raw_valid,vd,states = generate_raw_predictions(validation,history,specs,keep_states=True,progress=progress)
    vd.to_csv(out/'validation_fit_diagnostics.csv',index=False)
    ref_valid = load_references(validation,final=True)
    reference_frame = pd.concat([validation[['match_id']].reset_index(drop=True),ref_valid.reset_index(drop=True)],axis=1)
    bundle = dict(specs=specs,states=states,slopes=final_slopes,frozen=frozen,
        reference_names=list(ref_valid),source_metadata=source,registry=registry)
    predictions = predict_joint_points(bundle,validation,reference_frame)
    comparison = score_table(validation,predictions,recommendation=recommendation)
    comparison.to_csv(out/'validation_comparison.csv',index=False)
    pd.concat([validation.reset_index(drop=True),predictions.reset_index(drop=True)],axis=1).to_csv(out/'validation_predictions.csv',index=False)
    reference_frame.to_csv(out/'reference_predictions.csv',index=False)
    validation[['match_id','season','tourney_date','player_a_id','player_b_id','surface','event_key','best_of']].to_csv(out/'inference_targets.csv',index=False)
    pd.concat([validation[['match_id']].reset_index(drop=True),raw_valid.reset_index(drop=True)],axis=1).to_csv(out/'raw_validation_predictions.csv',index=False)
    joblib.dump(bundle,out/'frozen_models.joblib')
    matched = matched_controls(orows,oof,'pooled')
    for year in FOLDS:
        mask = orows.season.eq(year).to_numpy()
        matched.extend(matched_controls(orows.loc[mask],{n:p[mask] for n,p in oof.items()},str(year)))
    matched.extend(matched_controls(validation,predictions,'2023'))
    pd.DataFrame(matched).to_csv(out/'matched_controls.csv',index=False)
    selected = predictions[pooled['choice']].to_numpy()
    comparisons = {name:dict(accuracy=accuracy_interval(validation,predictions[name].to_numpy(),selected),
        log_loss=cluster_interval(validation,predictions[name].to_numpy(),selected)) for name in (REFERENCE,R5_ACCURACY,R5_LOSS)}
    summary = dict(round=8,branch='joint_points',recommendation=recommendation,pooled_choice=pooled['choice'],
        fit_candidates=5,fixed_blends=15,selection_entries=21,validation_matches=len(validation),
        correct_required_for_70=1812,source=source,comparisons=comparisons,
        development_snapshots=len(diagnostics),validation_snapshots=len(vd),elapsed_seconds=time.monotonic()-started,
        max_scaled_gradient=float(max(diagnostics.max_scaled_gradient.max(),vd.max_scaled_gradient.max())),
        final_test_evaluated=False,exploratory_after_2023=True)
    (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    write_report(out)
    print(comparison.to_string(index=False),flush=True)


if __name__ == '__main__':
    from scripts.train_joint_points_round8 import main as canonical_main
    with threadpool_limits(limits=4):
        canonical_main()
