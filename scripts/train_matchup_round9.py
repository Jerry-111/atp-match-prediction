"""Three fixed observed-style matchup structures on the original full cohort."""
from __future__ import annotations

import argparse
from datetime import datetime,timezone
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from scripts.matchup_features import MatchupClassifier,candidate_specs
from scripts.metrics import score_probabilities
from scripts.train_models import ROOT,FOLDS,SEED,sha256,temporal_masks,mean_loss,cluster_interval,mirror_features
from scripts.train_round5 import REFERENCE,predict_round5,accuracy,accuracy_interval
from scripts.train_distillation_round7 import predict_distillation,validate_rows
from scripts.train_transfer_round6 import verify_tour_targets
from scripts.train_venue_round8 import select_single,choose_procedure

STUDENT='previous_round7_market_student'


def predict_matchup(bundle,features):
    required=bundle['feature_names']
    if features.empty or features.columns.duplicated().any() or not set(required).issubset(features):
        raise ValueError('Complete unique frozen historical columns required')
    x=features[required].astype(float)
    if not np.isfinite(x.to_numpy()).all():raise ValueError('Finite frozen inputs required')
    old=predict_round5(bundle['previous_bundle'],x)
    values={name:model.predict_proba(x) for name,model in bundle['models'].items()}
    if set(values)&set(bundle['previous_aliases']):raise ValueError('Cannot overwrite previous references')
    values.update({alias:old[name].to_numpy() for alias,name in bundle['previous_aliases'].items()})
    student=predict_distillation(bundle['student_bundle'],x)
    values[STUDENT]=student[bundle['student_name']].to_numpy()
    for name,p in values.items():
        if np.asarray(p).shape!=(len(x),) or not np.isfinite(p).all() or np.any((p<0)|(p>1)):
            raise ValueError(f'Invalid frozen probability: {name}')
    return pd.DataFrame(values,index=features.index)


def markdown(frame):
    def show(value):return f'{value:.6f}' if isinstance(value,(float,np.floating)) else str(value)
    return '\n'.join(['| '+' | '.join(frame.columns)+' |','| '+' | '.join(['---']*len(frame.columns))+' |']+
        ['| '+' | '.join(show(v) for v in row)+' |' for row in frame.itertuples(index=False,name=None)])


def write_report(out,summary,frozen,table,matched,contributions):
    selected=table.set_index('model').loc[frozen['pooled_selection']['choice']]
    recommended=table.set_index('model').loc[frozen['recommendation']]
    best=table.iloc[0]
    text=f'''# Round 9: fixed antisymmetric style matchups

This tests new interactions in the existing T−21 information, not a new data
source. All 2,588 2023 targets and the 0.5 threshold remain unchanged. 2023 has
been repeatedly inspected and these results are exploratory; 2024–2025 stay sealed.

The branch-local rule **{frozen['procedure_gate']['procedure']}**. Its early
choice `{frozen['procedure_gate']['early_selection']['choice']}` changed 2022
accuracy by {frozen['procedure_gate']['accuracy_improvement']:+.4%}, versus the
required +0.5 percentage points and loss increase ≤0.005. The retained/adopted
choice `{frozen['recommendation']}` scores {int(recommended.correct)}/2,588,
{recommended.accuracy:.3%}, log loss {recommended.mean_log_loss:.6f}.

The separately pooled pre-2023 choice **{frozen['pooled_selection']['choice']}**
scores **{int(selected.correct)}/2,588 = {selected.accuracy:.3%}**, loss
**{selected.mean_log_loss:.6f}**. The highest observed row `{best.model}` scores
{int(best.correct)}/2,588; it does not overwrite the frozen decision.
70% requires 1,812 correct. No threshold, subset or blend was selected on 2023.

## Full-cohort comparison

{markdown(table[['model','correct','accuracy','mean_log_loss','recommended']])}

## Matched structural tests

Positive accuracy change and negative loss change favor the candidate.

{markdown(matched)}

All three models use the same {summary['main_features']} old odd main-effect
columns, C=0.01 and three-year recency weighting. Shared adds 21 skew terms;
surface adds 63 strongly shrunk deviations. No learned player-ID vocabulary,
extra neural network, low-rank optimization or hyperparameter grid. The older
matrix already has a few handcrafted matchup terms; “no new interaction” does
not mean globally interaction-free. The 326 inputs are preserved exactly.

## Descriptive logit contributions

{markdown(contributions)}

These magnitudes describe fitted predictions, not causal effects of style.
The observed-style skew matrix has at most rank six by its dimension, without
an imposed low-rank factorization. Surface and shared columns partly overlap;
ridge yields a unique, shrunk decomposition. All centers/scales are fitted
only on each training fold; factors 0.2/0.1 are applied after RMS normalization.

Matched date-block intervals and model coefficients are saved in summary.json
and coefficients.csv. They condition on the fitted predictions and ignore
repeated search and some shared-player dependence. The primary Chen–Joachims
tennis experiment had no accuracy gain, so a null result here is meaningful;
this observed-feature adaptation is not a replication of that latent model.
R7's market-supervised student has a distinct training-information condition.

Reproduce: `.venv/bin/python -m scripts.train_matchup_round9`.
Inference only: append `--predict-features path/to/historical_features.csv`.
[Protocol](../../../docs/research/round9_matchup.md) ·
[Frozen choice](frozen_selection.json) · [Source registry](search_registry.json).
'''
    (out/'report_summary.md').write_text(text)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=ROOT/'results/round9/matchup')
    parser.add_argument('--predict-features',type=Path)
    args=parser.parse_args();out=args.output;out.mkdir(parents=True,exist_ok=True)
    if args.predict_features:
        features=pd.read_csv(args.predict_features)
        p=predict_matchup(joblib.load(out/'frozen_models.joblib'),features)
        if 'match_id' in features:p.insert(0,'match_id',features.match_id.to_numpy())
        p.to_csv(out/'inference_predictions.csv',index=False)
        return
    previous,student_dir=ROOT/'results/round5',ROOT/'results/round7/distillation'
    audit=json.loads((previous/'artifact_audit.json').read_text())
    for name in ('development_features.csv','frozen_models.joblib','oof_predictions.csv','search_registry.json'):
        if sha256(previous/name)!=audit['artifact_sha256'][name]:raise ValueError(f'Audited input changed: {name}')
    student_audit=json.loads((student_dir/'artifact_audit.json').read_text())
    for name in ('frozen_models.joblib','frozen_selection.json'):
        if sha256(student_dir/name)!=student_audit['artifact_sha256'][name]:raise ValueError('Audited student changed')
    target_path=ROOT/'results/round3/development_targets.csv'
    rows=pd.read_csv(target_path);validate_rows(rows)
    inventory_file=ROOT/'data/processed/mens_singles_2006_2025.csv'
    old_registry=json.loads((previous/'search_registry.json').read_text())
    if sha256(inventory_file)!=old_registry['inventory_sha256']:raise ValueError('Inventory changed')
    inventory=pd.read_csv(inventory_file,dtype={'a_won':'string'});inventory=inventory[inventory.season.le(2023)]
    verify_tour_targets(rows,inventory)
    x=pd.read_csv(previous/'development_features.csv')
    if x.shape!=(len(rows),326) or not np.isfinite(x.to_numpy()).all():raise ValueError('Expected original 326 finite features')
    specs=candidate_specs(x.columns)
    if any(len(s['main_columns'])!=188 for s in specs):raise ValueError('Expected identical 188 main effects')
    orows=rows[rows.season.isin(FOLDS)];y=orows.a_won.to_numpy()
    old=pd.read_csv(previous/'oof_predictions.csv');identity=['match_id','season','a_won']
    if not orows[identity].reset_index(drop=True).equals(old[identity]):raise ValueError('OOF identity changed')
    oof={REFERENCE:old[REFERENCE].to_numpy()}
    registry=dict(round=9,branch='antisymmetric_style_matchup',candidates=specs,selection_entries=4,
        fitted_structures=3,total_fit_instances=12,feature_names=list(x),folds=list(FOLDS),seed=SEED,
        feature_code_sha256=sha256(ROOT/'scripts/matchup_features.py'),runner_sha256=sha256(Path(__file__)),
        selector_code_sha256=sha256(ROOT/'scripts/train_venue_round8.py'),
        protocol_sha256=sha256(ROOT/'docs/research/round9_matchup.md'),target_sha256=sha256(target_path),
        inventory_sha256=sha256(inventory_file),previous_artifact_sha256=audit['artifact_sha256'],
        student_artifact_sha256={n:sha256(student_dir/n) for n in ('frozen_models.joblib','frozen_selection.json')},
        mixture_search=False,decision_threshold=.5,exploratory_after_2023=True,final_test_evaluated=False)
    (out/'search_registry.json').write_text(json.dumps(registry,indent=2)+'\n')
    x.to_csv(out/'development_features.csv',index=False)
    cv=[];population=[]
    for i,spec in enumerate(specs,1):
        p=pd.Series(index=orows.index,dtype=float)
        scores={}
        for year in FOLDS:
            train,valid=temporal_masks(rows,year)
            model=MatchupClassifier(spec).fit(x[train],rows[train])
            p.loc[rows.index[valid]]=model.predict_proba(x[valid])
            scores[f'{year}_accuracy']=accuracy(rows.a_won[valid],p.loc[rows.index[valid]])
            scores[f'{year}_log_loss']=mean_loss(rows.a_won[valid].to_numpy(),p.loc[rows.index[valid]].to_numpy())
            population.append(dict(model=spec['name'],fold=year,training_rows=int(train.sum()),validation_rows=int(valid.sum()),
                last_training_date=str(pd.to_datetime(rows.tourney_date[train]).max()),
                cutoff=str(pd.to_datetime(rows.tourney_date[valid]).min()-pd.Timedelta(days=21)),
                design_width=len(model.design_names_),optimizer_iterations=int(model.estimator_.n_iter_[0])))
        oof[spec['name']]=p.to_numpy()
        cv.append(dict(name=spec['name'],accuracy=accuracy(y,p),mean_log_loss=mean_loss(y,p.to_numpy()),**scores))
        print(f"[{i}/3] {spec['name']}: acc={cv[-1]['accuracy']:.6f} loss={cv[-1]['mean_log_loss']:.6f}",flush=True)
    scores={}
    for year in FOLDS:
        mask=orows.season.eq(year).to_numpy()
        scores[f'{year}_accuracy']=accuracy(y[mask],oof[REFERENCE][mask]);scores[f'{year}_log_loss']=mean_loss(y[mask],oof[REFERENCE][mask])
    cv.append(dict(name=REFERENCE,accuracy=accuracy(y,oof[REFERENCE]),mean_log_loss=mean_loss(y,oof[REFERENCE]),**scores))
    pd.DataFrame(cv).to_csv(out/'temporal_cv.csv',index=False)
    gate=choose_procedure(y,oof,orows.season)
    pooled=select_single(y,oof,np.ones(len(y),bool))
    choice=pooled['choice'] if gate['procedure']=='adopt_accuracy_model' else REFERENCE
    frozen=dict(round=9,branch='antisymmetric_style_matchup',recommendation=choice,procedure_gate=gate,
        pooled_selection=pooled,decision_threshold=.5,frozen_at_utc=datetime.now(timezone.utc).isoformat(),
        historical_lag_days=21,fit_end_season=2022,exploratory_after_2023=True,final_test_evaluated=False)
    (out/'frozen_selection.json').write_text(json.dumps(frozen,indent=2)+'\n');print('Frozen',json.dumps(frozen),flush=True)
    train,valid=temporal_masks(rows,2023);vr=rows[valid]
    models={spec['name']:MatchupClassifier(spec).fit(x[train],rows[train]) for spec in specs}
    for spec in specs:
        m=models[spec['name']]
        population.append(dict(model=spec['name'],fold=2023,training_rows=int(train.sum()),validation_rows=int(valid.sum()),
            last_training_date=str(pd.to_datetime(rows.tourney_date[train]).max()),
            cutoff=str(pd.to_datetime(rows.tourney_date[valid]).min()-pd.Timedelta(days=21)),
            design_width=len(m.design_names_),optimizer_iterations=int(m.estimator_.n_iter_[0])))
    old_frozen=json.loads((previous/'frozen_selection.json').read_text())
    student_frozen=json.loads((student_dir/'frozen_selection.json').read_text())
    bundle=dict(models=models,feature_names=list(x),frozen=frozen,previous_bundle=joblib.load(previous/'frozen_models.joblib'),
        previous_aliases={REFERENCE:REFERENCE,'previous_round5_exploration':old_frozen['pooled_selection']['choice'],
            'previous_round5_best_loss':'all_inputs_spline_C0.01','rank_reference':'rank_reference'},
        student_bundle=joblib.load(student_dir/'frozen_models.joblib'),student_name=student_frozen['recommendation'])
    p=predict_matchup(bundle,x[valid]);scores=[]
    for name in p:
        score=score_probabilities(zip(vr.match_id,vr.a_won),zip(vr.match_id,p[name]))
        scores.append(dict(model=name,recommended=name==choice,**score,brier=float(np.mean((vr.a_won-p[name])**2))))
    table=pd.DataFrame(scores).sort_values(['accuracy','mean_log_loss'],ascending=[False,True]);table.to_csv(out/'validation_comparison.csv',index=False)
    pd.concat([vr.reset_index(drop=True),p.reset_index(drop=True)],axis=1).to_csv(out/'validation_predictions.csv',index=False)
    pd.concat([orows[identity].reset_index(drop=True),pd.DataFrame(oof)],axis=1).to_csv(out/'oof_predictions.csv',index=False)
    pd.DataFrame(population).to_csv(out/'fit_populations.csv',index=False)
    matched=[];matched_intervals={}
    for a,b in [('shared','no_new_interaction'),('surface','no_new_interaction'),('surface','shared')]:
        a,b='matchup_'+a,'matchup_'+b
        matched.append(dict(candidate=a,control=b,oof_accuracy_change=accuracy(y,oof[a])-accuracy(y,oof[b]),
            oof_loss_change=mean_loss(y,oof[a])-mean_loss(y,oof[b]),
            validation_accuracy_change=accuracy(vr.a_won,p[a])-accuracy(vr.a_won,p[b]),
            validation_loss_change=mean_loss(vr.a_won.to_numpy(),p[a])-mean_loss(vr.a_won.to_numpy(),p[b])))
        matched_intervals[a+'_vs_'+b]=dict(accuracy=accuracy_interval(vr,p[b].to_numpy(),p[a].to_numpy()),
            log_loss=cluster_interval(vr,p[b].to_numpy(),p[a].to_numpy()))
    matched=pd.DataFrame(matched);matched.to_csv(out/'matched_controls.csv',index=False)
    coefficients=[];contributions=[]
    for name,m in models.items():
        for feature,beta in zip(m.design_names_,m.estimator_.coef_[0]):coefficients.append(dict(model=name,feature=feature,coefficient=float(beta)))
        parts=m.contributions(x[valid])
        contributions.append(dict(model=name,design_width=len(m.design_names_),main_mean_absolute_logit=float(parts.main.abs().mean()),
            shared_mean_absolute_logit=float(parts.shared.abs().mean()),surface_mean_absolute_logit=float(parts.surface.abs().mean()),
            coefficient_l2=float(np.linalg.norm(m.estimator_.coef_))))
    pd.DataFrame(coefficients).to_csv(out/'coefficients.csv',index=False)
    contributions=pd.DataFrame(contributions);contributions.to_csv(out/'logit_contributions.csv',index=False)
    summary=dict(round=9,branch='antisymmetric_style_matchup',recommendation=choice,pooled_choice=pooled['choice'],
        main_features=188,input_features=326,training_rows=int(train.sum()),validation_matches=len(vr),
        structures=3,fit_instances=12,matched_intervals=matched_intervals,
        comparisons_vs_r4={n:dict(accuracy=accuracy_interval(vr,p[REFERENCE].to_numpy(),p[n].to_numpy()),
            log_loss=cluster_interval(vr,p[REFERENCE].to_numpy(),p[n].to_numpy())) for n in p},
        correct_required_for_70=1812,exploratory_after_2023=True,final_test_evaluated=False)
    (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    joblib.dump(bundle,out/'frozen_models.joblib')
    reloaded=predict_matchup(joblib.load(out/'frozen_models.joblib'),x[valid])
    swapped=predict_matchup(bundle,mirror_features(x[valid]))
    old_linear=joblib.load(ROOT/'results/round8/venue/frozen_models.joblib')['models']['venue_base_linear']
    old_linear_p=old_linear.predict_proba(x[valid])
    old_oof=pd.read_csv(ROOT/'results/round8/venue/oof_predictions.csv')
    if not old_oof[identity].equals(orows[identity].reset_index(drop=True)):raise ValueError('Matched old linear identity mismatch')
    check=dict(reload_max_error=float(np.max(np.abs(p.to_numpy()-reloaded.to_numpy()))),
        swap_max_error=float(np.max(np.abs(p.to_numpy()+swapped.to_numpy()-1))),
        original326_csv_values_max_error=float(np.max(np.abs(x.to_numpy()-pd.read_csv(previous/'development_features.csv').to_numpy()))),
        prior_linear_validation_max_error=float(np.max(np.abs(p.matchup_no_new_interaction-old_linear_p))),
        prior_linear_oof_max_error=float(np.max(np.abs(oof['matchup_no_new_interaction']-old_oof.venue_base_linear))),
        validation_scoring_only_after_freeze=True)
    if check['reload_max_error']>1e-12 or check['swap_max_error']>1e-12:raise ValueError('Frozen inference check failed')
    (out/'self_check.json').write_text(json.dumps(check,indent=2)+'\n')
    write_report(out,summary,frozen,table,matched,contributions)
    print(table.to_string(index=False),flush=True)


if __name__=='__main__':
    from scripts.train_matchup_round9 import main as canonical_main
    with threadpool_limits(limits=4):canonical_main()
