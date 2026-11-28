"""Two full-data supervised neural recipes with chronological inner stopping."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import platform
import subprocess
import sys
import time

import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from scripts.advanced_features import _date
from scripts.expanded_training import build_expanded_targets
from scripts.tabular_neural import (UPSTREAM_REVISION, candidate_specs, configure_device,
                                    fit_ensemble, predict_ensemble)
from scripts.train_models import ROOT,FOLDS,sha256,temporal_masks,mean_loss,cluster_interval
from scripts.train_round5 import accuracy,accuracy_interval
from scripts.train_transfer_round6 import verify_tour_targets
from scripts.train_joint_points_round8 import (REFERENCE,R5_ACCURACY,R5_LOSS,score_table,
                                              _verify_identity)
from scripts.train_margin_round7 import select_candidate as _select_candidate
from scripts.train_margin_round7 import choose_procedure as _choose_procedure

MIXTURES=(.25,.5,.75)
REFERENCES=(REFERENCE,R5_ACCURACY,R5_LOSS)


def blend_registry(specs):
    return {f"{s['name']}_r4blend{w:g}":dict(model=s['name'],model_weight=w)
            for s in specs for w in MIXTURES}


def add_blends(predictions,specs):
    result={n:np.asarray(p,dtype=float) for n,p in predictions.items()}
    for name,blend in blend_registry(specs).items():
        w=blend['model_weight']
        result[name]=w*result[blend['model']]+(1-w)*result[REFERENCE]
    return result


def selectable(predictions,specs):
    return {n:predictions[n] for n in [REFERENCE]+[s['name'] for s in specs]+list(blend_registry(specs))}


def select_candidate(y,predictions,mask,specs):
    return _select_candidate(y,selectable(predictions,specs),mask)


def choose_procedure(y,predictions,seasons,specs):
    gate=_choose_procedure(y,selectable(predictions,specs),seasons)
    if gate['procedure']=='adopt_margin':gate['procedure']='adopt_neural'
    return gate


def load_references(rows,final=False):
    directory=ROOT/'results/round7/foundation'
    name='validation_predictions.csv' if final else 'oof_predictions.csv'
    audit=json.loads((directory/'artifact_audit.json').read_text())
    if sha256(directory/name)!=audit['artifact_sha256'][name]:
        raise ValueError('Audited reference probabilities changed')
    old=_verify_identity(rows,pd.read_csv(directory/name))
    result=old[list(REFERENCES)].copy();result.index=rows.index
    return result


def verify_targets(rows,fresh):
    for key in ('match_id','season','player_a_id','player_b_id','surface','rank_a','rank_b','a_won'):
        if not np.array_equal(rows[key].to_numpy(),fresh[key].to_numpy()):
            raise ValueError(f'Cached target changed: {key}')
    if list(rows.tourney_date.map(_date))!=list(fresh.tourney_date.map(_date)):
        raise ValueError('Cached target dates changed')


def predict_neural(bundle,features,references,*,device='cpu',batch_size=1024):
    if features.empty or features.match_id.duplicated().any() or references.match_id.duplicated().any():
        raise ValueError('Unique nonempty inference IDs required')
    refs=references.set_index('match_id')
    if not set(features.match_id).issubset(refs.index):
        raise ValueError('Missing reference probability for requested target')
    p={name:refs.loc[features.match_id,name].to_numpy() for name in REFERENCES}
    x=features[bundle['columns']]
    for spec in bundle['specs']:
        p[spec['name']]=predict_ensemble(bundle['models'][spec['name']],x,
                                       device=device,batch_size=batch_size)
    result=pd.DataFrame(add_blends(p,bundle['specs']),index=features.index)
    if not np.isfinite(result.to_numpy()).all() or ((result<0)|(result>1)).any().any():
        raise ValueError('Invalid inference probabilities')
    return result


def matched_controls(rows,predictions,period):
    y=rows.a_won.to_numpy();a=np.asarray(predictions['plain_mlp']);b=np.asarray(predictions['realmlp_td_s'])
    return dict(period=period,baseline='plain_mlp',candidate='realmlp_td_s',matches=len(rows),
        accuracy_gain=accuracy(y,b)-accuracy(y,a),log_loss_reduction=mean_loss(y,a)-mean_loss(y,b))


def write_report(out):
    frozen=json.loads((out/'frozen_selection.json').read_text())
    summary=json.loads((out/'summary.json').read_text())
    table=pd.read_csv(out/'validation_comparison.csv').set_index('model')
    gate=frozen['procedure_gate'];choice=frozen['pooled_selection']['choice'];r=table.loc[choice]
    lines=['# Round 9: supervised tabular neural networks','',
        'Same 2,588 original targets and 0.5 decision threshold. These are repeated 2023 exploratory results; 2024–2025 remain sealed. No current-market inputs or market teacher labels are used.','',
        f"Branch-local recommendation: **{frozen['recommendation']}**. Early-selected 2022 gain is "
        f"**{100*gate['accuracy_improvement']:+.3f} percentage points**, with log-loss change "
        f"**{gate['loss_increase']:+.6f}**; required gain +0.5 pp and allowed loss increase +0.005.",'',
        f"Pooled-CV frozen comparison `{choice}`: **{int(r.correct)}/2,588 = {r.accuracy:.3%}**, "
        f"loss **{r.mean_log_loss:.6f}**. It needs **{1812-int(r.correct)}** more correct to reach 70%.",'',
        'Two recipes each average the same three fixed seeds. All 33,833 original training targets and 326 columns are used. '
        'Each seed selects an epoch only on a purged inner training-period tail, then refits on all eligible training rows. '
        'There are 24 final outer/2023 network fits and 24 inner probes, two selectable fitted configurations and six deterministic R4 mixtures. '
        'R5 comparisons and individual seeds do not enter the nine-entry selector.','',
        '| Model | Correct / 2,588 | Accuracy | Log loss | Role |','| --- | ---: | ---: | ---: | --- |']
    for name,row in table.iterrows():
        role='branch-local choice' if name==frozen['recommendation'] else 'pooled frozen comparison' if name==choice else 'report-only prior comparator' if name in (R5_ACCURACY,R5_LOSS) else 'fixed control / reference'
        lines.append(f'| `{name}` | {int(row.correct)} | {row.accuracy:.3%} | {row.mean_log_loss:.6f} | {role} |')
    controls=pd.read_csv(out/'matched_controls.csv')
    lines+=['','## Matched recipe comparison','',
        '| Period | TD-S accuracy change vs plain / pp | TD-S loss reduction vs plain |',
        '| --- | ---: | ---: |']
    for row in controls.itertuples():
        lines.append(f'| {row.period} | {100*row.accuracy_gain:+.3f} | {row.log_loss_reduction:+.6f} |')
    lines+=['',
        'This isolates a bundled training recipe, not a single architecture component: TD-S changes initialization, activation, optimizer/schedule and label smoothing. '
        'Both have the same robust preprocessing, three 256-wide layers, uniform sample weights, seeds and chronology. '
        'It is an adaptation of the author’s standalone TD-S, not the full RealMLP-TD system or a replication of its random-split benchmark. '
        'Epoch transport to the full training population, historical drift and repeated development search limit interpretation. '
        'A posthoc highest row is not substituted for the frozen choice.','',
        f"Run took {summary['elapsed_seconds']:.1f} seconds on CPU with four threads. Saved weights retain all seeds and folds; "
        'incomplete randomized training batches are omitted within each epoch as in the upstream recipe, while every prediction target remains in the cohort.','',
        'Reproduce: `.venv-foundation/bin/python -m scripts.train_neural_round9`. '
        'Reload: `.venv-foundation/bin/python -m scripts.train_neural_round9 --predict-features results/round9/neural/inference_features.csv`. '
        'Report only: `.venv-foundation/bin/python -m scripts.train_neural_round9 --report-only`.','',
        '[Protocol and evidence](../../../docs/research/round9_neural.md) · '
        '[Frozen selection](frozen_selection.json) · [Matched controls](matched_controls.csv) · '
        '[Inner stopping metadata](training_metadata.json) · [Synthetic timing](synthetic_smoke.json) · '
        '[Upstream MIT license](upstream/LICENSE.txt)','']
    (out/'report_summary.md').write_text('\n'.join(lines))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=ROOT/'results/round9/neural')
    parser.add_argument('--predict-features',type=Path)
    parser.add_argument('--reference-predictions',type=Path)
    parser.add_argument('--report-only',action='store_true')
    args=parser.parse_args();out=args.output;out.mkdir(parents=True,exist_ok=True)
    if args.report_only:write_report(out);return
    configure_device('cpu')
    if args.predict_features:
        bundle=joblib.load(out/'frozen_models.joblib')
        features=pd.read_csv(args.predict_features)
        refs=pd.read_csv(args.reference_predictions or out/'reference_predictions.csv')
        pred=predict_neural(bundle,features,refs)
        pd.concat([features[['match_id']],pred],axis=1).to_csv(out/'inference_predictions.csv',index=False)
        return
    smoke=json.loads((out/'synthetic_smoke.json').read_text())
    if not smoke['synthetic_only'] or smoke['columns']!=326:
        raise ValueError('Unscored synthetic timing must precede empirical fitting')
    invpath=ROOT/'data/processed/mens_singles_2006_2025.csv'
    oldreg=json.loads((ROOT/'results/round5/search_registry.json').read_text())
    if sha256(invpath)!=oldreg['inventory_sha256']:raise ValueError('Pinned inventory changed')
    inventory=pd.read_csv(invpath,dtype={'a_won':'string'})
    inventory=inventory[inventory.season.le(2023)].copy()
    rowpath=ROOT/'results/round3/development_targets.csv';rows=pd.read_csv(rowpath)
    verify_tour_targets(rows,inventory)
    fresh=build_expanded_targets(pd.read_csv(ROOT/'data/processed/model_features.csv'),inventory)
    verify_targets(rows,fresh)
    xpath=ROOT/'results/round5/development_features.csv'
    audit=json.loads((ROOT/'results/round5/artifact_audit.json').read_text())
    if sha256(xpath)!=audit['artifact_sha256']['development_features.csv']:
        raise ValueError('Audited original features changed')
    x=pd.read_csv(xpath)
    if x.shape!=(36421,326) or len(rows)!=len(x) or not np.isfinite(x.to_numpy()).all():
        raise ValueError('Expected complete finite original 326-column cohort')
    specs=candidate_specs();started=time.monotonic()
    registry=dict(round=9,branch='neural',specs=specs,blends=blend_registry(specs),selection_entries=9,
        fitted_configurations=2,outer_and_final_seed_fits=24,inner_stopping_probes=24,
        device='cpu',threads=4,sample_weight='uniform',threshold=.5,features=list(x.columns),
        inventory_sha256=sha256(invpath),targets_sha256=sha256(rowpath),features_sha256=sha256(xpath),
        protocol_sha256=sha256(ROOT/'docs/research/round9_neural.md'),
        source_sha256=sha256(Path(__file__)),engine_sha256=sha256(ROOT/'scripts/tabular_neural.py'),
        upstream_revision=UPSTREAM_REVISION,
        upstream_sha256={p.name:sha256(p) for p in sorted((out/'upstream').iterdir())},
        synthetic_smoke_sha256=sha256(out/'synthetic_smoke.json'),python=sys.version,
        platform=platform.platform(),final_test_evaluated=False)
    (out/'search_registry.json').write_text(json.dumps(registry,indent=2)+'\n')
    freeze=subprocess.check_output([sys.executable,'-m','pip','freeze'],text=True)
    (out/'dependencies.txt').write_text(freeze)
    orows=rows[rows.season.isin(FOLDS)];refs=load_references(orows)
    oof={n:refs[n].to_numpy() for n in refs}
    for spec in specs:oof[spec['name']]=np.full(len(orows),np.nan)
    fold_models={};metadata={};seed_predictions=[]
    def logger(year,name):
        def log(event):
            print(json.dumps(dict(fold=year,model=name,elapsed_seconds=time.monotonic()-started,**event)),flush=True)
        return log
    for year in FOLDS:
        train,valid=temporal_masks(rows,year)
        fold_models[str(year)]={};metadata[str(year)]={}
        for spec in specs:
            name=spec['name'];log=logger(year,name)
            fitted=fit_ensemble(x.loc[train],rows.loc[train],spec,progress=log)
            fold_models[str(year)][name]=fitted
            metadata[str(year)][name]=dict(inner=fitted['inner_metadata'],tuning=fitted['inner_tuning'],
                training_ids=fitted['training_ids'],refit_epochs=[m['epochs'] for m in fitted['members']])
            p=predict_ensemble(fitted,x.loc[valid])
            oof[name][orows.season.eq(year).to_numpy()]=p
            for member in fitted['members']:
                single=dict(fitted,members=[member]);ps=predict_ensemble(single,x.loc[valid])
                seed_predictions.append(dict(period=str(year),model=name,seed=member['seed'],
                    epochs=member['epochs'],accuracy=accuracy(rows.loc[valid,'a_won'].to_numpy(),ps),
                    mean_log_loss=mean_loss(rows.loc[valid,'a_won'].to_numpy(),ps)))
            log(dict(stage='outer_prediction_complete',rows=int(valid.sum())))
        joblib.dump(fold_models,out/'oof_models.joblib')
    if any(not np.isfinite(p).all() for p in oof.values()):raise ValueError('Incomplete OOF predictions')
    oof=add_blends(oof,specs)
    gate=choose_procedure(orows.a_won.to_numpy(),oof,orows.season.to_numpy(),specs)
    pooled=select_candidate(orows.a_won.to_numpy(),oof,np.ones(len(orows),bool),specs)
    recommendation=pooled['choice'] if gate['procedure']=='adopt_neural' else REFERENCE
    frozen=dict(round=9,branch='neural',recommendation=recommendation,procedure_gate=gate,
        pooled_selection=pooled,threshold=.5,fit_end_season=2022,final_test_evaluated=False,
        exploratory_after_2023=True)
    (out/'frozen_selection.json').write_text(json.dumps(frozen,indent=2)+'\n')
    print('Frozen selection: '+json.dumps(frozen),flush=True)
    cv=score_table(orows,oof)
    for year in FOLDS:
        mask=orows.season.eq(year).to_numpy()
        for index,row in cv.iterrows():
            cv.loc[index,f'{year}_accuracy']=accuracy(orows.a_won.to_numpy()[mask],oof[row.model][mask])
            cv.loc[index,f'{year}_log_loss']=mean_loss(orows.a_won.to_numpy()[mask],oof[row.model][mask])
    cv.to_csv(out/'temporal_cv.csv',index=False)
    pd.concat([orows[['match_id','season','a_won']].reset_index(drop=True),pd.DataFrame(oof)],axis=1).to_csv(out/'oof_predictions.csv',index=False)
    train,valid=temporal_masks(rows,2023);validation=rows.loc[valid]
    if int(train.sum())!=33833 or int(valid.sum())!=2588:raise ValueError('Final cohort changed')
    models={};metadata['2023']={}
    for spec in specs:
        name=spec['name'];fitted=fit_ensemble(x.loc[train],rows.loc[train],spec,progress=logger(2023,name))
        models[name]=fitted
        metadata['2023'][name]=dict(inner=fitted['inner_metadata'],tuning=fitted['inner_tuning'],
            training_ids=fitted['training_ids'],refit_epochs=[m['epochs'] for m in fitted['members']])
    (out/'training_metadata.json').write_text(json.dumps(metadata,indent=2)+'\n')
    bundle=dict(specs=specs,columns=list(x.columns),models=models,frozen=frozen,registry=registry)
    ref_valid=load_references(validation,final=True)
    references=pd.concat([validation[['match_id']].reset_index(drop=True),ref_valid.reset_index(drop=True)],axis=1)
    features=pd.concat([validation[['match_id']].reset_index(drop=True),x.loc[valid].reset_index(drop=True)],axis=1)
    pred=predict_neural(bundle,features,references)
    comparison=score_table(validation,pred,recommendation=recommendation)
    comparison.to_csv(out/'validation_comparison.csv',index=False)
    pd.concat([validation.reset_index(drop=True),pred.reset_index(drop=True)],axis=1).to_csv(out/'validation_predictions.csv',index=False)
    features.to_csv(out/'inference_features.csv',index=False);references.to_csv(out/'reference_predictions.csv',index=False)
    joblib.dump(bundle,out/'frozen_models.joblib')
    for name,fitted in models.items():
        for member in fitted['members']:
            ps=predict_ensemble(dict(fitted,members=[member]),x.loc[valid])
            seed_predictions.append(dict(period='2023',model=name,seed=member['seed'],epochs=member['epochs'],
                accuracy=accuracy(validation.a_won.to_numpy(),ps),mean_log_loss=mean_loss(validation.a_won.to_numpy(),ps)))
    pd.DataFrame(seed_predictions).to_csv(out/'seed_diagnostics.csv',index=False)
    matched=[matched_controls(orows,oof,'pooled')]
    for year in FOLDS:
        mask=orows.season.eq(year).to_numpy()
        matched.append(matched_controls(orows.loc[mask],{n:p[mask] for n,p in oof.items()},str(year)))
    matched.append(matched_controls(validation,pred,'2023'))
    pd.DataFrame(matched).to_csv(out/'matched_controls.csv',index=False)
    selected=pred[pooled['choice']].to_numpy()
    intervals={n:dict(accuracy=accuracy_interval(validation,pred[n].to_numpy(),selected),
        log_loss=cluster_interval(validation,pred[n].to_numpy(),selected)) for n in REFERENCES}
    summary=dict(round=9,branch='neural',recommendation=recommendation,pooled_choice=pooled['choice'],
        validation_matches=2588,training_matches=33833,features=326,fitted_configurations=2,
        fixed_blends=6,selection_entries=9,outer_and_final_seed_fits=24,inner_stopping_probes=24,
        device='cpu',threads=4,elapsed_seconds=time.monotonic()-started,comparisons=intervals,
        correct_required_for_70=1812,final_test_evaluated=False,exploratory_after_2023=True)
    (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    write_report(out);print(comparison.to_string(index=False),flush=True)


if __name__=='__main__':
    from scripts.train_neural_round9 import main as canonical_main
    with threadpool_limits(limits=4):canonical_main()
