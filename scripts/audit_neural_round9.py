"""Independent Round9 neural artifact audit, without optimization or refitting.

Run in .venv-foundation after the complete training run has finished.
"""
from __future__ import annotations

import json
import math

import joblib
import numpy as np
import pandas as pd
from scipy.special import expit
import torch
from threadpoolctl import threadpool_limits

from scripts.metrics import score_probabilities
from scripts.tabular_neural import predict_ensemble,configure_device
from scripts.train_models import ROOT,sha256,mirror_features,cluster_interval
from scripts.train_round5 import accuracy_interval
from scripts.train_neural_round9 import predict_neural,REFERENCE,R5_ACCURACY,R5_LOSS

TOLERANCE=2e-6
SEEDS=[1741,1742,1743]


def error(a,b):
    a,b=np.asarray(a),np.asarray(b)
    assert a.shape==b.shape,(a.shape,b.shape)
    return float(np.max(np.abs(a-b)))


def accuracy(y,p):
    return float(np.mean((np.asarray(p)>=.5)==np.asarray(y)))


def log_loss(y,p):
    p=np.clip(np.asarray(p),1e-8,1-1e-8)
    return float(-np.mean(np.asarray(y)*np.log(p)+(1-np.asarray(y))*np.log1p(-p)))


def equivalent(a,b):
    if isinstance(a,dict):
        assert set(a)==set(b)
        for key in a:equivalent(a[key],b[key])
    elif isinstance(a,list):
        assert len(a)==len(b)
        for first,second in zip(a,b):equivalent(first,second)
    elif isinstance(a,(float,int)) and not isinstance(a,bool):assert abs(a-b)<1e-12,(a,b)
    else:assert a==b,(a,b)


def independent_scaler(values,columns):
    reverse=values.copy();reverse[:,[c.endswith('_diff') for c in columns]]*=-1
    both=np.concatenate([values,reverse])
    median=np.median(both,axis=0)
    q=np.quantile(both,.75,axis=0)-np.quantile(both,.25,axis=0)
    q=np.where(q==0,.5*(both.max(axis=0)-both.min(axis=0)),q)
    factors=np.zeros_like(q);usable=q!=0;factors[usable]=1/(q[usable]+1e-30)
    return median,factors


def manual_forward(member,values):
    """NumPy float64 state-only forward; no Torch layer or upstream helper."""
    prep=member['prep']
    scaled=(np.asarray(values,dtype=float)-prep.median_)*prep.factors_
    h=(scaled/np.sqrt(1+(scaled/3)**2)).astype(np.float32).astype(float)
    state={key:tensor.detach().cpu().numpy().astype(float) for key,tensor in member['state_dict'].items()}
    if member['kind']=='realmlp':
        assert set(state)=={'0.scale'}|{f'{j}.{part}' for j in (1,3,5,7) for part in ('weight','bias')}
        h*=state['0.scale']
        for j in (1,3,5,7):
            h=h@state[f'{j}.weight']/math.sqrt(h.shape[1])+state[f'{j}.bias']
            if j!=7:
                h=1.0507009873554805*np.where(h>0,h,1.6732632423543772*np.expm1(np.minimum(h,0)))
    elif member['kind']=='plain':
        assert set(state)=={f'{j}.{part}' for j in (0,2,4,6) for part in ('weight','bias')}
        for j in (0,2,4,6):
            h=h@state[f'{j}.weight'].T+state[f'{j}.bias']
            if j!=6:h=np.maximum(h,0)
    else:raise AssertionError('Unexpected frozen recipe')
    assert h.shape==(len(values),2) and np.isfinite(h).all()
    return expit(h[:,1]-h[:,0])


def manual_symmetric(member,x,columns):
    values=x.to_numpy(dtype=float)
    reverse=values.copy();reverse[:,[c.endswith('_diff') for c in columns]]*=-1
    return .5*(manual_forward(member,values)+1-manual_forward(member,reverse))


def check_fitted(fitted,train_rows,train_x,validation_x,metadata):
    columns=list(train_x)
    assert fitted['columns']==columns and fitted['training_ids']==train_rows.match_id.tolist()
    assert metadata['training_ids']==fitted['training_ids']
    dates=pd.to_datetime(train_rows.tourney_date)
    latest=int(train_rows.season.max());tail=train_rows.season.eq(latest)
    cutoff=dates[tail].min()-pd.Timedelta(days=21)
    inner=train_rows.season.lt(latest)&dates.le(cutoff)
    expected=dict(tail_season=latest,purge_cutoff=cutoff.date().isoformat(),
        train_ids=train_rows.loc[inner,'match_id'].tolist(),tail_ids=train_rows.loc[tail,'match_id'].tolist(),
        train_rows=int(inner.sum()),tail_rows=int(tail.sum()),purged_rows=int((~inner&~tail).sum()))
    assert expected==fitted['inner_metadata']==metadata['inner']
    assert dates[inner].max()<=cutoff and train_rows.season.max()<=2022
    # Inner probe preprocessors were discarded. This computes the required
    # deterministic values; it does not claim comparison to a saved probe.
    inner_center,inner_factor=independent_scaler(train_x[inner].to_numpy(dtype=float),columns)
    assert np.isfinite(inner_center).all() and np.isfinite(inner_factor).all()
    center,factor=independent_scaler(train_x.to_numpy(dtype=float),columns)
    assert [m['seed'] for m in fitted['members']]==SEEDS
    assert [m['seed'] for m in fitted['inner_tuning']]==SEEDS
    assert metadata['tuning']==fitted['inner_tuning']
    assert metadata['refit_epochs']==[m['epochs'] for m in fitted['members']]
    scaler_error=manual_error=0.;predictions=[];details=[]
    subset=validation_x.iloc[np.linspace(0,len(validation_x)-1,min(64,len(validation_x)),dtype=int)]
    for member,tuning in zip(fitted['members'],fitted['inner_tuning']):
        assert member['kind']==fitted['spec']['kind']==tuning['kind']
        losses=np.asarray(tuning['inner_losses'],dtype=float)
        assert losses.shape==(256,) and np.isfinite(losses).all()
        chosen=int(np.flatnonzero(losses==losses.min())[-1])+1
        assert chosen==tuning['selected_epoch']==member['epochs']==member['selected_epoch']
        assert tuning['epochs']==256 and tuning['schedule_epochs']==member['schedule_epochs']==256
        assert member['inner_losses']==[]
        scaler_error=max(scaler_error,error(member['prep'].median_,center),error(member['prep'].factors_,factor))
        assert all(t.device.type=='cpu' and torch.isfinite(t).all() for t in member['state_dict'].values())
        assert sum(t.numel() for t in member['state_dict'].values())==member['parameter_count']
        assert member['parameter_count']==(216136 if member['kind']=='realmlp' else 215810)
        single=dict(fitted,members=[member])
        p=predict_ensemble(single,validation_x,device='cpu')
        predictions.append(p)
        expected_p=manual_symmetric(member,subset,columns)
        actual_p=predict_ensemble(single,subset,device='cpu')
        manual_error=max(manual_error,error(expected_p,actual_p))
        details.append(dict(seed=member['seed'],selected_epoch=chosen,inner_min_loss=float(losses.min()),
            full_training_rows=len(train_rows),inner_training_rows=int(inner.sum()),inner_tail_rows=int(tail.sum())))
    assert scaler_error==0,scaler_error
    assert manual_error<TOLERANCE,manual_error
    averaged=np.mean(predictions,axis=0)
    full=predict_ensemble(fitted,validation_x,device='cpu')
    assert error(averaged,full)==0
    return full,dict(scaler_max_error=scaler_error,manual_forward_max_error=manual_error,
        seed_average_max_error=0.,details=details),predictions


def pick(y,predictions,mask,names):
    limit=log_loss(y[mask],predictions[REFERENCE][mask])+.005
    allowed=[n for n in names if log_loss(y[mask],predictions[n][mask])<=limit+1e-12]
    name=min(allowed,key=lambda n:(-accuracy(y[mask],predictions[n][mask]),log_loss(y[mask],predictions[n][mask]),n))
    return dict(choice=name,accuracy=accuracy(y[mask],predictions[name][mask]),
        mean_log_loss=log_loss(y[mask],predictions[name][mask]),selection_rows=int(mask.sum()),eligible_models=len(allowed),loss_limit=limit)


def main():
    configure_device('cpu')
    out=ROOT/'results/round9/neural'
    required=('frozen_models.joblib','oof_models.joblib','training_metadata.json','summary.json','validation_predictions.csv')
    assert all((out/name).exists() for name in required),'Wait for the complete neural run before auditing'
    registry=json.loads((out/'search_registry.json').read_text())
    bindings={'source_sha256':'scripts/train_neural_round9.py','engine_sha256':'scripts/tabular_neural.py',
        'protocol_sha256':'docs/research/round9_neural.md','targets_sha256':'results/round3/development_targets.csv',
        'features_sha256':'results/round5/development_features.csv','inventory_sha256':'data/processed/mens_singles_2006_2025.csv'}
    for key,path in bindings.items():assert sha256(ROOT/path)==registry[key],path
    for name,digest in registry['upstream_sha256'].items():assert sha256(out/'upstream'/name)==digest
    assert registry['upstream_revision']=='a8c73f75dbeae4ab1ba5bc92372c444ffa0f766e'
    assert registry['upstream_revision'] in (out/'upstream/commit.txt').read_text()
    assert 'Permission is hereby granted' in (out/'upstream/LICENSE.txt').read_text()
    assert registry['device']=='cpu' and registry['sample_weight']=='uniform' and registry['selection_entries']==9
    rows=pd.read_csv(ROOT/'results/round3/development_targets.csv');x=pd.read_csv(ROOT/'results/round5/development_features.csv')
    assert x.shape==(36421,326) and list(x)==registry['features']
    dates=pd.to_datetime(rows.tourney_date)
    bundle=joblib.load(out/'frozen_models.joblib');folds=joblib.load(out/'oof_models.joblib')
    metadata=json.loads((out/'training_metadata.json').read_text())
    assert bundle['registry']==registry and set(folds)=={'2020','2021','2022'}
    frozen=json.loads((out/'frozen_selection.json').read_text());assert bundle['frozen']==frozen
    specs=registry['specs'];new_names=[s['name'] for s in specs]
    assert new_names==['realmlp_td_s','plain_mlp']
    for spec in specs:assert spec['seeds']==SEEDS and spec['max_epochs']==256 and spec['hidden_sizes']==[256]*3
    oof=pd.read_csv(out/'oof_predictions.csv');omask=rows.season.isin([2020,2021,2022]);orows=rows[omask]
    identity=['match_id','season','a_won'];assert oof[identity].equals(orows[identity].reset_index(drop=True))
    refs_old=pd.read_csv(ROOT/'results/round7/foundation/oof_predictions.csv')
    assert refs_old[identity].equals(oof[identity])
    references=[REFERENCE,R5_ACCURACY,R5_LOSS]
    for name in references:assert error(oof[name],refs_old[name])==0
    checks={};reconstructed={n:np.zeros(len(orows)) for n in new_names};seed_scores=[]
    final_predictions={}
    for year in (2020,2021,2022,2023):
        valid=rows.season.eq(year);cutoff=dates[valid].min()-pd.Timedelta(days=21)
        train=rows.season.lt(year)&dates.le(cutoff)
        assert rows[train].season.le(2022).all() and dates[train].max()<=cutoff
        if year==2023:assert int(train.sum())==33833 and int(valid.sum())==2588
        model_set=bundle['models'] if year==2023 else folds[str(year)]
        assert set(model_set)==set(new_names)
        for name in new_names:
            probability,check,members=check_fitted(model_set[name],rows[train],x[train],x[valid],metadata[str(year)][name])
            checks[f'{year}_{name}']=check
            if year==2023:final_predictions[name]=probability
            else:reconstructed[name][orows.season.eq(year).to_numpy()]=probability
            for member,p in zip(model_set[name]['members'],members):
                seed_scores.append(dict(period=str(year),model=name,seed=member['seed'],epochs=member['epochs'],
                    accuracy=accuracy(rows.a_won[valid].to_numpy(),p),mean_log_loss=log_loss(rows.a_won[valid].to_numpy(),p)))
            print(f'Audited weights, transforms and inner chronology: {year} {name}',flush=True)
    oof_reload=max(error(reconstructed[n],oof[n]) for n in new_names);assert oof_reload<1e-12
    preds={n:oof[n].to_numpy() for n in references};preds.update(reconstructed)
    for name,blend in registry['blends'].items():
        weight=blend['model_weight'];assert weight in (.25,.5,.75)
        preds[name]=weight*preds[blend['model']]+(1-weight)*preds[REFERENCE]
        assert error(preds[name],oof[name])<1e-12
    assert set(oof)-set(identity)==set(preds) and len(preds)==11
    seed_saved=pd.read_csv(out/'seed_diagnostics.csv').astype({'period':str})
    seed_new=pd.DataFrame(seed_scores)
    pd.testing.assert_frame_equal(seed_new,seed_saved,check_exact=False,atol=1e-12,rtol=0)
    y=orows.a_won.to_numpy();eligible=[REFERENCE]+new_names+list(registry['blends'])
    assert len(eligible)==9 and not set(eligible)&{R5_ACCURACY,R5_LOSS}
    early=np.isin(orows.season,(2020,2021));gate_mask=orows.season.eq(2022).to_numpy()
    selected_early=pick(y,preds,early,eligible);pooled=pick(y,preds,np.ones(len(y),bool),eligible)
    equivalent(selected_early,frozen['procedure_gate']['early_selection']);equivalent(pooled,frozen['pooled_selection'])
    pnew,pref=preds[selected_early['choice']][gate_mask],preds[REFERENCE][gate_mask]
    gain=accuracy(y[gate_mask],pnew)-accuracy(y[gate_mask],pref)
    delta=log_loss(y[gate_mask],pnew)-log_loss(y[gate_mask],pref)
    gate=frozen['procedure_gate'];assert abs(gain-gate['accuracy_improvement'])<1e-12 and abs(delta-gate['loss_increase'])<1e-12
    passed=gain>=.005-1e-12 and delta<=.005+1e-12
    assert gate['procedure']==('adopt_neural' if passed else 'retain_round4')
    assert frozen['recommendation']==(pooled['choice'] if passed else REFERENCE)
    cv=pd.read_csv(out/'temporal_cv.csv').set_index('model');cv_error=0.
    for name,p in preds.items():
        cv_error=max(cv_error,abs(accuracy(y,p)-cv.loc[name,'accuracy']),abs(log_loss(y,p)-cv.loc[name,'mean_log_loss']))
        for year in (2020,2021,2022):
            mask=orows.season.eq(year).to_numpy()
            cv_error=max(cv_error,abs(accuracy(y[mask],p[mask])-cv.loc[name,f'{year}_accuracy']),abs(log_loss(y[mask],p[mask])-cv.loc[name,f'{year}_log_loss']))
    assert cv_error<1e-12
    valid=rows.season.eq(2023);vr=rows[valid]
    features=pd.read_csv(out/'inference_features.csv');references_frame=pd.read_csv(out/'reference_predictions.csv')
    assert features.match_id.tolist()==references_frame.match_id.tolist()==vr.match_id.tolist()
    assert error(features[list(x)],x[valid])==0
    original_refs=pd.read_csv(ROOT/'results/round7/foundation/validation_predictions.csv').set_index('match_id').loc[vr.match_id]
    for name in references:assert error(references_frame[name],original_refs[name])==0
    p=predict_neural(bundle,features,references_frame,device='cpu')
    saved=pd.read_csv(out/'validation_predictions.csv')
    assert saved[list(rows)].equals(vr.reset_index(drop=True)) and p.shape==(2588,11)
    reload_error=error(p,saved[list(p)]);assert reload_error<1e-12
    for name in new_names:assert error(p[name],final_predictions[name])==0
    swapped_references=references_frame.copy();swapped_references[references]=1-swapped_references[references]
    swapped=predict_neural(bundle,mirror_features(features),swapped_references,device='cpu')
    swap_error=error(p+swapped,np.ones(p.shape));assert swap_error<TOLERANCE
    chunked=predict_neural(bundle,features,references_frame,device='cpu',batch_size=257)
    permutation=np.random.default_rng(158).permutation(len(features))
    permuted=predict_neural(bundle,features.iloc[permutation],references_frame,device='cpu',batch_size=511).loc[features.index]
    batch_error=max(error(p,chunked),error(p,permuted));assert batch_error<TOLERANCE
    assert np.array_equal(p.to_numpy()>=.5,chunked.to_numpy()>=.5)
    assert np.array_equal(p.to_numpy()>=.5,permuted.to_numpy()>=.5)
    assert np.array_equal(p.to_numpy()>=.5,(1-swapped.to_numpy())>=.5)
    poisoned=predict_neural(bundle,features[features.columns[::-1]].assign(a_won='POISON',score='POISON'),references_frame,device='cpu')
    assert error(p,poisoned)==0
    table=pd.read_csv(out/'validation_comparison.csv').set_index('model');metric_error=0.
    for name in p:
        score=score_probabilities(zip(vr.match_id,vr.a_won),zip(vr.match_id,p[name]))
        for key,value in score.items():metric_error=max(metric_error,abs(value-table.loc[name,key]))
        assert abs(np.mean((vr.a_won.to_numpy()-p[name].to_numpy())**2)-table.loc[name,'brier'])<1e-12
    assert metric_error<1e-12
    summary=json.loads((out/'summary.json').read_text())
    for name in references:
        expected=dict(accuracy=accuracy_interval(vr,p[name].to_numpy(),p[pooled['choice']].to_numpy()),
            log_loss=cluster_interval(vr,p[name].to_numpy(),p[pooled['choice']].to_numpy()))
        equivalent(expected,summary['comparisons'][name])
    matched=pd.read_csv(out/'matched_controls.csv').astype({'period':str})
    for _,row in matched.iterrows():
        if row.period=='2023':yy=vr.a_won.to_numpy();a,b=p.plain_mlp.to_numpy(),p.realmlp_td_s.to_numpy()
        else:
            mask=np.ones(len(y),bool) if row.period=='pooled' else orows.season.eq(int(row.period)).to_numpy()
            yy,a,b=y[mask],preds['plain_mlp'][mask],preds['realmlp_td_s'][mask]
        assert row.matches==len(yy) and abs(row.accuracy_gain-(accuracy(yy,b)-accuracy(yy,a)))<1e-12
        assert abs(row.log_loss_reduction-(log_loss(yy,a)-log_loss(yy,b)))<1e-12
    result=dict(audit_agent='independent /root/audit',status='passed',validation_rows=2588,
        probability_columns=11,original_feature_columns=326,original_feature_value_max_error=0.,
        saved_outer_and_final_seed_states_checked=24,chronological_inner_selections_checked=24,
        original_final_training_rows=33833,fold_checks=checks,independent_oof_reload_max_error=oof_reload,
        validation_reload_max_error=reload_error,swap_max_error=swap_error,batch_permutation_max_error=batch_error,
        batch_permutation_changed_classifications=0,unused_label_column_poison_error=0.,
        oof_metrics_max_error=cv_error,validation_metrics_max_error=metric_error,
        historical_selection_gate_and_nine_eligible_entries_independently_reproduced=True,
        seed_diagnostics_and_equal_average_checked=True,refit_schedule_horizon_all_members=256,
        saved_preprocessor_rebuild_max_error=max(c['scaler_max_error'] for c in checks.values()),
        independent_numpy_forward_max_error=max(c['manual_forward_max_error'] for c in checks.values()),
        probability_tolerance=TOLERANCE,paired_intervals_recomputed=3,
        limitations=['No network optimizer stage was repeated; all 24 returned outer/final states were reloaded.',
          'Inner probe preprocessors and per-epoch weights were discarded by the trainer. Inner scaler requirements were deterministically reconstructed from exact purged IDs and the source fit path reviewed, but cannot be compared to an unsaved probe object.',
          'Stored 256-step inner loss traces support last-minimum epoch selection checks; full training trajectories and probe losses were not replayed.',
          'Float32 hardware/batch rounding may vary within the measured tolerance; audited full-cohort classifications were unchanged.',
          'Repeated 2023 and 2022 reuse remains exploratory. No 2024–2025 labels were parsed, no final annual raw file was opened, and no final evaluation was performed; the pinned mixed-year inventory was only hashed.'],
        runtime=dict(torch=torch.__version__,device='cpu',threads=4),
        source_sha256=registry['source_sha256'],engine_sha256=registry['engine_sha256'],protocol_sha256=registry['protocol_sha256'],
        audit_code_sha256=sha256(ROOT/'scripts/audit_neural_round9.py'),
        artifact_sha256={str(path.relative_to(out)):sha256(path) for path in sorted(out.rglob('*'))
            if path.is_file() and path.name!='artifact_audit.json'})
    (out/'artifact_audit.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ('artifact_sha256','fold_checks')},indent=2),flush=True)


if __name__=='__main__':
    with threadpool_limits(limits=4):main()
