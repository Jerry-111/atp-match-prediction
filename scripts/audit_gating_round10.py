"""Independent numerical and artifact audit for the Round10 fixed expert gate.

Only saved estimators are evaluated. No primitive or meta optimizer is refitted.
"""
from __future__ import annotations

import json

import joblib
import numpy as np
import pandas as pd
from scipy.optimize._numdiff import approx_derivative

from scripts.metrics import score_probabilities
from scripts.train_models import ROOT,sha256,cluster_interval
from scripts.train_round5 import accuracy_interval
from scripts.train_gating_round10 import predict_inputs,predict_stage

EXPERT_NAMES=['add_age_spline_C0.01','r5_linear_logistic_C0.003','realmlp_td_s','sequence_mean']
CONTEXT_NAMES=['rank_gap','career_coverage','surface_coverage','point_coverage','best_of_five']
PROBABILITY_NAMES=['equal_mix','global_mix','conditional_mix','round4_reference']
BRANCHES=['round4','round5','round9/neural','round9/sequence']


def equivalent(a,b):
    if isinstance(a,dict):
        assert set(a)==set(b),(set(a),set(b))
        for key in a:equivalent(a[key],b[key])
    elif isinstance(a,list):
        assert len(a)==len(b)
        for first,second in zip(a,b):equivalent(first,second)
    elif isinstance(a,(float,int)) and not isinstance(a,bool):assert abs(a-b)<1e-12,(a,b)
    else:assert a==b,(a,b)


def accuracy(y,p):return float(np.mean((np.asarray(p)>=.5)==np.asarray(y)))


def loss(y,p):
    p=np.clip(np.asarray(p,dtype=float),1e-8,1-1e-8);y=np.asarray(y)
    return float(np.mean(-y*np.log(p)-(1-y)*np.log1p(-p)))


def max_error(first, second):
    a,b=np.asarray(first),np.asarray(second)
    assert a.shape==b.shape,(a.shape,b.shape)
    return float(np.max(np.abs(a-b)))


def independent_contexts(features):
    names=['log_rank_abs_gap','career_min_count','surface_career_min_count',
           'points_overall_364_serve_points_min','format_best_of_five']
    value=features[names].to_numpy(dtype=float).copy()
    assert np.isfinite(value).all() and np.all(value[:,:4]>=0)
    assert np.isin(value[:,4],[0,1]).all()
    value[:,1:4]=np.log1p(value[:,1:4])
    return value


def independent_contrasts():
    return np.array([[1/np.sqrt(2),1/np.sqrt(6),1/np.sqrt(12)],
                     [-1/np.sqrt(2),1/np.sqrt(6),1/np.sqrt(12)],
                     [0,-2/np.sqrt(6),1/np.sqrt(12)],
                     [0,0,-3/np.sqrt(12)]])


def independent_forward(global_weights,theta,z,probabilities):
    p=np.clip(np.asarray(probabilities,dtype=float),1e-6,1-1e-6)
    logits=np.log(global_weights)[None,:]+z@theta@independent_contrasts().T
    shifted=logits-logits.max(axis=1,keepdims=True)
    weights=np.exp(shifted);weights/=weights.sum(axis=1,keepdims=True)
    weights[np.all(z==0,axis=1)]=global_weights
    return (weights*p).sum(axis=1),weights


def independent_objective(theta_flat,global_weights,z,probabilities,y):
    theta=np.asarray(theta_flat).reshape(5,3)
    p=np.clip(np.asarray(probabilities,dtype=float),1e-6,1-1e-6)
    y=np.asarray(y,dtype=float)
    q,w=independent_forward(global_weights,theta,z,p)
    loss=float(np.mean(-y*np.log(q)-(1-y)*np.log1p(-q))+.005*np.sum(theta**2))
    d=(q-y)/(q*(1-q))
    derivative=z.T@(d[:,None]*w*(p-q[:,None]))@independent_contrasts()/len(y)+.01*theta
    return loss,derivative.ravel()


def check_saved_fit(global_model,conditional_model,probabilities,y,features):
    """Rebuild train-only transforms and check both saved optimality conditions."""
    p=np.clip(np.asarray(probabilities,dtype=float),1e-6,1-1e-6)
    y=np.asarray(y,dtype=float)
    x=independent_contexts(features)
    center=x.mean(axis=0);sd=x.std(axis=0);active=sd>1e-12
    scale=np.where(active,np.maximum(sd,1e-6),1.)
    saved=conditional_model.scaler
    scaler_error=max(max_error(saved.mean,center),max_error(saved.scale,scale))
    # Independent contiguous-array reductions may round differently from
    # pandas' column-major view; retain the measured error rather than forcing
    # identical reduction order or claiming bitwise equality.
    assert scaler_error<1e-12 and np.array_equal(saved.active,active)
    z=np.clip((x-center)/scale,-3,3);z[:,~active]=0
    weights=np.asarray(global_model.weights)
    assert weights.shape==(4,) and np.isfinite(weights).all()
    assert weights.min()>=.02-1e-12 and weights.max()<=.94+1e-12
    assert abs(weights.sum()-1)<1e-12
    assert np.array_equal(weights,conditional_model.global_weights)
    assert conditional_model.theta.shape==(5,3) and np.isfinite(conditional_model.theta).all()
    assert global_model.diagnostics['rows']==conditional_model.diagnostics['rows']==len(y)
    assert global_model.diagnostics['weight_floor']==.02
    assert conditional_model.diagnostics['penalty']==.01
    q=p@weights
    loss=float(np.mean(-y*np.log(q)-(1-y)*np.log1p(-q)))
    gradient=p.T@((q-y)/(q*(1-q)))/len(y)
    free=weights>.0200001
    multiplier=gradient[free].mean()
    kkt=float(np.max(np.abs(gradient[free]-multiplier)))
    if (~free).any():kkt=max(kkt,float(np.maximum(multiplier-gradient[~free],0).max()))
    assert abs(loss-global_model.diagnostics['objective'])<1e-12 and kkt<=1e-5
    objective,gradient=independent_objective(conditional_model.theta.ravel(),weights,z,p,y)
    assert abs(objective-conditional_model.diagnostics['objective'])<1e-12
    assert np.max(np.abs(gradient))<=1e-6
    numerical=approx_derivative(lambda b:independent_objective(b,weights,z,p,y)[0],
        conditional_model.theta.ravel(),method='3-point').ravel()
    gradient_error=max_error(gradient,numerical)
    assert gradient_error<1e-8
    return dict(training_rows=len(y),scaler_rebuild_max_error=scaler_error,
        shared_global_anchor_max_error=0.,global_kkt_error=kkt,
        conditional_max_gradient=float(np.max(np.abs(gradient))),
        conditional_finite_difference_error=gradient_error)


def load_bound_experts(rows,registry,final=False):
    name='validation_predictions.csv' if final else 'oof_predictions.csv'
    values={}
    for expert,branch in zip(EXPERT_NAMES,BRANCHES):
        info=registry['source_bindings'][expert]
        assert info['branch']==branch
        directory=ROOT/'results'/branch
        audit=json.loads((directory/'artifact_audit.json').read_text())
        assert audit['status']=='passed' and sha256(directory/'artifact_audit.json')==info['audit_sha256']
        assert sha256(directory/name)==info['files'][name]==audit['artifact_sha256'][name]
        original=pd.read_csv(directory/name)
        assert not original.match_id.duplicated().any() and set(original.match_id)==set(rows.match_id)
        original=original.set_index('match_id').loc[rows.match_id]
        assert np.array_equal(original.season,rows.season) and np.array_equal(original.a_won,rows.a_won)
        p=original[expert].to_numpy(dtype=float)
        assert np.isfinite(p).all() and np.all((p>=0)&(p<=1))
        values[expert]=p
    return pd.DataFrame(values,index=rows.index)


def check_table(table,rows,predictions):
    table=table.set_index('model');assert set(table.index)==set(PROBABILITY_NAMES)
    maximum=0.
    for name in PROBABILITY_NAMES:
        score=score_probabilities(zip(rows.match_id,rows.a_won),zip(rows.match_id,predictions[name]))
        for key,value in score.items():maximum=max(maximum,abs(value-table.loc[name,key]))
        maximum=max(maximum,abs(np.mean((rows.a_won.to_numpy()-predictions[name])**2)-table.loc[name,'brier']))
    assert maximum<1e-12
    return maximum


def main():
    out=ROOT/'results/round10/gating'
    assert (out/'summary.json').exists() and (out/'report_summary.md').exists(),'Wait for complete results'
    registry=json.loads((out/'search_registry.json').read_text())
    bindings={'source_sha256':'scripts/train_gating_round10.py','engine_sha256':'scripts/conditional_gate.py',
        'protocol_sha256':'docs/research/round10_gating_protocol.md',
        'feasibility_sha256':'docs/research/round10_oof_feasibility.md',
        'targets_sha256':'results/round3/development_targets.csv',
        'features_sha256':'results/round5/development_features.csv'}
    for key,path in bindings.items():assert sha256(ROOT/path)==registry[key],path
    assert registry['experts']==EXPERT_NAMES and registry['context_columns']==CONTEXT_NAMES
    assert registry['selection_entries']==PROBABILITY_NAMES and registry['procedures']==PROBABILITY_NAMES[:3]
    assert registry['global_weight_floor']==.02 and registry['gate_penalty']==.01
    assert registry['optimizer_runs']==6 and registry['base_expert_fits']==0 and registry['gate_parameters']==15
    assert registry['meta_warmup_season']==2020 and registry['selection_season']==2021 and registry['gate_season']==2022
    bundle=joblib.load(out/'frozen_models.joblib');assert bundle['registry']==registry
    assert set(bundle['models'])=={'2021','2022','2023'}
    frozen=json.loads((out/'frozen_selection.json').read_text());assert bundle['frozen']==frozen
    metadata=json.loads((out/'training_metadata.json').read_text())
    rows=pd.read_csv(ROOT/bindings['targets_sha256']).set_index('match_id',drop=False)
    x=pd.read_csv(ROOT/bindings['features_sha256']);x.index=rows.index
    assert len(rows)==36421 and x.shape==(36421,326)
    assert not rows.match_id.duplicated().any() and rows.season.between(2009,2023).all()
    dates=pd.to_datetime(rows.tourney_date)
    orows=rows[rows.season.isin([2020,2021,2022])];assert len(orows)==6130
    assert orows.groupby('season').size().to_dict()=={2020:1214,2021:2388,2022:2528}
    inputs=load_bound_experts(orows,registry)
    cached_inputs=pd.read_csv(out/'primitive_oof_inputs.csv').set_index('match_id',drop=False)
    assert cached_inputs[['match_id','season','tourney_date','a_won']].equals(orows[['match_id','season','tourney_date','a_won']])
    assert max_error(cached_inputs[EXPERT_NAMES],inputs)==0
    contexts=pd.DataFrame(independent_contexts(x),index=rows.index,columns=CONTEXT_NAMES)
    exposure=(x.points_overall_364_serve_points_log_sum-abs(x.points_overall_364_serve_points_log_diff))/2
    exposure_error=max_error(contexts.point_coverage,exposure);assert exposure_error<1e-12
    stage_predictions={};stage_weights={};fit_checks={};manual_error=0.;weight_rows=[]
    for year in (2021,2022,2023):
        query=rows[rows.season.eq(year)]
        cutoff=dates[query.index].min()-pd.Timedelta(days=21)
        train=orows.season.lt(year)&dates[orows.index].le(cutoff)
        info=metadata[str(year)]
        expected=dict(target_year=year,cutoff=cutoff.date().isoformat(),
            training_ids=orows.loc[train,'match_id'].tolist(),query_ids=query.match_id.tolist(),
            training_rows=int(train.sum()),query_rows=len(query),
            purged_rows=int((orows.season.lt(year)&~train).sum()))
        assert all(info[k]==v for k,v in expected.items()) and info['stage']==year
        assert int(train.sum())=={2021:1214,2022:3602,2023:6130}[year]
        assert not set(expected['training_ids'])&set(expected['query_ids'])
        models=bundle['models'][str(year)];assert set(models)=={'global','conditional'}
        fit_checks[str(year)]=check_saved_fit(models['global'],models['conditional'],inputs[train],
            orows.a_won[train],x.loc[orows.index[train]])
        equivalent(info['global_optimizer'],models['global'].diagnostics)
        equivalent(info['conditional_optimizer'],models['conditional'].diagnostics)
        assert max_error(info['global_weights'],models['global'].weights)==0
        assert max_error(info['context_mean'],models['conditional'].scaler.mean)==0
        assert max_error(info['context_scale'],models['conditional'].scaler.scale)==0
        assert np.array_equal(info['active_contexts'],models['conditional'].scaler.active)
        qp=inputs.loc[query.index] if year<2023 else load_bound_experts(query,registry,final=True)
        predicted=predict_stage(models,qp,contexts.loc[query.index]);assert list(predicted)==PROBABILITY_NAMES
        qclip=np.clip(qp.to_numpy(),1e-6,1-1e-6)
        normalized=(contexts.loc[query.index].to_numpy()-models['conditional'].scaler.mean)/models['conditional'].scaler.scale
        normalized=np.clip(normalized,-3,3);normalized[:,~models['conditional'].scaler.active]=0
        cq,weights=independent_forward(models['global'].weights,models['conditional'].theta,normalized,qclip)
        independent=np.column_stack([qclip.mean(axis=1),qclip@models['global'].weights,cq,qp.iloc[:,0]])
        manual_error=max(manual_error,max_error(predicted,independent))
        stage_predictions[str(year)]=predicted;stage_weights[str(year)]=qp
        for population,population_inputs,ids in [('train',inputs[train],orows.index[train]),('query',qp,query.index)]:
            clipped={name:int(((population_inputs[name]<1e-6)|(population_inputs[name]>1-1e-6)).sum()) for name in EXPERT_NAMES}
            assert info[('training' if population=='train' else 'query')+'_clipped_probabilities']==clipped
            transformed=(contexts.loc[ids].to_numpy()-models['conditional'].scaler.mean)/models['conditional'].scaler.scale
            transformed=np.clip(transformed,-3,3);transformed[:,~models['conditional'].scaler.active]=0
            _,weights=independent_forward(models['global'].weights,models['conditional'].theta,transformed,population_inputs)
            for i,name in enumerate(EXPERT_NAMES):
                weight_rows.append(dict(stage=year,population=population,expert=name,rows=len(ids),
                    global_weight=models['global'].weights[i],mean=weights[:,i].mean(),std=weights[:,i].std(),
                    minimum=weights[:,i].min(),maximum=weights[:,i].max()))
    assert manual_error<1e-12
    saved_weights=pd.read_csv(out/'weight_spread.csv')
    pd.testing.assert_frame_equal(pd.DataFrame(weight_rows),saved_weights,check_exact=False,atol=1e-12,rtol=0)
    meta_rows=orows[orows.season.isin([2021,2022])]
    pred=pd.concat([stage_predictions['2021'],stage_predictions['2022']]).loc[meta_rows.index]
    saved=pd.read_csv(out/'oof_predictions.csv').set_index('match_id',drop=False)
    assert len(saved)==4916 and set(saved.season)=={2021,2022} and len(saved.match_id.unique())==4916
    assert saved[['match_id','season','tourney_date','a_won']].equals(meta_rows[['match_id','season','tourney_date','a_won']])
    oof_error=max_error(pred,saved[PROBABILITY_NAMES]);assert oof_error<1e-12
    cv=pd.read_csv(out/'temporal_cv.csv');assert len(cv)==12
    metric_error=0.
    for period in ('2021','2022','pooled_2021_2022_descriptive_only'):
        ids=meta_rows.index if period.startswith('pooled') else meta_rows.index[meta_rows.season.eq(int(period))]
        metric_error=max(metric_error,check_table(cv[cv.period.eq(period)],rows.loc[ids],pred.loc[ids]))
    early=rows[rows.season.eq(2021)];ep=stage_predictions['2021'];y=early.a_won.to_numpy()
    limit=loss(y,ep.round4_reference)+.005
    allowed=[name for name in PROBABILITY_NAMES if loss(y,ep[name])<=limit+1e-12]
    selected=min(allowed,key=lambda name:(-accuracy(y,ep[name]),loss(y,ep[name]),name))
    choice=dict(choice=selected,selection_season=2021,selection_rows=2388,accuracy=accuracy(y,ep[selected]),
        mean_log_loss=loss(y,ep[selected]),eligible_entries=allowed,loss_limit=limit,decision_threshold=.5)
    equivalent(choice,json.loads((out/'frozen_2021_choice.json').read_text()));equivalent(choice,frozen['selection'])
    assert frozen['original_choice_sha256']==sha256(out/'frozen_2021_choice.json')
    assert frozen['no_pooled_reselection'] is True and frozen['selected_procedure']==selected
    gate_rows=rows[rows.season.eq(2022)];gp=stage_predictions['2022'];gy=gate_rows.a_won.to_numpy()
    gain=accuracy(gy,gp[selected])-accuracy(gy,gp.round4_reference)
    change=loss(gy,gp[selected])-loss(gy,gp.round4_reference)
    passed=gain>=.005-1e-12 and change<=.005+1e-12
    gate=dict(passed=passed,procedure='adopt_gating' if passed else 'retain_round4',selected_procedure=selected,
        gate_season=2022,gate_rows=2528,accuracy_improvement=gain,loss_increase=change,net_correct=int(round(gain*2528)),
        required_accuracy_improvement=.005,allowed_loss_increase=.005,reference_accuracy=accuracy(gy,gp.round4_reference),
        candidate_accuracy=accuracy(gy,gp[selected]),reference_loss=loss(gy,gp.round4_reference),candidate_loss=loss(gy,gp[selected]))
    equivalent(gate,frozen['procedure_gate']);assert frozen['recommendation']==(selected if passed else 'round4_reference')
    vr=rows[rows.season.eq(2023)];final=stage_predictions['2023']
    saved_final=pd.read_csv(out/'validation_predictions.csv').set_index('match_id',drop=False)
    assert len(saved_final)==2588 and saved_final[['match_id','season','tourney_date','a_won']].equals(vr[['match_id','season','tourney_date','a_won']])
    final_error=max_error(final,saved_final[PROBABILITY_NAMES]);assert final_error<1e-12
    metric_error=max(metric_error,check_table(pd.read_csv(out/'validation_comparison.csv'),vr,final))
    queries=pd.read_csv(out/'inference_inputs.csv')
    assert queries.match_id.tolist()==vr.match_id.tolist()
    assert max_error(queries[CONTEXT_NAMES],contexts.loc[vr.index])<1e-12
    assert max_error(queries[EXPERT_NAMES],stage_weights['2023'])==0
    reloaded=predict_inputs(bundle,queries);inference_error=max_error(reloaded,final);assert inference_error<1e-12
    swapped=queries.copy();swapped[EXPERT_NAMES]=1-swapped[EXPERT_NAMES]
    reverse=predict_inputs(bundle,swapped);swap_error=max_error(reloaded+reverse,np.ones(reloaded.shape));assert swap_error<1e-12
    rng=np.random.default_rng(1010);permutation=rng.permutation(len(queries))
    permuted=predict_inputs(bundle,queries.iloc[permutation]).loc[vr.index]
    chunks=pd.concat([predict_inputs(bundle,queries.iloc[:1000]),predict_inputs(bundle,queries.iloc[1000:])])
    order_error=max(max_error(reloaded,permuted),max_error(reloaded,chunks));assert order_error<1e-12
    assert np.array_equal(reloaded.to_numpy()>=.5,permuted.to_numpy()>=.5)
    assert np.array_equal(reloaded.to_numpy()>=.5,chunks.to_numpy()>=.5)
    poisoned=predict_inputs(bundle,queries.assign(a_won='POISON',score='POISON'))
    assert max_error(reloaded,poisoned)==0
    for invalid in (2015,2020,2024,2025):
        bad=queries.copy();bad['season']=invalid
        try:predict_inputs(bundle,bad)
        except ValueError:pass
        else:raise AssertionError(f'Invalid query season accepted: {invalid}')
    manifest=json.loads((out/'final_fit_manifest.json').read_text())
    assert manifest['model_sha256']==sha256(out/'frozen_models.joblib')
    assert manifest['selection_sha256']==sha256(out/'frozen_selection.json')
    assert manifest['final_training_ids']==metadata['2023']['training_ids']
    assert manifest['final_parameters_frozen_before_new_2023_prediction'] is True
    matched=pd.read_csv(out/'matched_routing.csv').astype({'period':str})
    assert matched.period.tolist()==['2021','2022','2023']
    for year in ('2021','2022','2023'):
        r=matched[matched.period.eq(year)].iloc[0];p=stage_predictions[year];yy=rows.loc[p.index,'a_won'].to_numpy()
        assert r.matches==len(p)
        assert abs(r.accuracy_improvement-(accuracy(yy,p.conditional_mix)-accuracy(yy,p.global_mix)))<1e-12
        assert abs(r.log_loss_reduction-(loss(yy,p.global_mix)-loss(yy,p.conditional_mix)))<1e-12
        assert abs(r.max_probability_change-np.abs(p.conditional_mix-p.global_mix).max())<1e-12
        assert abs(r.mean_probability_change-np.abs(p.conditional_mix-p.global_mix).mean())<1e-12
    summary=json.loads((out/'summary.json').read_text())
    pairs={'selected_vs_reference':('round4_reference',selected),'conditional_vs_global':('global_mix','conditional_mix')}
    for key,(before,after) in pairs.items():
        comparison=dict(accuracy=accuracy_interval(vr,final[before].to_numpy(),final[after].to_numpy()),
            log_loss=cluster_interval(vr,final[before].to_numpy(),final[after].to_numpy()))
        equivalent(comparison,summary['comparisons'][key])
    result=dict(status='passed',auditor='independent /root/audit',no_refitting=True,final_test_evaluated=False,
        primitive_oof_rows=6130,honest_meta_validation_rows=4916,meta_validation_seasons=[2021,2022],
        no_manufactured_2020_meta_test=True,validation_matches=2588,probability_columns=4,
        saved_meta_optimizers_checked=6,base_expert_fits=0,fit_checks=fit_checks,
        exact_four_primitive_hash_identity_label_bindings=True,
        context_exposure_semantics_max_error=exposure_error,independent_forward_max_error=manual_error,
        oof_reload_max_error=oof_error,validation_reload_max_error=final_error,inference_reload_max_error=inference_error,
        swap_max_error=swap_error,permutation_batch_max_error=order_error,changed_order_classifications=0,
        label_poisoning_max_error=0.,metrics_max_error=metric_error,
        independent_2021_only_selection=choice,independent_2022_gate=gate,no_pooled_reselection=True,
        paired_intervals_recomputed=2,all_clipping_counts_and_weight_spread_checked=True,
        final_frozen_manifest_verified=True,invalid_season_probes_rejected=[2015,2020,2024,2025],
        limitations=['No optimizer, primitive model, or historical training trajectory was rerun.',
          'R4/R5 primitive fold weights were not retained. Their OOF legality relies on audited caches and generation paths, unlike the directly reloaded R9 fold models.',
          'Source ordering proves the freeze-before-new-prediction code path; file hashes alone are not independent wall-clock proof.',
          'Prior recipe selection, repeated 2022 gate use and repeated 2023 exploration remain adaptive research limitations; no 2024–2025 data were opened.'],
        audit_source_sha256=sha256(ROOT/'scripts/audit_gating_round10.py'),
        artifact_sha256={path.name:sha256(path) for path in sorted(out.iterdir()) if path.is_file() and path.name!='artifact_audit.json'})
    (out/'artifact_audit.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ('artifact_sha256','fit_checks')},indent=2))


if __name__=='__main__':main()
