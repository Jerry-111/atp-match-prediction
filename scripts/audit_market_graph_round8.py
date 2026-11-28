"""Independent frozen-artifact checks; never fit classifiers or inspect holdout."""
from __future__ import annotations

import json
from unittest.mock import patch

import joblib
import numpy as np
import pandas as pd
from scipy.special import expit,logit
from threadpoolctl import threadpool_limits

from scripts.market_graph_features import build_market_graph_features,load_opening_history,SOURCE_COLUMNS
from scripts.train_market_graph_round8 import predict_market_graph
from scripts.train_models import ROOT,sha256,mirror_features,cluster_interval
from scripts.train_round5 import REFERENCE,accuracy_interval
from scripts.metrics import score_probabilities


def difference(a,b):
    a,b=np.asarray(a),np.asarray(b)
    assert a.shape==b.shape
    return float(np.max(np.abs(a-b)))


def loss(y,p):
    p=np.clip(p,1e-8,1-1e-8)
    return float(-np.mean(y*np.log(p)+(1-y)*np.log1p(-p)))


def accuracy(y,p):
    return float(np.mean((np.asarray(p)>=.5)==np.asarray(y)))


def equivalent(a,b):
    if isinstance(a,dict):
        assert set(a)==set(b)
        for key in a:equivalent(a[key],b[key])
    elif isinstance(a,list):
        assert len(a)==len(b)
        for first,second in zip(a,b):equivalent(first,second)
    elif isinstance(a,(float,int)) and not isinstance(a,bool):
        assert abs(a-b)<1e-12,(a,b)
    else:assert a==b,(a,b)


def convert_probability(p,source_format,target_format):
    """Independent binary search of ordinary best-of-series polynomials."""
    p=np.asarray(p);source_format=np.asarray(source_format)
    def series(q,fmt):
        return np.where(np.asarray(fmt)==3,3*q*q-2*q**3,10*q**3-15*q**4+6*q**5)
    low,high=np.zeros_like(p),np.ones_like(p)
    for _ in range(60):
        middle=(low+high)/2
        below=series(middle,source_format)<p
        low=np.where(below,middle,low);high=np.where(below,high,middle)
    return series((low+high)/2,target_format)


def dense_queries(targets,history,graph,settings):
    """Independent dense normal equations on nine fixed chronological targets."""
    subset=targets[targets.season.between(2021,2023)]
    selected=subset.iloc[np.linspace(0,len(subset)-1,9,dtype=int)]
    max_error,checks=0.,0
    for index,row in selected.iterrows():
        cutoff=pd.Timestamp(row.tourney_date,tz='UTC')-pd.Timedelta(days=21)
        past=history[pd.to_datetime(history.match_utc,utc=True).le(cutoff)]
        if past.empty:
            assert graph.loc[index,'market_graph_available']==0
            continue
        nodes=sorted(set(past.player_a_id)|set(past.player_b_id));lookup={p:i for i,p in enumerate(nodes)}
        a,b=past.player_a_id.map(lookup).to_numpy(),past.player_b_id.map(lookup).to_numpy()
        if row.player_a_id not in lookup or row.player_b_id not in lookup:
            assert graph.loc[index,'market_graph_available']==0
            continue
        matrix=np.zeros((len(past),len(nodes)));matrix[np.arange(len(past)),a]=1;matrix[np.arange(len(past)),b]=-1
        parent=list(range(len(nodes)))
        def find(a):
            while parent[a]!=a:a=parent[a]
            return a
        for first,second in zip(a,b):parent[find(first)]=find(second)
        ai,bi=lookup[row.player_a_id],lookup[row.player_b_id]
        if find(ai)!=find(bi):
            assert graph.loc[index,'market_graph_available']==0
            continue
        assert graph.loc[index,'market_graph_available']==1
        z=logit(np.clip(convert_probability(expit(past.opening_logit_diff.to_numpy()),past.best_of.to_numpy(),3),1e-12,1-1e-12))
        age=(cutoff-pd.to_datetime(past.match_utc,utc=True)).dt.total_seconds().to_numpy()/86400
        assert np.all(age>=0)
        for spec in settings:
            weight=np.ones(len(past)) if spec['half_life'] is None else np.exp2(-age/spec['half_life'])
            weight*=np.where(past.surface.eq(row.surface),1.,spec['other_surface'])
            lap=matrix.T@(weight[:,None]*matrix)+.001*np.eye(len(nodes))
            ratings=np.linalg.solve(lap,matrix.T@(weight*z))
            p=convert_probability(np.array([expit(ratings[ai]-ratings[bi])]),3,row.best_of)[0]
            expected=logit(np.clip(p,1e-12,1-1e-12))
            max_error=max(max_error,abs(expected-graph.loc[index,spec['name']+'_diff']))
            checks+=1
    assert max_error<1e-10,max_error
    return max_error,checks


def main():
    out=ROOT/'results/round8/market_graph'
    registry=json.loads((out/'search_registry.json').read_text())
    assert registry['source_code_sha256']==sha256(ROOT/'scripts/train_market_graph_round8.py')
    assert registry['feature_code_sha256']==sha256(ROOT/'scripts/market_graph_features.py')
    assert registry['protocol_sha256']==sha256(ROOT/'docs/research/round8_market_graph.md')
    rows=pd.read_csv(ROOT/'results/round3/development_targets.csv')
    old=pd.read_csv(ROOT/'results/round5/development_features.csv')
    targets=rows.drop(columns=['a_won']).assign(best_of=np.where(old.format_best_of_five,5,3))
    bundle=joblib.load(out/'frozen_models.joblib')
    inputs=pd.read_csv(out/'inference_features.csv')
    assert difference(inputs[list(old)],old)==0
    reads=[];reader=pd.read_csv
    def traced(path,*args,**kwargs):
        reads.append((str(path),kwargs.get('usecols')))
        return reader(path,*args,**kwargs)
    with patch('scripts.market_graph_features.pd.read_csv',side_effect=traced):
        history=load_opening_history(targets)
    names=[p.rsplit('/',1)[-1] for p,_ in reads]
    assert names==['linked_matches.csv','2021.csv','2022.csv','2023.csv']
    for _,columns in reads[1:]:assert columns==SOURCE_COLUMNS
    saved_history=pd.read_csv(out/'opening_history.csv')
    saved_history['match_utc']=pd.to_datetime(saved_history.match_utc,utc=True)
    pd.testing.assert_frame_equal(history,saved_history,check_exact=False,atol=1e-14,rtol=0)
    pd.testing.assert_frame_equal(history,bundle['history'],check_exact=True)
    assert len(history)==7341 and not history.source_match_id.duplicated().any()
    graph=build_market_graph_features(targets,history)
    locked=build_market_graph_features(targets,history,locked_before='2023-01-01')
    rebuilt=pd.concat([old,graph,locked.add_prefix('locked_')],axis=1)
    assert list(rebuilt)==list(inputs)==bundle['feature_names']
    rebuild_error=difference(rebuilt,inputs);assert rebuild_error<1e-12
    dense_error,dense_count=dense_queries(targets,history,graph,registry['graph_settings'])
    sample=targets.iloc[np.linspace(0,len(targets)-1,100,dtype=int)].copy()
    sample[['player_a_id','player_b_id']]=sample[['player_b_id','player_a_id']]
    swapped_features=build_market_graph_features(sample,history)
    feature_swap=difference(swapped_features,mirror_features(graph.loc[sample.index]));assert feature_swap<1e-12
    valid=rows.season.eq(2023);vr=rows[valid];vx=inputs[valid]
    probabilities=predict_market_graph(bundle,vx)
    saved=pd.read_csv(out/'validation_predictions.csv')
    assert saved[list(rows)].equals(vr.reset_index(drop=True))
    assert list(saved.columns[len(rows.columns):])==list(probabilities) and probabilities.shape==(2588,17)
    reload_error=difference(probabilities,saved[list(probabilities)]);assert reload_error<1e-12
    swapped=predict_market_graph(bundle,mirror_features(vx))
    swap_error=difference(probabilities+swapped,np.ones(probabilities.shape));assert swap_error<1e-12
    poisoned=predict_market_graph(bundle,vx[vx.columns[::-1]].assign(a_won='DO NOT READ',cote1_ouverture='DO NOT READ',score='DO NOT READ'))
    assert difference(poisoned,probabilities)==0
    fallback=graph.loc[valid,'market_graph_available'].eq(0)
    graph_names=[s['name'] for s in bundle['specs']]
    assert difference(probabilities.loc[fallback,graph_names],np.repeat(probabilities.loc[fallback,REFERENCE].to_numpy()[:,None],12,axis=1))==0
    table=pd.read_csv(out/'validation_comparison.csv').set_index('model');metric_error=0.
    for name in probabilities:
        scored=score_probabilities(zip(vr.match_id,vr.a_won),zip(vr.match_id,probabilities[name]))
        for key,value in scored.items():metric_error=max(metric_error,abs(value-table.loc[name,key]))
        assert abs(loss(vr.a_won.to_numpy(),probabilities[name])-table.loc[name,'mean_log_loss'])<1e-12
        assert abs(np.mean((vr.a_won-probabilities[name])**2)-table.loc[name,'brier'])<1e-12
    assert metric_error<1e-12
    oof=pd.read_csv(out/'oof_predictions.csv')
    omask=(rows.season.eq(2021)&pd.to_datetime(rows.tourney_date).ge('2021-07-01'))|rows.season.eq(2022)
    orows=rows[omask];identity=['match_id','season','a_won']
    assert oof[identity].equals(orows[identity].reset_index(drop=True))
    oldoof=pd.read_csv(ROOT/'results/round5/oof_predictions.csv').set_index('match_id').loc[orows.match_id]
    assert difference(oof[REFERENCE],oldoof[REFERENCE])<1e-12
    available=graph.loc[omask,'market_graph_available'].eq(1).to_numpy()
    oof_rebuild_error=0.
    for spec in bundle['specs']:
        raw=expit(graph.loc[omask,spec['graph']+'_diff'].to_numpy())
        expected=oldoof[REFERENCE].to_numpy().copy()
        expected[available]=spec['weight']*raw[available]+(1-spec['weight'])*expected[available]
        oof_rebuild_error=max(oof_rebuild_error,difference(expected,oof[spec['name']]))
    assert oof_rebuild_error<1e-12
    selectable=graph_names+[REFERENCE];y=oof.a_won.to_numpy()
    def pick(mask):
        limit=loss(y[mask],oof[REFERENCE].to_numpy()[mask])+.005
        eligible=[n for n in selectable if loss(y[mask],oof[n].to_numpy()[mask])<=limit]
        return min(eligible,key=lambda n:(-accuracy(y[mask],oof[n].to_numpy()[mask]),loss(y[mask],oof[n].to_numpy()[mask]),n))
    early=oof.season.eq(2021).to_numpy();gate=~early
    early_name,pooled=pick(early),pick(np.ones(len(y),bool))
    frozen=json.loads((out/'frozen_selection.json').read_text());assert frozen==bundle['frozen']
    assert frozen['procedure_gate']['early_selection']['choice']==early_name
    assert frozen['pooled_selection']['choice']==pooled
    gain=accuracy(y[gate],oof[early_name].to_numpy()[gate])-accuracy(y[gate],oof[REFERENCE].to_numpy()[gate])
    delta=loss(y[gate],oof[early_name].to_numpy()[gate])-loss(y[gate],oof[REFERENCE].to_numpy()[gate])
    passed=gain>=.005-1e-12 and delta<=.005+1e-12
    assert frozen['recommendation']==(pooled if passed else REFERENCE)
    assert abs(gain-frozen['procedure_gate']['accuracy_improvement'])<1e-12
    assert abs(delta-frozen['procedure_gate']['loss_increase'])<1e-12
    cv=pd.read_csv(out/'temporal_cv.csv');cv_error=0.
    for _,row in cv.iterrows():
        mask=np.ones(len(y),bool) if row.fold=='pooled' else early if row.fold=='2021H2' else gate
        cv_error=max(cv_error,abs(accuracy(y[mask],oof[row.model].to_numpy()[mask])-row.accuracy),abs(loss(y[mask],oof[row.model].to_numpy()[mask])-row.mean_log_loss))
        assert row.n==int(mask.sum())
    assert cv_error<1e-12 and len(cv)==48
    summary=json.loads((out/'summary.json').read_text())
    for name in probabilities:
        expected=dict(accuracy=accuracy_interval(vr,probabilities[REFERENCE].to_numpy(),probabilities[name].to_numpy()),
            log_loss=cluster_interval(vr,probabilities[REFERENCE].to_numpy(),probabilities[name].to_numpy()))
        equivalent(expected,summary['comparisons_vs_r4'][name])
    selected=next(s for s in bundle['specs'] if s['name']==pooled) if pooled!=REFERENCE else None
    locked_fallback=locked.loc[valid,'market_graph_available'].eq(0)
    if selected:
        locked_p=probabilities[REFERENCE].to_numpy().copy()
        use=~locked_fallback.to_numpy()
        locked_p[use]=(1-selected['weight'])*locked_p[use]+selected['weight']*expit(locked.loc[valid,selected['graph']+'_diff'].to_numpy()[use])
        assert difference(locked_p,probabilities.selected_graph_locked_pre2023)<1e-12
    assert summary['exact_fallback_rows_2023']==int(fallback.sum())
    result=dict(audit_agent='independent /root/audit',status='passed',validation_rows=2588,
        validation_probability_columns=17,development_feature_rows=len(inputs),rebuilt_feature_columns=inputs.shape[1],
        opening_edges=len(history),source_files_opened=names[1:],source_columns_checked=SOURCE_COLUMNS,
        source_outcomes_or_closing_fields_read=False,feature_rebuild_max_error=rebuild_error,
        independent_dense_normal_equations_queries=dense_count,independent_dense_max_error=dense_error,
        real_target_feature_swap_checks=100,feature_swap_max_error=feature_swap,
        reload_max_error=reload_error,probability_swap_max_error=swap_error,label_poison_error=0.,
        validation_metrics_max_error=metric_error,oof_metrics_max_error=cv_error,
        historical_graph_oof_rebuild_max_error=oof_rebuild_error,oof_prediction_columns=16,selection_eligible_columns=13,
        early_choice=early_name,pooled_choice=pooled,selection_and_gate_independently_reproduced=True,
        exact_current_graph_fallback_rows=int(fallback.sum()),locked_graph_fallback_rows=int(locked_fallback.sum()),
        paired_date_block_intervals_recomputed=17,
        limitations=['No old binary classifiers or OOF heads refitted; graph estimates and historical graph predictions were fully rebuilt.',
          'Historical opening-price inputs and updates from earlier2023 edges are a distinct information condition from pure history or the static market student.',
          'Actual match UTC is an upper bound only under provider prematch provenance; no quote capture timestamps independently verified.',
          '2023 remains exploratory; no2024/25 source or final evaluation opened.'],
        source_code_sha256=registry['source_code_sha256'],feature_code_sha256=registry['feature_code_sha256'],
        protocol_sha256=registry['protocol_sha256'],
        audit_code_sha256=sha256(ROOT/'scripts/audit_market_graph_round8.py'),
        artifact_sha256={p.name:sha256(p) for p in sorted(out.iterdir()) if p.is_file() and p.name!='artifact_audit.json'})
    (out/'artifact_audit.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='artifact_sha256'},indent=2),flush=True)


if __name__=='__main__':
    with threadpool_limits(limits=4):main()
