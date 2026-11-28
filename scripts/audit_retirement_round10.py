"""Independent raw-state reconstruction and saved-model audit for Round10 RET.

No model optimizer is repeated. The official evidence remains 10 verified and
20 unresolved sampled cases; this audit does not re-adjudicate the PDFs.
"""
from __future__ import annotations

from collections import Counter,defaultdict
import csv
import hashlib
import json
from math import exp,log1p
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.special import expit
from threadpoolctl import threadpool_limits

from scripts.audit_gating_round10 import equivalent,max_error,accuracy,loss
from scripts.event_state_features import EXCLUDED_PRIOR_IDS
from scripts.metrics import score_probabilities
from scripts.model_event_metadata import EVENT_METADATA
from scripts.train_models import ROOT,sha256,mirror_features,cluster_interval
from scripts.train_round5 import accuracy_interval
from scripts.train_retirement_round10 import predict_retirement

REFERENCE='previous_probability_recommendation'
NEW_NAMES=['ret_I_spline','ret_R_spline','ret_RC_spline']
SCALARS=['ordinary_gap','ordinary_return_recency','ordinary_return_state','ret_recency','ret_presence','ret_return_state']
NEW_COLUMNS=[f'{name}_{suffix}' for name in SCALARS for suffix in ('diff','sum')]


def day_number(value):return int(pd.Timestamp(str(value)).to_datetime64().astype('datetime64[D]').astype(int))


def independent_states(rows,inventory,entries):
    """Independent date-array masks and direct counts, without the feature builder."""
    completed=defaultdict(set);rets=defaultdict(set);statuses=Counter();ret_years=Counter()
    latest=max(day_number(v) for v in rows.tourney_date)-21
    completed_ids=set()
    for r in inventory.itertuples(index=False):
        d=day_number(r.tourney_date)
        if d>latest:continue
        assert r.match_id not in completed_ids
        completed_ids.add(r.match_id)
        assert str(r.player_a_id)!=str(r.player_b_id)
        completed[str(r.player_a_id)].add(d);completed[str(r.player_b_id)].add(d)
    observed_ret_ids=set()
    for entry in entries:
        season=int(entry['season']);assert 2006<=season<=2023
        path=ROOT/'data/raw'/f'{season}.csv';assert sha256(path)==entry['sha256']
        with path.open(encoding='utf-8-sig',newline='') as handle:
            for r in csv.DictReader(handle):
                if r['tourney_level'] not in ('250','500','M','G'):continue
                date_value=EVENT_METADATA[r['tourney_id']][0] if r['tourney_id'] in EVENT_METADATA and r['round'] not in ('Q1','Q2','Q3','Q4') else r['tourney_date']
                d=day_number(date_value)
                if d>latest:continue
                score=r.get('score','').strip().upper()
                flags=[]
                if 'RET' in score:flags.append('retirement')
                if 'W/O' in score or 'WALKOVER' in score:flags.append('walkover')
                if 'DEF' in score:flags.append('default')
                if not flags:continue
                status=flags[0] if len(flags)==1 else 'ambiguous';statuses[status]+=1
                if status!='retirement':continue
                key=f"{r['tourney_id']}:{int(r['match_num'])}"
                if key in EXCLUDED_PRIOR_IDS:
                    statuses['known_excluded_ret']+=1;continue
                assert key not in observed_ret_ids and r['winner_id']!=r['loser_id']
                observed_ret_ids.add(key);rets[r['loser_id']].add(d);ret_years[season]+=1
    completed={p:np.array(sorted(v),dtype=int) for p,v in completed.items()}
    returned={p:days[1:][np.diff(days)>=28] for p,days in completed.items()}
    rets={p:np.array(sorted(v),dtype=int) for p,v in rets.items()}
    empty=np.array([],dtype=int);cache={}
    def query(player,t):
        key=(player,t)
        if key in cache:return cache[key]
        cutoff=t-21;first=t-364
        c=completed.get(player,empty);past=c[c<=cutoff]
        gap=min(365,cutoff-int(past[-1])) if len(past) else 365
        def state(anchors):
            anchors=anchors[(anchors>=first)&(anchors<=cutoff)]
            if not len(anchors):return 0.,0.,0.
            a=int(anchors[-1]);recency=exp(-(cutoff-a)/90.)
            appearances=int(np.count_nonzero((c>a)&(c<=cutoff)))
            return recency,1.,recency*exp(-appearances/5.)
        ir,_,iq=state(returned.get(player,empty));rr,rp,rq=state(rets.get(player,empty))
        cache[key]=(log1p(gap)/log1p(365),ir,iq,rr,rp,rq)
        return cache[key]
    values=[];coverage=defaultdict(Counter)
    for row in rows.itertuples(index=False):
        t=day_number(row.tourney_date);a=query(str(row.player_a_id),t);b=query(str(row.player_b_id),t)
        values.append([value for left,right in zip(a,b) for value in (left-right,left+right)])
        coverage[int(row.season)]['targets']+=1
        coverage[int(row.season)]['either_ret_364']+=int(a[4]>0 or b[4]>0)
        coverage[int(row.season)]['both_ret_364']+=int(a[4]>0 and b[4]>0)
        coverage[int(row.season)]['either_ordinary_return']+=int(a[1]>0 or b[1]>0)
    return pd.DataFrame(values,index=rows.index,columns=NEW_COLUMNS),dict(
        source_rows_by_status=dict(statuses),ret_rows_by_season={str(k):v for k,v in ret_years.items()},
        completed_inventory_rows=len(completed_ids),ret_rows=len(observed_ret_ids),
        coverage_by_season={str(k):dict(v) for k,v in coverage.items()})


def rms(x):
    scale=np.sqrt(np.mean(np.asarray(x)**2,axis=0))
    return np.where(scale>1e-10,scale,1.)


def verify_preprocessor(model,train):
    spec=model.spec;values=train[spec['odd_columns']].to_numpy()
    maximum=max_error(model.odd_scale_,rms(values));z=np.clip(values/model.odd_scale_,-8,8)
    shapes=[]
    assert list(model.splines_)==spec['spline_columns']
    for name,t in model.splines_.items():
        assert t.degree==2 and t.include_bias and t.extrapolation=='constant'
        assert max_error(t.bsplines_[0].t,np.arange(-6,6.1,1.5))==0
        column=z[:,[spec['odd_columns'].index(name)]]
        shapes.append((t.transform(column)-t.transform(-column))/2)
    maximum=max(maximum,max_error(model.spline_scale_,rms(np.column_stack(shapes))))
    interaction=[]
    for odd,even in spec['interactions']:
        c=train[even].to_numpy();mean=float(c.mean());sd=max(float(c.std()),1e-10)
        maximum=max(maximum,abs(model.context_mean_[even]-mean),abs(model.context_scale_[even]-sd))
        interaction.append(z[:,spec['odd_columns'].index(odd)]*np.clip((c-mean)/sd,-3,3))
    maximum=max(maximum,max_error(model.interaction_scale_,rms(np.column_stack(interaction))))
    assert maximum<1e-12,maximum
    assert model.estimator_.C==.01 and not model.estimator_.fit_intercept
    assert spec['params']==dict(C=.01,shape_weight=.3,interaction_weight=.3,half_life=3.)
    design=model._design(train)
    assert design.shape[1]==model.design_width_==model.estimator_.n_features_in_
    manual=np.clip(expit(design@model.estimator_.coef_[0]),1e-8,1-1e-8)
    probability_error=max_error(manual,model.predict_proba(train));assert probability_error<1e-12
    return dict(scaler_max_error=maximum,training_rows=len(train),design_width=model.design_width_,
        manual_linear_head_error=probability_error,iterations=int(model.estimator_.n_iter_[0]))


def select(y,predictions,mask):
    limit=loss(y[mask],predictions[REFERENCE][mask])+.005
    allowed=[name for name,p in predictions.items() if loss(y[mask],p[mask])<=limit]
    name=min(allowed,key=lambda n:(-accuracy(y[mask],predictions[n][mask]),loss(y[mask],predictions[n][mask]),n))
    return dict(choice=name,selected_accuracy=accuracy(y[mask],predictions[name][mask]),
        selected_log_loss=loss(y[mask],predictions[name][mask]),loss_limit=limit,
        eligible_models=allowed,selection_rows=int(mask.sum()),mixture_search=False)


def main():
    out=ROOT/'results/round10/retirement';registry=json.loads((out/'search_registry.json').read_text())
    for key,path in {'protocol_sha256':'docs/research/round10_retirement_protocol.md',
        'feature_code_sha256':'scripts/retirement_features.py','runner_sha256':'scripts/train_retirement_round10.py',
        'learner_sha256':'scripts/round2_learning.py','inventory_sha256':'data/processed/mens_singles_2006_2025.csv',
        'target_sha256':'data/processed/model_features.csv','expanded_target_sha256':'results/round3/development_targets.csv'}.items():
        assert sha256(ROOT/path)==registry[key],path
    for key,name in {'source_audit_sha256':'source_feasibility.json','sample_registry_sha256':'sample_registry.json',
                     'official_reviews_sha256':'official_reviews.json'}.items():assert sha256(out/name)==registry[key]
    evidence=json.loads((out/'source_feasibility.json').read_text())
    assert evidence['review_counts']=={'verified':10,'unverified':20} and evidence['observed_official_contradictions']==0
    assert evidence['sample_rows']==30 and evidence['source_semantics_fully_verified'] is False
    assert evidence['model_fitting_ready'] is False and evidence['root_limited_experiment_authorized'] is True
    for key,name in [('sample_registry_sha256','sample_registry.json'),('official_reviews_sha256','official_reviews.json'),('official_sources_sha256','official_sources.json')]:
        assert evidence[key]==sha256(out/name)
    assert evidence['audit_code_sha256']==sha256(ROOT/'scripts/audit_retirement_source_round10.py')
    reviews=json.loads((out/'official_reviews.json').read_text());checks=evidence['sampled_checks']
    assert Counter(row['review']['status'] for row in checks)==Counter(verified=10,unverified=20)
    for row in checks:
        assert row['review']==reviews.get(row['match_id'],dict(status='unverified'))
        if row['review']['status']=='verified':
            source=row['official_source'];assert sha256(Path(source['cache_path']))==source['sha256']==row['review']['source_sha256']
    rows=pd.read_csv(ROOT/'results/round3/development_targets.csv');dates=pd.to_datetime(rows.tourney_date)
    base=pd.read_csv(ROOT/'results/round5/development_features.csv');x=pd.read_csv(out/'development_features.csv')
    assert x.shape==(36421,338) and list(x)==registry['features'] and max_error(x[list(base)],base)==0
    inventory=pd.read_csv(ROOT/'data/processed/mens_singles_2006_2025.csv',
        usecols=['source','season','match_id','tourney_date','player_a_id','player_b_id'])
    inventory=inventory[inventory.source.eq('tour')&inventory.season.between(2006,2023)]
    entries=registry['feature_metadata']['source_files'];assert [int(e['season']) for e in entries]==list(range(2006,2024))
    rebuilt,source=independent_states(rows,inventory,entries)
    feature_error=max_error(rebuilt,x[NEW_COLUMNS]);assert feature_error<1e-12
    for key in ('source_rows_by_status','ret_rows_by_season','completed_inventory_rows','coverage_by_season'):
        equivalent(source[key],registry['feature_metadata'][key])
    assert source['ret_rows']==1452 and source['ret_rows_by_season']['2023']==65
    groups=registry['feature_groups'];assert groups['base']==list(base)
    expected={'I':SCALARS[:3],'R':SCALARS[:5],'RC':SCALARS[:3]+[SCALARS[3],SCALARS[5]]}
    specs=registry['candidates'];assert [s['name'] for s in specs]==NEW_NAMES
    for spec,(kind,scalar_names) in zip(specs,expected.items()):
        added=[f'{n}_{suffix}' for n in scalar_names for suffix in ('diff','sum')]
        assert groups[kind]==added and spec['candidate_input_columns']==list(base)+added
        assert len(spec['candidate_input_columns'])=={'I':332,'R':336,'RC':336}[kind]
        assert set(groups['I'])<=set(added)
        assert all(c in spec['spline_columns'] for c in added if c.endswith('_diff'))
        assert all(['log_rank_diff',c] in spec['interactions'] for c in added if c.endswith('_sum'))
        assert not any(c.endswith('_sum') for c in spec['odd_columns'])
    assert registry['fit_configurations']==3 and registry['total_fits']==12 and registry['mixture_search'] is False
    bundle=joblib.load(out/'frozen_models.joblib');frozen=json.loads((out/'frozen_selection.json').read_text())
    assert bundle['frozen']==frozen and set(bundle['models'])==set(NEW_NAMES)
    assert bundle['feature_names']==list(x)
    records=json.loads((out/'fit_records.json').read_text());assert len(records)==12
    oof=pd.read_csv(out/'oof_predictions.csv');orows=rows[rows.season.isin([2020,2021,2022])]
    assert oof[['match_id','season','a_won']].equals(orows[['match_id','season','a_won']].reset_index(drop=True))
    assert list(oof.columns[3:])==[REFERENCE]+NEW_NAMES
    old_oof=pd.read_csv(ROOT/'results/round5/oof_predictions.csv');assert max_error(oof[REFERENCE],old_oof[REFERENCE])==0
    preds={REFERENCE:oof[REFERENCE].to_numpy()};fit_checks={};oof_error=0.;final_new={}
    for spec in specs:
        folds=joblib.load(out/f"fold_models_{spec['name']}.joblib");assert set(folds)=={2020,2021,2022}
        p=np.zeros(len(orows))
        for year in (2020,2021,2022,2023):
            valid=rows.season.eq(year);cutoff=dates[valid].min()-pd.Timedelta(days=21)
            train=rows.season.lt(year)&dates.le(cutoff)
            model=bundle['models'][spec['name']] if year==2023 else folds[year]
            assert model.spec==spec
            record=next(r for r in records if r['model']==spec['name'] and r['validation_season']==year)
            assert record['train_rows']==int(train.sum())
            assert record['train_max_date']==str(dates[train].max()) and record['validation_min_date']==str(dates[valid].min())
            assert record['training_match_ids_sha256']==hashlib.sha256('\n'.join(rows.loc[train,'match_id']).encode()).hexdigest()
            assert rows.season[train].max()<=2022 and dates[train].max()<=cutoff
            fit_checks[f"{spec['name']}_{year}"]=verify_preprocessor(model,x[train])
            probability=model.predict_proba(x[valid])
            if year==2023:final_new[spec['name']]=probability
            else:p[orows.season.eq(year).to_numpy()]=probability
        preds[spec['name']]=p;oof_error=max(oof_error,max_error(p,oof[spec['name']]))
    assert oof_error<1e-12
    y=orows.a_won.to_numpy();early=orows.season.isin([2020,2021]).to_numpy();gm=orows.season.eq(2022).to_numpy()
    early_choice=select(y,preds,early);pooled=select(y,preds,np.ones(len(y),dtype=bool))
    equivalent(early_choice,frozen['procedure_gate']['early_selection']);equivalent(pooled,frozen['pooled_selection'])
    gate=frozen['procedure_gate'];candidate=preds[early_choice['choice']][gm];reference=preds[REFERENCE][gm]
    gain=accuracy(y[gm],candidate)-accuracy(y[gm],reference);delta=loss(y[gm],candidate)-loss(y[gm],reference)
    assert abs(gain-gate['accuracy_improvement'])<1e-12
    assert abs(delta-(gate['candidate_2022_loss']-gate['reference_2022_loss']))<1e-12
    passed=gain>=.005-1e-12 and delta<=.005+1e-12
    assert gate['procedure']==('adopt_accuracy_model' if passed else 'retain_round4')
    assert frozen['recommendation']==(pooled['choice'] if passed else REFERENCE)
    cv=pd.read_csv(out/'temporal_cv.csv').set_index('name');metric_error=0.
    for name,p in preds.items():
        metric_error=max(metric_error,abs(accuracy(y,p)-cv.loc[name,'accuracy']),abs(loss(y,p)-cv.loc[name,'mean_log_loss']))
        for year in (2020,2021,2022):
            mask=orows.season.eq(year).to_numpy()
            metric_error=max(metric_error,abs(accuracy(y[mask],p[mask])-cv.loc[name,f'{year}_accuracy']),abs(loss(y[mask],p[mask])-cv.loc[name,f'{year}_log_loss']))
    valid=rows.season.eq(2023);vr=rows[valid];final=predict_retirement(bundle,x[valid])
    saved=pd.read_csv(out/'validation_predictions.csv');assert saved[list(rows)].equals(vr.reset_index(drop=True))
    assert final.shape==(2588,7);reload_error=max_error(final,saved[list(final)]);assert reload_error<1e-12
    for name in NEW_NAMES:assert max_error(final[name],final_new[name])==0
    old_final=pd.read_csv(ROOT/'results/round5/validation_predictions.csv')
    assert old_final.match_id.tolist()==vr.match_id.tolist()
    for alias,name in bundle['previous_aliases'].items():assert max_error(final[alias],old_final[name])<1e-12
    reverse=predict_retirement(bundle,mirror_features(x[valid]));swap_error=max_error(final+reverse,np.ones(final.shape));assert swap_error<1e-12
    shuffled=predict_retirement(bundle,x[valid].sample(frac=1,random_state=1010)).loc[x[valid].index]
    permutation_error=max_error(final,shuffled);assert permutation_error<1e-12
    assert np.array_equal(final.to_numpy()>=.5,shuffled.to_numpy()>=.5)
    poison=predict_retirement(bundle,x[valid].assign(a_won='POISON',score='POISON')[list(x)[::-1]+['a_won','score']])
    assert max_error(final,poison)==0
    table=pd.read_csv(out/'validation_comparison.csv').set_index('model')
    for name in final:
        score=score_probabilities(zip(vr.match_id,vr.a_won),zip(vr.match_id,final[name]))
        for key,value in score.items():metric_error=max(metric_error,abs(value-table.loc[name,key]))
        metric_error=max(metric_error,abs(np.mean((vr.a_won.to_numpy()-final[name].to_numpy())**2)-table.loc[name,'brier']))
    assert metric_error<1e-12
    matched=pd.read_csv(out/'matched_controls.csv')
    for r in matched.itertuples():
        assert abs(r.oof_accuracy_change-(accuracy(y,preds[r.candidate])-accuracy(y,preds[r.control])))<1e-12
        assert abs(r.oof_loss_change-(loss(y,preds[r.candidate])-loss(y,preds[r.control])))<1e-12
        assert abs(r.validation_accuracy_change-(accuracy(vr.a_won,final[r.candidate])-accuracy(vr.a_won,final[r.control])))<1e-12
        assert abs(r.validation_loss_change-(loss(vr.a_won,final[r.candidate])-loss(vr.a_won,final[r.control])))<1e-12
    summary=json.loads((out/'summary.json').read_text())
    for baseline,comparisons in summary['comparisons'].items():
        for name,record in comparisons.items():
            equivalent(dict(accuracy=accuracy_interval(vr,final[baseline].to_numpy(),final[name].to_numpy()),
                log_loss=cluster_interval(vr,final[baseline].to_numpy(),final[name].to_numpy())),record)
    result=dict(status='passed',auditor='independent /root/audit',no_model_refitting=True,final_test_evaluated=False,
        validation_matches=2588,probability_columns=7,oof_rows=6130,oof_probability_columns=4,
        saved_fitted_models_reloaded=12,fit_checks=fit_checks,old326_feature_max_error=0.,
        independently_rebuilt_state_values=36421*12,independent_feature_max_error=feature_error,
        independent_source_counts=source,source_years_read=list(range(2006,2024)),
        candidate_input_dimensions=[332,336,336],I_nested_in_R_and_RC=True,R_RC_same_design_width=True,
        oof_reload_max_error=oof_error,validation_reload_max_error=reload_error,
        swap_max_error=swap_error,permutation_max_error=permutation_error,changed_order_classifications=0,
        unused_label_column_error=0.,metrics_max_error=metric_error,
        independent_early_choice=early_choice,independent_pooled_choice=pooled,
        independent_gate_accuracy_gain=gain,independent_gate_loss_increase=delta,
        matched_controls_checked=True,paired_intervals_recomputed=14,
        official_verified_sample_rows=10,official_unresolved_sample_rows=20,
        observed_official_contradictions=0,source_semantics_fully_verified=False,
        bounded_source_authorization_hashes_verified=True,
        limitations=['No model optimization or full gradient trajectory was repeated; twelve saved fold/final estimators and their preprocessing were checked.',
          'This audit verifies the existing official-evidence manifest and cached source hashes, not a second human adjudication of PDFs. Twenty sampled records remain unresolved; no population error rate or clinical injury status is established.',
          'Feature reconstruction shares the established official date-correction table and known-exclusion list, while independently parsing dates/statuses and counting appearances.',
          'Raw sources were limited to 2006–2023. The mixed-year inventory was read only for identity/date/source metadata, then filtered; no final-year labels or annual 2024–2025 raw files were opened.',
          'Repeated 2023 exploration and 2022 gate reuse remain development limitations.'],
        audit_source_sha256=sha256(ROOT/'scripts/audit_retirement_round10.py'),
        artifact_sha256={p.name:sha256(p) for p in sorted(out.iterdir()) if p.is_file() and p.name!='artifact_audit.json'})
    assert all(fit_checks[f'ret_R_spline_{year}']['design_width']==fit_checks[f'ret_RC_spline_{year}']['design_width'] for year in (2020,2021,2022,2023))
    (out/'artifact_audit.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ('artifact_sha256','fit_checks','independent_source_counts')},indent=2))


if __name__=='__main__':
    with threadpool_limits(limits=4):main()
