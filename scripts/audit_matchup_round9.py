"""Independently reconstruct the frozen skew-style design and its evidence."""
from __future__ import annotations
from itertools import combinations
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from scipy.special import expit
from threadpoolctl import threadpool_limits
from scripts.train_models import ROOT, sha256, mirror_features, temporal_masks
from scripts.train_matchup_round9 import predict_matchup
from scripts.report_round7 import verify_scores

OUT=ROOT/'results/round9/matchup'
STYLES=('ace','double_fault','first_in','first_won','second_won','return_first_won','return_second_won')
PAIRS=list(combinations(range(7),2))


def loss(y,p):
    y,p=np.asarray(y),np.clip(np.asarray(p),1e-15,1-1e-15)
    return float(np.mean(-y*np.log(p)-(1-y)*np.log1p(-p)))


def styles(x):
    sums=x[[f'style_{name}_sum' for name in STYLES]].to_numpy()
    diffs=x[[f'style_{name}_diff' for name in STYLES]].to_numpy()
    return (sums+diffs)/2,(sums-diffs)/2


def independent_parameters(train):
    a,b=styles(train)
    players=np.r_[a,b]
    center=players.mean(axis=0); sd=players.std(axis=0)
    sd=np.where(sd>1e-10,sd,1.)
    aa,bb=np.clip((a-center)/sd,-4,4),np.clip((b-center)/sd,-4,4)
    z=np.column_stack([aa[:,i]*bb[:,j]-aa[:,j]*bb[:,i] for i,j in PAIRS])
    rms=np.sqrt(np.mean(z*z,axis=0)); rms=np.where(rms>1e-10,rms,1.)
    odd=[c for c in train if c.endswith('_diff')]
    main=train[odd].to_numpy()
    scale=np.sqrt(np.mean(main*main,axis=0));scale=np.where(scale>0,scale,1.)
    return dict(center=center,sd=sd,rms=rms,odd=odd,main_scale=scale)


def design(x,parameters,structure):
    main=x[parameters['odd']].to_numpy()/parameters['main_scale']
    if structure=='no_new_interaction':return main
    a,b=styles(x)
    aa=np.clip((a-parameters['center'])/parameters['sd'],-4,4)
    bb=np.clip((b-parameters['center'])/parameters['sd'],-4,4)
    z=np.column_stack([aa[:,i]*bb[:,j]-aa[:,j]*bb[:,i] for i,j in PAIRS])/parameters['rms']
    parts=[main,.2*z]
    if structure=='surface':
        parts += [.1*z*x[f'surface_is_{name}'].to_numpy()[:,None] for name in ('hard','clay','grass')]
    return np.column_stack(parts)


def audit():
    read=lambda name:json.loads((OUT/name).read_text())
    bundle,registry,frozen=joblib.load(OUT/'frozen_models.joblib'),read('search_registry.json'),read('frozen_selection.json')
    assert bundle['frozen']==frozen and not frozen['final_test_evaluated']
    for field,file in [('feature_code_sha256','scripts/matchup_features.py'),
                       ('runner_sha256','scripts/train_matchup_round9.py'),
                       ('selector_code_sha256','scripts/train_venue_round8.py'),
                       ('protocol_sha256','docs/research/round9_matchup.md'),
                       ('target_sha256','results/round3/development_targets.csv')]:
        assert registry[field]==sha256(ROOT/file),field
    rows=pd.read_csv(ROOT/'results/round3/development_targets.csv')
    x=pd.read_csv(OUT/'development_features.csv')
    old=pd.read_csv(ROOT/'results/round5/development_features.csv')
    assert np.array_equal(x,old) and x.shape==(36421,326)
    train,valid=temporal_masks(rows,2023)
    assert train.sum()==33833 and valid.sum()==2588 and rows.loc[train,'season'].max()==2022
    vr=rows[valid]
    saved=pd.read_csv(OUT/'validation_predictions.csv')
    table=pd.read_csv(OUT/'validation_comparison.csv')
    verify_scores(table,saved,vr)
    p=predict_matchup(bundle,x[valid]); reverse=predict_matchup(bundle,mirror_features(x[valid]))
    reload_error=float(np.abs(p.to_numpy()-saved[p.columns].to_numpy()).max())
    swap_error=float(np.abs(p.to_numpy()+reverse.to_numpy()-1).max())
    assert reload_error<1e-14 and swap_error<1e-12
    poisoned=x[valid].copy();poisoned['a_won']=-999;poisoned['target_score']=999
    assert np.array_equal(p,predict_matchup(bundle,poisoned))
    params=independent_parameters(x[train]); design_error=0.;prob_error=0.;gradient=0.;transform_error=0.
    coefficients=pd.read_csv(OUT/'coefficients.csv')
    contribution=pd.read_csv(OUT/'logit_contributions.csv').set_index('model')
    for name,m in bundle['models'].items():
        assert m.training_rows_==33833 and m.fit_max_season_==2022 and m.estimator_.n_iter_[0]<2000
        assert m.spec['C']==.01 and m.spec['half_life']==3.
        assert m.spec['main_columns']==params['odd']
        transform_error=max(transform_error,float(np.abs(m.main_scale_-params['main_scale']).max()))
        if m.transformer_ is not None:
            for actual,expected in [(m.transformer_.center_,params['center']),(m.transformer_.scale_,params['sd']),
                                    (m.transformer_.interaction_rms_,params['rms'])]:
                transform_error=max(transform_error,float(np.abs(actual-expected).max()))
        xx=design(x[train],params,m.spec['structure'])
        vv=design(x[valid],params,m.spec['structure'])
        design_error=max(design_error,float(np.abs(xx-m.design(x[train])).max()),float(np.abs(vv-m.design(x[valid])).max()))
        beta=m.estimator_.coef_[0]
        prob_error=max(prob_error,float(np.abs(np.clip(expit(vv@beta),1e-8,1-1e-8)-p[name]).max()))
        yy=rows.loc[train,'a_won'].to_numpy(); weights=np.exp2(-(2022-rows.loc[train,'season'].to_numpy())/3.)
        residual=weights*(expit(xx@beta)-yy)
        gradient=max(gradient,float(np.max(np.abs(xx.T@residual+beta/.01))/weights.sum()))
        cc=coefficients[coefficients.model.eq(name)]
        assert list(cc.feature)==m.design_names_ and np.allclose(cc.coefficient,beta,rtol=0,atol=1e-15)
        products=vv*beta
        for key,values in [('main',products[:,:188]),('shared',products[:,188:209]),('surface',products[:,209:])]:
            magnitude=float(np.abs(values.sum(axis=1)).mean())
            assert abs(magnitude-contribution.loc[name,key+'_mean_absolute_logit'])<1e-12
    assert transform_error<1e-10 and design_error<1e-10 and prob_error<1e-12 and gradient<1e-4
    oof=pd.read_csv(OUT/'oof_predictions.csv'); expected=rows[rows.season.isin([2020,2021,2022])]
    assert oof[['match_id','season','a_won']].equals(expected[['match_id','season','a_won']].reset_index(drop=True))
    cv=pd.read_csv(OUT/'temporal_cv.csv')
    for _,r in cv.iterrows():
        for period in (None,2020,2021,2022):
            mask=np.ones(len(oof),bool) if period is None else oof.season.eq(period).to_numpy()
            yy,pp=oof.loc[mask,'a_won'].to_numpy(),oof.loc[mask,r['name']].to_numpy()
            prefix='' if period is None else str(period)+'_'
            assert abs(r[prefix+'accuracy']-np.mean((pp>=.5)==yy))<1e-14
            assert abs(r[prefix+'log_loss' if period is not None else 'mean_log_loss']-loss(yy,pp))<1e-14
    ref='previous_probability_recommendation';menu=[ref]+list(bundle['models'])
    def choice(mask):
        yy=oof.loc[mask,'a_won'];limit=loss(yy,oof.loc[mask,ref])+.005
        allowed=[n for n in menu if loss(yy,oof.loc[mask,n])<=limit+1e-12]
        return min(allowed,key=lambda n:(-np.mean((oof.loc[mask,n]>=.5)==yy),loss(yy,oof.loc[mask,n]),n))
    early=choice(oof.season.lt(2022));pooled=choice(np.ones(len(oof),bool))
    assert early==frozen['procedure_gate']['early_selection']['choice'] and pooled==frozen['pooled_selection']['choice']
    gate=oof[oof.season.eq(2022)]
    gain=np.mean((gate[early]>=.5)==gate.a_won)-np.mean((gate[ref]>=.5)==gate.a_won)
    assert abs(gain-frozen['procedure_gate']['accuracy_improvement'])<1e-14
    assert not (gain>=.005-1e-12 and loss(gate.a_won,gate[early])-loss(gate.a_won,gate[ref])<=.005+1e-12)
    assert frozen['recommendation']==ref
    populations=pd.read_csv(OUT/'fit_populations.csv')
    for r in populations.itertuples():
        fit,query=temporal_masks(rows,int(r.fold))
        assert r.training_rows==fit.sum() and r.validation_rows==query.sum()
        assert pd.to_datetime(rows.loc[fit,'tourney_date']).max()<=pd.to_datetime(rows.loc[query,'tourney_date']).min()-pd.Timedelta(days=21)
    for r in pd.read_csv(OUT/'matched_controls.csv').itertuples():
        for source,prefix in [(oof,'oof'),(saved,'validation')]:
            yy=source.a_won; a,b=source[r.candidate],source[r.control]
            assert abs(getattr(r,prefix+'_accuracy_change')-(np.mean((a>=.5)==yy)-np.mean((b>=.5)==yy)))<1e-14
            assert abs(getattr(r,prefix+'_loss_change')-(loss(yy,a)-loss(yy,b)))<1e-14
    oldp=pd.read_csv(ROOT/'results/round8/venue/validation_predictions.csv').venue_base_linear
    parity_error=float(np.abs(saved.matchup_no_new_interaction-oldp).max())
    assert parity_error<1e-8 and np.array_equal(saved.matchup_no_new_interaction>=.5,oldp>=.5)
    result=dict(status='passed',auditor='coordinator, separate from implementation',validation_matches=2588,
        probability_columns=len(p.columns),reload_max_error=reload_error,swap_max_error=swap_error,
        independent_design_max_error=design_error,independent_transform_max_error=transform_error,
        independent_prediction_max_error=prob_error,independent_scaled_gradient_max=gradient,
        source326_unchanged=True,cohort_and_metrics_verified=True,all_four_oof_columns_and_temporal_cv_verified=True,
        independent_selection_and_purge_verified=True,all_matched_controls_verified=True,
        coefficient_and_contribution_tables_verified=True,earlier_venue_baseline_probability_error=parity_error,
        earlier_venue_baseline_classifications_identical=True,target_label_and_score_poisoning_unchanged=True,
        final_test_evaluated=False,
        limitations='No OOF model refit and no independent rerun of paired bootstrap intervals; final designs/transforms/objective gradients and all OOF metrics were independently checked.',
        artifact_sha256={f.name:sha256(f) for f in OUT.iterdir() if f.is_file() and f.name!='artifact_audit.json'},
        audit_script_sha256=sha256(Path(__file__)))
    (OUT/'artifact_audit.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='artifact_sha256'},indent=2))


if __name__=='__main__':
    with threadpool_limits(limits=4):audit()
