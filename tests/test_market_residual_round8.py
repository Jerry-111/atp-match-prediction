import joblib
import numpy as np
import pandas as pd
import pytest
from scipy.optimize import check_grad
from scipy.special import expit
from scripts.train_market_residual_round8 import (ResidualModel, offset_objective, build_inputs,
    raw_probability, mirror_input, candidate_specs, predict_bundle, select_candidate, choose_procedure)


def fixture():
    rng=np.random.default_rng(158);n=80
    history=pd.DataFrame({'log_rank_diff':rng.normal(size=n),'event_sets_diff':rng.normal(size=n),
        'format_best_of_five':np.zeros(n),'surface_is_clay':np.zeros(n),'surface_is_grass':np.zeros(n)})
    market=pd.DataFrame({'market_open_logit_diff':rng.normal(size=n),'market_open_available':np.ones(n),
        'market_open_overround':np.full(n,.03),'market_close_logit_diff':rng.normal(size=n),
        'market_close_available':np.ones(n),'market_close_overround':np.full(n,.03),
        'market_movement_logit_diff':rng.normal(size=n),'market_both_available':np.ones(n)})
    reference=expit(rng.normal(size=n))
    rows=pd.DataFrame({'season':np.full(n,2021),'tourney_date':['2021-01-01']*n,'a_won':rng.integers(0,2,n)})
    return history,market,reference,rows


def test_offset_gradient_and_closed_form_zero_gradient():
    rng=np.random.default_rng(158);x=rng.normal(size=(50,5));y=rng.integers(0,2,50);z=rng.normal(size=50);b=rng.normal(size=5)
    assert check_grad(lambda b:offset_objective(b,x,y,z,.03)[0],lambda b:offset_objective(b,x,y,z,.03)[1],b)<1e-5
    q=expit(z)
    loss,gradient=offset_objective(np.zeros(5),x,q,z,.03)
    assert np.isfinite(loss)
    np.testing.assert_array_equal(gradient,np.zeros(5))


def test_quote_conditions_and_exact_no_quote_fallback():
    h,m,ref,_=fixture()
    opened=build_inputs(h,m,ref,'opening')
    changed=m.copy();changed.loc[:,[n for n in changed if '_close_' in n or '_movement_' in n]]=99999
    pd.testing.assert_frame_equal(opened,build_inputs(h,changed,ref,'opening'))
    m.loc[0,['market_close_available','market_open_available']]=0
    m.loc[1,'market_close_available']=0
    x=build_inputs(h,m,ref,'latest');p=raw_probability(x)
    assert p[0]==ref[0]
    assert p[1]==expit(m.market_open_logit_diff[1])
    assert p[2]==expit(m.market_close_logit_diff[2])
    np.testing.assert_allclose(raw_probability(mirror_input(x)),1-p,atol=2e-16)
    with pytest.raises(ValueError):
        build_inputs(h,m.assign(market_open_available=2),ref,'opening')


@pytest.mark.parametrize('kind',['linear','catboost'])
def test_models_swap_reload_poison_and_fallback(kind,tmp_path):
    h,m,ref,rows=fixture();x=build_inputs(h,m,ref,'latest')
    specs=candidate_specs([n for n in h if not n.startswith('event_')],['event_sets_diff'],'latest')
    spec=next(s for s in specs if s['group']=='event' and s['kind']==kind)
    if kind=='catboost': spec={**spec,'iterations':12}
    model=ResidualModel(spec).fit(x,rows)
    query=x.copy();query.loc[[0,3],'quote_available']=0
    bundle=dict(models={spec['name']:model},feature_names=list(x))
    p=predict_bundle(bundle,query)
    reverse=predict_bundle(bundle,mirror_input(query))
    np.testing.assert_allclose(reverse,1-p,atol=3e-16,rtol=0)
    for c in p:
        np.testing.assert_array_equal(p.loc[[0,3],c],ref[[0,3]])
    poison=query.assign(a_won=999,score=999,current_stat=999)
    pd.testing.assert_frame_equal(p,predict_bundle(bundle,poison))
    joblib.dump(bundle,tmp_path/'model.joblib')
    pd.testing.assert_frame_equal(p,predict_bundle(joblib.load(tmp_path/'model.joblib'),query))
    with pytest.raises(ValueError,match='2021/22'):
        ResidualModel(spec).fit(x,rows.assign(season=2023))


def test_catboost_prediction_baseline_is_added_once():
    from catboost import CatBoostClassifier,Pool
    x=np.arange(20)[:,None];y=np.arange(20)%2;z=np.linspace(-1,1,20)
    model=CatBoostClassifier(iterations=2,depth=1,verbose=False,allow_writing_files=False).fit(Pool(x,y,baseline=z))
    residual=model.predict(x,prediction_type='RawFormulaVal')
    total=model.predict(Pool(x,baseline=z),prediction_type='RawFormulaVal')
    np.testing.assert_allclose(total-residual,z,atol=5e-8,rtol=0)


def test_finite_registry_and_selection_uses_own_market_guard():
    h,_,_,_=fixture()
    for condition in ('opening','latest'):
        specs=candidate_specs(list(h),[],condition)
        assert len(specs)==9
        assert all(len(s['columns'])==len(set(s['columns'])) for s in specs)
        if condition=='opening': assert all('quote_movement_diff' not in s['columns'] for s in specs)
    y=np.tile([0,1],1000);ref=np.where(y==1,.6,.4)
    bad=ref.copy();bad[:200]=1-bad[:200]
    ps={'raw_market':ref,'worse':bad}
    mask=np.arange(len(y))<1000
    assert select_candidate(y,ps,mask)['choice']=='raw_market'
    gate=choose_procedure(y,ps,np.where(mask,'2021H2','2022'))
    assert gate['procedure']=='retain_raw_market'
