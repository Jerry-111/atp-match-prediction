"""Independently reproduce frozen residual artifacts; no fitting or tuning.

Run from the repository root with python -m scripts.audit_market_residual_round8.
"""
from pathlib import Path
import sys,json,hashlib
sys.path.insert(0,str(Path.cwd()))
import numpy as np,pandas as pd,joblib
from scipy.special import expit,logit
from threadpoolctl import threadpool_limits
from scripts.train_market_residual_round8 import predict_bundle,mirror_input
from scripts.metrics import score_probabilities
from scripts.train_models import sha256,cluster_interval
from scripts.train_round5 import accuracy_interval

root=Path(__file__).resolve().parents[1]; output=root/'results/round8/market_residual'
allrows=pd.read_csv('results/round3/development_targets.csv')
rows=allrows[allrows.season.isin([2021,2022,2023])].copy()
history=pd.read_csv('results/round7/event/development_features.csv').loc[rows.index]
market=pd.read_csv('results/round5/market/features.csv').set_index('match_id').loc[rows.match_id].set_axis(rows.index)
links=pd.read_csv('results/round5/market/linked_matches.csv').set_index('match_id').loc[rows.match_id].set_axis(rows.index)
utc=pd.to_datetime(links.market_match_utc,utc=True)
prior=pd.concat([pd.read_csv('results/round5/oof_predictions.csv'),pd.read_csv('results/round5/validation_predictions.csv')]).set_index('match_id').loc[rows.match_id].set_axis(rows.index)
assert prior[['season','a_won']].equals(rows[['season','a_won']])
ref=prior.previous_probability_recommendation.to_numpy()
dates=pd.to_datetime(rows.tourney_date)
def maxdiff(a,b):
    a,b=np.asarray(a),np.asarray(b)
    assert a.shape==b.shape
    return float(np.max(np.abs(a-b)))
def loss(y,p):
    p=np.clip(p,1e-8,1-1e-8);return float(-np.mean(y*np.log(p)+(1-y)*np.log1p(-p)))
def acc(y,p):return float(np.mean((np.asarray(p)>=.5)==np.asarray(y)))
def pick(y,p,mask):
    limit=loss(y[mask],p['raw_market'][mask])+.005
    eligible=[n for n,v in p.items() if loss(y[mask],v[mask])<=limit]
    return min(eligible,key=lambda n:(-acc(y[mask],p[n][mask]),loss(y[mask],p[n][mask]),n)),len(eligible),limit
def recursive(a,b):
    if isinstance(a,dict):
        assert set(a)==set(b)
        for k in a:recursive(a[k],b[k])
    elif isinstance(a,list):
        assert len(a)==len(b)
        for x,y in zip(a,b):recursive(x,y)
    elif isinstance(a,(float,int)) and not isinstance(a,bool):assert abs(a-b)<1e-12,(a,b)
    else:assert a==b,(a,b)
with threadpool_limits(limits=2):
  for condition in ['opening','latest']:
    out=output/condition
    registry=json.loads((out/'search_registry.json').read_text())
    assert registry['runner_sha256']==sha256(root/'scripts/train_market_residual_round8.py')
    revision=json.loads((output/'report_revision.json').read_text())
    current_protocol=sha256(root/'docs/research/round8_market_residual.md')
    assert current_protocol==revision['current_protocol_sha256']
    assert registry['runner_sha256']==revision['runner_unchanged_sha256']
    clarification='''
Post-fit interpretation clarification (no numerical or protocol-menu change):
"Opening" identifies the price field, not a verified prediction decision time.
For event-input variants, a player's earlier-round result is known before the
target match, but is not established to precede capture of that match's opening
quote. Consequently these are later-prematch predictions using opening prices,
not validated forecasts made at the instant those opening prices were posted.
'''
    doc=(root/'docs/research/round8_market_residual.md').read_text()
    assert doc.count(clarification)==1
    if registry['protocol_sha256']!=current_protocol:
        assert registry['protocol_sha256']==revision['at_fit_protocol_sha256']
        assert hashlib.sha256(doc.replace(clarification,'').encode()).hexdigest()==registry['protocol_sha256']
    for path,value in registry['source_sha256'].items():assert sha256(root/path)==value
    bundle=joblib.load(out/'frozen_models.joblib')
    x=pd.read_csv(out/'development_features.csv').set_axis(rows.index)
    expected=history.copy()
    close=market.market_close_available.eq(1).to_numpy() if condition=='latest' else np.zeros(len(rows),bool)
    available=close|market.market_open_available.eq(1).to_numpy()
    quote=np.where(close,market.market_close_logit_diff,market.market_open_logit_diff)
    expected['quote_logit_diff']=quote
    expected['quote_absolute_logit']=np.abs(quote)
    expected['quote_overround']=np.where(close,market.market_close_overround,market.market_open_overround)
    expected['quote_available']=available.astype(float)
    expected['reference_logit_diff']=logit(ref);expected['reference_probability']=ref
    if condition=='latest':
        expected['quote_movement_diff']=market.market_movement_logit_diff
        expected['quote_both_available']=market.market_both_available
        expected['quote_closing_available']=close.astype(float)
    assert list(expected)==list(x)==bundle['feature_names']==registry['feature_names']
    input_error=maxdiff(x,expected);assert input_error<1e-12
    if condition=='opening':assert not any(('movement' in c or 'closing' in c or 'close_' in c) for s in registry['candidates'] for c in s['columns'])
    v=rows.season.eq(2023);vr=rows[v];vx=x[v]
    p=predict_bundle(bundle,vx)
    saved=pd.read_csv(out/'validation_predictions.csv')
    assert saved[['match_id','season','tourney_date','a_won']].equals(vr[['match_id','season','tourney_date','a_won']].reset_index(drop=True))
    assert list(saved.columns[4:])==list(p) and len(p)==2588 and len(p.columns)==28
    reload_error=maxdiff(p,saved[list(p)]);assert reload_error<1e-12
    sw=predict_bundle(bundle,mirror_input(vx))
    swap_error=maxdiff(p+sw,np.ones(p.shape));assert swap_error<1e-12
    poison=predict_bundle(bundle,vx[vx.columns[::-1]].assign(a_won='NO',winner_id='NO',score='NO'))
    poison_error=maxdiff(p,poison);assert poison_error==0
    fallback=vx.quote_available.eq(0)
    assert maxdiff(p[fallback],np.repeat(vx.reference_probability[fallback].to_numpy()[:,None],28,axis=1))==0
    raw=np.where(available[v],expit(quote[v]),ref[v]);assert maxdiff(raw,p.raw_market)<1e-12
    table=pd.read_csv(out/'validation_comparison.csv').set_index('model')
    metric_error=0.
    for n in p:
        scored=score_probabilities(zip(vr.match_id,vr.a_won),zip(vr.match_id,p[n]))
        for k,value in scored.items():metric_error=max(metric_error,abs(value-table.loc[n,k]))
        assert scored['correct']==int(((p[n]>=.5)==vr.a_won).sum())
        assert abs(scored['mean_log_loss']-loss(vr.a_won.to_numpy(),p[n]))<1e-12
        assert abs(table.loc[n,'brier']-np.mean((vr.a_won-p[n])**2))<1e-12
    assert metric_error<1e-12
    oo=pd.read_csv(out/'oof_predictions.csv');orows=rows[(rows.season.eq(2021)&dates.ge('2021-07-01'))|rows.season.eq(2022)]
    assert oo[['match_id','season','a_won']].equals(orows[['match_id','season','a_won']].reset_index(drop=True))
    preds={n:oo[n].to_numpy() for n in p};y=oo.a_won.to_numpy()
    assert set(preds)==set(oo)-{'match_id','season','a_won'} and len(preds)==28
    early=oo.season.eq(2021).to_numpy();gate=~early
    early_name,early_n,early_limit=pick(y,preds,early)
    pooled_name,pooled_n,pooled_limit=pick(y,preds,np.ones(len(y),bool))
    frozen=json.loads((out/'frozen_selection.json').read_text())
    assert bundle['frozen']==frozen
    assert frozen['procedure_gate']['early_selection']['choice']==early_name
    assert frozen['pooled_selection']['choice']==pooled_name
    for select,name,n,limit,mask in [(frozen['procedure_gate']['early_selection'],early_name,early_n,early_limit,early),(frozen['pooled_selection'],pooled_name,pooled_n,pooled_limit,np.ones(len(y),bool))]:
        assert select['eligible_models']==n and select['selection_rows']==int(mask.sum())
        assert abs(select['loss_limit']-limit)<1e-12
        assert abs(select['accuracy']-acc(y[mask],preds[name][mask]))<1e-12
        assert abs(select['mean_log_loss']-loss(y[mask],preds[name][mask]))<1e-12
    gain=acc(y[gate],preds[early_name][gate])-acc(y[gate],preds['raw_market'][gate])
    lgain=loss(y[gate],preds[early_name][gate])-loss(y[gate],preds['raw_market'][gate])
    passed=gain>=.005-1e-12 and lgain<=.005+1e-12
    assert frozen['recommendation']==(pooled_name if passed else 'raw_market')
    assert frozen['procedure_gate']['procedure']==('adopt_residual' if passed else 'retain_raw_market')
    assert abs(frozen['procedure_gate']['accuracy_improvement']-gain)<1e-12
    assert abs(frozen['procedure_gate']['loss_change']-lgain)<1e-12
    cv=pd.read_csv(out/'temporal_cv.csv');cv_error=0.
    for _,row in cv.iterrows():
        mask=np.ones(len(y),bool) if row.fold=='pooled' else (early if row.fold=='2021H2' else gate)
        cv_error=max(cv_error,abs(row.accuracy-acc(y[mask],preds[row.model][mask])),abs(row.mean_log_loss-loss(y[mask],preds[row.model][mask])))
    assert cv_error<1e-12 and len(cv)==84
    populations=pd.read_csv(out/'fit_populations.csv');fold_counts={}
    for fold in ['2021H2','2022','2023']:
        valid=(rows.season.eq(2021)&dates.ge('2021-07-01')) if fold=='2021H2' else rows.season.eq(int(fold))
        cutoff=dates[valid].min()-pd.Timedelta(days=21)
        tr=rows.season.between(2021,2022)&dates.le(cutoff)&~valid&available&utc.le(cutoff.tz_localize('UTC'))
        part=populations[populations.fold.eq(fold)]
        assert len(part)==(1 if fold=='2023' else 9)
        assert part.fit_rows.eq(tr.sum()).all() and part.validation_rows.eq(valid.sum()).all()
        assert (pd.to_datetime(part.cutoff)==cutoff).all()
        assert (pd.to_datetime(part.last_market_utc,utc=True)==utc[tr].max()).all()
        assert utc[tr].max()<=cutoff.tz_localize('UTC') and rows[tr].season.le(2022).all()
        fold_counts[fold]={'fit':int(tr.sum()),'validation':int(valid.sum()),'cutoff':str(cutoff)}
        if fold=='2023':final=tr
    independent_prediction_error=0.;scale_error=0.;max_gradient=0.
    for name,m in bundle['models'].items():
        assert m.training_rows_==int(final.sum())
        s=m.spec;xx=vx[s['columns']].to_numpy()
        if s['kind']=='linear':
            training=x.loc[final,s['columns']].to_numpy();scale=np.sqrt(np.mean(training**2,axis=0));scale[scale<1e-10]=1
            scale_error=max(scale_error,maxdiff(scale,m.scale_))
            residual=np.clip(xx/scale,-8,8)@m.beta_
            design=np.clip(training/scale,-8,8);eta=x.quote_logit_diff[final].to_numpy()+design@m.beta_
            gradient=(design.T@(expit(eta)-rows.a_won[final].to_numpy())+m.beta_/s['C'])/int(final.sum())
            max_gradient=max(max_gradient,float(np.max(np.abs(gradient))))
        else:
            params=m.estimator_.get_all_params()
            assert params['depth']==3 and params['iterations']==300 and params['l2_leaf_reg']==50
            assert params['boost_from_average']==False
            a=m.estimator_.predict(xx,prediction_type='RawFormulaVal')
            b=m.estimator_.predict(mirror_input(vx)[s['columns']].to_numpy(),prediction_type='RawFormulaVal')
            residual=.5*(a-b)
        for alpha in [.25,.5,1.]:
            probability=np.where(available[v],np.clip(expit(quote[v]+alpha*residual),1e-8,1-1e-8),ref[v])
            independent_prediction_error=max(independent_prediction_error,maxdiff(probability,p[f'{name}_alpha{alpha:g}']))
    assert scale_error<1e-12 and independent_prediction_error<1e-12 and max_gradient<1e-5
    summary=json.loads((out/'summary.json').read_text())
    for name in p:
        interval={'accuracy':accuracy_interval(vr,p.raw_market.to_numpy(),p[name].to_numpy()),'log_loss':cluster_interval(vr,p.raw_market.to_numpy(),p[name].to_numpy())}
        recursive(interval,summary['comparisons_vs_raw_market'][name])
    assert summary['fallback_rows']==int(fallback.sum()) and summary['available_quotes']==int(available[v].sum())
    result={'audit_agent':'independent /root/audit','status':'passed','condition':condition,
      'validation_rows':2588,'validation_probability_columns':28,'oof_rows':len(oo),'oof_columns':28,
      'input_rebuild_max_error':input_error,'reload_max_error':reload_error,'swap_max_error':swap_error,
      'label_poison_and_column_order_max_error':poison_error,'validation_metrics_max_error':metric_error,
      'oof_metrics_max_error':cv_error,'single_selection_and_gate_independently_reproduced':True,
      'final_train_only_rms_max_error':scale_error,'independent_residual_formula_max_error':independent_prediction_error,
      'final_offset_logistic_gradient_max_abs':max_gradient,'exact_r4_fallback_rows':int(fallback.sum()),
      'date_block_intervals_recomputed':28,'fold_populations':fold_counts,
      'opening_has_no_closing_or_movement_input':condition=='opening',
      'temporal_fit_checks':'Earlier 2021/22 only, tournament date and actual UTC each <= fold start minus21 days.',
      'limits':['No OOF training or final model fitting repeated. Stored OOF predictions, selectors, final scales and residual formulas checked.',
                'Later-prematch input condition; opening quote timestamps unknown and earlier-round information is not established to precede opening capture.',
                'Latest condition uses opening in2021 because closing is entirely missing; later coverage shift remains.',
                '2023 comparisons exploratory after repeated prior inspection; 2024/25 not read.'],
      'source_code_sha256':sha256(root/'scripts/train_market_residual_round8.py'),
      'at_fit_protocol_sha256':registry['protocol_sha256'],
      'current_protocol_sha256':sha256(root/'docs/research/round8_market_residual.md'),
      'report_revision':revision,
      'audit_code_sha256':sha256(root/'scripts/audit_market_residual_round8.py'),
      'artifact_sha256':{p.name:sha256(p) for p in sorted(out.iterdir()) if p.is_file() and p.name!='artifact_audit.json'}}
    (out/'artifact_audit.json').write_text(json.dumps(result,indent=2)+'\n')
    print(condition,json.dumps({k:v for k,v in result.items() if k not in ['artifact_sha256','limits']}),flush=True)
