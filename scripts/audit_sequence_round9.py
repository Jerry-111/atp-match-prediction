"""Independent source, normalizer, NumPy-forward and frozen-output checks."""
from __future__ import annotations
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from scipy.special import expit
import torch
from threadpoolctl import threadpool_limits
from scripts.train_models import ROOT,sha256,temporal_masks,mirror_features
from scripts.sequence_features import load_sequences,build_sequence_features
from scripts.sequence_learning import predict_average
from scripts.train_sequence_round9 import predict_sequence_bundle

OUT=ROOT/'results/round9/sequence'


def loss(y,p):
    y,p=np.asarray(y),np.clip(np.asarray(p),1e-15,1-1e-15)
    return float(np.mean(-y*np.log(p)-(1-y)*np.log1p(-p)))


def verify_scores(table,predictions,identity):
    assert len(predictions)==2588 and not predictions.match_id.duplicated().any()
    expected=identity.set_index('match_id').a_won
    actual=predictions.set_index('match_id').a_won
    assert set(actual.index)==set(expected.index) and actual.loc[expected.index].equals(expected)
    y=predictions.a_won.to_numpy()
    for r in table.itertuples():
        p=predictions[r.model].to_numpy()
        assert np.isfinite(p).all() and np.all((p>=0)&(p<=1))
        correct=int(np.sum((p>=.5)==y))
        assert r.matches==2588 and r.correct==correct and abs(r.accuracy-correct/2588)<1e-14
        assert abs(r.mean_log_loss-loss(y,p))<1e-14 and abs(r.brier-np.mean((p-y)**2))<1e-14


def normalize_parameters(x,seq):
    values=x.to_numpy(dtype=float);center=values.mean(0)
    center[[c.endswith('_diff') for c in x]]=0
    scale=np.maximum(np.sqrt(np.mean((values-center)**2,axis=0)),1e-6)
    tokens=seq.values[seq.mask].astype(float)
    tc=tokens.mean(0) if len(tokens) else np.zeros(22)
    ts=np.maximum(tokens.std(0),1e-6) if len(tokens) else np.ones(22)
    return center,scale,tc,ts,len(tokens)


def manual_forward(member,x,seq):
    """Float64 NumPy reconstruction, independent of the Torch module forward."""
    norm=member.normalizer;s=member.state;n=len(x)
    static=((x.to_numpy()-norm.static_center)/norm.static_scale).astype(np.float32).astype(float)
    tokens=((seq.values-norm.token_center)/norm.token_scale).astype(np.float32).astype(float)
    tokens[~seq.mask]=0
    tokens=tokens.reshape(n*2,16,22);mask=seq.mask.reshape(n*2,16)
    if member.architecture=='sequence_shuffled_gru':
        reordered=tokens.copy()
        for i,length in enumerate(mask.sum(1)):
            if length>1:
                permutation=np.random.default_rng(901+int(length)).permutation(int(length))
                reordered[i,-length:,:20]=tokens[i,16-length+permutation,:20]
        tokens=reordered
    projected=np.tanh(tokens@s['projection.weight'].astype(float).T+s['projection.bias'].astype(float))
    if member.architecture=='sequence_mean':
        state=(projected*mask[:,:,None]).sum(1)/np.maximum(mask.sum(1),1)[:,None]
    else:
        state=np.zeros((n*2,16))
        for t in range(16):
            a=projected[:,t]@s['cell.weight_ih'].astype(float).T+s['cell.bias_ih'].astype(float)
            b=state@s['cell.weight_hh'].astype(float).T+s['cell.bias_hh'].astype(float)
            reset=expit(a[:,:16]+b[:,:16]);update=expit(a[:,16:32]+b[:,16:32])
            candidate=np.tanh(a[:,32:]+reset*b[:,32:])
            proposed=(1-update)*candidate+update*state
            state=np.where(mask[:,t,None],proposed,state)
    state=(state*mask.any(1)[:,None]).reshape(n,2,16)
    def head(v):
        hidden=np.tanh(v@s['head.0.weight'].astype(float).T+s['head.0.bias'].astype(float))
        return (hidden@s['head.2.weight'].astype(float).T+s['head.2.bias'].astype(float)).ravel()
    first=head(np.c_[static,state[:,0],state[:,1]])
    second=head(np.c_[static*s['parity'],state[:,1],state[:,0]])
    return expit((first-second)/2)


def audit():
    read=lambda name:json.loads((OUT/name).read_text())
    bundle=joblib.load(OUT/'frozen_models.joblib');folds=joblib.load(OUT/'oof_models.joblib')['models']
    registry,frozen=read('search_registry.json'),read('frozen_selection.json')
    assert bundle['frozen']==frozen and not frozen['final_test_evaluated']
    for field,file in [('runner_sha256','scripts/train_sequence_round9.py'),('learner_sha256','scripts/sequence_learning.py'),
        ('feature_code_sha256','scripts/sequence_features.py'),('protocol_sha256','docs/research/round9_sequence.md'),
        ('target_sha256','results/round3/development_targets.csv'),('sequence_sha256','results/round9/sequence/sequence_features.npz')]:
        assert registry[field]==sha256(ROOT/file),field
    rows=pd.read_csv(ROOT/'results/round3/development_targets.csv');x=pd.read_csv(ROOT/'results/round5/development_features.csv')
    seq=load_sequences(OUT/'sequence_features.npz');train,valid=temporal_masks(rows,2023)
    assert len(rows)==36421 and train.sum()==33833 and valid.sum()==2588
    assert np.array_equal(seq.match_ids,rows.match_id) and np.array_equal(seq.indices,rows.index)
    cutoff=np.array([(pd.Timestamp(d)-pd.Timedelta(days=21)).date().toordinal() for d in rows.tourney_date])[:,None,None]
    assert np.all((~seq.mask)|((seq.dates<=cutoff)&(seq.dates>=cutoff-1460)))
    assert not np.any((np.diff(seq.dates,axis=2)<=0)&seq.mask[:,:,1:]&seq.mask[:,:,:-1])
    inventory=pd.read_csv(ROOT/'data/processed/mens_singles_2006_2025.csv',dtype={'a_won':'string'})
    rebuilt=build_sequence_features(rows.drop(columns=['a_won']),inventory[inventory.season.le(2023)])
    assert np.array_equal(seq.values,rebuilt.values) and np.array_equal(seq.mask,rebuilt.mask) and np.array_equal(seq.dates,rebuilt.dates)
    assert max(e['season'] for e in registry['source_metadata']['source_files'])<=2023
    vr,vs=rows[valid],seq.take(valid.to_numpy());table=pd.read_csv(OUT/'validation_comparison.csv')
    saved=pd.read_csv(OUT/'validation_predictions.csv');verify_scores(table,saved,vr)
    p=predict_sequence_bundle(bundle,x[valid],vs)
    reverse=predict_sequence_bundle(bundle,mirror_features(x[valid]),vs.swapped())
    reload_error=float(np.abs(p.to_numpy()-saved[p.columns].to_numpy()).max())
    swap_error=float(np.abs(p.to_numpy()+reverse.to_numpy()-1).max())
    assert reload_error<1e-12 and swap_error<2e-7
    oof=pd.read_csv(OUT/'oof_predictions.csv');seed_oof=pd.read_csv(OUT/'oof_seed_predictions.csv')
    expected=rows[rows.season.isin([2020,2021,2022])]
    assert oof[['match_id','season','a_won']].equals(expected[['match_id','season','a_won']].reset_index(drop=True))
    assert seed_oof[['match_id','season','a_won']].equals(oof[['match_id','season','a_won']])
    final_seed=pd.read_csv(OUT/'validation_seed_predictions.csv')
    normalizer_error=0.;manual_error=0.;oof_error=0.;chunk_error=0.;members_checked=0;classification_changes=0
    for year in (2020,2021,2022,2023):
        fit,query=temporal_masks(rows,year);tr=rows[fit];tx=x[fit];ts=seq.take(fit.to_numpy())
        current=normalize_parameters(tx,ts)
        tail=tr.season.eq(tr.season.max());inner_cutoff=pd.to_datetime(tr.loc[tail,'tourney_date']).min()-pd.Timedelta(days=21)
        inner=tr.season.lt(tr.season.max())&pd.to_datetime(tr.tourney_date).le(inner_cutoff)
        earlier=normalize_parameters(tx[inner],ts.take(inner.to_numpy()))
        qx=x[query];qs=seq.take(query.to_numpy())
        for kind in registry['architectures']:
            models=bundle['models'][kind] if year==2023 else folds[kind,year]
            assert tuple(m.seed for m in models)==(158,2718)
            individual=[]
            for m in models:
                members_checked+=1;meta=m.metadata;norm=m.normalizer
                assert meta['refit_rows']==fit.sum() and meta['inner_training_rows']==inner.sum() and meta['inner_validation_rows']==tail.sum()
                assert meta['inner_cutoff']==str(inner_cutoff.date()) and meta['inner_validation_season']==int(tr.season.max())
                assert norm.training_rows==fit.sum() and norm.training_valid_tokens==current[-1]
                for actual,want in zip((norm.static_center,norm.static_scale,norm.token_center,norm.token_scale),current[:4]):
                    normalizer_error=max(normalizer_error,float(np.max(np.abs(actual-want))))
                for key,want in zip(('static_center','static_scale','token_center','token_scale'),earlier[:4]):
                    normalizer_error=max(normalizer_error,float(np.max(np.abs(np.array(meta['inner_'+key])-want))))
                path=meta['inner_history'];best=min(path,key=lambda r:r['inner_log_loss'])['epoch']
                assert best==meta['selected_epoch'] and len(meta['refit_history'])==best
                assert len(path)<=30 and len(path)>=8
                mp=m.predict_proba(qx,qs);individual.append(mp)
                observed=final_seed[f'{kind}_seed{m.seed}'].to_numpy() if year==2023 else seed_oof.loc[seed_oof.season.eq(year),f'{kind}_seed{m.seed}'].to_numpy()
                oof_error=max(oof_error,float(np.max(np.abs(mp-observed))))
                if year==2023:
                    subset=np.unique(np.linspace(0,len(qx)-1,128,dtype=int))
                    manual_error=max(manual_error,float(np.max(np.abs(manual_forward(m,qx.iloc[subset],qs.take(subset))-mp[subset]))))
                    small=m.predict_proba(qx,qs,batch_size=17)
                    chunk_error=max(chunk_error,float(np.max(np.abs(small-mp))))
                    classification_changes+=int(np.sum((small>=.5)!=(mp>=.5)))
            avg=np.mean(individual,axis=0)
            observed=p[kind].to_numpy() if year==2023 else oof.loc[oof.season.eq(year),kind].to_numpy()
            assert np.max(np.abs(avg-observed))<1e-12
    assert members_checked==24 and normalizer_error<1e-12 and oof_error<1e-12 and manual_error<2e-6 and chunk_error<2e-6
    assert classification_changes==0
    ref='previous_probability_recommendation'
    for source in (oof,saved):
        for spec in registry['candidates']:
            assert np.max(np.abs(source[spec['name']]-(spec['weight']*source[spec['architecture']]+(1-spec['weight'])*source[ref])))<1e-12
    for metric in pd.read_csv(OUT/'temporal_cv.csv').itertuples():
        part=oof if metric.fold=='pooled' else oof[oof.season.eq(int(metric.fold))]
        assert metric.matches==len(part)
        assert abs(metric.accuracy-np.mean((part[metric.model]>=.5)==part.a_won))<1e-14
        assert abs(metric.mean_log_loss-loss(part.a_won,part[metric.model]))<1e-14
    menu=[s['name'] for s in registry['candidates']]+[ref]
    def choose(mask):
        yy=oof.loc[mask,'a_won'];limit=loss(yy,oof.loc[mask,ref])+.005
        allowed=[n for n in menu if loss(yy,oof.loc[mask,n])<=limit+1e-12]
        return min(allowed,key=lambda n:(-np.mean((oof.loc[mask,n]>=.5)==yy),loss(yy,oof.loc[mask,n]),n))
    early=choose(oof.season.lt(2022));pooled=choose(np.ones(len(oof),bool))
    assert early==frozen['procedure_gate']['early_selection']['choice'] and pooled==frozen['pooled_selection']['choice']
    gate=oof[oof.season.eq(2022)]
    gain=np.mean((gate[early]>=.5)==gate.a_won)-np.mean((gate[ref]>=.5)==gate.a_won)
    adopted=gain>=.005-1e-12 and loss(gate.a_won,gate[early])-loss(gate.a_won,gate[ref])<=.005+1e-12
    assert abs(gain-frozen['procedure_gate']['accuracy_improvement'])<1e-14
    assert frozen['recommendation']==(pooled if adopted else ref)
    result=dict(status='passed',auditor='coordinator, independent of implementation',validation_matches=2588,
        probability_columns=len(p.columns),individual_seed_probability_columns=6,
        full_sequence_shape=list(seq.values.shape),full_sequence_rebuild_exact=True,all_sequence_dates_causal=True,
        verified_saved_seed_models=members_checked,normalizer_max_error=normalizer_error,
        oof_and_individual_seed_reload_error=oof_error,reload_max_error=reload_error,swap_max_error=swap_error,
        independent_numpy_forward_max_error=manual_error,independent_numpy_forward_rows_per_final_seed=128,
        chunk17_max_error=chunk_error,chunk_classification_changes=classification_changes,
        independent_selection_gate_and_fixed_blends_verified=True,all_metrics_and_cv_verified=True,
        final_test_evaluated=False,
        limitations='No retraining or replay of gradient trajectories; recorded inner losses choose epochs correctly, but their numeric values are not independently replayed. Full feature rebuild uses original builder; separate raw-token source audit checks independent arithmetic on sampled histories.',
        artifact_sha256={f.name:sha256(f) for f in OUT.iterdir() if f.is_file() and f.name!='artifact_audit.json'},
        audit_script_sha256=sha256(Path(__file__)))
    (OUT/'artifact_audit.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='artifact_sha256'},indent=2))


if __name__=='__main__':
    torch.set_num_threads(4)
    with threadpool_limits(limits=4):audit()
