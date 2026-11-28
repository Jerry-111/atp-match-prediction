"""Independent artifact checks for the frozen Round 8 joint point branch."""
from __future__ import annotations
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from scipy.optimize import brentq
from scipy.special import expit, logit
from threadpoolctl import threadpool_limits
from scripts.train_models import ROOT, sha256
from scripts.joint_point_model import load_point_history, attach_metadata
from scripts.train_joint_points_round8 import predict_joint_points
from scripts.paper_point_model import match_probability
from scripts.report_round7 import verify_scores

OUT = ROOT/'results/round8/joint_points'


def read_json(name):
    return json.loads((OUT/name).read_text())


def loss(y,p):
    p = np.clip(np.asarray(p),1e-15,1-1e-15)
    return float(np.mean(-np.asarray(y)*np.log(p)-(1-np.asarray(y))*np.log1p(-p)))


def manual_terms(a,b,s,e,spec):
    terms = [('intercept',1.),('serve:'+str(a),1.)]
    if s!='Hard': terms += [('baseline:'+s,1.)]
    if spec['opponent']: terms += [('return:'+str(b),-1.)]
    if spec['surface']: terms += [('preference:'+str(a)+':'+s,1.),('preference:'+str(b)+':'+s,-1.)]
    if spec['event'] and e: terms += [('event:'+e,1.)]
    return terms


def audit():
    bundle = joblib.load(OUT/'frozen_models.joblib')
    frozen, registry = read_json('frozen_selection.json'), read_json('search_registry.json')
    assert not frozen['final_test_evaluated'] and bundle['frozen']==frozen
    for field,file in [('source_code_sha256','scripts/train_joint_points_round8.py'),
                       ('engine_sha256','scripts/joint_point_model.py'),
                       ('protocol_sha256','docs/research/round8_joint_points.md')]:
        assert registry[field]==sha256(ROOT/file), field
    rows = pd.read_csv(ROOT/'results/round3/development_targets.csv')
    targets = pd.read_csv(OUT/'inference_targets.csv')
    saved = pd.read_csv(OUT/'validation_predictions.csv')
    refs = pd.read_csv(OUT/'reference_predictions.csv')
    table = pd.read_csv(OUT/'validation_comparison.csv')
    verify_scores(table,saved,rows[rows.season.eq(2023)])
    p = predict_joint_points(bundle,targets,refs)
    names = list(p)
    error = float(np.abs(p.to_numpy()-saved[names].to_numpy()).max())
    assert error<1e-14
    flipped = targets.copy()
    flipped[['player_a_id','player_b_id']] = targets[['player_b_id','player_a_id']].to_numpy()
    reverse_refs = refs.copy()
    for name in bundle['reference_names']: reverse_refs[name] = 1-refs[name]
    reverse = predict_joint_points(bundle,flipped,reverse_refs)
    exchange = float(np.abs(p.to_numpy()+reverse.to_numpy()-1).max())
    assert exchange<1e-12
    poisoned = targets.copy(); poisoned['a_won'] = 999
    assert np.array_equal(p,predict_joint_points(bundle,poisoned,refs))
    # Reconstruct allowed history and preknown metadata; do not open 2024/2025 raw files.
    inventory = pd.read_csv(ROOT/'data/processed/mens_singles_2006_2025.csv',dtype={'a_won':'string'})
    history,metadata,source = load_point_history(inventory[inventory.season.le(2023)])
    rebuilt = attach_metadata(rows[rows.season.eq(2023)],metadata).reset_index(drop=True)
    for name in targets:
        assert np.array_equal(rebuilt[name].astype(str),targets[name].astype(str)), name
    raw = pd.read_csv(OUT/'raw_validation_predictions.csv')
    manual_error, max_gradient, snapshots, gradient_samples = 0.,0.,0,0
    cutoffs = (pd.to_datetime(targets.tourney_date)-pd.Timedelta(days=21)).dt.strftime('%Y-%m-%d')
    for spec in bundle['specs']:
        states = bundle['states'][spec['name']]
        sorted_cutoffs = sorted(states)
        check_gradient = {sorted_cutoffs[0],sorted_cutoffs[len(sorted_cutoffs)//2],sorted_cutoffs[-1]}
        for cutoff,state in states.items():
            snapshots += 1
            assert state.cutoff==cutoff and state.spec==spec
            day = pd.Timestamp(cutoff)
            past = history[history.date.between(day-pd.Timedelta(days=1460),day)]
            assert len(past)==state.diagnostics['observations']
            assert past.match_id.nunique()==state.diagnostics['matches']
            assert str(past.date.min())==state.diagnostics['oldest_observation']
            assert str(past.date.max())==state.diagnostics['newest_observation']
            coefficients = dict(zip(state.keys,state.coefficients))
            index = targets.index[cutoffs.eq(cutoff)]
            query = targets.loc[index]
            service = []
            for r in query.itertuples():
                service.append([expit(logit(.65)+sum(coefficients.get(key,0)*value for key,value in
                    manual_terms(a,b,r.surface,r.event_key,spec))) for a,b in
                    ((r.player_a_id,r.player_b_id),(r.player_b_id,r.player_a_id))])
            service = np.asarray(service)
            manual = np.clip(match_probability(service[:,0],service[:,1],query.best_of.to_numpy()),1e-8,1-1e-8)
            manual = np.where(service[:,0]==service[:,1],.5,manual)
            manual_error = max(manual_error,float(np.max(np.abs(manual-raw.loc[index,spec['name']]))))
            if cutoff in check_gradient:
                gradient_samples += 1
                lookup = {key:i for i,key in enumerate(state.keys)}
                precision = np.array([1/spec['prior_sd'][key.split(':',1)[0]]**2 for key in state.keys])
                gradient = precision*state.coefficients
                for r in past.itertuples():
                    terms = manual_terms(r.server,r.returner,r.surface,r.event,spec)
                    eta = logit(.65)+sum(coefficients[key]*value for key,value in terms)
                    w = 1. if spec['half_life'] is None else 2.**(-(day-r.date).days/spec['half_life'])
                    residual = w*(r.total*expit(eta)-r.won)
                    for key,value in terms: gradient[lookup[key]] += value*residual
                max_gradient = max(max_gradient,float(np.max(np.abs(gradient))/len(past)))
    assert manual_error<1e-13 and max_gradient<1e-5
    # Refit scalar calibration via an independent root of the likelihood derivative.
    rawdev = pd.read_csv(OUT/'raw_development_predictions.csv').set_index('match_id')
    calibration = read_json('calibration_metadata.json')
    slope_error = 0.
    for year,cal in calibration.items():
        year = int(year)
        valid = rows[rows.season.eq(year)]
        cutoff = pd.to_datetime(valid.tourney_date).min()-pd.Timedelta(days=21)
        train = rows[rows.season.between(2017,year-1)&pd.to_datetime(rows.tourney_date).le(cutoff)]
        assert train.match_id.tolist()==cal['ids'] and len(train)==cal['rows']
        for name,slope in cal['slopes'].items():
            z = logit(np.clip(rawdev.loc[train.match_id,name].to_numpy(),1e-8,1-1e-8))
            y = train.a_won.to_numpy()
            def derivative(value): return float(np.mean(z*(expit(value*z)-y)))
            root = .05 if derivative(.05)>=0 else 20. if derivative(20.)<=0 else brentq(derivative,.05,20.,xtol=1e-12)
            slope_error = max(slope_error,abs(root-slope))
    assert slope_error<1e-6
    oof = pd.read_csv(OUT/'oof_predictions.csv')
    expected_oof = rows[rows.season.isin([2020,2021,2022])].set_index('match_id')
    assert len(oof)==len(expected_oof) and not oof.match_id.duplicated().any()
    assert np.array_equal(oof.a_won,expected_oof.loc[oof.match_id,'a_won'])
    for _,metric in pd.read_csv(OUT/'temporal_cv.csv').iterrows():
        yy,pp = oof.a_won.to_numpy(),oof[metric.model].to_numpy()
        assert metric.matches==len(oof) and metric.correct==int(np.sum((pp>=.5)==yy))
        assert abs(metric.mean_log_loss-loss(yy,pp))<1e-13
        assert abs(metric.brier-np.mean((yy-pp)**2))<1e-13
        for year in (2020,2021,2022):
            mask = oof.season.eq(year).to_numpy()
            assert abs(metric[f'{year}_accuracy']-np.mean((pp[mask]>=.5)==yy[mask]))<1e-13
            assert abs(metric[f'{year}_log_loss']-loss(yy[mask],pp[mask]))<1e-13
    menu = ['round4_reference']+[s['name'] for s in bundle['specs']]+list(registry['blends'])
    def choice(mask):
        yy = oof.loc[mask,'a_won'].to_numpy()
        limit = loss(yy,oof.loc[mask,'round4_reference'])+.005
        allowed = [n for n in menu if loss(yy,oof.loc[mask,n])<=limit+1e-12]
        return min(allowed,key=lambda n:(-np.mean((oof.loc[mask,n].to_numpy()>=.5)==yy),loss(yy,oof.loc[mask,n]),n))
    early = choice(oof.season.lt(2022))
    assert early==frozen['procedure_gate']['early_selection']['choice']
    assert choice(np.ones(len(oof),bool))==frozen['pooled_selection']['choice']
    gate_rows = oof[oof.season.eq(2022)]
    gain = np.mean((gate_rows[early]>=.5)==gate_rows.a_won)-np.mean((gate_rows.round4_reference>=.5)==gate_rows.a_won)
    delta_loss = loss(gate_rows.a_won,gate_rows[early])-loss(gate_rows.a_won,gate_rows.round4_reference)
    assert abs(gain-frozen['procedure_gate']['accuracy_improvement'])<1e-14
    assert abs(delta_loss-frozen['procedure_gate']['loss_increase'])<1e-14
    adopted = gain>=.005-1e-12 and delta_loss<=.005+1e-12
    assert adopted==(frozen['procedure_gate']['procedure']=='adopt_joint_points')
    result = dict(status='passed',auditor='coordinator, independent of branch implementation',
        validation_matches=len(targets),probability_columns=len(names),reload_max_abs_error=error,
        swap_max_abs_error=exchange,independent_raw_max_abs_error=manual_error,
        full_validation_snapshots_checked=snapshots,independent_gradient_snapshots=gradient_samples,
        independent_max_scaled_gradient=max_gradient,independent_calibration_slope_error=slope_error,
        calibration_fits=20,calibration_ids_and_purge_verified=True,
        label_poisoning_unchanged=True,source_metadata_rebuilt=True,
        source_valid_matches=source['valid_history_matches'],all_metrics_and_oof_pooled_verified=True,
        independent_early_pooled_gate_verified=True,final_test_evaluated=False,
        limitations='No full OOF point-model refit; independently reconstructed all validation raw forecasts and 15 of 175 gradients. Paired intervals not separately rerun here.',
        artifact_sha256={p.name:sha256(p) for p in OUT.iterdir() if p.is_file() and p.name!='artifact_audit.json'},
        audit_script_sha256=sha256(Path(__file__)))
    (OUT/'artifact_audit.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='artifact_sha256'},indent=2))


if __name__=='__main__':
    with threadpool_limits(limits=4): audit()
