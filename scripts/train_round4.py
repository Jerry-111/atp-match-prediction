"""Paper-grounded follow-up experiments on unchanged chronological cohorts.

All2023follow-up results are exploratory. The replacement gate compares against
round3's earlier-selected2022predictions, not a weaker historical baseline.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.special import expit
from threadpoolctl import threadpool_limits

from scripts.metrics import score_probabilities
from scripts.round2_learning import candidate_specs as spline_specs
from scripts.round2_learning import make_model as make_spline
from scripts.train_models import (FOLDS, ROOT, SEED, SymmetricClassifier,
    cluster_interval, mean_loss, sha256, temporal_masks)
from scripts.train_round3 import predict_round3, select_configurations

GATE_IMPROVEMENT = .001


def candidate_specs(x,groups):
    specs=[]
    def add_group(name,columns,only_spline=False):
        odd=[c for c in columns if c.endswith('_diff')]
        for c in (.003,.01):
            if not only_spline:
                specs.append(dict(name=f'{name}_linear_C{c:g}',family=name,kind='logistic',
                    columns=odd,params={'C':c},half_life=3.))
            spline=next(s for s in spline_specs(columns) if s['params']['C']==c and s['params']['half_life']==3.)
            specs.append(dict(spline,name=f'{name}_spline_C{c:g}',family=name))
    for group in ('margin','graph','point','profile'):
        add_group('add_'+group,groups['base']+groups[group])
    # Follow-up diagnostic groups distinguish the two profile information sources.
    add_group('add_age',groups['base']+[c for c in groups['profile'] if c.startswith('profile_age')])
    add_group('add_points',groups['base']+[c for c in groups['profile'] if not c.startswith('profile_age')])
    add_group('all_papers',list(x))
    compact=list(dict.fromkeys(['log_rank_diff']+[c for c in groups['base'] if c.startswith(('format_','surface_is_'))]
        +sum([groups[g] for g in ('margin','graph','point','profile')],[])))
    add_group('paper_compact',compact)
    for group in ('margin','graph','point','profile'):
        add_group('without_'+group,[c for c in x if c not in groups[group]],only_spline=True)
    for depth in (3,4):
        specs.append(dict(name=f'papers_cat_d{depth}',family='paper_catboost',kind='catboost',columns=list(x),
            params=dict(depth=depth,iterations=600,learning_rate=.03,l2_leaf_reg=20),half_life=3.))
    for leaves in (7,15):
        specs.append(dict(name=f'papers_hist_l{leaves}',family='paper_histboost',kind='histboost',columns=list(x),
            params=dict(max_iter=300,max_leaf_nodes=leaves,learning_rate=.03,l2_regularization=50,min_samples_leaf=100),half_life=3.))
    for col in groups['point']:
        if col.endswith('_match_logit_diff'):
            specs.append(dict(name='raw_'+col,kind='raw_logit',family='raw_point_model',columns=[col],params={}))
    for col in ('paper_overall_welo_games_diff','paper_overall_welo_sets_diff','paper_overall_elo_diff'):
        if col in x:
            specs.append(dict(name='raw_'+col,kind='raw_elo',family='raw_margin_model',columns=[col],params={}))
    return specs


def make_model(spec):
    return make_spline(spec) if spec['kind']=='odd_spline' else SymmetricClassifier(spec)


def raw_probability(spec,x):
    value=x[spec['columns'][0]].to_numpy()
    return np.clip(expit(value*(np.log(10)/400 if spec['kind']=='raw_elo' else 1)),1e-8,1-1e-8)


def choose_procedure(y,oof,specs,seasons,reference):
    seasons=np.asarray(seasons)
    early=np.isin(seasons,(2020,2021))
    gate=seasons==2022
    if not early.any() or not gate.any():
        raise ValueError('Need early folds and2022gate')
    selected,_=select_configurations(specs,oof,y,early)
    reference_loss=mean_loss(y[gate],reference[gate])
    loss=mean_loss(y[gate],oof[selected][gate])
    improvement=reference_loss-loss
    return dict(procedure='adopt_paper_model' if improvement>=GATE_IMPROVEMENT else 'retain_round3',
        early_selected=selected,reference_2022_loss=reference_loss,candidate_2022_loss=loss,
        absolute_improvement=improvement,required_improvement=GATE_IMPROVEMENT,
        selection_seasons=[2020,2021],gate_season=2022)


def predict_round4(bundle,features):
    if features.empty or features.columns.duplicated().any():
        raise ValueError('Prediction features must have rows and unique columns')
    missing=set(bundle['feature_names'])-set(features)
    if missing:
        raise ValueError(f'Missing frozen features: {sorted(missing)}')
    x=features[bundle['feature_names']].astype(float)
    if not np.isfinite(x.to_numpy()).all():
        raise ValueError('Frozen features must be finite')
    p={n:model.predict_proba(x) for n,model in bundle['models'].items()}
    p.update({n:raw_probability(s,x) for n,s in bundle['raw_specs'].items()})
    previous=predict_round3(bundle['round3_bundle'],x)
    for alias,name in bundle['previous_aliases'].items():
        p[alias]=previous[name].to_numpy()
    for name,values in p.items():
        if np.asarray(values).shape!=(len(x),) or not np.isfinite(values).all() or np.any((values<0)|(values>1)):
            raise ValueError(f'Invalid probabilities: {name}')
    return pd.DataFrame(p,index=features.index)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=ROOT/'results/round4')
    parser.add_argument('--round3',type=Path,default=ROOT/'results/round3')
    parser.add_argument('--predict-features',type=Path)
    args=parser.parse_args()
    out=args.output
    out.mkdir(parents=True,exist_ok=True)
    if args.predict_features is not None:
        features=pd.read_csv(args.predict_features)
        p=predict_round4(joblib.load(out/'frozen_models.joblib'),features)
        if 'match_id' in features:p.insert(0,'match_id',features.match_id.to_numpy())
        p.to_csv(out/'inference_predictions.csv',index=False)
        return
    original_path=ROOT/'data/processed/model_features.csv'
    inventory_path=ROOT/'data/processed/mens_singles_2006_2025.csv'
    old_registry=json.loads((args.round3/'search_registry.json').read_text())
    for key,path in [('target_sha256',original_path),('inventory_sha256',inventory_path)]:
        if old_registry[key]!=sha256(path):raise ValueError('Round3sourcehash mismatch')
    inventory=pd.read_csv(inventory_path,dtype={'a_won':'string'})
    history=inventory[inventory.season.le(2023)].copy()
    del inventory
    from scripts.expanded_training import build_expanded_targets,build_expanded_features
    print('Rebuilding matched expanded history matrix',flush=True)
    rows=build_expanded_targets(pd.read_csv(original_path),history)
    base=build_expanded_features(rows,history)
    previous_rows=pd.read_csv(args.round3/'development_targets.csv')
    if not rows[['match_id','season','a_won']].equals(previous_rows[['match_id','season','a_won']]):
        raise ValueError('Round3 target identity mismatch')
    previous_x=pd.read_csv(args.round3/'development_features.csv')
    if list(base)!=list(previous_x) or not np.allclose(base,previous_x,atol=1e-10,rtol=0):
        raise ValueError('Existing features changed')
    from scripts.paper_margin_features import build_paper_margin_features
    from scripts.paper_graph_features import build_paper_graph_features
    from scripts.paper_point_model import build_paper_point_features
    from scripts.paper_profile_features import build_paper_profile_features
    blocks={'base':base}
    for group,builder,inputs in [
        ('margin',build_paper_margin_features,(rows,history)),
        ('graph',build_paper_graph_features,(rows,history)),
        ('point',build_paper_point_features,(rows,base)),
        ('profile',build_paper_profile_features,(rows,history))]:
        print('Building paper branch:',group,flush=True)
        blocks[group]=builder(*inputs)
        if not blocks[group].index.equals(rows.index):raise ValueError('Feature rowalignment mismatch')
    x=pd.concat(list(blocks.values()),axis=1)
    if x.columns.duplicated().any() or not np.isfinite(x.to_numpy()).all():raise ValueError('Invalid numeric features')
    groups={g:list(b) for g,b in blocks.items()}
    specs=candidate_specs(x,groups)
    registry=dict(round=4,seed=SEED,folds=list(FOLDS),feature_groups=groups,candidates=specs,
        paper_metadata={g:b.attrs for g,b in blocks.items() if g!='base'},
        target_sha256=sha256(original_path),inventory_sha256=sha256(inventory_path),
        exploratory_after_2023=True,final_test_evaluated=False,
        selection='Early2020--21 winner must improve current round3 early2022reference by at least.001; refit pooledchoice.')
    (out/'search_registry.json').write_text(json.dumps(registry,indent=2)+'\n')
    x.to_csv(out/'development_features.csv',index=False)
    oof_rows=rows[rows.season.isin(FOLDS)]
    y=oof_rows.a_won.to_numpy()
    old_oof=pd.read_csv(args.round3/'oof_predictions.csv')
    if not old_oof[['match_id','season','a_won']].equals(oof_rows[['match_id','season','a_won']].reset_index(drop=True)):
        raise ValueError('Round3OOF identity mismatch')
    old_frozen=json.loads((args.round3/'frozen_selection.json').read_text())
    reference=old_oof[old_frozen['procedure_gate']['early_selected']].to_numpy()
    oof,scores={},[]
    for i,spec in enumerate(specs,1):
        p=pd.Series(index=oof_rows.index,dtype=float)
        folds={}
        for year in FOLDS:
            train,valid=temporal_masks(rows,year)
            if spec['kind'].startswith('raw_'):
                probability=raw_probability(spec,x[valid])
            else:
                model=make_model(spec).fit(x[train],rows.a_won[train],rows.season[train])
                probability=model.predict_proba(x[valid])
            p.loc[rows.index[valid]]=probability
            folds[str(year)]=mean_loss(rows.a_won[valid].to_numpy(),probability)
        oof[spec['name']]=p.to_numpy()
        scores.append(dict(name=spec['name'],family=spec['family'],mean_log_loss=mean_loss(y,p.to_numpy()),**folds))
        pd.DataFrame(scores).to_csv(out/'temporal_cv.csv',index=False)
        print(f"[{i}/{len(specs)}] {spec['name']}: {scores[-1]['mean_log_loss']:.6f}",flush=True)
    gate=choose_procedure(y,oof,specs,oof_rows.season,reference)
    best,selected=select_configurations(specs,oof,y,np.ones(len(y),dtype=bool))
    choice=best if gate['procedure']=='adopt_paper_model' else 'round3_recommended'
    frozen=dict(round=4,recommendation=choice,best_single=best,selected_by_family=selected,procedure_gate=gate,
        previous_recommendation=old_frozen['recommendation'],fit_end_season=2022,history_lag_days=21,
        exploratory_after_2023=True,final_test_evaluated=False)
    (out/'frozen_selection.json').write_text(json.dumps(frozen,indent=2)+'\n')
    print('Frozen paper procedure:',choice,gate,flush=True)
    by_name={s['name']:s for s in specs}
    train,valid=temporal_masks(rows,2023)
    models,raw_specs={},{}
    for name in sorted(set(selected.values())|{best}):
        spec=by_name[name]
        if spec['kind'].startswith('raw_'):raw_specs[name]=spec
        else:models[name]=make_model(spec).fit(x[train],rows.a_won[train],rows.season[train])
    aliases={'round3_recommended':old_frozen['recommendation'],'round1_best_single':'round1_best_single',
             'round1_ensemble':'round1_ensemble','rank_reference':'rank_reference',
             'round3_linear':old_frozen['selected_by_family']['history2009_linear_format1']}
    bundle=dict(models=models,raw_specs=raw_specs,feature_names=list(x),frozen=frozen,
        round3_bundle=joblib.load(args.round3/'frozen_models.joblib'),previous_aliases=aliases)
    p=predict_round4(bundle,x[valid]);vr=rows[valid]
    table=[]
    for name in p:
        scores=score_probabilities(zip(vr.match_id,vr.a_won),zip(vr.match_id,p[name]))
        table.append(dict(model=name,recommended=name==choice,**scores,brier=float(np.mean((vr.a_won.to_numpy()-p[name].to_numpy())**2))))
    result=pd.DataFrame(table).sort_values('mean_log_loss')
    result.to_csv(out/'validation_comparison.csv',index=False)
    pd.concat([vr.reset_index(drop=True),p.reset_index(drop=True)],axis=1).to_csv(out/'validation_predictions.csv',index=False)
    pd.concat([oof_rows[['match_id','season','a_won']].reset_index(drop=True),pd.DataFrame(oof)],axis=1).to_csv(out/'oof_predictions.csv',index=False)
    intervals={n:cluster_interval(vr,p.round3_recommended.to_numpy(),p[n].to_numpy()) for n in p}
    summary=dict(round=4,recommendation=choice,new_candidates=len(specs),features=x.shape[1],training_rows=int(train.sum()),
        validation_matches=len(vr),comparisons_vs_round3=intervals,exploratory_after_2023=True,final_test_evaluated=False)
    (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    joblib.dump(bundle,out/'frozen_models.joblib')
    print(result.to_string(index=False),flush=True)


if __name__=='__main__':
    from scripts.train_round4 import main as canonical_main
    with threadpool_limits(limits=4):canonical_main()
