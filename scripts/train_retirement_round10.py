"""Three fixed nested inactivity / provider-recorded RET controls; no mixtures."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from scripts.metrics import score_probabilities
from scripts.round2_learning import candidate_specs as spline_specs
from scripts.train_distillation_round7 import validate_rows
from scripts.train_models import ROOT, FOLDS, SEED, sha256, temporal_masks, mean_loss, cluster_interval, mirror_features
from scripts.train_round5 import REFERENCE, ACCURACY_GATE, LOSS_GUARD, accuracy, accuracy_interval, fit_model, predict_round5
from scripts.train_transfer_round6 import verify_tour_targets
from scripts.retirement_features import build_retirement_features


def candidate_specs(groups):
    specs=[]
    for name in ('I','R','RC'):
        added=groups[name];columns=groups['base']+added
        if len(columns)!=len(set(columns)):raise ValueError('Repeated feature')
        spec=next(s for s in spline_specs(columns) if s['params']['C']==.01 and s['params']['half_life']==3.)
        spec=dict(spec,name=f'ret_{name}_spline',family=name,method='spline',candidate_input_columns=columns)
        spec['spline_columns']=list(spec['spline_columns'])+[c for c in added if c.endswith('_diff')]
        spec['interactions']=list(spec['interactions'])+[['log_rank_diff',c] for c in added if c.endswith('_sum')]
        spec['columns']=list(dict.fromkeys(spec['columns']+[c for c in added if c.endswith('_sum')]))
        specs.append(spec)
    return specs


def verify_source_authorization(out):
    audit=json.loads((out/'source_feasibility.json').read_text())
    if (audit.get('source_semantics_fully_verified') is not False
            or audit.get('model_fitting_ready') is not False
            or audit.get('root_limited_experiment_authorized') is not True
            or audit.get('observed_official_contradictions')!=0
            or audit.get('review_counts')!={'verified':10,'unverified':20}
            or audit.get('sample_rows')!=30):
        raise ValueError('Source semantics differ from the approved partial-evidence scope')
    for field,filename in [('sample_registry_sha256','sample_registry.json'),
                           ('official_reviews_sha256','official_reviews.json'),
                           ('official_sources_sha256','official_sources.json')]:
        if audit.get(field)!=sha256(out/filename):
            raise ValueError('Stale source adjudication: '+filename)
    if audit.get('audit_code_sha256')!=sha256(ROOT/'scripts/audit_retirement_source_round10.py'):
        raise ValueError('Source-audit code changed without rebuilding its evidence manifest')
    return audit


def select_single(y, predictions, mask):
    y, mask = np.asarray(y), np.asarray(mask, dtype=bool)
    if (y.ndim != 1 or not np.isin(y, (0, 1)).all() or mask.shape != y.shape or not mask.any()
            or REFERENCE not in predictions):
        raise ValueError('Aligned binary selection labels, mask and reference required')
    for name, p in predictions.items():
        p = np.asarray(p)
        if p.shape != y.shape or not np.isfinite(p).all() or np.any((p < 0) | (p > 1)):
            raise ValueError(f'Invalid selection probability: {name}')
    limit = mean_loss(y[mask], predictions[REFERENCE][mask]) + LOSS_GUARD
    eligible = [n for n,p in predictions.items() if mean_loss(y[mask], p[mask]) <= limit]
    best = min(eligible, key=lambda n: (-accuracy(y[mask], predictions[n][mask]),
               mean_loss(y[mask], predictions[n][mask]), n))
    return dict(choice=best, selected_accuracy=accuracy(y[mask], predictions[best][mask]),
        selected_log_loss=mean_loss(y[mask], predictions[best][mask]), loss_limit=limit,
        eligible_models=eligible, selection_rows=int(mask.sum()), mixture_search=False)


def choose_procedure(y, predictions, seasons):
    seasons = np.asarray(seasons)
    if seasons.shape != np.asarray(y).shape or not np.isin(seasons, FOLDS).all():
        raise ValueError('Only 2020..2022 OOF predictions may select models')
    early, held = np.isin(seasons, (2020, 2021)), seasons == 2022
    if not early.any() or not held.any():
        raise ValueError('Need early folds and separate 2022 gate')
    selection = select_single(y, predictions, early)
    y = np.asarray(y)
    ref, candidate = predictions[REFERENCE][held], predictions[selection['choice']][held]
    before, after = accuracy(y[held], ref), accuracy(y[held], candidate)
    ref_loss, loss = mean_loss(y[held], ref), mean_loss(y[held], candidate)
    passed = after-before >= ACCURACY_GATE-1e-12 and loss <= ref_loss+LOSS_GUARD+1e-12
    return dict(procedure='adopt_accuracy_model' if passed else 'retain_round4', early_selection=selection,
        reference=REFERENCE, reference_2022_accuracy=before, candidate_2022_accuracy=after,
        accuracy_improvement=after-before, required_accuracy_improvement=ACCURACY_GATE,
        reference_2022_loss=ref_loss, candidate_2022_loss=loss, allowed_loss_increase=LOSS_GUARD,
        selection_seasons=[2020, 2021], gate_season=2022)


def predict_retirement(bundle, features):
    required = bundle['feature_names']
    if features.empty or features.columns.duplicated().any() or not set(required).issubset(features):
        raise ValueError('Missing, duplicate or empty frozen features')
    x = features[required].astype(float)
    if not np.isfinite(x.to_numpy()).all():
        raise ValueError('Frozen features must be finite')
    old = predict_round5(bundle['previous_bundle'], x)
    values = {name: model.predict_proba(x) for name,model in bundle['models'].items()}
    if set(values) & set(bundle['previous_aliases']):
        raise ValueError('New models may not overwrite old references')
    values.update({alias: old[name].to_numpy() for alias,name in bundle['previous_aliases'].items()})
    for name, p in values.items():
        if np.asarray(p).shape != (len(x),) or not np.isfinite(p).all() or np.any((p < 0) | (p > 1)):
            raise ValueError(f'Invalid prediction: {name}')
    return pd.DataFrame(values, index=features.index)


def write_report(out, summary, frozen, comparison, matched):
    def table(frame):
        def show(v):return f'{v:.6f}' if isinstance(v,(float,np.floating)) else str(v)
        return '\n'.join(['| '+' | '.join(frame.columns)+' |','| '+' | '.join(['---']*len(frame.columns))+' |']+
            ['| '+' | '.join(show(v) for v in row)+' |' for row in frame.itertuples(index=False,name=None)])
    gate=frozen['procedure_gate'];chosen=comparison.set_index('model').loc[frozen['recommendation']]
    selected=comparison.set_index('model').loc[frozen['pooled_selection']['choice']]
    highest=comparison.iloc[0]
    text=f'''# Round 10：历史 RET 状态与普通停赛对照

来源仍是部分核验：30 条预先固定样本中 10 条官方确认、20 条未确认；已确认中
未发现 RET 方向矛盾，但年代偏早。协调者接受此限制，授权三个有限配置。
研究对象是提供方记录的历史非完赛状态，不是全面核验的伤病数据；没有按官方
资料可得性筛选历史记录。W/O 和 DEF 分开，不加入 RET，也不增加训练败局。

同一完整 2023 年 2,588 场，阈值 0.5、T−21 历史不变。2023 已反复查看，以下
比较均为探索性；2024–2025 未解封，不宣称 SOTA 或预设能达到 70%。

本分支冻结推荐 **{frozen['recommendation']}**：{round(chosen.accuracy*2588)}/2,588，
accuracy {chosen.accuracy:.4%}，log loss {chosen.mean_log_loss:.6f}。
2022 gate：**{gate['procedure']}**，accuracy 变化 {gate['accuracy_improvement']:+.4%}，
loss 变化 {gate['candidate_2022_loss']-gate['reference_2022_loss']:+.6f}；门槛为
accuracy ≥+0.5pp 且 loss ≤+0.005。
2020–2022 pooled 候选 **{frozen['pooled_selection']['choice']}**：
{round(selected.accuracy*2588)}/2,588、{selected.accuracy:.4%}、loss {selected.mean_log_loss:.6f}。
2023 表中事后最高行 **{highest.model}**：{round(highest.accuracy*2588)}/2,588；
这不改变冻结推荐。70% 需要 1,812 场，最高行仍差 {max(0,1812-round(highest.accuracy*2588))} 场。

三个固定 spline 配置、每个三次 forward CV 和一次最终 refit，共 12 fits；
没有混合或阈值搜索。I 的六个普通 inactivity 字段在三组都保留；R 加四个
历史 RET recency/presence 字段，RC 加四个 recency/后续出场衰减字段。
候选矩阵维数分别 332/336/336；I 是嵌套对照，R/RC 同容量。联合存储矩阵338列。
新增统计均以日期分组，不虚构同日顺序。零历史行保留；重拟合模型在零新状态
时不保证与 R4 概率完全相同。所有比例/缩放只能在当期训练数据拟合。

## 同批 2023 比较

{table(comparison[['model','accuracy','mean_log_loss','recommended']])}

## 配对机制对照

差值为候选减对照；accuracy 正、loss 负表示更好。R−I/RC−I 包含新增维数，
R−RC 才是同容量机制对照，均不解释为 RET 的因果效应。

{table(matched)}

完整 CV 在 temporal_cv.csv，来源/覆盖在 search_registry.json 和
coverage_by_season.csv，按赛事日期块的配对区间在 summary.json。区间没有多重
搜索校正；重复使用 2022 gate 也不是独立确认。

[精确协议](../../../docs/research/round10_retirement_protocol.md) ·
[来源审计](source_feasibility.json) · [固定样本](sample_registry.json)
'''
    (out/'report_summary.md').write_text(text)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT/'results/round10/retirement')
    parser.add_argument('--predict-features', type=Path)
    args = parser.parse_args()
    out, previous = args.output, ROOT/'results/round5'
    out.mkdir(parents=True, exist_ok=True)
    if args.predict_features:
        x = pd.read_csv(args.predict_features)
        result = predict_retirement(joblib.load(out/'frozen_models.joblib'), x)
        if 'match_id' in x:
            result.insert(0, 'match_id', x.match_id.to_numpy())
        result.to_csv(out/'inference_predictions.csv', index=False)
        return
    audit = json.loads((previous/'artifact_audit.json').read_text())
    for name in ('development_features.csv','frozen_models.joblib','oof_predictions.csv','search_registry.json'):
        if sha256(previous/name) != audit['artifact_sha256'][name]:
            raise ValueError(f'Previous audited artifact changed: {name}')
    old_registry = json.loads((previous/'search_registry.json').read_text())
    inventory_file = ROOT/'data/processed/mens_singles_2006_2025.csv'
    target_file = ROOT/'data/processed/model_features.csv'
    if sha256(inventory_file) != old_registry['inventory_sha256'] or sha256(target_file) != old_registry['target_sha256']:
        raise ValueError('Pinned source hashes changed')
    rows = pd.read_csv(ROOT/'results/round3/development_targets.csv')
    validate_rows(rows)
    inventory = pd.read_csv(inventory_file, dtype={'a_won':'string'})
    inventory = inventory[inventory.season.le(2023)].copy()
    verify_tour_targets(rows, inventory)
    base = pd.read_csv(previous/'development_features.csv')
    if base.shape != (len(rows),326):
        raise ValueError('Expected unchanged prior 326-column matrix')
    print('Building provider-recorded RET and ordinary inactivity states', flush=True)
    source_audit=verify_source_authorization(out)
    retirement = build_retirement_features(rows, inventory)
    metadata = dict(retirement.attrs)
    x = pd.concat([base,retirement],axis=1)
    if not retirement.index.equals(rows.index) or x.columns.duplicated().any() or not np.isfinite(x.to_numpy()).all():
        raise ValueError('Invalid retirement matrix')
    groups = dict(base=list(base),**metadata['groups'])
    specs = candidate_specs(groups)
    if x.shape != (36421,338) or [len(s['candidate_input_columns']) for s in specs] != [332,336,336]:
        raise ValueError('Fixed full cohort or nested dimensions changed')
    old = pd.read_csv(previous/'oof_predictions.csv')
    orows = rows[rows.season.isin(FOLDS)]
    identity = ['match_id','season','a_won']
    if not orows[identity].reset_index(drop=True).equals(old[identity]):
        raise ValueError('Previous OOF identities changed')
    oof = {REFERENCE:old[REFERENCE].to_numpy()}
    registry = dict(round=10,branch='historical_retirement_state',seed=SEED,folds=list(FOLDS),
        features=list(x),feature_groups=groups,feature_metadata=metadata,candidates=specs,
        selection_eligible=[s['name'] for s in specs]+[REFERENCE],mixture_search=False,
        protocol_sha256=sha256(ROOT/'docs/research/round10_retirement_protocol.md'),
        feature_code_sha256=sha256(ROOT/'scripts/retirement_features.py'),runner_sha256=sha256(Path(__file__)),
        inventory_sha256=sha256(inventory_file),target_sha256=sha256(target_file),
        expanded_target_sha256=sha256(ROOT/'results/round3/development_targets.csv'),
        previous_artifact_sha256=audit['artifact_sha256'],accuracy_gate=ACCURACY_GATE,loss_guard=LOSS_GUARD,
        prior_round5_models_selection_eligible=False,final_test_evaluated=False,exploratory_after_2023=True,
        source_audit_sha256=sha256(out/'source_feasibility.json'),sample_registry_sha256=sha256(out/'sample_registry.json'),
        official_reviews_sha256=sha256(out/'official_reviews.json'),official_verified=10,official_unverified=20,
        source_semantics_fully_verified=False,root_limited_experiment_authorized=True,fit_configurations=3,total_fits=12,
        learner_sha256=sha256(ROOT/'scripts/round2_learning.py'),candidate_input_dimensions={s['name']:len(s['candidate_input_columns']) for s in specs})
    (out/'search_registry.json').write_text(json.dumps(registry,indent=2)+'\n')
    x.to_csv(out/'development_features.csv',index=False)
    y, cv = orows.a_won.to_numpy(), []
    fit_records=[]
    for i,spec in enumerate(specs,1):
        p = pd.Series(index=orows.index,dtype=float)
        scores = {}
        fold_models={}
        for year in FOLDS:
            train,valid = temporal_masks(rows,year)
            validate_rows(rows[train],fitting=True)
            model = fit_model(spec,x[train],rows[train])
            fold_models[year]=model
            fit_records.append(dict(model=spec['name'],validation_season=year,train_rows=int(train.sum()),
                train_max_date=str(pd.to_datetime(rows.loc[train,'tourney_date']).max()),
                validation_min_date=str(pd.to_datetime(rows.loc[valid,'tourney_date']).min()),
                training_match_ids_sha256=hashlib.sha256('\n'.join(rows.loc[train,'match_id']).encode()).hexdigest()))
            prediction = model.predict_proba(x[valid])
            p.loc[rows.index[valid]] = prediction
            scores[f'{year}_accuracy'] = accuracy(rows.a_won[valid],prediction)
            scores[f'{year}_log_loss'] = mean_loss(rows.a_won[valid].to_numpy(),prediction)
        joblib.dump(fold_models,out/f"fold_models_{spec['name']}.joblib")
        oof[spec['name']] = p.to_numpy()
        cv.append(dict(name=spec['name'],family=spec['family'],method=spec['method'],
            accuracy=accuracy(y,p),mean_log_loss=mean_loss(y,p),**scores))
        pd.DataFrame(cv).to_csv(out/'temporal_cv.csv',index=False)
        print(f"[{i}/{len(specs)}] {spec['name']}: acc={cv[-1]['accuracy']:.5f} loss={cv[-1]['mean_log_loss']:.6f}",flush=True)
    ref = oof[REFERENCE]
    scores = {}
    for year in FOLDS:
        mask = orows.season.eq(year).to_numpy()
        scores[f'{year}_accuracy'],scores[f'{year}_log_loss'] = accuracy(y[mask],ref[mask]),mean_loss(y[mask],ref[mask])
    cv.append(dict(name=REFERENCE,family='previous_reference',method='previous',accuracy=accuracy(y,ref),
        mean_log_loss=mean_loss(y,ref),**scores))
    pd.DataFrame(cv).to_csv(out/'temporal_cv.csv',index=False)
    gate = choose_procedure(y,oof,orows.season)
    pooled = select_single(y,oof,np.ones(len(y),dtype=bool))
    choice = pooled['choice'] if gate['procedure']=='adopt_accuracy_model' else REFERENCE
    frozen = dict(round=10,branch='historical_retirement_state',recommendation=choice,procedure_gate=gate,
        pooled_selection=pooled,frozen_at_utc=datetime.now(timezone.utc).isoformat(),decision_threshold=.5,
        historical_lag_days=21,provider_recorded_ret_status=True,source_semantics_fully_verified=False,fit_end_season=2022,
        exploratory_after_2023=True,final_test_evaluated=False)
    (out/'frozen_selection.json').write_text(json.dumps(frozen,indent=2)+'\n')
    print('Frozen:',json.dumps(frozen),flush=True)
    train,valid = temporal_masks(rows,2023)
    validate_rows(rows[train],fitting=True)
    models = {s['name']:fit_model(s,x[train],rows[train]) for s in specs}
    for spec in specs:
        fit_records.append(dict(model=spec['name'],validation_season=2023,train_rows=int(train.sum()),
            train_max_date=str(pd.to_datetime(rows.loc[train,'tourney_date']).max()),
            validation_min_date=str(pd.to_datetime(rows.loc[valid,'tourney_date']).min()),
            training_match_ids_sha256=hashlib.sha256('\n'.join(rows.loc[train,'match_id']).encode()).hexdigest()))
    (out/'fit_records.json').write_text(json.dumps(fit_records,indent=2)+'\n')
    old_frozen = json.loads((previous/'frozen_selection.json').read_text())
    aliases = {REFERENCE:REFERENCE,'previous_round5_exploration':old_frozen['pooled_selection']['choice'],
        'previous_round5_best_loss':'all_inputs_spline_C0.01','rank_reference':'rank_reference'}
    bundle = dict(models=models,feature_names=list(x),frozen=frozen,previous_aliases=aliases,
        previous_bundle=joblib.load(previous/'frozen_models.joblib'))
    p,vr = predict_retirement(bundle,x[valid]),rows[valid]
    scores = []
    for name in p:
        score = score_probabilities(zip(vr.match_id,vr.a_won),zip(vr.match_id,p[name]))
        scores.append(dict(model=name,recommended=name==choice,**score,
            brier=float(np.mean((vr.a_won.to_numpy()-p[name].to_numpy())**2))))
    comparison = pd.DataFrame(scores).sort_values(['accuracy','mean_log_loss'],ascending=[False,True])
    comparison.to_csv(out/'validation_comparison.csv',index=False)
    pd.concat([vr.reset_index(drop=True),p.reset_index(drop=True)],axis=1).to_csv(out/'validation_predictions.csv',index=False)
    pd.concat([orows[identity].reset_index(drop=True),pd.DataFrame(oof)],axis=1).to_csv(out/'oof_predictions.csv',index=False)
    matched = []
    for candidate,control in (('R','I'),('RC','I'),('RC','R')):
        cn,bn=f'ret_{candidate}_spline',f'ret_{control}_spline'
        matched.append(dict(candidate=cn,control=bn,
            oof_accuracy_change=accuracy(y,oof[cn])-accuracy(y,oof[bn]),
            oof_loss_change=mean_loss(y,oof[cn])-mean_loss(y,oof[bn]),
            validation_accuracy_change=accuracy(vr.a_won,p[cn])-accuracy(vr.a_won,p[bn]),
            validation_loss_change=mean_loss(vr.a_won.to_numpy(),p[cn])-mean_loss(vr.a_won.to_numpy(),p[bn])))
    matched = pd.DataFrame(matched)
    matched.to_csv(out/'matched_controls.csv',index=False)
    intervals = {r:{n:dict(accuracy=accuracy_interval(vr,p[r].to_numpy(),p[n].to_numpy()),
        log_loss=cluster_interval(vr,p[r].to_numpy(),p[n].to_numpy())) for n in p}
        for r in (REFERENCE,'previous_round5_exploration')}
    pd.DataFrame.from_dict(metadata['coverage_by_season'],orient='index').rename_axis('season').to_csv(out/'coverage_by_season.csv')
    summary = dict(round=10,branch='historical_retirement_state',recommendation=choice,pooled_choice=pooled['choice'],
        new_candidates=len(specs),total_fits=12,features=x.shape[1],training_rows=int(train.sum()),validation_matches=len(vr),
        coverage=metadata['coverage_by_season'],source_rows_by_status=metadata['source_rows_by_status'],
        official_verified=10,official_unverified=20,source_semantics_fully_verified=False,root_limited_experiment_authorized=True,
        candidate_input_dimensions={s['name']:len(s['candidate_input_columns']) for s in specs},
        comparisons=intervals,correct_required_for_70=1812,exploratory_after_2023=True,final_test_evaluated=False)
    (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    joblib.dump(bundle,out/'frozen_models.joblib')
    reloaded = predict_retirement(joblib.load(out/'frozen_models.joblib'),x[valid])
    swapped = predict_retirement(bundle,mirror_features(x[valid]))
    check = dict(reload_max_error=float(np.max(np.abs(p.to_numpy()-reloaded.to_numpy()))),
        swap_max_error=float(np.max(np.abs(p.to_numpy()+swapped.to_numpy()-1))),
        old_features_max_error=float(np.max(np.abs(x[list(base)].to_numpy()-base.to_numpy()))),
        feature_sha256=sha256(out/'development_features.csv'),validation_scoring_only_after_freeze=True)
    if check['reload_max_error'] > 1e-12 or check['swap_max_error'] > 1e-12:
        raise ValueError(f'Frozen prediction self-check failed: {check}')
    (out/'self_check.json').write_text(json.dumps(check,indent=2)+'\n')
    write_report(out,summary,frozen,comparison,matched)
    print(comparison.to_string(index=False),flush=True)


if __name__ == '__main__':
    from scripts.train_retirement_round10 import main as canonical_main
    with threadpool_limits(limits=4):
        canonical_main()
