"""Three fixed player-history encoders, averaged over two seeds; T-21 only."""
from __future__ import annotations

import argparse
from datetime import datetime,timezone
import importlib.metadata
import json
from pathlib import Path
import time

import joblib
import numpy as np
import pandas as pd
import torch
from threadpoolctl import threadpool_limits

from scripts.baseline import DEFAULT_CACHE,fetch_csv
from scripts.sequence_features import build_sequence_features,save_sequences,load_sequences
from scripts.sequence_learning import ARCHITECTURES,SEEDS,fit_sequence,predict_average,check_sequence
from scripts.train_models import ROOT,FOLDS,sha256,temporal_masks,mean_loss,cluster_interval,mirror_features
from scripts.train_round5 import REFERENCE,accuracy,accuracy_interval,predict_round5
from scripts.train_distillation_round7 import validate_rows,predict_distillation,R5_COMPARATOR,R5_LOSS
from scripts.train_venue_round8 import select_single,choose_procedure
from scripts.metrics import score_probabilities

STUDENT='previous_round7_market_student'


def candidate_specs():
    return [dict(name=name+(f'_r4blend{w:g}' if w<1 else ''),architecture=name,weight=w)
            for name in ARCHITECTURES for w in (1.,.25,.5,.75)]


def compose_predictions(averages,reference):
    reference=np.asarray(reference,dtype=float)
    if reference.ndim!=1 or not np.isfinite(reference).all() or np.any((reference<0)|(reference>1)):
        raise ValueError('Finite reference probability vector required')
    output={}
    for spec in candidate_specs():
        p=np.asarray(averages[spec['architecture']],dtype=float)
        if p.shape!=reference.shape or not np.isfinite(p).all() or np.any((p<0)|(p>1)):
            raise ValueError('Finite aligned seed-average probabilities required')
        output[spec['name']]=spec['weight']*p+(1-spec['weight'])*reference
    output[REFERENCE]=reference.copy()
    return pd.DataFrame(output)


def prepare_sequences(rows,out):
    path=out/'sequence_features.npz';meta_path=out/'sequence_metadata.json'
    target_path=ROOT/'results/round3/development_targets.csv';inventory_path=ROOT/'data/processed/mens_singles_2006_2025.csv'
    signature=dict(target_sha256=sha256(target_path),inventory_sha256=sha256(inventory_path),
        feature_code_sha256=sha256(ROOT/'scripts/sequence_features.py'))
    metadata=json.loads(meta_path.read_text()) if meta_path.exists() else {}
    cached=(path.exists() and all(metadata.get(k)==v for k,v in signature.items())
            and metadata.get('sequence_sha256')==sha256(path))
    if cached:
        for entry in metadata['source_files']:
            if not 2006<=int(entry['season'])<=2023:raise ValueError('Final source year forbidden')
            fetch_csv(DEFAULT_CACHE,entry)
        sequences=load_sequences(path,metadata=metadata)
    else:
        inventory=pd.read_csv(inventory_path,dtype={'a_won':'string'})
        sequences=build_sequence_features(rows.drop(columns=['a_won']),inventory)
        save_sequences(path,sequences)
        metadata=dict(sequences.metadata,**signature,sequence_sha256=sha256(path))
        meta_path.write_text(json.dumps(metadata,indent=2)+'\n')
    check_sequence(sequences)
    if (not np.array_equal(sequences.match_ids,rows.match_id.to_numpy())
            or not np.array_equal(sequences.indices,rows.index.to_numpy())):
        raise ValueError('Cached sequence target identity/order changed')
    cutoffs=np.array([(pd.Timestamp(d)-pd.Timedelta(days=21)).date().toordinal() for d in rows.tourney_date])[:,None,None]
    if (np.any(sequences.mask & (sequences.dates>cutoffs))
            or np.any(sequences.mask & (sequences.dates<cutoffs-1460))
            or np.any((np.diff(sequences.dates,axis=-1)<=0)&sequences.mask[:,:,1:]&sequences.mask[:,:,:-1])):
        raise ValueError('Cached sequence dates violate strict chronological boundaries')
    return sequences,metadata


def predict_sequence_bundle(bundle,x,sequences,*,device='cpu'):
    if not set(bundle['feature_names']).issubset(x) or x.columns.duplicated().any():
        raise ValueError('Frozen static feature schema missing/duplicated')
    x=x[bundle['feature_names']]
    old=predict_round5(bundle['previous_bundle'],x)
    averages={name:predict_average(models,x,sequences,device=device) for name,models in bundle['models'].items()}
    output=compose_predictions(averages,old[REFERENCE].to_numpy())
    output.index=x.index
    for alias,name in bundle['previous_aliases'].items():output[alias]=old[name].to_numpy()
    student=predict_distillation(bundle['student_bundle'],x)
    output[STUDENT]=student[bundle['student_name']].to_numpy()
    return output


def write_report(out,summary,frozen,table,matched):
    pooled=frozen['pooled_selection']['choice'];gate=frozen['procedure_gate']
    selected=table.set_index('model').loc[pooled]
    text=['# Round 9：球员历史赛事序列', '',
        '全2588场2023、固定0.5阈值和原T−21历史信息规则保持不变；2024–2025未读取。'
        '这是反复使用2023后的探索结果，不能证明SOTA。每人最近16个不同历史赛事日期组，'
        '同日比赛先聚合，四年窗口内共22字段；技术统计缺失和零历史均保留。', '',
        f"本分支推荐 `{frozen['recommendation']}`。2020–2021早选 `{gate['early_selection']['choice']}`；"
        f"2022准确率变化 {100*gate['accuracy_improvement']:+.3f} 个百分点（门槛+0.5），"
        f"loss变化 {gate['candidate_2022_loss']-gate['reference_2022_loss']:+.6f}（最多+0.005）。", '',
        f"pooled冻结候选 `{pooled}`：{int(selected.correct)}/2588，{selected.accuracy:.3%}，"
        f"loss {selected.mean_log_loss:.6f}。距离70%所需1812场还差 {max(0,1812-int(selected.correct))} 场。", '',
        '3个结构（mean、GRU、固定打乱GRU）×2固定种子，按概率平均；从不选择较好的种子。'
        '每个结构另有3个预先固定R4混合：3raw+9mix+R4=13可选项，混合不增加训练。'
        '共24个返回的seed模型（18个OOF、6个最终），各自先进行一次严格内层时间尾段epoch选择，'
        '合计48个优化阶段。所有结构接收相同326静态字段、时间字段和mask。', '',
        '| 模型 | 正确数 | 准确率 | Log loss |', '| --- | ---: | ---: | ---: |']
    for r in table.itertuples():text.append(f'| `{r.model}` | {r.correct} | {r.accuracy:.3%} | {r.mean_log_loss:.6f} |')
    text+=['','| 原始编码器对照 | Pooled CV准确率变化 | Pooled CV loss变化 | 2023准确率变化 | 2023 loss变化 |',
        '| --- | ---: | ---: | ---: | ---: |']
    for r in matched.itertuples():text.append(f'| {r.candidate} − {r.control} | {100*r.oof_accuracy_change:+.3f}pp | {r.oof_loss_change:+.6f} | {100*r.validation_accuracy_change:+.3f}pp | {r.validation_loss_change:+.6f} |')
    comp=summary['comparisons_vs_r4'][pooled];ci=comp['log_loss']['ci95']
    text+=['',f"冻结候选比R4净多判对 {comp['accuracy']['net_correct']:+d} 场；loss降低 "
        f"{comp['log_loss']['mean_log_loss_reduction']:.6f}，条件日期块95%区间 [{ci[0]:.6f}, {ci[1]:.6f}]。"
        '区间不覆盖反复搜索、重复门槛或全部球员相关性。', '',
        f"2023共有 {summary['zero_history_player_appearances_2023']} 次球员出场无可用序列，"
        '编码为零向量并继续使用同一静态输入，没有删除目标。技术统计不完整的历史比赛仍保留其胜负、排名和场地。', '',
        'mean池化同样看到时间字段，因此GRU差异涉及归纳偏置与容量；固定shuffle可被足够强的模型学会逆转，'
        '它不等于彻底删除时间信息。公开通用序列论文或其他筛选样本的网球准确率不能替代本实验结果。', '',
        '复跑：`.venv-foundation/bin/python -m scripts.train_sequence_round9 --device cpu`。',
        '重载：同一命令追加 `--predict-features results/round5/development_features.csv '
        '--predict-sequences results/round9/sequence/sequence_features.npz`。这些全开发集推理输出不是早年OOF；'
        '真正OOF及18个相应fold模型单独保存。',
        '协议：[round9_sequence.md](../../../docs/research/round9_sequence.md)。','']
    (out/'report_summary.md').write_text('\n'.join(text))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=ROOT/'results/round9/sequence')
    parser.add_argument('--device',choices=['cpu','mps'],default='cpu')
    parser.add_argument('--predict-features',type=Path)
    parser.add_argument('--predict-sequences',type=Path)
    args=parser.parse_args();out=args.output;out.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(4)
    if args.predict_features:
        if args.predict_sequences is None:raise ValueError('Frozen sequence archive required')
        sequences=load_sequences(args.predict_sequences)
        x=pd.read_csv(args.predict_features);x.index=sequences.indices
        result=predict_sequence_bundle(joblib.load(out/'frozen_models.joblib'),x,sequences,device=args.device)
        result.insert(0,'match_id',sequences.match_ids);result.to_csv(out/'inference_predictions.csv',index=False)
        return
    start=time.perf_counter();r5=ROOT/'results/round5';d7=ROOT/'results/round7/distillation'
    audit=json.loads((r5/'artifact_audit.json').read_text())
    for name in ('development_features.csv','frozen_models.joblib','oof_predictions.csv','frozen_selection.json'):
        if sha256(r5/name)!=audit['artifact_sha256'][name]:raise ValueError('Audited static artifact changed: '+name)
    rows=pd.read_csv(ROOT/'results/round3/development_targets.csv');validate_rows(rows)
    prior_registry=json.loads((d7/'search_registry.json').read_text())
    if sha256(ROOT/'results/round3/development_targets.csv')!=prior_registry['target_sha256']:
        raise ValueError('Audited targets changed')
    x=pd.read_csv(r5/'development_features.csv')
    if x.shape!=(len(rows),326):raise ValueError('Expected unchanged 326 static features')
    sequences,metadata=prepare_sequences(rows,out)
    print(f'Validated sequence archive {sequences.values.shape}, {metadata["historical_matches"]} past matches',flush=True)
    fold_plan={year:temporal_masks(rows,year) for year in (*FOLDS,2023)}
    registry=dict(round=9,branch='historical_player_sequence',architectures=list(ARCHITECTURES),seeds=list(SEEDS),
        candidates=candidate_specs(),selectable_entries=13,fixed_blends=9,returned_seed_fits=24,optimization_stages=48,
        protocol_sha256=sha256(ROOT/'docs/research/round9_sequence.md'),runner_sha256=sha256(Path(__file__)),
        learner_sha256=sha256(ROOT/'scripts/sequence_learning.py'),feature_code_sha256=sha256(ROOT/'scripts/sequence_features.py'),
        target_sha256=sha256(ROOT/'results/round3/development_targets.csv'),sequence_sha256=sha256(out/'sequence_features.npz'),
        source_metadata=metadata,previous_artifact_sha256=audit['artifact_sha256'],feature_names=list(x),
        device=args.device,cpu_threads=4,epochs='last earlier season inner-tail, 21-day purge, then reset/refit',
        dependency_versions={name:importlib.metadata.version(name) for name in ('torch','numpy','pandas','scikit-learn')},
        decision_threshold=.5,history_lag_days=21,exploratory_after_2023=True,final_test_evaluated=False)
    (out/'search_registry.json').write_text(json.dumps(registry,indent=2)+'\n')
    omask=rows.season.isin(FOLDS);orows=rows[omask];y=orows.a_won.to_numpy()
    old=pd.read_csv(r5/'oof_predictions.csv')
    if not old[['match_id','season','a_won']].equals(orows[['match_id','season','a_won']].reset_index(drop=True)):
        raise ValueError('R4 honest OOF alignment mismatch')
    seed_oof=pd.DataFrame(index=orows.index);average_oof={};oof_models={};fit_log=[]
    for kind in ARCHITECTURES:
        predictions=pd.Series(index=orows.index,dtype=float)
        for year in FOLDS:
            train,valid=fold_plan[year];tr,va=train.to_numpy(),valid.to_numpy();models=[]
            for seed in SEEDS:
                print(f'Fitting {kind}, fold{year}, seed{seed}',flush=True)
                model=fit_sequence(kind,seed,x[train],rows[train],sequences.take(tr),device=args.device)
                models.append(model);fit_log.append(dict(fold=year,**model.metadata))
                seed_oof.loc[rows.index[valid],f'{kind}_seed{seed}']=model.predict_proba(x[valid],sequences.take(va),device=args.device)
                print(f'  epoch={model.metadata["selected_epoch"]}, elapsed={model.metadata["elapsed_seconds"]:.2f}s',flush=True)
            oof_models[kind,year]=models
            predictions.loc[rows.index[valid]]=seed_oof.loc[rows.index[valid],[f'{kind}_seed{s}' for s in SEEDS]].mean(axis=1)
            (out/'fit_log.json').write_text(json.dumps(fit_log,indent=2)+'\n')
        average_oof[kind]=predictions.to_numpy()
        print(f'{kind} pooled CV accuracy={accuracy(y,predictions.to_numpy()):.6f}, loss={mean_loss(y,predictions.to_numpy()):.6f}',flush=True)
    oof=compose_predictions(average_oof,old[REFERENCE].to_numpy());values={c:oof[c].to_numpy() for c in oof}
    cv=[]
    for name,p in values.items():
        for fold in (*FOLDS,'pooled'):
            m=np.ones(len(y),bool) if fold=='pooled' else orows.season.eq(fold).to_numpy()
            cv.append(dict(model=name,fold=fold,matches=int(m.sum()),accuracy=accuracy(y[m],p[m]),mean_log_loss=mean_loss(y[m],p[m])))
    pd.DataFrame(cv).to_csv(out/'temporal_cv.csv',index=False)
    identity=orows[['match_id','season','a_won']].reset_index(drop=True)
    pd.concat([identity,oof],axis=1).to_csv(out/'oof_predictions.csv',index=False)
    pd.concat([identity,seed_oof.reset_index(drop=True)],axis=1).to_csv(out/'oof_seed_predictions.csv',index=False)
    joblib.dump(dict(models=oof_models,feature_names=list(x),registry=registry),out/'oof_models.joblib')
    gate=choose_procedure(y,values,orows.season.to_numpy());pooled=select_single(y,values,np.ones(len(y),bool))
    chosen=pooled['choice'] if gate['procedure']=='adopt_accuracy_model' else REFERENCE
    frozen=dict(round=9,branch='historical_player_sequence',recommendation=chosen,pooled_selection=pooled,
        procedure_gate=gate,decision_threshold=.5,frozen_at_utc=datetime.now(timezone.utc).isoformat(),
        fit_end_season=2022,history_lag_days=21,seeds=list(SEEDS),seed_aggregation='arithmetic probability average',
        exploratory_after_2023=True,final_test_evaluated=False)
    (out/'frozen_selection.json').write_text(json.dumps(frozen,indent=2)+'\n')
    print('Frozen selection:',json.dumps(frozen),flush=True)
    train,valid=fold_plan[2023];tr,va=train.to_numpy(),valid.to_numpy();models={}
    for kind in ARCHITECTURES:
        models[kind]=[]
        for seed in SEEDS:
            print(f'Final refit {kind}, seed{seed}',flush=True)
            model=fit_sequence(kind,seed,x[train],rows[train],sequences.take(tr),device=args.device)
            models[kind].append(model);fit_log.append(dict(fold='final',**model.metadata))
            print(f'  epoch={model.metadata["selected_epoch"]}, elapsed={model.metadata["elapsed_seconds"]:.2f}s',flush=True)
            (out/'fit_log.json').write_text(json.dumps(fit_log,indent=2)+'\n')
    old_frozen=json.loads((r5/'frozen_selection.json').read_text())
    student_frozen=json.loads((d7/'frozen_selection.json').read_text())
    bundle=dict(models=models,feature_names=list(x),frozen=frozen,registry=registry,
        previous_bundle=joblib.load(r5/'frozen_models.joblib'),student_bundle=joblib.load(d7/'frozen_models.joblib'),
        student_name=student_frozen['recommendation'],
        previous_aliases={R5_COMPARATOR:old_frozen['pooled_selection']['choice'],R5_LOSS:'all_inputs_spline_C0.01','rank_reference':'rank_reference'})
    joblib.dump(bundle,out/'frozen_models.joblib')
    vr=rows[valid];vs=sequences.take(va);p=predict_sequence_bundle(bundle,x[valid],vs,device='cpu')
    table=[]
    for name in p:
        score=score_probabilities(zip(vr.match_id,vr.a_won),zip(vr.match_id,p[name]))
        table.append(dict(model=name,recommended=name==chosen,**score,brier=float(np.mean((vr.a_won.to_numpy()-p[name].to_numpy())**2))))
    table=pd.DataFrame(table).sort_values(['accuracy','mean_log_loss'],ascending=[False,True])
    table.to_csv(out/'validation_comparison.csv',index=False)
    pd.concat([vr.reset_index(drop=True),p.reset_index(drop=True)],axis=1).to_csv(out/'validation_predictions.csv',index=False)
    seed_val={f'{kind}_seed{m.seed}':m.predict_proba(x[valid],vs) for kind,mm in models.items() for m in mm}
    pd.concat([vr[['match_id','season','a_won']].reset_index(drop=True),pd.DataFrame(seed_val)],axis=1).to_csv(out/'validation_seed_predictions.csv',index=False)
    matched=[]
    for a,b in [('sequence_gru','sequence_mean'),('sequence_gru','sequence_shuffled_gru'),('sequence_shuffled_gru','sequence_mean')]:
        matched.append(dict(candidate=a,control=b,oof_accuracy_change=accuracy(y,values[a])-accuracy(y,values[b]),
            oof_loss_change=mean_loss(y,values[a])-mean_loss(y,values[b]),
            validation_accuracy_change=accuracy(vr.a_won,p[a])-accuracy(vr.a_won,p[b]),
            validation_loss_change=mean_loss(vr.a_won.to_numpy(),p[a].to_numpy())-mean_loss(vr.a_won.to_numpy(),p[b].to_numpy())))
    matched=pd.DataFrame(matched);matched.to_csv(out/'matched_controls.csv',index=False)
    swapped=predict_sequence_bundle(bundle,mirror_features(x[valid]),vs.swapped())
    reloaded=predict_sequence_bundle(joblib.load(out/'frozen_models.joblib'),x[valid],vs)
    swap_error=float(np.max(np.abs(swapped.to_numpy()+p.to_numpy()-1)))
    reload_error=float(np.max(np.abs(reloaded.to_numpy()-p.to_numpy())))
    if swap_error>2e-7 or reload_error>1e-12:raise ValueError('Sequence frozen reload/exchange failed')
    summary=dict(round=9,branch='historical_player_sequence',recommendation=chosen,pooled_choice=pooled['choice'],
        validation_matches=len(vr),training_rows=int(train.sum()),new_architectures=3,fixed_blends=9,selectable_entries=13,
        returned_seed_fits=24,optimization_stages=48,prediction_columns=len(p.columns),
        parameter_counts={k:ms[0].metadata['parameter_count'] for k,ms in models.items()},
        zero_history_player_appearances_2023=int((~vs.mask.any(-1)).sum()),
        max_reload_error=reload_error,max_exchange_error=swap_error,
        comparisons_vs_r4={c:dict(accuracy=accuracy_interval(vr,p[REFERENCE].to_numpy(),p[c].to_numpy()),
            log_loss=cluster_interval(vr,p[REFERENCE].to_numpy(),p[c].to_numpy())) for c in p},
        elapsed_seconds=time.perf_counter()-start,correct_required_for_70=1812,
        inference_device='cpu',training_device=args.device,exploratory_after_2023=True,final_test_evaluated=False)
    (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    write_report(out,summary,frozen,table,matched)
    print(table.to_string(index=False),flush=True)


if __name__=='__main__':
    from scripts.train_sequence_round9 import main as canonical_main
    with threadpool_limits(limits=4):canonical_main()
