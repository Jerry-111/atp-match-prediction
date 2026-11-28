"""Honest forward meta learning on four frozen primitive probability streams."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import joblib
import numpy as np
import pandas as pd
import scipy
from threadpoolctl import threadpool_limits

from scripts.advanced_features import _date
from scripts.multilevel_features import _season_date
from scripts.conditional_gate import (EXPERTS,CONTEXTS,FEATURE_FIELDS,PROBABILITY_EPS,
    WEIGHT_FLOOR,PENALTY,context_frame,fit_global,fit_conditional,equal_mixture,meta_training_mask)
from scripts.train_models import ROOT,sha256,mean_loss,cluster_interval
from scripts.train_round5 import accuracy,accuracy_interval
from scripts.train_joint_points_round8 import score_table

REFERENCE='round4_reference'
PROCEDURES=('equal_mix','global_mix','conditional_mix')
REPORT_NAMES=(*PROCEDURES,REFERENCE)
SOURCE_BRANCHES=('round4','round5','round9/neural','round9/sequence')


def source_bindings():
    bindings={}
    for branch,expert in zip(SOURCE_BRANCHES,EXPERTS):
        directory=ROOT/'results'/branch
        audit=json.loads((directory/'artifact_audit.json').read_text())
        if audit['status']!='passed':raise ValueError(f'Unverified primitive branch: {branch}')
        files={}
        for filename in ('oof_predictions.csv','validation_predictions.csv'):
            actual=sha256(directory/filename)
            if actual!=audit['artifact_sha256'][filename]:raise ValueError(f'Audited source changed: {branch}/{filename}')
            files[filename]=actual
        bindings[expert]=dict(branch=branch,files=files,audit_sha256=sha256(directory/'artifact_audit.json'))
    return bindings


def read_experts(rows,bindings,*,final=False):
    filename='validation_predictions.csv' if final else 'oof_predictions.csv'
    result=pd.DataFrame(index=rows.match_id.to_numpy())
    for expert in EXPERTS:
        info=bindings[expert];path=ROOT/'results'/info['branch']/filename
        if sha256(path)!=info['files'][filename]:raise ValueError('Primitive source changed after registry')
        source=pd.read_csv(path)
        if source.match_id.duplicated().any() or set(source.match_id)!=set(rows.match_id):
            raise ValueError('Primitive source target cohort changed')
        source=source.set_index('match_id').loc[rows.match_id]
        if not np.array_equal(source.a_won.to_numpy(),rows.a_won.to_numpy()):
            raise ValueError('Primitive source label mismatch')
        if not np.array_equal(source.season.to_numpy(),rows.season.to_numpy()):
            raise ValueError('Primitive source season mismatch')
        p=source[expert].to_numpy(dtype=float)
        if not np.isfinite(p).all() or np.any((p<0)|(p>1)):raise ValueError('Invalid primitive probability')
        result[expert]=p
    result.index.name='match_id'
    return result


def _valid_predictions(y,predictions):
    y=np.asarray(y)
    if y.ndim!=1 or not len(y) or not np.isin(y,[0,1]).all():raise ValueError('Binary selection labels required')
    if tuple(predictions.columns)!=REPORT_NAMES or len(predictions)!=len(y):
        raise ValueError('Exact four-entry report/selection schema required')
    p=predictions.to_numpy(dtype=float)
    if not np.isfinite(p).all() or np.any((p<0)|(p>1)):raise ValueError('Invalid selection probabilities')
    return y


def select_2021(rows,predictions):
    if not rows.season.eq(2021).all():raise ValueError('Only 2021 selects the meta procedure')
    if not rows.index.equals(predictions.index):raise ValueError('Selection probability/label row indices must match')
    y=_valid_predictions(rows.a_won.to_numpy(),predictions)
    limit=mean_loss(y,predictions[REFERENCE].to_numpy())+.005
    allowed=[n for n in REPORT_NAMES if mean_loss(y,predictions[n].to_numpy())<=limit+1e-12]
    name=min(allowed,key=lambda n:(-accuracy(y,predictions[n].to_numpy()),
                                    mean_loss(y,predictions[n].to_numpy()),n))
    return dict(choice=name,selection_season=2021,selection_rows=len(rows),
        accuracy=accuracy(y,predictions[name].to_numpy()),mean_log_loss=mean_loss(y,predictions[name].to_numpy()),
        eligible_entries=allowed,loss_limit=limit,decision_threshold=.5)


def gate_2022(choice,rows,predictions):
    if not rows.season.eq(2022).all() or choice not in REPORT_NAMES:
        raise ValueError('2022 evaluates the already selected procedure')
    if not rows.index.equals(predictions.index):raise ValueError('Gate probability/label row indices must match')
    y=_valid_predictions(rows.a_won.to_numpy(),predictions)
    p=predictions[choice].to_numpy();reference=predictions[REFERENCE].to_numpy()
    gain=accuracy(y,p)-accuracy(y,reference);increase=mean_loss(y,p)-mean_loss(y,reference)
    passed=gain>=.005-1e-12 and increase<=.005+1e-12
    return dict(passed=passed,procedure='adopt_gating' if passed else 'retain_round4',
        selected_procedure=choice,gate_season=2022,gate_rows=len(rows),
        accuracy_improvement=gain,loss_increase=increase,
        net_correct=int(round(gain*len(rows))),required_accuracy_improvement=.005,allowed_loss_increase=.005,
        reference_accuracy=accuracy(y,reference),candidate_accuracy=accuracy(y,p),
        reference_loss=mean_loss(y,reference),candidate_loss=mean_loss(y,p))


def clipping_counts(probabilities):
    return {name:int(((probabilities[name]<PROBABILITY_EPS)|(probabilities[name]>1-PROBABILITY_EPS)).sum())
            for name in EXPERTS}


def predict_stage(models,probabilities,contexts):
    if not probabilities.index.equals(contexts.index):raise ValueError('Inference row IDs must align')
    return pd.DataFrame(dict(equal_mix=equal_mixture(probabilities),
        global_mix=models['global'].predict_proba(probabilities),
        conditional_mix=models['conditional'].predict_proba(probabilities,contexts),
        round4_reference=probabilities[EXPERTS[0]].to_numpy()),index=probabilities.index)


def fit_stage(year,oof_rows,probabilities,contexts,target_rows):
    if not target_rows.season.eq(year).all():raise ValueError('Stage and query year must agree')
    if not probabilities.index.equals(oof_rows.index) or not contexts.index.equals(oof_rows.index):
        raise ValueError('OOF IDs and contexts must be aligned before meta fitting')
    mask,metadata=meta_training_mask(oof_rows,target_rows)
    p=probabilities.loc[mask];x=contexts.loc[mask];y=oof_rows.loc[mask,'a_won']
    global_model=fit_global(p,y);conditional=fit_conditional(p,y,x,global_model)
    metadata.update(stage=year,training_clipped_probabilities=clipping_counts(p),
        global_optimizer=global_model.diagnostics,conditional_optimizer=conditional.diagnostics,
        global_weights=global_model.weights.tolist(),context_mean=conditional.scaler.mean.tolist(),
        context_scale=conditional.scaler.scale.tolist(),active_contexts=conditional.scaler.active.tolist())
    return {'global':global_model,'conditional':conditional},metadata


def weight_summary(model,contexts,stage,population):
    weights=model.weights_for(contexts)
    return [dict(stage=stage,population=population,expert=name,rows=len(contexts),
        global_weight=float(model.global_weights[k]),mean=float(weights[:,k].mean()),
        std=float(weights[:,k].std()),minimum=float(weights[:,k].min()),maximum=float(weights[:,k].max()))
        for k,name in enumerate(EXPERTS)]


def routing_contrast(rows,predictions,period):
    y=rows.a_won.to_numpy();g=predictions.global_mix.to_numpy();c=predictions.conditional_mix.to_numpy()
    return dict(period=period,matches=len(rows),accuracy_improvement=accuracy(y,c)-accuracy(y,g),
        log_loss_reduction=mean_loss(y,g)-mean_loss(y,c),
        max_probability_change=float(np.max(np.abs(c-g))),mean_probability_change=float(np.mean(np.abs(c-g))))


def predict_inputs(bundle,inputs,stage='2023'):
    if stage not in bundle['models'] or inputs.empty or inputs.match_id.duplicated().any():
        raise ValueError('Unique nonempty query IDs and saved development stage required')
    if not inputs.season.eq(int(stage)).all():raise ValueError('Inference year must match the saved development stage')
    for day,season in zip(inputs.tourney_date,inputs.season):_season_date(_date(day),int(season))
    data=inputs.set_index('match_id')
    return predict_stage(bundle['models'][stage],data[list(EXPERTS)],data[list(CONTEXTS)])


def write_report(out):
    frozen=json.loads((out/'frozen_selection.json').read_text());summary=json.loads((out/'summary.json').read_text())
    table=pd.read_csv(out/'validation_comparison.csv').set_index('model')
    selected=table.loc[frozen['selected_procedure']];gate=frozen['procedure_gate']
    lines=['# Round 10: conditional reliability of four frozen experts','',
        'Original historical T−21 condition; all 2,588 targets, labels and threshold 0.5 remain fixed. '
        '2023 is repeatedly used exploratory validation. The 2024–2025 final test remains sealed.','',
        f"The procedure frozen on 2021 alone is **{frozen['selected_procedure']}**. Its 2022 gate gains "
        f"**{gate['net_correct']:+d}/2,528** ({100*gate['accuracy_improvement']:+.3f} pp), with loss change "
        f"{gate['loss_increase']:+.6f}. Branch recommendation: **{frozen['recommendation']}**.",'',
        f"The unchanged selected procedure scores **{int(selected.correct)}/2,588 = {selected.accuracy:.3%}**, "
        f"log loss **{selected.mean_log_loss:.6f}** on 2023; it is **{1812-int(selected.correct)}** correct short of 70%.",'',
        '2020 supplies 1,214 meta-training observations, not a learned-meta test. The honest forward '
        'meta comparison contains 4,916 rows in 2021–2022; it is not used for pooled reselection. '
        'Final meta fitting uses 6,130 original primitive OOF forecasts through 2022. No base network or classifier was retrained.','',
        '| Procedure | Correct / 2,588 | Accuracy | Log loss | Role |','| --- | ---: | ---: | ---: | --- |']
    for name in REPORT_NAMES:
        r=table.loc[name]
        role='2021-selected procedure' if name==frozen['selected_procedure'] else 'fixed control/reference'
        if name==frozen['recommendation']:role+='; branch recommendation'
        lines.append(f'| `{name}` | {int(r.correct)} | {r.accuracy:.3%} | {r.mean_log_loss:.6f} | {role} |')
    matched=pd.read_csv(out/'matched_routing.csv')
    lines+=['','## Does context help over the same global mixture?','',
        '| Period | Conditional accuracy change / pp | Conditional loss reduction | Mean probability change |',
        '| --- | ---: | ---: | ---: |']
    for r in matched.itertuples():
        lines.append(f'| {r.period} | {100*r.accuracy_improvement:+.3f} | {r.log_loss_reduction:+.6f} | {r.mean_probability_change:.6f} |')
    lines+=['',
        'Equal/global/conditional procedures share the same four primitive forecasts. The global weights '
        'have a fixed 0.02 floor; conditional routing uses exactly that anchor and a fixed 0.01 quadratic penalty '
        'on 15 identifiable coefficients. Its five even contexts use only preknown rank separation, coverage '
        'and format. A win over equal averaging alone is not proof of useful routing. Training losses are '
        'optimizer diagnostics, not forward scores. No best-2023 row replaces the frozen choice.','',
        'R4/R5 old fold estimators were not stored: their primitive chronology relies on frozen masks/source/cache audits. '
        'R9 fold models were independently reloaded. Reused earlier model selection and the repeated 2022 gate '
        'limit confirmation. Neither a mixture nor this audit restores an untouched historical test.','',
        f"Six optimizer runs completed in {summary['elapsed_seconds']:.2f}s. The numerical routine saves all three "
        'stages, context scalers, exact training IDs and weight spread. Probability clipping counts are explicit.',
        '', 'Reproduce: `.venv/bin/python -m scripts.train_gating_round10`. '
        'Reload: `.venv/bin/python -m scripts.train_gating_round10 --predict-inputs results/round10/gating/inference_inputs.csv`. '
        'Report only: `.venv/bin/python -m scripts.train_gating_round10 --report-only`.','',
        '[Protocol](../../../docs/research/round10_gating_protocol.md) · '
        '[OOF source legality](../../../docs/research/round10_oof_feasibility.md) · '
        '[Frozen choice](frozen_selection.json) · [Routing contrasts](matched_routing.csv) · '
        '[Weight spread](weight_spread.csv) · [Exact meta chronology](training_metadata.json)','']
    (out/'report_summary.md').write_text('\n'.join(lines))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=ROOT/'results/round10/gating')
    parser.add_argument('--predict-inputs',type=Path);parser.add_argument('--stage',choices=['2021','2022','2023'],default='2023')
    parser.add_argument('--report-only',action='store_true')
    args=parser.parse_args();out=args.output;out.mkdir(parents=True,exist_ok=True)
    if args.report_only:write_report(out);return
    if args.predict_inputs:
        p=predict_inputs(joblib.load(out/'frozen_models.joblib'),pd.read_csv(args.predict_inputs),args.stage)
        p.reset_index().to_csv(out/'inference_predictions.csv',index=False);return
    started=time.monotonic();bindings=source_bindings()
    rowpath=ROOT/'results/round3/development_targets.csv';xpath=ROOT/'results/round5/development_features.csv'
    prior=json.loads((ROOT/'results/round9/neural/search_registry.json').read_text())
    if sha256(rowpath)!=prior['targets_sha256'] or sha256(xpath)!=prior['features_sha256']:
        raise ValueError('Independently verified original inputs changed')
    rows=pd.read_csv(rowpath);x=pd.read_csv(xpath)
    if len(rows)!=36421 or x.shape!=(36421,326) or rows.match_id.duplicated().any():raise ValueError('Original input cohort changed')
    if not rows.season.between(2009,2023).all():raise ValueError('Only development target metadata allowed')
    for day,season in zip(rows.tourney_date,rows.season):_season_date(_date(day),int(season))
    rows.index=rows.match_id.to_numpy();x.index=rows.index
    contexts=context_frame(x);orows=rows[rows.season.between(2020,2022)].copy()
    if len(orows)!=6130 or rows.season.eq(2023).sum()!=2588:raise ValueError('Fixed meta/validation cohorts changed')
    experts=read_experts(orows,bindings);contexts=contexts.loc[rows.index]
    registry=dict(round=10,branch='gating',experts=list(EXPERTS),source_bindings=bindings,
        procedures=list(PROCEDURES),selection_entries=list(REPORT_NAMES),context_fields=list(FEATURE_FIELDS),
        context_columns=list(CONTEXTS),probability_clip=PROBABILITY_EPS,global_weight_floor=WEIGHT_FLOOR,
        gate_penalty=PENALTY,gate_parameters=15,optimizer_runs=6,base_expert_fits=0,
        meta_warmup_season=2020,selection_season=2021,gate_season=2022,threshold=.5,
        targets_sha256=sha256(rowpath),features_sha256=sha256(xpath),
        source_sha256=sha256(Path(__file__)),engine_sha256=sha256(ROOT/'scripts/conditional_gate.py'),
        protocol_sha256=sha256(ROOT/'docs/research/round10_gating_protocol.md'),
        feasibility_sha256=sha256(ROOT/'docs/research/round10_oof_feasibility.md'),
        dependency_versions=dict(numpy=np.__version__,pandas=pd.__version__,scipy=scipy.__version__,joblib=joblib.__version__),
        final_test_evaluated=False,exploratory_after_2023=True)
    (out/'search_registry.json').write_text(json.dumps(registry,indent=2)+'\n')
    pd.concat([orows[['match_id','season','tourney_date','a_won']],experts],axis=1).to_csv(out/'primitive_oof_inputs.csv',index=False)
    models={};metadata={};weight_rows=[];oof_frames=[];contrasts=[];choice=None
    for year in (2021,2022):
        query=orows[orows.season.eq(year)]
        models[str(year)],metadata[str(year)]=fit_stage(year,orows,experts,contexts.loc[orows.index],query)
        pred=predict_stage(models[str(year)],experts.loc[query.index],contexts.loc[query.index])
        metadata[str(year)]['query_clipped_probabilities']=clipping_counts(experts.loc[query.index])
        train_ids=metadata[str(year)]['training_ids']
        for population,ids in [('train',train_ids),('query',query.match_id.tolist())]:
            weight_rows+=weight_summary(models[str(year)]['conditional'],contexts.loc[ids],year,population)
        oof_frames.append(pd.concat([query[['match_id','season','tourney_date','a_won']],pred],axis=1))
        contrasts.append(routing_contrast(query,pred,str(year)))
        if year==2021:
            choice=select_2021(query,pred)
            (out/'frozen_2021_choice.json').write_text(json.dumps(choice,indent=2)+'\n')
            print('Frozen 2021 procedure: '+json.dumps(choice),flush=True)
        else:
            gate=gate_2022(choice['choice'],query,pred)
            frozen=dict(round=10,branch='gating',selected_procedure=choice['choice'],selection=choice,
                procedure_gate=gate,recommendation=choice['choice'] if gate['passed'] else REFERENCE,
                no_pooled_reselection=True,selection_rows=2388,gate_rows=2528,final_meta_fit_rows=6130,
                threshold=.5,exploratory_after_2023=True,final_test_evaluated=False,
                original_choice_sha256=sha256(out/'frozen_2021_choice.json'))
            (out/'frozen_selection.json').write_text(json.dumps(frozen,indent=2)+'\n')
            print('Frozen 2022 gate: '+json.dumps(gate),flush=True)
    forward=pd.concat(oof_frames)
    if len(forward)!=4916 or not set(forward.season)=={2021,2022}:raise ValueError('Learned-meta OOF population changed')
    forward.to_csv(out/'oof_predictions.csv',index=False)
    cv=[]
    for period in ('2021','2022','pooled_2021_2022_descriptive_only'):
        group=forward if period.startswith('pooled') else forward[forward.season.eq(int(period))]
        table=score_table(group,group[list(REPORT_NAMES)]);table['period']=period;cv.append(table)
    pd.concat(cv,ignore_index=True).to_csv(out/'temporal_cv.csv',index=False)
    validation=rows[rows.season.eq(2023)]
    models['2023'],metadata['2023']=fit_stage(2023,orows,experts,contexts.loc[orows.index],validation)
    train_ids=metadata['2023']['training_ids']
    weight_rows+=weight_summary(models['2023']['conditional'],contexts.loc[train_ids],2023,'train')
    bundle=dict(models=models,frozen=frozen,registry=registry,experts=EXPERTS,contexts=CONTEXTS)
    joblib.dump(bundle,out/'frozen_models.joblib')
    (out/'final_fit_manifest.json').write_text(json.dumps(dict(
        model_sha256=sha256(out/'frozen_models.joblib'),selection_sha256=sha256(out/'frozen_selection.json'),
        final_training_ids=train_ids,final_parameters_frozen_before_new_2023_prediction=True),indent=2)+'\n')
    # Only after all meta choices and parameters are frozen are 2023 expert
    # probability columns loaded and the new mixtures evaluated.
    final_experts=read_experts(validation,bindings,final=True)
    final_contexts=contexts.loc[validation.index]
    pred=predict_stage(models['2023'],final_experts,final_contexts)
    metadata['2023']['query_clipped_probabilities']=clipping_counts(final_experts)
    weight_rows+=weight_summary(models['2023']['conditional'],final_contexts,2023,'query')
    pd.DataFrame(weight_rows).to_csv(out/'weight_spread.csv',index=False)
    (out/'training_metadata.json').write_text(json.dumps(metadata,indent=2)+'\n')
    inputs=pd.concat([validation[['match_id','season','tourney_date']],final_experts,final_contexts],axis=1)
    inputs.to_csv(out/'inference_inputs.csv',index=False)
    pd.concat([validation[['match_id','season','a_won']],final_experts],axis=1).to_csv(out/'validation_experts.csv',index=False)
    pd.concat([validation[['match_id','season','tourney_date','a_won']],pred],axis=1).to_csv(out/'validation_predictions.csv',index=False)
    comparison=score_table(validation,pred,recommendation=frozen['recommendation'])
    comparison.to_csv(out/'validation_comparison.csv',index=False)
    contrasts.append(routing_contrast(validation,pred,'2023'))
    pd.DataFrame(contrasts).to_csv(out/'matched_routing.csv',index=False)
    selected=pred[frozen['selected_procedure']].to_numpy()
    intervals=dict(selected_vs_reference=dict(
        accuracy=accuracy_interval(validation,pred[REFERENCE].to_numpy(),selected),
        log_loss=cluster_interval(validation,pred[REFERENCE].to_numpy(),selected)),
        conditional_vs_global=dict(accuracy=accuracy_interval(validation,pred.global_mix.to_numpy(),pred.conditional_mix.to_numpy()),
        log_loss=cluster_interval(validation,pred.global_mix.to_numpy(),pred.conditional_mix.to_numpy())))
    summary=dict(round=10,branch='gating',selected_procedure=frozen['selected_procedure'],
        recommendation=frozen['recommendation'],validation_matches=2588,meta_forward_rows=4916,
        final_meta_training_rows=6130,base_expert_fits=0,optimizer_runs=6,gate_parameters=15,
        correct_required_for_70=1812,comparisons=intervals,elapsed_seconds=time.monotonic()-started,
        final_test_evaluated=False,exploratory_after_2023=True)
    (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    write_report(out);print(comparison.to_string(index=False),flush=True)


if __name__=='__main__':
    from scripts.train_gating_round10 import main as canonical_main
    with threadpool_limits(limits=4):canonical_main()
