"""Read-only diagnostics of already-saved R1--R4 development predictions.

No estimator is fitted, no decision threshold optimized, no row excluded from
full-cohort scores, and no 2024--2025 outcome file is opened. Subgroups and the
label-aware oracle are descriptive diagnostics, never replacement test scores.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from math import ceil
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.metrics import score_probabilities

ROOT = Path(__file__).resolve().parents[1]
ROUNDS = ('modeling', 'round2', 'round3', 'round4')
CURRENT = 'R4 age spline'
CORE = ['R1 linear', 'R2 spline', 'R3 spline', 'R3 linear', CURRENT, 'R4 CatBoost']
IDEAS = [
    dict(id='A1', stage='independent-before-other-agent-ideas', origin='AI-assisted',
         idea='Abrupt changes or sparse history may dominate stale T-21 skill estimates.',
         assumption='Existing historical summaries miss a useful state-change signal.',
         prediction='OOF errors concentrate in sparse-history or long-inactivity groups.',
         falsifier='Those groups are no worse, or contain too few errors to matter.',
         next_check='Investigate uncertainty-aware dynamics without relaxing T-21.'),
    dict(id='A2', stage='independent-before-other-agent-ideas', origin='AI-assisted',
         idea='Existing learner families may share errors because their input information is similar.',
         assumption='Additional independent pre-match information is available.',
         prediction='High error intersections limit gains from another small same-feature grid.',
         falsifier='Current families disagree widely and useful routing is learnable in early folds.',
         next_check='Consider cross-level history or opponent-style representations, evaluated in training folds.'),
    dict(id='A3', stage='independent-before-other-agent-ideas', origin='AI-assisted',
         idea='Probability-loss improvements may improve confidence more than winner ordering.',
         assumption='A substantial error share is near the fixed 0.5 decision boundary.',
         prediction='R4 changes few classifications or offsets corrections with regressions.',
         falsifier='Most errors are confident mistakes, rather than near-boundary errors.',
         next_check='Seek new predictive ordering information; monotone symmetric calibration cannot fix accuracy.'),
]


def read_json(path):
    return json.loads(path.read_text())


def aligned_prediction(frame, rows, column):
    if frame.match_id.duplicated().any() or rows.match_id.duplicated().any():
        raise ValueError('Duplicate match IDs')
    if set(frame.match_id) != set(rows.match_id):
        raise ValueError('Saved predictions have a different cohort')
    aligned = frame.set_index('match_id').loc[rows.match_id]
    if not np.array_equal(aligned.a_won, rows.a_won):
        raise ValueError('Outcome/ID alignment mismatch')
    p = aligned[column].to_numpy(dtype=float)
    if not np.isfinite(p).all() or np.any((p < 0) | (p > 1)):
        raise ValueError('Invalid probabilities')
    return p


def collect_predictions(result_root, rows):
    frozen = {r:read_json(result_root/r/'frozen_selection.json') for r in ROUNDS}
    definitions = {
        'Rank logistic': ('modeling', frozen['modeling']['selected_by_family']['lr_000']),
        'R1 linear': ('modeling', frozen['modeling']['best_single']),
        'R2 spline': ('round2', frozen['round2']['best_single']),
        'R3 spline': ('round3', frozen['round3']['recommendation']),
        'R3 linear': ('round3', frozen['round3']['selected_by_family']['history2009_linear_format1']),
        CURRENT: ('round4', frozen['round4']['recommendation']),
        'R4 CatBoost': ('round4', frozen['round4']['selected_by_family']['paper_catboost']),
        'R4 all papers': ('round4', frozen['round4']['selected_by_family']['all_papers']),
    }
    outputs, provenance, observed = {}, {}, []
    for split, suffix in [('OOF 2020-2022', 'oof'), ('2023 exploratory', 'validation')]:
        target = rows.loc[rows.season.isin([2020,2021,2022]) if suffix=='oof' else rows.season.eq(2023)].copy()
        frames = {r:pd.read_csv(result_root/r/f'{suffix}_predictions.csv') for r in ROUNDS}
        p = {label:aligned_prediction(frames[r],target,name) for label,(r,name) in definitions.items()}
        first = frozen['modeling']
        if suffix=='oof':
            # Reconstruct only the ALREADY FROZEN convex weights, without any fit.
            p['R1 ensemble'] = sum(w*aligned_prediction(frames['modeling'],target,n)
                for n,w in zip(first['ensemble_components'],first['ensemble_weights']))
        else:
            p['R1 ensemble'] = aligned_prediction(frames['modeling'],target,'oof_convex_ensemble')
            for r in ROUNDS:
                table = pd.read_csv(result_root/r/'validation_comparison.csv')
                for record in table.to_dict('records'):
                    model = record['model']
                    score = metric(target, aligned_prediction(frames[r],target,model))
                    for key in ('matches','correct','accuracy','mean_log_loss'):
                        if abs(float(record[key])-score[key]) > 1e-12:
                            raise ValueError('Saved score differs from rescore')
                    observed.append(dict(round=r,model=model,**score))
        outputs[split] = (target,p)
    provenance['definitions'] = definitions
    provenance['R1 ensemble'] = dict(components=frozen['modeling']['ensemble_components'],
        weights=frozen['modeling']['ensemble_weights'],
        caution='OOF weights and family configuration choices used OOF labels; meta-training diagnostic, not independent validation.')
    return outputs,provenance,observed


def metric(rows,p):
    score=score_probabilities(zip(rows.match_id,rows.a_won),zip(rows.match_id,p))
    score.update(errors=score['matches']-score['correct'],
                 brier=float(np.mean((rows.a_won.to_numpy()-p)**2)),
                 mean_confidence=float(np.mean(np.maximum(p,1-p))))
    return score


def fixed_bins(values,edges,labels):
    return pd.cut(values,edges,labels=labels,right=False,include_lowest=True).astype(str).to_numpy()


def subgroup_labels(rows,x,p):
    confidence=np.maximum(p,1-p)
    # Reverse log1p only for a descriptive grouping. The original feature caps
    # gaps at 1095 days and assigns that sentinel to unseen players: keep unseen
    # history separate, and restore integer days before binning roundoff.
    inactive=np.rint(np.expm1(.5*(x.overall_inactivity_log_sum+abs(x.overall_inactivity_log_diff))))
    inactive_bins=fixed_bins(inactive,[0,45,90,180,np.inf],['0-<45','45-<90','90-<180','180+'])
    inactive_bins=np.where(x.overall_unseen_sum.gt(0),'unseen Tour history',inactive_bins)
    return {
        'surface':rows.surface.astype(str).to_numpy(),
        'format':np.where(x.format_best_of_five.eq(1),'best of 5','best of 3'),
        'absolute_rank_gap':fixed_bins(abs(rows.rank_a-rows.rank_b),[0,11,26,51,101,np.inf],
                                      ['0-10','11-25','26-50','51-100','101+']),
        'rank_ratio':fixed_bins(np.maximum(rows.rank_a,rows.rank_b)/np.minimum(rows.rank_a,rows.rank_b),
                               [1,1.5,2,4,np.inf],['1-<1.5','1.5-<2','2-<4','4+']),
        'start_month':pd.to_datetime(rows.tourney_date).dt.strftime('%m').to_numpy(),
        'season':rows.season.astype(str).to_numpy(),
        'R4_confidence':fixed_bins(confidence,[.5,.55,.6,.7,.8,.9,1.00000001],
                                  ['.50-<.55','.55-<.60','.60-<.70','.70-<.80','.80-<.90','.90-1']),
        'minimum_career_matches':fixed_bins(x.career_min_count,[0,10,30,100,np.inf],
                                           ['0-9','10-29','30-99','100+']),
        'minimum_surface_matches':fixed_bins(x.surface_career_min_count,[0,10,30,100,np.inf],
                                            ['0-9','10-29','30-99','100+']),
        'maximum_inactivity_days':inactive_bins,
    }


def diagnose_split(rows,x,p):
    y=rows.a_won.to_numpy()
    errors={n:(q>=.5)!=y for n,q in p.items()}
    full={n:metric(rows,q) for n,q in p.items()}
    pairs=[]
    for first in p:
        for second in p:
            a,b=errors[first],errors[second]
            union=int(np.sum(a|b)); intersection=int(np.sum(a&b))
            pairs.append(dict(first=first,second=second,both_wrong=intersection,
                both_correct=int(np.sum(~a&~b)),first_wrong_second_correct=int(np.sum(a&~b)),
                first_correct_second_wrong=int(np.sum(~a&b)),disagreements=int(np.sum(a!=b)),
                error_jaccard=intersection/union if union else 0.,
                probability_correlation=float(np.corrcoef(p[first],p[second])[0,1])))
    groups=[]
    labels=subgroup_labels(rows,x,p[CURRENT])
    for dimension,values in labels.items():
        if len(values)!=len(rows) or 'nan' in set(values):
            raise ValueError('Subgroups must partition the full cohort')
        for value in sorted(set(values)):
            mask=values==value
            for name,prob in p.items():
                groups.append(dict(dimension=dimension,group=value,model=name,
                    **metric(rows.loc[mask],prob[mask]),
                    errors_fraction_of_model_errors=float(errors[name][mask].sum()/errors[name].sum())))
    core_errors=np.stack([errors[n] for n in CORE],axis=1)
    unanimous_wrong=core_errors.all(axis=1)
    unanimous_correct=(~core_errors).all(axis=1)
    oracle_correct=int((~unanimous_wrong).sum())
    confidence=np.maximum(p[CURRENT],1-p[CURRENT])
    boundary={}
    for limit in (.55,.6,.7,.8,.9):
        mask=confidence<limit
        boundary[str(limit)]=dict(matches=int(mask.sum()),errors=int(errors[CURRENT][mask].sum()),
            fraction_of_errors=float(errors[CURRENT][mask].sum()/errors[CURRENT].sum()))
    return dict(full_cohort=full,pairwise=pairs,subgroups=groups,boundary=boundary,
        core_agreement=dict(models=CORE,unanimous_wrong=int(unanimous_wrong.sum()),
            unanimous_correct=int(unanimous_correct.sum()),
            disputed=int((~unanimous_wrong&~unanimous_correct).sum()),
            label_aware_oracle_correct=oracle_correct,label_aware_oracle_accuracy=oracle_correct/len(rows),
            oracle_caveat='Uses outcomes to pick an already-existing model per match; unattainable diagnostic bound, NOT a fitted model or achievable accuracy claim.',
            current_errors_shared_by_all=int(np.sum(errors[CURRENT]&unanimous_wrong)),
            current_errors_some_other_core_model_gets_right=int(np.sum(errors[CURRENT]&~unanimous_wrong))))


def markdown_table(columns,records):
    return '\n'.join(['| '+' | '.join(columns)+' |','| '+' | '.join(['---']*len(columns))+' |']+
                     ['| '+' | '.join(str(r[c]) for c in columns)+' |' for r in records])


def write_report(path,report):
    validation=report['splits']['2023 exploratory']
    oof=report['splits']['OOF 2020-2022']
    gap=report['seventy_percent_gap']
    lines=['# 第五轮：既有模型错误诊断（不训练）','',
        '固定0.5分类规则；全量2023仍为2,588场。以下为既有预测的描述性复核，不能用于宣称子集70%等于全量70%。2024–2025未读取。',
        '',f"全量70%至少需要 **{gap['required_correct']}** 场正确。当前推荐正确 **{gap['current_correct']}** 场，需净增 **{gap['additional_from_current']}** 场；历史最高观察准确率为 **{100*gap['best_observed_accuracy']:.2f}%**（{gap['best_observed_correct']}场），仍需净增 **{gap['additional_from_observed_best']}** 场。最高观察值仅用于描述，没有更改推荐。",'',
        '## 全量比较','',
        'OOF列是固定候选的逐年外推预测，但候选选择已用过这些折，具有选择偏差；第一轮融合权重也在OOF上拟合，因此该行OOF还包含融合层训练乐观偏差。', '']
    records=[]
    for name,s in validation['full_cohort'].items():
        v=oof['full_cohort'][name]
        records.append({'模型':name,'OOF准确率':f"{100*v['accuracy']:.2f}%",'2023正确':s['correct'],
            '2023准确率':f"{100*s['accuracy']:.2f}%",'2023 log loss':f"{s['mean_log_loss']:.6f}"})
    lines.extend([markdown_table(list(records[0]),records),'','## 错误重叠与边界',''])
    for split,d in report['splits'].items():
        agreement=d['core_agreement']; boundary=d['boundary']
        lines.append(f"- {split}：六个预先指定代表模型共同错 {agreement['unanimous_wrong']} 场；共同对 {agreement['unanimous_correct']} 场；存在分歧 {agreement['disputed']} 场。当前模型错误中，其他代表至少一个能做对 {agreement['current_errors_some_other_core_model_gets_right']} 场。这只是标签已知的互补空间，不能假定可以自动辨认这些比赛。")
        lines.append(f"- {split}：当前模型置信度低于55%的 {boundary['0.55']['matches']} 场中错 {boundary['0.55']['errors']} 场，占全部错误 {100*boundary['0.55']['fraction_of_errors']:.1f}%；低于60%的错误占比 {100*boundary['0.6']['fraction_of_errors']:.1f}%。没有调整任何阈值。")
    contrast=[r for r in validation['pairwise'] if r['second']==CURRENT and r['first'] in CORE and r['first']!=CURRENT]
    records=[{'前模型':r['first'],'R4修正':r['first_wrong_second_correct'],
              'R4新增错误':r['first_correct_second_wrong'],'共同错误':r['both_wrong'],
              '错误Jaccard':f"{r['error_jaccard']:.3f}"} for r in contrast]
    lines.extend(['',markdown_table(list(records[0]),records),'', '## 描述性分组（所有组保持原分母）','',
        '分组边界在计算前固定。月份为赛事开始月；置信度分组采用当前R4预测；不把任何高分组替换为全量指标。活动间隔来自T−21以前最后一场的日期，最长截为1095天，无Tour历史者单列；它不是未经删减的现实休赛天数。多重比较和球员重复出现未校正，不能把组间差当作机制已验证。',''])
    for dimension in ('surface','format','absolute_rank_gap','start_month','minimum_career_matches','maximum_inactivity_days'):
        records=[]
        for r in validation['subgroups']:
            if r['dimension']!=dimension or r['model']!=CURRENT: continue
            older=next(z for z in validation['subgroups'] if z['dimension']==dimension and z['group']==r['group'] and z['model']=='R3 spline')
            train=next(z for z in oof['subgroups'] if z['dimension']==dimension and z['group']==r['group'] and z['model']==CURRENT)
            records.append({'组':r['group'],'2023场数':r['matches'],'2023错误':r['errors'],
                'R4准确率':f"{100*r['accuracy']:.2f}%",'相对R3净增正确':r['correct']-older['correct'],
                'OOF场数':train['matches'],'OOF准确率':f"{100*train['accuracy']:.2f}%"})
        lines.extend([f'### {dimension}','',markdown_table(list(records[0]),records),''])
    agreement=validation['core_agreement']
    correct_without_disputed=agreement['unanimous_correct']
    needed_in_disputed=gap['required_correct']-correct_without_disputed
    format_rows={r['group']:r for r in validation['subgroups'] if r['dimension']=='format' and r['model']==CURRENT}
    near_ratio=[r for r in validation['subgroups'] if r['dimension']=='rank_ratio' and r['model']==CURRENT and r['group'] in ('1-<1.5','1.5-<2')]
    lines.extend(['## 对下一轮最有用的结论','',
        f"1. **单靠现有模型路由空间很紧。** 六模型共同错误占当前错误 {100*agreement['unanimous_wrong']/validation['full_cohort'][CURRENT]['errors']:.1f}%。只在已有分歧中选模型而不改变共同判断，达到70%需要在{agreement['disputed']}场分歧比赛中判对{needed_in_disputed}场（{100*needed_in_disputed/agreement['disputed']:.1f}%）。这是标签已知后才能计算的要求，不是已可达到的模型性能；优先检验真正新增信息，而非承诺小幅融合即可到70%。",
        f"2. **主要分母是三盘两胜制（best of 3）。** 它有{format_rows['best of 3']['matches']}场、{format_rows['best of 3']['errors']}个错误；五盘三胜仅{format_rows['best of 5']['matches']}场。后续模型仍须同时对全部比赛输出，重点比较大量普通巡回赛的净正确增量。",
        f"3. **相近实力仍难区分。** 排名比例小于2的{sum(r['matches'] for r in near_ratio)}场含{sum(r['errors'] for r in near_ratio)}个错误。R3到R4的2023修正64场与新增64错抵消，提示需要能改变相对排序的信息；单调、保持交换对称的概率校准不能改变0.5胜负边界。这个结论不能证明这些错误必然可预测。",
        '4. **保留对初始A1的反证。** 低于10场Tour历史组并未明显更差，长间隔组也不是稳定最差组；简单“新手/复出造成主要瓶颈”的说法不受这些边际分组支持。对手强弱、赛制及排名比例可能混杂该比较，新增跨级别历史应在训练折验证，而不是根据2023分组直接定策略。',
        '5. **不追逐单年月/场地噪声。** 2023草地准确率低于硬地，但OOF草地较高；六月亦呈方向变化。任何场地专用或时间自适应方案需先通过训练期逐年前推，不能从一次探索分组确认泛化收益。','',
        '## 独立初始假设与后续检查','',
        '这些想法在查看其他agent的新方案之前记录；已知前四轮总体结果，因此不是首次未见数据的预注册假设。按信息增量、可反证性、训练期可验证性、实现成本判断，未作伪精确打分。',''])
    for idea in IDEAS:
        lines.append(f"- **{idea['id']}**：{idea['idea']} 预期：{idea['prediction']} 反证：{idea['falsifier']} 下一步：{idea['next_check']}")
    lines.extend(['','## 复跑与限制','',
        '`.venv/bin/python -m scripts.diagnose_round5` 只读取四轮已保存的开发预测和既有特征；不反序列化模型、不训练、不调用最终测试数据。完整分组、全部模型对的重叠、来源哈希及所有已报告2023行保存在 `diagnostics.json`。',
        '', '本轮使用 scientific-brainstorming 技能组织独立想法与反证；只是诊断，不增加SOTA或70%已达到的证据。当前模型选择按log loss，70%是用户目标，不是从数据推导的可达保证。'])
    path.write_text('\n'.join(lines)+'\n')


def write_plot(path,report):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter
    fig,axes=plt.subplots(2,2,figsize=(12,8),layout='constrained')
    colors=['#65758b','#147d92']
    d=report['splits']['2023 exploratory']
    names=['Rank logistic','R1 linear','R2 spline','R3 spline','R3 linear',CURRENT]
    ax=axes[0,0]
    ax.barh(names,[d['full_cohort'][n]['accuracy'] for n in names],color=colors[1])
    ax.axvline(.7,color='#b23b3b',linestyle='--',label='70% target')
    ax.set_xlim(.5,.72);ax.xaxis.set_major_formatter(PercentFormatter(1));ax.invert_yaxis()
    ax.set_title('Full 2023 cohort: 2,588 matches');ax.legend(loc='lower right')
    ax=axes[0,1]
    for (split,data),color in zip(report['splits'].items(),colors):
        bins=[g for g in data['subgroups'] if g['model']==CURRENT and g['dimension']=='R4_confidence']
        ax.plot(range(len(bins)),[g['accuracy'] for g in bins],marker='o',color=color,label=split)
    ax.set_xticks(range(len(bins)),[g['group'] for g in bins],rotation=25)
    ax.yaxis.set_major_formatter(PercentFormatter(1));ax.set_ylim(.45,1)
    ax.set_title('Accuracy by fixed R4 confidence bins');ax.legend(fontsize=8)
    ax=axes[1,0]
    matrix=np.array([[next(r['error_jaccard'] for r in d['pairwise'] if r['first']==a and r['second']==b)
                       for b in CORE] for a in CORE])
    im=ax.imshow(matrix,cmap='Blues',vmin=.7,vmax=1)
    for i in range(len(CORE)):
        for j in range(len(CORE)):ax.text(j,i,f'{matrix[i,j]:.2f}',ha='center',va='center',fontsize=8,
                                         color='white' if matrix[i,j]>.91 else 'black')
    ax.set_xticks(range(len(CORE)),CORE,rotation=35,ha='right',fontsize=8)
    ax.set_yticks(range(len(CORE)),CORE,fontsize=8)
    ax.set_title('2023 error overlap (Jaccard)');fig.colorbar(im,ax=ax,shrink=.8)
    ax=axes[1,1]
    v=d['core_agreement']
    vals=[v['unanimous_correct'],v['unanimous_wrong'],v['disputed']]
    ax.bar(['All six correct','All six wrong','Disputed'],vals,color=['#147d92','#b45a5a','#b9a06d'])
    for i,value in enumerate(vals):ax.text(i,value+20,str(value),ha='center')
    ax.set_ylim(0,max(vals)*1.15);ax.set_ylabel('Matches');ax.set_title('2023 shared and differing decisions')
    fig.suptitle('Existing predictions only — descriptive diagnostics, no fitting or threshold tuning',fontsize=13)
    fig.savefig(path,dpi=160)
    plt.close(fig)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results',type=Path,default=ROOT/'results')
    parser.add_argument('--output',type=Path,default=ROOT/'results/round5')
    args=parser.parse_args()
    rows=pd.read_csv(args.results/'round3/development_targets.csv')
    x=pd.read_csv(args.results/'round4/development_features.csv')
    if not rows.season.between(2009,2023).all() or len(rows)!=len(x):
        raise ValueError('Unexpected or held-out target cohort')
    outputs,provenance,observed=collect_predictions(args.results,rows)
    split_report={split:diagnose_split(target,x.loc[target.index],p) for split,(target,p) in outputs.items()}
    n=split_report['2023 exploratory']['full_cohort'][CURRENT]['matches']
    current=split_report['2023 exploratory']['full_cohort'][CURRENT]['correct']
    best_correct=max(r['correct'] for r in observed)
    source_paths=[args.results/'round3/development_targets.csv',args.results/'round4/development_features.csv']
    source_paths += [args.results/r/name for r in ROUNDS for name in
                    ('frozen_selection.json','oof_predictions.csv','validation_predictions.csv','validation_comparison.csv')]
    report=dict(scope='Read-only diagnostics of already-saved development predictions',
        decision_threshold=.5,no_training=True,no_threshold_tuning=True,no_subset_replaces_full_cohort=True,
        final_test_evaluated=False,initial_independent_ideas=IDEAS,provenance=provenance,
        source_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in source_paths},
        splits=split_report,all_previously_reported_2023_models=observed,
        seventy_percent_gap=dict(matches=n,required_correct=ceil(.7*n),current_correct=current,
            additional_from_current=ceil(.7*n)-current,best_observed_correct=best_correct,
            best_observed_accuracy=best_correct/n,additional_from_observed_best=ceil(.7*n)-best_correct,
            best_observed_models=[dict(round=r['round'],model=r['model']) for r in observed if r['correct']==best_correct]))
    args.output.mkdir(parents=True,exist_ok=True)
    (args.output/'diagnostics.json').write_text(json.dumps(report,indent=2)+'\n')
    write_report(args.output/'diagnostics.md',report)
    write_plot(args.output/'diagnostics.png',report)
    combined=[dict(split=split,**row) for split,d in split_report.items() for row in d['subgroups']]
    pd.DataFrame(combined).to_csv(args.output/'diagnostics.csv',index=False)
    print(json.dumps(report['seventy_percent_gap'],indent=2))
    for split,d in split_report.items():
        print(split,d['core_agreement'],d['boundary'])


if __name__=='__main__':
    main()
