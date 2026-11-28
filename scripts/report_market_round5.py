"""Report the separate market-information experiment without fitting any model."""
from __future__ import annotations

import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/round5/market'
CLOSING='closing_market_descriptive'


def load_results():
    data={k:pd.read_csv(OUT/f) for k,f in (
        ('scores','validation_comparison.csv'),('predictions','validation_predictions.csv'),
        ('cv','temporal_cv.csv'),('features','features.csv'),('links','linked_matches.csv'))}
    for name in ('summary','frozen_selection','search_registry'):
        data[name]=json.loads((OUT/(name+'.json')).read_text())
        if data[name].get('final_test_evaluated') is not False:
            raise ValueError('Market exploratory report requires unopened final holdout')
    data['frozen']=data['frozen_selection'];data['registry']=data['search_registry']
    p=data['predictions'];scores=data['scores']
    if len(p)!=2588 or p.match_id.duplicated().any() or not p.season.eq(2023).all():
        raise ValueError('Expected full unchanged 2,588-match validation cohort')
    if not scores.matches.eq(len(p)).all() or scores.model.duplicated().any():
        raise ValueError('Scoring denominator or model identity mismatch')
    chosen=data['frozen']['recommendation']
    if data['summary']['recommendation']!=chosen or not bool(scores.set_index('model').loc[chosen,'recommended']):
        raise ValueError('Frozen recommendation mismatch')
    f=data['features'].set_index('match_id').loc[p.match_id]
    if data['features'].match_id.duplicated().any():raise ValueError('Duplicate market feature IDs')
    source=p.history_only.to_numpy()
    missing=f.market_open_available.eq(0).to_numpy()
    for name in scores.model:
        probability=p[name].to_numpy()
        if not np.isfinite(probability).all() or np.any((probability<0)|(probability>1)):
            raise ValueError('Invalid reported probabilities')
        actual_correct=int(np.sum((probability>=.5)==p.a_won.to_numpy()))
        if actual_correct!=int(scores.set_index('model').loc[name,'correct']):
            raise ValueError('Correct-count mismatch')
        mask=f.market_close_available.eq(0).to_numpy() if name==CLOSING else missing
        if not np.array_equal(probability[mask],source[mask]):
            raise ValueError('Missing-quote fallback changed historical probabilities')
    old=pd.read_csv(ROOT/'results/round4/validation_predictions.csv')
    base=data['registry']['base_model']
    if list(old.match_id)!=list(p.match_id) or not np.array_equal(old[base],p.history_only):
        raise ValueError('Historical reference is not the fixed Round4 base')
    for field,column in [('opening_available','market_open_available'),('closing_available','market_close_available')]:
        if int(f[column].sum())!=data['summary'][field]:raise ValueError('Quote coverage mismatch')
    data['validation_features']=f
    return data


def label(name):
    labels={'history_only':'Frozen Round4 history only','opening_market':'Raw opening market + fallback',
            CLOSING:'Raw closing market + fallback (descriptive)'}
    return labels.get(name,name.replace('_',' '))


def best_observed(data,include_closing=True):
    scores=data['scores']
    if not include_closing:scores=scores[scores.model.ne(CLOSING)]
    return scores.sort_values(['accuracy','mean_log_loss','model'],ascending=[False,True,True]).iloc[0]


def selected_names(data):
    return list(dict.fromkeys(['history_only','opening_market',CLOSING,data['frozen']['recommendation'],
                              best_observed(data,False).model,best_observed(data).model]))


def comparison_table(data):
    scores=data['scores'].copy()
    scores['description']=scores.model.map(label)
    scores['role']='Exploratory fixed opening candidate'
    scores.loc[scores.model.eq('history_only'),'role']='Fixed Round4 historical reference'
    scores.loc[scores.model.eq('opening_market'),'role']='Raw opening benchmark'
    scores.loc[scores.model.eq(CLOSING),'role']='Later closing benchmark; never selected'
    scores.loc[scores.model.eq(data['frozen']['recommendation']),'role']='Frozen recommendation in market-information regime'
    need=data['summary']['correct_required_for_70']
    scores['additional_correct_for_70']=(need-scores.correct).clip(lower=0)
    reference=scores.set_index('model').loc['opening_market']
    scores['accuracy_gain_vs_opening_pp']=100*(scores.accuracy-reference.accuracy)
    scores['log_loss_reduction_vs_opening']=reference.mean_log_loss-scores.mean_log_loss
    return scores[['model','description','role','matches','correct','accuracy','mean_log_loss','brier',
                   'additional_correct_for_70','accuracy_gain_vs_opening_pp','log_loss_reduction_vs_opening']]


def coverage_table(data):
    coverage=data['registry']['market_metadata']['coverage_by_season']
    return pd.DataFrame([dict(season=year,**counts,
        opening_missing_fallback=counts['targets']-counts['opening_available'],
        closing_missing_fallback=counts['targets']-counts['closing_available']) for year,counts in coverage.items()])


def intervals_table(data):
    rows=[]
    for name,v in data['summary']['comparisons_vs_history'].items():
        opening=data['summary']['accuracy_comparisons_vs_opening'][name]
        scores=data['scores'].set_index('model')
        rows.append(dict(model=name,accuracy_gain_vs_history_pp=100*(scores.loc[name,'accuracy']-scores.loc['history_only','accuracy']),
            history_accuracy_lower95_pp=100*v['accuracy']['ci95'][0],history_accuracy_upper95_pp=100*v['accuracy']['ci95'][1],
            log_loss_reduction_vs_history=float(scores.loc['history_only','mean_log_loss']-scores.loc[name,'mean_log_loss']),
            history_loss_lower95=v['log_loss']['ci95'][0],history_loss_upper95=v['log_loss']['ci95'][1],
            accuracy_gain_vs_opening_pp=100*(scores.loc[name,'accuracy']-scores.loc['opening_market','accuracy']),
            opening_accuracy_lower95_pp=100*opening['ci95'][0],opening_accuracy_upper95_pp=100*opening['ci95'][1]))
    return pd.DataFrame(rows)


def plot_comparison(data):
    names=selected_names(data);score=data['scores'].set_index('model').loc[names]
    chosen=data['frozen']['recommendation']
    colors=['#8a929a' if n=='history_only' else '#b47e27' if n==CLOSING else '#117f88' if n==chosen else '#3c5975' for n in names]
    fig,(acc,loss)=plt.subplots(1,2,figsize=(13,max(4.8,.65*len(names))),sharey=True,
                               gridspec_kw={'width_ratios':[1.3,1]})
    y=np.arange(len(names));acc.scatter(score.accuracy,y,c=colors,s=65)
    acc.set_yticks(y,[label(n)+(' *' if n==chosen else '') for n in names]);acc.invert_yaxis()
    acc.axvline(.7,color='#b34545',linestyle='--',linewidth=1,label='70% = 1,812 correct')
    acc.set_xlim(min(.63,float(score.accuracy.min())-.012),max(.72,float(score.accuracy.max())+.04))
    for yy,row in enumerate(score.itertuples()):acc.text(row.accuracy+.003,yy,f'{row.accuracy:.2%} ({row.correct})',va='center',fontsize=9)
    acc.xaxis.set_major_formatter(PercentFormatter(1));acc.set_xlabel('Accuracy on all 2,588 matches');acc.legend(loc='lower right',fontsize=8)
    loss.scatter(score.mean_log_loss,y,c=colors,s=65)
    span=max(.025,float(score.mean_log_loss.max()-score.mean_log_loss.min()))
    loss.set_xlim(float(score.mean_log_loss.min())-.01,float(score.mean_log_loss.max())+span*.5)
    for yy,val in enumerate(score.mean_log_loss):loss.text(val+.002,yy,f'{val:.5f}',va='center',fontsize=9)
    loss.tick_params(axis='y',left=False,labelleft=False);loss.set_xlabel('Mean log loss (lower is better)')
    for ax in (acc,loss):ax.grid(axis='x',alpha=.2);ax.spines[['top','right']].set_visible(False)
    fig.suptitle('Separate market-information branch · reused exploratory 2023',fontsize=14,x=.01,ha='left')
    fig.text(.01,.012,'* Frozen opening-based recommendation. Closing is a later-information descriptive benchmark.\n'
             'All missing quotes fall back exactly to the frozen Round4 history model; no matches are removed.',fontsize=9)
    fig.tight_layout(rect=(0,.08,1,.94))
    fig.savefig(OUT/'market_comparison.png',dpi=180,bbox_inches='tight',facecolor='white');plt.close(fig)


def narrative(data,notebook=False):
    frozen=data['frozen'];summary=data['summary'];gate=frozen['procedure_gate'];chosen=frozen['recommendation']
    scores=data['scores'].set_index('model');row=scores.loc[chosen];opening=scores.loc['opening_market'];old=scores.loc['history_only']
    best=best_observed(data);best_open=best_observed(data,False);needed=summary['correct_required_for_70']
    remaining=max(0,needed-int(row.correct))
    reach=(f'达到70%所需的{needed}场门槛，超出{int(row.correct)-needed}场' if remaining==0 else f'距离70%仍需净增{remaining}场正确判断')
    status='通过市场信息升级门槛' if gate['procedure']=='adopt_market_information' else '未通过门槛，仍推荐原历史概率'
    low,high=summary['accuracy_comparisons_vs_opening'][chosen]['ci95']
    gain=float(row.accuracy-opening.accuracy)
    if chosen=='opening_market':increment='冻结推荐就是未经拟合的开盘市场基准；相对开盘价的额外算法提升为0。'
    elif low<=0<=high:increment='相对开盘市场的配对区间包含0，不能确认存在额外预测增益。'
    elif high<0:increment='相对开盘市场的配对区间为负；该探索比较不支持优于原始开盘价。'
    else:increment='条件配对区间未跨0，但已复用2023和多轮选择，仍不能宣称独立验证优于市场。'
    table=['| 角色/模型 | 正确数 / 2588 | 准确率 | log loss | Brier |',
           '| --- | ---: | ---: | ---: | ---: |']
    for name in selected_names(data):
        r=scores.loc[name]
        role='冻结推荐' if name==chosen else '固定R4历史' if name=='history_only' else '原始开盘+缺失回退' if name=='opening_market' else '收盘独立描述+缺失回退' if name==CLOSING else '最佳观察（不更改推荐）'
        table.append(f'| {role}：`{name}` | {int(r.correct)} | {r.accuracy:.2%} | {r.mean_log_loss:.6f} | {r.brier:.6f} |')
    coverage=['| 年份 | 完整目标 | 匹配 | opening有效 | opening回退 | closing有效 | closing回退 |',
              '| --- | ---: | ---: | ---: | ---: | ---: | ---: |']
    for r in coverage_table(data).itertuples():
        coverage.append(f'| {r.season} | {r.targets} | {r.matched} | {r.opening_available} | {r.opening_missing_fallback} | {r.closing_available} | {r.closing_missing_fallback} |')
    prefix='' if notebook else '../../../'
    image='results/round5/market/market_comparison.png' if notebook else 'market_comparison.png'
    return f'''# 第五轮附加市场实验：更晚的赛前信息，不是纯历史算法增益

**{status}，冻结推荐为 `{chosen}`。** 在完整2,588场2023比赛上判对 **{int(row.correct)}场（{row.accuracy:.2%}）**，{reach}。

这个结果使用供应方称为开盘/赛前最后报价的市场信息。原任务只用赛事开始周及T−21历史；这里的信息通常更晚、来源也不同。**不能把加入市场后的改善归功于仅靠排名、状态和历史交手的模型达到同样效果。**2023已反复查看，本报告是探索性结果；2024–2025未打开。

## 同一全量分母，缺失精确回退

底模固定为第四轮 `{data['registry']['base_model']}`，不根据本轮或主第五轮2023成绩换底模。训练阶段使用其2021/2022真实向前OOF概率，2023使用冻结预测。参数选择曾使用这些开发折，因而这些OOF不是一个全新独立测试集。

所有市场模型在无法匹配或双边赔率无效时，**逐行精确回退到该底模**。Opening与closing的缺失位置不同，两个全量分数因此包含不同的市场覆盖成分。

{chr(10).join(coverage)}

2021没有closing，故跨年可训练候选仅使用opening。Closing只作为独立描述基准，不参与门槛或最终推荐选择。不能仅保留有价格的样本，再把其分数称作完整2,588场成绩。

## 全量比较与70%目标

{chr(10).join(table)}

全部开盘候选中的最高观察准确率为 `{best_open.model}`（{best_open.accuracy:.2%}，{int(best_open.correct)}场）。含独立closing行的最高观察值为 `{best.model}`（{best.accuracy:.2%}，{int(best.correct)}场）。**这些观察排序不覆盖冻结选择，也不能把closing的更晚报价条件混同于opening。**全量70%需要至少 **{needed}场正确**；达到目标本身不等于超过市场或达到SOTA。

![市场信息条件下的完整比赛比较]({image})

## 市场信息贡献与额外算法贡献分开看

冻结推荐相对固定R4历史模型：准确率变化 **{100*(row.accuracy-old.accuracy):+.2f}个百分点**，log-loss减少 **{old.mean_log_loss-row.mean_log_loss:+.6f}**。这包含市场带来的额外信息。

相对**原始去水opening+相同缺失回退**：准确率变化 **{100*gain:+.2f}个百分点**，log-loss减少 **{opening.mean_log_loss-row.mean_log_loss:+.6f}**；准确率变化的配对日期块95%区间为 **[{100*low:+.2f}, {100*high:+.2f}]个百分点**。{increment}

区间按目标赛事开始日期分块，不校正多轮探索、参数选择及跨日期重复球员依赖。它是对已保存候选的条件描述，不是独立最终检验；Notebook保留全部模型相对历史与opening的区间。

## 冻结选择经过什么检验

共{summary['candidate_count']}个固定候选：历史底模、原始opening、固定概率/logit融合，以及三种对称logistic校准。阈值始终为0.5。2021H2在更早2021数据上拟合后选择 `{gate['early_selected']}`；2022检查其准确率 **{gate['candidate_2022_accuracy']:.2%}** 对历史 **{gate['reference_2022_accuracy']:.2%}**，提升 **{100*gate['accuracy_improvement']:+.2f}个百分点**（要求至少{100*gate['required_accuracy_improvement']:.2f}）；候选log loss **{gate['candidate_2022_loss']:.6f}**，历史 **{gate['reference_2022_loss']:.6f}**，允许最多增加{gate['allowed_loss_increase']:.3f}。

之后用2021H2+2022时间外预测重选固定候选，并只用2021–2022拟合最终校准。冻结后才计算本轮2023成绩。2022本身也曾被前轮使用，所以这个门槛是明确的探索性决策规则，不是恢复了独立性。

## 来源、限制与复跑

数据：[Valuebetennis公开CSV](https://www.valuebetennis.com/en/donnees.htm)，[CC BY4.0](https://creativecommons.org/licenses/by/4.0/)。仅下载2021–2023。开盘与收盘字段没有逐条报价采集时间，只有供应方的赛前定义；无法独立证明每笔报价在规定预测时刻已公开。单一来源的历史导出不等于多家公司市场共识，也不构成投注收益结论。匹配只用身份、日期、赛事、场地及轮次，不用胜者或比分。

[来源与匹配审计]({prefix}docs/research/round5_market.md) · [独立市场协议]({prefix}docs/research/round5_market_protocol.md) · [执行Notebook]({prefix}round5_market.ipynb) · [HTML]({prefix}round5_market.html)

```sh
.venv/bin/python -m scripts.train_market_round5
.venv/bin/python -m scripts.report_market_round5
```

保留主T−21第五轮与本市场分支两个结论。最终测试应按各自冻结协议独立进行，继续保留原始opening、closing和历史模型对照。
'''


def notebook_cells(data):
    import nbformat as nbf
    md,code=nbf.v4.new_markdown_cell,nbf.v4.new_code_cell
    return [md(narrative(data,True)),
        md('## 全部已冻结候选的完整2023比较\n\nClosing不参与选择；额外正确数以全量1812门槛计算。'),
        code("from IPython.display import display, Image\nimport pandas as pd\nfrom scripts.report_market_round5 import load_results, comparison_table, coverage_table, intervals_table, OUT\n"
             "data=load_results()\npd.set_option('display.max_columns',None)\npd.set_option('display.precision',6)\n"
             "display(comparison_table(data))\ndisplay(Image(filename=str(OUT/'market_comparison.png')))"),
        md('## 报价覆盖、匹配状态和缺失回退\n\n此处是数据覆盖，不是删减后的性能。'),
        code("display(coverage_table(data))\ndisplay(data['links'].groupby(['season','status']).size().rename('matches').reset_index())\n"
             "print('Every missing opening/closing forecast was checked against exact frozen history fallback.')"),
        md('## 全部时间外候选与2022门槛\n\nH2选择、2022门槛、再pool重选；不使用2023选择。'),
        code("display(data['cv'].sort_values(['accuracy','mean_log_loss'],ascending=[False,True]))\n"
             "display(pd.DataFrame([data['frozen']['procedure_gate']]))"),
        md('## 相对历史与原始opening的条件不确定性\n\n日期块区间不能消除已反复查看2023或多轮选择带来的不确定性。'),
        code("display(intervals_table(data))\nassert data['summary']['final_test_evaluated'] is False\n"
             "print('Separate market-information regime; 2024–2025 remains unopened.')")]


def main():
    import nbformat as nbf
    from nbclient import NotebookClient
    from nbconvert import HTMLExporter
    from jupyter_client import KernelManager
    data=load_results();plot_comparison(data)
    notebook=nbf.v4.new_notebook(cells=notebook_cells(data),metadata={'kernelspec':{'display_name':'Python3','language':'python','name':'python3'}})
    manager=KernelManager(kernel_name='python3')
    manager.kernel_spec.argv=[sys.executable,'-m','ipykernel_launcher','-f','{connection_file}']
    NotebookClient(notebook,km=manager,timeout=180,resources={'metadata':{'path':str(ROOT)}}).execute()
    nbf.write(notebook,ROOT/'round5_market.ipynb')
    exporter=HTMLExporter(template_name='lab');exporter.embed_images=True
    html,_=exporter.from_notebook_node(notebook)
    (ROOT/'round5_market.html').write_text(html)
    (OUT/'report_summary.md').write_text(narrative(data))
    print('Wrote independent market report, executed notebook, HTML and figure.')


if __name__=='__main__':main()
