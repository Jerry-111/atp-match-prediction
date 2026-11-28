"""Render the frozen bounded-freshness experiment; no fitting or selection."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import nbformat
from nbclient import NotebookClient
from nbconvert import HTMLExporter
import numpy as np
import pandas as pd

from scripts.train_models import ROOT, sha256
from scripts.report_round7 import verify_scores, rebase_links

OUT = ROOT/'results/round11'
DATA = OUT/'freshness'
LABELS = {
    'overall_elo_stale': 'Overall Elo / original history',
    'overall_elo_fresh': 'Overall Elo / bounded fresher history',
    'rank_rating_logistic_stale': 'Rank + ratings / original history',
    'rank_rating_logistic_fresh': 'Rank + ratings / bounded fresher history',
    'previous_probability_recommendation': 'R4 historical reference',
    'previous_round5_exploration': 'R5 accuracy exploration',
}


def read(path):
    return json.loads(path.read_text())


def table(frame):
    def value(v):
        return f'{v:.6f}' if isinstance(v,(float,np.floating)) else str(v)
    return '\n'.join(['| '+' | '.join(frame.columns)+' |',
                      '| '+' | '.join(['---']*len(frame.columns))+' |']+
                     ['| '+' | '.join(value(v) for v in row)+' |'
                      for row in frame.itertuples(index=False,name=None)])


def main():
    rows=pd.read_csv(ROOT/'results/round3/development_targets.csv')
    rows=rows[rows.season.eq(2023)]
    comparison=pd.read_csv(DATA/'validation_comparison.csv')
    predictions=pd.read_csv(DATA/'validation_predictions.csv')
    verify_scores(comparison,predictions,rows)
    audit=read(DATA/'artifact_audit.json')
    assert audit['status']=='passed'
    frozen,summary=read(DATA/'frozen_selection.json'),read(DATA/'summary.json')
    assert not frozen['final_test_evaluated'] and summary['total_fits']==3
    ledger=read(OUT/'completion/qualified_pairs.json')
    assert sha256(OUT/'completion/qualified_pairs.json')==frozen['qualified_manifest_sha256']
    matched=pd.read_csv(DATA/'matched_controls.csv')
    prior=pd.read_csv(ROOT/'results/round10/overview_comparison.csv')
    prior=prior[prior.branch.eq('prior') & prior.model.isin(
        ['rank_reference','market_lambda1','history_linear_C0.03_alpha1','raw_market'])].copy()
    records=[]
    for r in comparison.to_dict('records'):
        r.update(plot_label=LABELS[r['model']], branch='freshness',
                 information_regime='Tier B有限近期结果，含公开时效假设' if r['model'].endswith('_fresh')
                 else '原T−21历史输入')
        records.append(r)
    for r in prior.to_dict('records'):
        records.append({k:r[k] for k in ['model','matches','correct','accuracy','mean_log_loss','brier',
                                         'plot_label','branch','information_regime']})
    display=pd.DataFrame(records)
    assert display.matches.eq(2588).all()
    display['additional_correct_for_70']=1812-display.correct
    display.to_csv(OUT/'overview_comparison.csv',index=False)
    # Separate information conditions remain visible in the labels and caption.
    order=list(LABELS)+['rank_reference','market_lambda1','history_linear_C0.03_alpha1','raw_market']
    view=display.set_index('model').loc[order].reset_index()
    fig,axes=plt.subplots(1,2,figsize=(16,8),sharey=True,gridspec_kw={'width_ratios':[1.3,1]})
    yy=np.arange(len(view));colors=['#398178' if n.endswith('_fresh') else '#6c7c91' for n in view.model]
    axes[0].barh(yy,100*view.accuracy,color=colors,height=.65)
    axes[1].scatter(view.mean_log_loss,yy,c=colors,s=55)
    for i,r in view.iterrows():
        axes[0].text(100*r.accuracy+.1,i,f'{r.accuracy:.2%} ({r.correct})',va='center',fontsize=9)
        axes[1].text(r.mean_log_loss+.0008,i,f'{r.mean_log_loss:.5f}',va='center',fontsize=9)
    axes[0].set_yticks(yy,view.plot_label);axes[0].invert_yaxis()
    axes[0].set_xlim(60.8,71.5);axes[0].axvline(70,color='#a94141',linestyle='--',linewidth=1)
    axes[0].set_xlabel('Accuracy (%) | higher is better')
    axes[1].set_xlim(view.mean_log_loss.min()-.004,view.mean_log_loss.max()+.011)
    axes[1].set_xlabel('Mean log loss | lower is better')
    for ax in axes:
        ax.grid(axis='x',alpha=.15);ax.set_axisbelow(True);ax.spines[['top','right']].set_visible(False)
    fig.suptitle('Round 11: same models, bounded fresher history',fontsize=16,weight='bold',y=.96)
    fig.text(.5,.915,'Same 2,588 targets. Historical market supervision and current-price references are separate conditions.',ha='center',fontsize=9)
    fig.text(.5,.025,'Tier B uses explicit publication/schedule assumptions, not immutable as-of archives. Reused validation is exploratory.',ha='center',fontsize=9)
    fig.subplots_adjust(left=.32,right=.97,top=.87,bottom=.11,wspace=.14)
    fig.savefig(OUT/'round11_comparison.png',dpi=170,facecolor='white');plt.close(fig)
    freeze=read(DATA/'search_registry.json')
    command=(f".venv/bin/python -m scripts.train_freshness_round11 --output /tmp/atp-round11-reproduction "
             f"--approved-protocol-sha256 {freeze['protocol_sha256']} "
             f"--approved-manifest-sha256 {freeze['qualified_manifest_sha256']}")
    cover=pd.DataFrame(ledger['summary'])[['season','total_target_matches','qualified_pairs',
        'qualified_source_editions','qualified_target_editions','target_edition_rows','direct_participant_target_rows']]
    m=matched[matched.season.eq(2023)]
    selected=frozen['early_selection']['choice'] or '无候选'
    coverage=pd.read_csv(DATA/'coverage_by_match.csv').set_index('match_id')
    eligible=coverage.loc[predictions.match_id,'rating_input_changed'].to_numpy(dtype=bool)
    ceilings=[]
    for name in ('overall_elo','rank_rating_logistic'):
        correct=(predictions[name+'_stale'].to_numpy()>=.5)==predictions.a_won.to_numpy()
        bound=int(correct[~eligible].sum()+eligible.sum())
        ceilings.append(dict(algorithm=name,modifiable_rows=int(eligible.sum()),
                             optimistic_correct_bound=bound,optimistic_accuracy_bound=bound/2588))
    pd.DataFrame(ceilings).to_csv(OUT/'fixed_support_ceiling.csv',index=False)
    body=f'''# 第十一轮：有限来源下的信息时效对照

固定完整2,588场2023比赛、0.5阈值，比较同一模型的旧历史与补充近期结果后的输入。
只运行两组成对算法、三次共享逻辑回归拟合；没有重新选参数或混合权重。
2021冻结选择为 **{selected}**，2022 gate为 **{frozen['procedure_gate']['procedure']}**。
此局部机制选择不能直接替换更强的既有模型。正式历史推荐保留规则及完整选择见
[frozen_selection.json](freshness/frozen_selection.json)。70%目标仍未达成，最终年份继续封存。

## 同批比赛比较

{table(view[['model','information_regime','correct','accuracy','mean_log_loss','additional_correct_for_70']])}

![第十一轮比较](round11_comparison.png)

最高旧市场观察1760/2588=68.01%仍是**未被开发期选择**的一行；与纯历史模型的信息条件不同，
不能称为本轮提升或SOTA。R4/R5上下文行的较早年份曾参与旧选择，不进入本轮机制选择。

## 配对变化与有限覆盖

{table(m[['algorithm','stale_correct','fresh_correct','accuracy_improvement','log_loss_change',
          'probability_changed_rows','classification_changed_rows','exact_fallback_rows']])}

{table(cover)}

30个按时间固定的来源赛事中，18个有当前官方完赛上界文本（含1个较晚回顾），12个未解决；
其中仅9个独立source edition实际提供增量，组成15对source-target关系。
2023有234行所在赛事获准增加历史，直接涉及118行球员；覆盖数字不是准确率分母。
概率实际变化数及分类翻转数在上表。所有没有增量的行精确沿用各自stale输入与概率。
只有有限早季来源经过核查，不能据此否定全赛季更新历史的价值。

即使把本轮输入发生变化的每一行都假设预测正确，其余固定行保持原预测，乐观上界仍为：

{table(pd.DataFrame(ceilings))}

这是使用已查看2023标签计算的事后诊断上界，不是新模型或来源选择规则。
它说明仅在这个固定输入支持范围里调模型的局限，不能替代更广泛的数据证据。

资格使用明确的**Tier B公开结果可用/稳定、赛程一致性假设**：官方已完赛报道日期的全球保守上界
必须早于独立目标开始日下界。当前正文不是历史不可变网页快照，来源完赛报道也不逐条认证完整签表。
拒绝、未核和时间重叠均保留fallback；没有依照2023错例选择赛事或缩小全体比赛。

按赛事日期块配对的accuracy/loss区间及2021/2022完整结果见
[summary.json](freshness/summary.json)、[matched_controls.csv](freshness/matched_controls.csv)。
这些区间没有校正已进行的广泛搜索；2022 gate及2023均已重复使用，不能当独立最终确认。

## 复跑与审计

在现有依赖和前五轮已复建的环境中，拟合到新的输出目录，避免覆盖冻结运行：

```bash
{command}
.venv/bin/python -m scripts.audit_freshness_round11
.venv/bin/python -m scripts.report_round11
```

第一条在独立目录复跑；后两条默认审计/呈现仓库原冻结产物，不自动切换到复跑目录。
无需重新抓取网页。资格ledger与所有引用文件必须匹配冻结哈希；发现证据变化应建新实验，不能重写旧结果。
模型保存在freshness目录的model_2021.joblib、model_2022.joblib和frozen_models.joblib，
权重/训练矩阵为本地可复建文件。完整审计范围及限制见
[artifact_audit.json](freshness/artifact_audit.json)，常规测试见[verification.json](verification.json)。

独立实现重新计算全部763项来源资格，以及36,421行的两项新旧rating特征，误差均为0。
三套保存模型的7,504行×4列概率、手工线性预测、两列旧参考概率与指标均精确复现；
交换球员/置换批次最大误差2.22e−16且不改分类。六组成对区间独立复算最大差4.34e−19。
没有重跑优化轨迹，没有重新审定原始每场结果，也没有把所有326项旧特征说成独立重建。

常规环境 **343 tests /109 subtests通过，8个可选Torch测试跳过**。独立代码审查发现并列准确率
用浮点相减可能影响tie-break，但协调者启动拟合早于暂停消息到达，修复在本次运行后落盘。
已保留与执行哈希逐字匹配的[原runner](freshness/original_train_freshness_round11.py)，
仅把selector改为整数净正确数；本次2021为−3和0，无候选，前后决定一致。
没有重拟合或改写原概率、模型和冻结选择。
[修正与源码差异记录](freshness/selection_correction.json)明确区分实际旧代码运行和今后复跑修正版。

[精确协议](../../docs/research/round11_freshness_protocol.md) ·
[来源资格](completion/qualified_pairs.json) ·
[独立证据审查](../../docs/research/round11_availability_review.md)
'''
    (OUT/'overview.md').write_text(body)
    notebook=nbformat.v4.new_notebook(cells=[
        nbformat.v4.new_markdown_cell(rebase_links(body,OUT,ROOT)),
        nbformat.v4.new_code_cell("from pathlib import Path\nimport pandas as pd\nfrom IPython.display import display, Image\nroot = Path.cwd()\ndisplay(pd.read_csv(root/'results/round11/overview_comparison.csv'))"),
        nbformat.v4.new_code_cell("display(pd.read_csv(root/'results/round11/freshness/matched_controls.csv'))"),
        nbformat.v4.new_code_cell("display(Image(filename=str(root/'results/round11/round11_comparison.png')))"),
    ],metadata={'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'}})
    NotebookClient(notebook,timeout=120,kernel_name='python3',resources={'metadata':{'path':str(ROOT)}}).execute()
    nbformat.write(notebook,ROOT/'round11.ipynb')
    html,_=HTMLExporter().from_notebook_node(notebook)
    (ROOT/'round11.html').write_text(html)
    print(f'Rendered {len(view)} rows and three executed report cells.')


if __name__=='__main__':
    main()
