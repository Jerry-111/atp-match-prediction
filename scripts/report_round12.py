"""Render audited conditional residual-state experiments without refitting."""
from __future__ import annotations

import json

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
from scripts.report_round11 import table

OUT = ROOT / 'results/round12'
DATA = OUT / 'residual_state'
LABELS = {
    'rank_reference': 'Rank baseline',
    'overall_elo_stale': 'Raw overall Elo K32',
    'previous_probability_recommendation': 'R4 historical reference',
    'shared_only': 'Same R5 shared-feature control',
    'shared_static_residual': '+ Static player residuals',
    'shared_dynamic_residual': '+ Dynamic player residuals',
    'previous_round5_exploration': 'R5 frozen accuracy exploration',
    'market_lambda1': 'Train-market supervision (prior)',
    'history_linear_C0.03_alpha1': 'Current opening + history (unselected)',
    'raw_market': 'Current latest quote (prior)',
}


def read(path):
    return json.loads(path.read_text())


def main():
    original = pd.read_csv(ROOT / 'results/round3/development_targets.csv')
    rows = original.loc[original.season.eq(2023)]
    comparison = pd.read_csv(DATA / 'validation_comparison.csv')
    predictions = pd.read_csv(DATA / 'validation_predictions.csv')
    verify_scores(comparison, predictions, rows)
    audit = read(DATA / 'artifact_audit.json')
    if audit['status'] != 'passed':
        raise ValueError('Independent audit must pass before rendering')
    registry, frozen, summary = (read(DATA / n) for n in
        ['search_registry.json', 'frozen_selection.json', 'summary.json'])
    if summary['final_test_evaluated'] or sha256(DATA/'frozen_selection.json') != summary['frozen_selection_sha256']:
        raise ValueError('Frozen decision/embargo mismatch')
    matched = pd.read_csv(DATA / 'matched_controls.csv')
    contributions = pd.read_csv(DATA / 'residual_contributions.csv')
    contributions = contributions.loc[contributions.year.eq(2023)]
    cv = pd.read_csv(DATA / 'temporal_cv.csv')
    prior = pd.read_csv(ROOT / 'results/round11/overview_comparison.csv')
    prior = prior.loc[prior.model.isin(['overall_elo_stale','market_lambda1',
                                      'history_linear_C0.03_alpha1','raw_market'])].copy()
    records = []
    for row in comparison.to_dict('records'):
        row.update(branch='residual_state', information_regime='原T−21历史输入')
        records.append(row)
    for row in prior.to_dict('records'):
        records.append({k: row[k] for k in ['model','matches','correct','accuracy','mean_log_loss','brier',
                                           'branch','information_regime']})
    display = pd.DataFrame(records).set_index('model').loc[list(LABELS)].reset_index()
    if not display.matches.eq(2588).all():
        raise ValueError('Full cohort changed')
    display['additional_correct_for_70'] = 1812-display.correct
    display.to_csv(OUT / 'overview_comparison.csv', index=False)
    fig, axes = plt.subplots(1,2,figsize=(16,7.8),sharey=True,gridspec_kw={'width_ratios':[1.3,1]})
    positions = np.arange(len(display))
    colors = ['#367d73' if n.startswith('shared_') else '#738399' for n in display.model]
    axes[0].barh(positions,100*display.accuracy,color=colors,height=.65)
    axes[1].scatter(display.mean_log_loss,positions,c=colors,s=55)
    for i,row in display.iterrows():
        axes[0].text(100*row.accuracy+.1,i,f'{row.accuracy:.2%} ({int(row.correct)})',va='center',fontsize=9)
        axes[1].text(row.mean_log_loss+.0008,i,f'{row.mean_log_loss:.5f}',va='center',fontsize=9)
    axes[0].set_yticks(positions,[LABELS[n] for n in display.model]);axes[0].invert_yaxis()
    axes[0].set_xlim(60.8,max(71.5,100*display.accuracy.max()+2));axes[0].axvline(70,color='#a94141',linestyle='--')
    axes[0].set_xlabel('Accuracy (%) | higher is better')
    axes[1].set_xlim(display.mean_log_loss.min()-.004,display.mean_log_loss.max()+.011)
    axes[1].set_xlabel('Mean log loss | lower is better')
    for axis in axes:
        axis.grid(axis='x',alpha=.15);axis.set_axisbelow(True);axis.spines[['top','right']].set_visible(False)
    fig.suptitle('Round 12: jointly fitted conditional player residuals',fontsize=16,weight='bold',y=.96)
    fig.text(.5,.915,'Same 2,588 matches; three matched historical structures. Market references use different information.',ha='center',fontsize=9)
    fig.text(.5,.025,'All reused 2023 results are exploratory. Neither the prior papers nor this comparison establishes SOTA.',ha='center',fontsize=9)
    fig.subplots_adjust(left=.32,right=.97,top=.87,bottom=.11,wspace=.14)
    fig.savefig(OUT / 'round12_comparison.png',dpi=170,facecolor='white');plt.close(fig)
    selected = comparison.set_index('model').loc[frozen['recommendation']]
    pooled = comparison.set_index('model').loc[frozen['pooled_selection']['choice']]
    gate = frozen['procedure_gate']
    threshold_reached = bool(comparison.correct.ge(1812).any())
    goal_text = ('有候选达到数字门槛；是否完成目标还须按冻结选择、审计和交付条件核对。' if threshold_reached
                 else '本轮没有候选达到1,812场正确，70%目标继续保持active。')
    body = f'''# 第十二轮：联合学习球员残差状态

本轮固定原2,588场2023比赛和0.5阈值，比较同一强历史特征模型的三种结构：
无玩家残差、静态玩家残差、随季度变化的玩家残差。{goal_text}
正式分支推荐 **{frozen['recommendation']}**：**{int(selected.correct)}/2,588 = {selected.accuracy:.3%}**，
log loss **{selected.mean_log_loss:.6f}**。全部2023结果仍是重复查看后的探索性证据；2024–2025继续封存。

## 完整队列对比

{table(display[['model','information_regime','correct','accuracy','mean_log_loss','additional_correct_for_70']])}

![第十二轮模型比较](round12_comparison.png)

市场上下文保留原信息条件，不能与本轮纯历史模型混称同一输入条件。
此前1,760场正确的opening+history行没有被开发期选择，不能按2023最好分数直接推荐。

## 选择与机制检验

2020–2021选择 **{gate['early_selection']['choice']}**；其2022相对R4净正确变化
**{gate['net_correct']:+d}/2,528**，预先门槛为至少+13且loss增加不超过.005。
因此gate为 **{gate['procedure']}**。独立记录的2020–2022 pooled选择
**{frozen['pooled_selection']['choice']}** 在2023为 **{int(pooled.correct)}/2,588**、
loss **{pooled.mean_log_loss:.6f}**。没有根据2023重选残差方差、阈值或混合权重。

{table(matched)}

static−shared检验稳定玩家残差是否增益；dynamic−static检验时间耦合是否有额外价值。
完整前推年度结果见[temporal_cv.csv](residual_state/temporal_cv.csv)，配对日期块区间见
[summary.json](residual_state/summary.json)。这些区间以拟合后的预测为条件，没有校正
多轮搜索或全部跨日期球员依赖。不能把训练拟合更好直接解释为预测更好。

静态模型相对原共享控制纠正59场、弄错54场，净增5；动态模型纠正73场、弄错87场。
两种状态扩展相对同输入控制在2020、2021、2022、2023每年的log loss都更差。
这说明本次固定先验和季度近似没有提供稳定的概率收益，并不否定所有二元BT或动态模型。

{table(contributions[['model','mean_absolute_state_logit','rms_state_logit','classifications_changed_by_state']])}

状态贡献并非全被压成零。表中分类变化是相对各扩展模型**自身联合拟合beta的共享部分**，
不是相对独立R5控制；beta也会因联合拟合而改变，因此不能把全部增减归因于状态项的单独因果作用。

## 方法与时间边界

精确复用R5 `all_inputs_spline_C0.01`：199个输入、256维设计、C=.01、三年样本权重。
每个outer年只在原21天purge后的历史上拟合变换和共享beta。静态或季度状态与beta
共同优化最终比赛二元似然；之后每个目标日期只在T−21以内的历史前缀重新求状态。
beta、缩放、样条和先验均冻结。允许前缀内部的历史状态平滑，不能称严格单向filter。

初始残差SD=.35、年度创新SD=.15均是预先固定的实用log-odds尺度，非论文验证过的ATP参数。
未出现的季度不建自由节点；预测取最后可用状态。没有合规历史的玩家残差为0；
全球训练未见但后来已有可用比赛的玩家能够获得状态。历史从有完整冻结特征的2009年开始。
未引入市场、新完赛日期或逐分charting数据。

实际有 **{summary['global_stages']}次global优化、{summary['state_only_solves']}次状态优化**，
不是只训练三个模型一次。每个年度的模型和各截止日状态都已保存。旧强控制的全部
6,130条OOF与2,588条2023概率须误差≤1e−10且分类完全一致才会继续。

## 复跑和独立核验

在已复建前五轮输入和当前依赖的环境中运行到新目录：

```bash
.venv/bin/python -m scripts.train_residual_state_round12 --output /tmp/atp-round12-reproduction --approved-protocol-sha256 {registry['protocol_sha256']}
.venv/bin/python -m scripts.audit_residual_state_round12
.venv/bin/python -m scripts.report_round12
```

第一条创建独立复跑产物；后两条默认核验和呈现仓库原冻结运行，不会自动切到复跑目录。
原路径已有产物时runner拒绝覆盖。模型权重在residual_state/model_2020.joblib至
model_2023.joblib以及frozen_models.joblib，本地保留且由运行清单绑定哈希，可按命令重建。
已保存截止日的推断用`--predict-features`与`--query-metadata`，query必须不含a_won；
新截止日需要按同一边界提供合规历史并调用引擎的predict_asof，不能冒用旧状态。

[artifact_audit.json](residual_state/artifact_audit.json)记录独立检查范围与局限；
[verification.json](verification.json)记录常规测试。重载状态复算不等于重新运行全部优化器，
也不等于独立重建326列历史特征或重新裁定每个原始结果。

低级别市场监督方向先完成了[可行性审查](../../docs/research/round12_market_transfer_feasibility.md)，
严格筛选留下29,050行，但有比分截断、短比赛偏差和缺少新排序信号的问题，因此没有拟合。
这不是把辅助数据数量当成准确率提升。核方法、排名轨迹、身高和左右手重复路线也已排除。

[执行协议](../../docs/research/round12_residual_state_protocol.md) ·
[一手文献与负证据](../../docs/research/round12_residual_state_proposal.md) ·
[独立结果解释](../../docs/research/round12_residual_state_findings.md) ·
[探索决策记录](../../docs/research/round12_decisions.md)
'''
    (OUT / 'overview.md').write_text(body)
    notebook = nbformat.v4.new_notebook(cells=[
        nbformat.v4.new_markdown_cell(rebase_links(body,OUT,ROOT)),
        nbformat.v4.new_code_cell("from pathlib import Path\nimport pandas as pd\nfrom IPython.display import display, Image\nroot=Path.cwd()\ndisplay(pd.read_csv(root/'results/round12/overview_comparison.csv'))"),
        nbformat.v4.new_code_cell("display(pd.read_csv(root/'results/round12/residual_state/matched_controls.csv'))"),
        nbformat.v4.new_code_cell("display(Image(filename=str(root/'results/round12/round12_comparison.png')))"),
    ],metadata={'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'}})
    NotebookClient(notebook,timeout=120,kernel_name='python3',resources={'metadata':{'path':str(ROOT)}}).execute()
    nbformat.write(notebook,ROOT / 'round12.ipynb')
    html,_ = HTMLExporter().from_notebook_node(notebook)
    (ROOT / 'round12.html').write_text(html)
    print(f'Rendered {len(display)} comparison rows and three executed report cells.')


if __name__ == '__main__':
    main()
