"""Render saved round-six evidence without retraining or changing selection."""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import nbformat
from nbclient import NotebookClient
from nbconvert import HTMLExporter
import pandas as pd

from scripts.train_models import ROOT


def matched_controls(cv):
    indexed = cv.set_index('name')
    rows = []
    for group in ('extended', 'trajectory', 'breakpoint', 'combined'):
        for learner in ('linear', 'spline'):
            for c in (.003, .01, .03):
                base, candidate = f'r6_base_{learner}_C{c:g}', f'r6_{group}_{learner}_C{c:g}'
                aa, bb = indexed.loc[base], indexed.loc[candidate]
                row = dict(feature_block=group, learner=learner, C=c, base=base, candidate=candidate,
                    accuracy_gain=bb.accuracy-aa.accuracy, log_loss_reduction=aa.mean_log_loss-bb.mean_log_loss)
                for year in (2020, 2021, 2022):
                    row[f'{year}_accuracy_gain'] = bb[f'{year}_accuracy']-aa[f'{year}_accuracy']
                    row[f'{year}_log_loss_reduction'] = aa[f'{year}_log_loss']-bb[f'{year}_log_loss']
                rows.append(row)
    return pd.DataFrame(rows)


def main():
    out = ROOT / 'results/round6'
    frozen = json.loads((out / 'frozen_selection.json').read_text())
    summary = json.loads((out / 'summary.json').read_text())
    registry = json.loads((out / 'search_registry.json').read_text())
    comparison = pd.read_csv(out / 'validation_comparison.csv')
    cv = pd.read_csv(out / 'temporal_cv.csv')
    controls = matched_controls(cv)
    controls.to_csv(out / 'matched_information_controls.csv', index=False)
    indexed = comparison.set_index('model')
    choice, exploration = frozen['recommendation'], frozen['pooled_selection']['choice']
    selected = indexed.loc[exploration]
    gate = frozen['procedure_gate']
    delta = summary['comparisons']['previous_round5_exploration'][exploration]['accuracy']
    best = comparison.iloc[0]
    minloss = comparison.sort_values('mean_log_loss').iloc[0]
    block_summary = controls.groupby('feature_block').agg(
        positive_accuracy_controls=('accuracy_gain', lambda a: int((a > 0).sum())),
        total_controls=('accuracy_gain', 'size'), mean_accuracy_gain=('accuracy_gain', 'mean'),
        mean_loss_reduction=('log_loss_reduction', 'mean')).reset_index()
    table = '\n'.join(f'| `{r.model}` | {int(r.correct)} | {r.accuracy:.2%} | {r.mean_log_loss:.5f} | {r.brier:.5f} |'
        for r in comparison.itertuples())
    block_table = '\n'.join(f'| {r.feature_block} | {r.positive_accuracy_controls}/{r.total_controls} | {r.mean_accuracy_gain*100:+.3f} | {r.mean_loss_reduction:+.5f} |'
        for r in block_summary.itertuples())
    body = f'''# 第六轮主实验：遗漏历史、排名轨迹与关键分

34项新配置，{summary['features']}项输入；仍用33,833条拟合目标与完整2,588场2023验证目标。所有2023结果属于反复使用样本上的探索，2024–2025最终测试未打开。没有市场价格输入，也没有放宽T−21历史截止。

## 开发期选择及2023结果

- 2022流程门槛：原参考准确率{gate['reference_2022_accuracy']:.4%}，早期选中流程{gate['candidate_2022_accuracy']:.4%}，增幅{gate['accuracy_improvement']*100:+.3f}个百分点，要求至少0.5个百分点且loss护栏通过。结论：`{gate['procedure']}`。
- 正式推荐：`{choice}`。pooled开发折选择的探索比较：`{exploration}`，2023正确**{int(selected.correct)}/2588 = {selected.accuracy:.2%}**，log loss **{selected.mean_log_loss:.5f}**。
- 相对第五轮冻结探索组合：纠正{delta['corrected']}场，同时新错{delta['newly_wrong']}场，净增{delta['net_correct']:+d}场；准确率差的条件日期块区间为[{delta['ci95'][0]*100:+.3f}, {delta['ci95'][1]*100:+.3f}]个百分点。该区间未覆盖全部搜索不确定性。
- 本表最高观察准确率`{best.model}`为{best.accuracy:.2%}；最低观察loss`{minloss.model}`为{minloss.mean_log_loss:.5f}。这些事后排序不改写冻结选择。
- 70%需要正确1,812场，探索比较仍差{max(0,1812-int(selected.correct))}场。

| 模型 | 正确数/2588 | Accuracy | Log loss ↓ | Brier ↓ |
| --- | ---: | ---: | ---: | ---: |
{table}

## 匹配信息对照

每块在相同学习器、C及half-life下与275项旧核心特征比较，计6个匹配对照。下表是pooled2020–2022结果的描述均值，不能当独立样本做显著性检验。完整各年差值在`matched_information_controls.csv`。

| 加入信息 | Accuracy改善的对照数 | 平均Accuracy变化（百分点） | 平均loss降低 |
| --- | ---: | ---: | ---: |
{block_table}

新增历史评级包含相同算法的旧历史控制；允许赛事与排除项详见来源审计。排名轨迹只用2006以来、每场T−21以前的可见记录，同日冲突快照不使用，不能称官方生涯最高排名。BP/保发残差固定收缩，并不被解释成心理能力或因果效应。

## 复跑与限制

先复现前五轮，然后运行`python -m scripts.train_round6`及`python -m scripts.report_round6`。本报告只读取已保存结果，不改变模型。使用`python -m scripts.train_round6 --predict-features path/to/features.csv`重载冻结模型，需要全部{summary['features']}项冻结输入。大特征表和本地joblib按固定来源重建，不提交Git。

跨级别监督迁移是单独的匹配实验，见[迁移报告](transfer/report_summary.md)；跨来源标签审计见[独立记录](source_audit.md)。既有市场结果来自更晚信息条件，不与本轮历史算法增益混为一谈。[实验协议](../../docs/research/round6_protocol.md) · [机器选择记录](frozen_selection.json) · [独立重载审计](artifact_audit.json)。
'''
    (out / 'report_summary.md').write_text(body)
    plot = comparison.sort_values('accuracy').reset_index(drop=True)
    colors = ['#c65c38' if n == exploration else '#316a9b' if n == choice else '#7e95aa' for n in plot.model]
    fig, ax = plt.subplots(1, 2, figsize=(14, max(6, .38*len(plot)+1.6)), sharey=True,
        gridspec_kw={'width_ratios': [1, 1]})
    yy = list(range(len(plot)))
    ax[0].barh(yy, plot.accuracy*100, color=colors, height=.66)
    ax[0].set_xlim(60, 71)
    ax[0].axvline(70, color='#a74444', linestyle='--', linewidth=1)
    ax[0].set_yticks(yy, plot.model)
    ax[0].set_xlabel('Accuracy (%) — full 2,588 matches')
    for i, value in enumerate(plot.accuracy):
        ax[0].text(value*100+.08, i, f'{value:.2%}', va='center', fontsize=8)
    ax[1].scatter(plot.mean_log_loss, yy, color=colors, s=40)
    ax[1].set_xlabel('Log loss (lower is better)')
    for axis in ax:
        axis.grid(axis='x', alpha=.2)
        axis.set_axisbelow(True)
        axis.spines[['top', 'right']].set_visible(False)
    fig.suptitle('Round 6: new historical information\nExploratory 2023 results; T−21 unchanged; final test sealed', fontsize=13)
    fig.tight_layout(rect=(0,0,1,.95))
    fig.savefig(out / 'round6_comparison.png', dpi=160)
    plt.close(fig)
    notebook = nbformat.v4.new_notebook(cells=[
        nbformat.v4.new_markdown_cell(body),
        nbformat.v4.new_code_cell("from pathlib import Path\nimport json\nimport pandas as pd\nfrom IPython.display import display, Image\nROOT = Path.cwd()\nOUT = ROOT / 'results/round6'\ncomparison = pd.read_csv(OUT / 'validation_comparison.csv')\ndisplay(comparison)"),
        nbformat.v4.new_code_cell("display(Image(filename=str(OUT / 'round6_comparison.png')))"),
        nbformat.v4.new_code_cell("display(pd.read_csv(OUT / 'matched_information_controls.csv'))"),
        nbformat.v4.new_code_cell("frozen = json.loads((OUT / 'frozen_selection.json').read_text())\nprint(json.dumps(frozen, indent=2))")],
        metadata={'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'}})
    executed = NotebookClient(notebook, timeout=180, kernel_name='python3', resources={'metadata': {'path': str(ROOT)}}).execute()
    nbformat.write(executed, ROOT / 'round6.ipynb')
    html, _ = HTMLExporter().from_notebook_node(executed)
    (ROOT / 'round6.html').write_text(html)
    print(body[:1800])


if __name__ == '__main__':
    main()
