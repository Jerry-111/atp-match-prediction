"""Render all four frozen Round 7 branches without fitting or selecting a model."""
from __future__ import annotations

import json
import os
from pathlib import Path
import re

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import nbformat
from nbclient import NotebookClient
from nbconvert import HTMLExporter
import numpy as np
import pandas as pd

from scripts.train_models import ROOT
from scripts.metrics import score_probabilities

BRANCHES = ('foundation', 'margin', 'distillation', 'event')
OUT = ROOT / 'results/round7'
REGIMES = {
    'baseline': 'T−21历史输入；原基准',
    'foundation': 'T−21历史输入；合成预训练',
    'margin': 'T−21历史输入；历史得分监督',
    'distillation': 'T−21历史输入；历史市场监督/匹配控制',
    'event': 'T−21历史＋本届本人更早轮次',
}
OLD_NAMES = {'round4_reference', 'round5_pooled_comparator', 'round5_probability_comparator',
             'previous_probability_recommendation', 'previous_round5_exploration',
             'previous_round5_best_loss', 'rank_reference'}


def read_json(path):
    return json.loads(Path(path).read_text())


def load_branch(branch, identity):
    directory = OUT / branch
    data = {key: read_json(directory / f'{key}.json') for key in ('summary', 'frozen_selection')}
    data['table'] = pd.read_csv(directory / 'validation_comparison.csv')
    data['predictions'] = pd.read_csv(directory / 'validation_predictions.csv')
    data['oof'] = pd.read_csv(directory / 'oof_predictions.csv')
    data['cv'] = pd.read_csv(directory / 'temporal_cv.csv')
    verify_scores(data['table'], data['predictions'], identity)
    if data['summary']['final_test_evaluated'] or data['frozen_selection']['final_test_evaluated']:
        raise ValueError('This development report requires a sealed final test')
    return data


def verify_scores(table, predictions, identity):
    """Check identities and recompute all saved metrics, including baseline rows."""
    if len(predictions) != 2588 or predictions.match_id.duplicated().any():
        raise ValueError('Expected the complete unique 2,588-match cohort')
    actual = predictions.set_index('match_id').a_won.sort_index()
    expected = identity.set_index('match_id').a_won.sort_index()
    if not actual.equals(expected):
        raise ValueError('Target identity or outcome changed across reports')
    for row in table.itertuples():
        p = predictions[row.model].to_numpy()
        score = score_probabilities(zip(predictions.match_id, predictions.a_won), zip(predictions.match_id, p))
        for key in ('matches', 'correct', 'accuracy', 'mean_log_loss'):
            if not np.isclose(getattr(row, key), score[key], atol=1e-12, rtol=0):
                raise ValueError(f'Metric mismatch for {row.model}: {key}')
        brier = np.mean((predictions.a_won.to_numpy() - p) ** 2)
        if not np.isclose(row.brier, brier, atol=1e-12, rtol=0):
            raise ValueError(f'Brier mismatch for {row.model}')


def event_matched_controls(data):
    """Matched information effects; every difference keeps learner and fallback fixed."""
    rows = []
    for old, new in (('base', 'context'), ('context', 'load'),
                     ('context', 'performance'), ('context', 'combined')):
        for learner in ('linear', 'spline'):
            baseline, candidate = f'event_{old}_{learner}', f'event_{new}_{learner}'
            for period in ('2020', '2021', '2022', 'pooled_2020_2022', '2023_exploratory'):
                source = data['predictions'] if period == '2023_exploratory' else data['oof']
                if period.isdigit():
                    source = source[source.season.eq(int(period))]
                y = source.a_won.to_numpy()
                a, b = source[baseline].to_numpy(), source[candidate].to_numpy()
                def loss(p):
                    p = np.clip(p, 1e-15, 1-1e-15)
                    return float(-np.mean(y * np.log(p) + (1-y) * np.log1p(-p)))
                acc_a, acc_b = np.mean((a >= .5) == y), np.mean((b >= .5) == y)
                rows.append(dict(comparison=f'{old}_to_{new}', learner=learner,
                    baseline=baseline, candidate=candidate, period=period, matches=len(y),
                    baseline_accuracy=acc_a, candidate_accuracy=acc_b, accuracy_gain=acc_b-acc_a,
                    baseline_log_loss=loss(a), candidate_log_loss=loss(b),
                    log_loss_reduction=loss(a)-loss(b)))
    return pd.DataFrame(rows)


def markdown_table(table, label='role'):
    lines = ['| 比较及角色 | 正确数 / 2,588 | Accuracy | Log loss ↓ | 距 70% 额外正确数 |',
             '| --- | ---: | ---: | ---: | ---: |']
    for row in table.itertuples():
        lines.append(f'| {getattr(row, label)} | {int(row.correct):,} | {row.accuracy:.2%} | '
                     f'{row.mean_log_loss:.6f} | {1812-int(row.correct)} |')
    return '\n'.join(lines)


def write_event_report(data):
    controls = event_matched_controls(data)
    controls.to_csv(OUT / 'event/matched_information_controls.csv', index=False)
    frozen, summary = data['frozen_selection'], data['summary']
    gate = frozen['procedure_gate']
    choice = frozen['pooled_selection']['choice']
    selected = data['table'].set_index('model').loc[choice]
    new = data['table'][data['table'].model.str.startswith('event_')]
    best = new.sort_values(['accuracy', 'mean_log_loss'], ascending=[False, True]).iloc[0]
    matched_lines = ['| 相同学习器的加入信息对照 | pooled CV 改善学习器数 | 开发折平均准确率变化 / pp | 开发折平均 loss 降低 | 2023 改善学习器数 | 2023 平均准确率变化 / pp |',
                     '| --- | ---: | ---: | ---: | ---: | ---: |']
    for comparison, group in controls.groupby('comparison', sort=False):
        cv = group[group.period.eq('pooled_2020_2022')]
        val = group[group.period.eq('2023_exploratory')]
        matched_lines.append(f'| `{comparison}` | {int(cv.accuracy_gain.gt(0).sum())}/2 | '
            f'{100*cv.accuracy_gain.mean():+.3f} | {cv.log_loss_reduction.mean():+.6f} | '
            f'{int(val.accuracy_gain.gt(0).sum())}/2 | {100*val.accuracy_gain.mean():+.3f} |')
    table = data['table'].copy()
    table['role'] = table.model.map(lambda n: f'`{n}`' + ('（分支保留）' if n == frozen['recommendation']
        else '（pooled 冻结探索）' if n == choice else '（对照/观察）'))
    text = f'''# 第七轮赛事进程分支：门槛未通过

这一分支允许加入两名球员在**本届赛事已完成、更早的本人淘汰赛轮次**，属于比原 T−21 更晚的信息条件。
27 项新输入与原历史特征组合，共 12 个拟合配置；线性和样条分别有 base/context/load/performance/combined 五组，另有两个 CatBoost 深度。
所有模型覆盖相同 2,588 场；其中 **{summary['validation_with_prior']:,}** 场有合格本人早轮历史，**{summary['validation_fallback']:,}** 场严格回退到原 R4 概率。
同一回退也应用于不含新增信息的 base 控制，因此不把回退差异误当信息效果。

2020–2021 选择的流程在 2022 比 R4 多判对 **{round(2528*gate['accuracy_improvement'])} / 2,528** 场，
准确率变化 **{100*gate['accuracy_improvement']:+.3f} 个百分点**，未达 +0.5 个百分点门槛；loss 护栏通过。
分支继续保留 `{frozen['recommendation']}`。pooled 2020–2022 冻结探索组合 `{choice}` 在 2023 为
**{int(selected.correct)} / 2,588 = {selected.accuracy:.2%}**、loss **{selected.mean_log_loss:.6f}**。
新单模型的最高观察准确率是 `{best.model}`（与 load_spline 同正确数）：**{int(best.correct)} / 2,588**、loss **{best.mean_log_loss:.6f}**。
这个事后排序不改变预先冻结选择；也没有超过旧 R5 探索组合的 1,715 场正确。

## 全部同队列结果

{markdown_table(table)}

## 匹配信息对照

context 包含已完成轮次/轮次情境；load 在它之上加入局数、盘数、时长等负荷；performance 加入本人更早轮次的表现；combined 同时加入两者。
下表每格是两种固定学习器的描述，不能把两个高度相关对照当独立试验。开发期和 2023 观察分开。

{chr(10).join(matched_lines)}

[完整逐年与 pooled 对照](matched_information_controls.csv)保留 40 行；positive accuracy_gain / log_loss_reduction 表示新信息更好。
CatBoost 同时改变学习器和输入，不能单独解释为赛事进程信息的增益。

## 复跑、来源与限制

在前序轮次就绪后运行 `.venv/bin/python -m scripts.train_event_round7`，再运行
`.venv/bin/python -m scripts.report_round7` 生成报告；报告器只读已冻结概率，不训练或改选模型。
原始历史的既有重复在拟合前按稳定键和原行内容核验处理，保留目标和标签；本人早轮次来源可逐项追溯。
不能从赛事开始日期推断未知实际比赛时间；轮次顺序仅支持严格更早的本人淘汰赛记录，排除同轮、未来轮及循环赛。
所有 2023 结果属于复用样本上的探索，2022 门槛也已复用，不是独立确认。2024–2025 继续封存。

[冻结选择](frozen_selection.json) · [独立审计](artifact_audit.json) ·
[协议](../../../docs/research/round7_event.md) · [来源审计](../../../docs/research/round7_event_source_audit.md) ·
[整轮总览](../overview.md)
'''
    (OUT / 'event/report_summary.md').write_text(text)
    return controls


def build_comparison(data, identity):
    original = ROOT / 'results/modeling'
    elo = read_json(original / 'frozen_selection.json')['selected_by_family']['raw_elo']
    elo_table = pd.read_csv(original / 'validation_comparison.csv').query('model == @elo')
    verify_scores(elo_table, pd.read_csv(original / 'validation_predictions.csv'), identity)
    records = []
    def add(branch, row, role, display=False, plot_label=''):
        records.append(dict(branch=branch, information_regime=REGIMES[branch], model=row.model,
            role=role, display=display, plot_label=plot_label,
            matches=int(row.matches), correct=int(row.correct), accuracy=float(row.accuracy),
            mean_log_loss=float(row.mean_log_loss), brier=float(row.brier),
            additional_correct_for_70=1812-int(row.correct)))
    event_table = data['event']['table'].set_index('model', drop=False)
    for name, role, label in (
        ('rank_reference', '排名 logistic 原基准', 'Ranking baseline'),
        ('previous_probability_recommendation', 'R4：保留的纯历史概率推荐', 'R4 historical reference'),
        ('previous_round5_exploration', 'R5：冻结准确率探索组合', 'R5 accuracy exploration'),
        ('previous_round5_best_loss', 'R5：此前最低观察 loss 对照', 'R5 lowest observed loss'),
    ):
        add('baseline', event_table.loc[name], role, True, label)
    add('baseline', elo_table.iloc[0], '整体 Elo 原基准（开发期选 K=32）', True, 'Overall Elo baseline')
    label_map = {'foundation': ('TabICLv2：分支冻结选择', 'TabICLv2 branch choice'),
                 'margin': ('得分监督：pooled 冻结探索，门槛失败', 'Performance-target exploration'),
                 'distillation': ('市场软标签：分支冻结选择', 'Market-supervised student'),
                 'event': ('本人早轮次：pooled 冻结探索，门槛失败', 'Earlier-round exploration')}
    for branch in BRANCHES:
        frozen, table = data[branch]['frozen_selection'], data[branch]['table']
        pooled = frozen['pooled_selection']['choice']
        selected = frozen['recommendation']
        for row in table.itertuples():
            role = '既有对照' if row.model in OLD_NAMES else '预设但未选中的控制；2023 观察'
            display, label = False, ''
            if row.model == selected:
                role = '分支保留原 R4 推荐' if row.model in OLD_NAMES else '分支冻结选择'
            if row.model == pooled:
                role, label = label_map[branch]
                display = True
            if branch == 'distillation' and row.model == 'uniform_lambda1':
                role, label, display = '均匀平滑：未选中的最高准确率观察', 'Uniform smoothing (unselected)', True
            add(branch, row, role, display, label)
    return pd.DataFrame(records)


def plot_comparison(table):
    view = table[table.display].copy()
    # Deliberate information-group order, not a retrospective winner ordering.
    order = ['Ranking baseline', 'Overall Elo baseline', 'R4 historical reference',
             'R5 accuracy exploration', 'R5 lowest observed loss', 'TabICLv2 branch choice',
             'Performance-target exploration', 'Market-supervised student',
             'Uniform smoothing (unselected)', 'Earlier-round exploration']
    view = view.set_index('plot_label').loc[order].reset_index()
    color = {'baseline': '#5d7189', 'foundation': '#286a91', 'margin': '#286a91',
             'distillation': '#b77924', 'event': '#38846b'}
    fig, axes = plt.subplots(1, 2, figsize=(15, 7.5), sharey=True,
                             gridspec_kw={'width_ratios': [1.2, 1]})
    positions = np.arange(len(view))
    colors = [color[b] for b in view.branch]
    bars = axes[0].barh(positions, 100*view.accuracy, color=colors, height=.62)
    bars[8].set_hatch('///')
    axes[0].set_yticks(positions, view.plot_label)
    axes[0].invert_yaxis()
    axes[0].set_xlim(61.5, 71.6)
    axes[0].set_xlabel('Accuracy (%)  |  higher is better')
    axes[0].axvline(70, color='#a84444', linestyle='--', linewidth=1.1)
    axes[0].text(70, -.75, '70% target', ha='center', color='#a84444', fontsize=9)
    for i, row in view.iterrows():
        axes[0].text(100*row.accuracy+.12, i, f'{100*row.accuracy:.2f}%  ({row.correct})', va='center', fontsize=9)
    axes[1].scatter(view.mean_log_loss, positions, c=colors, s=65)
    axes[1].set_xlim(.600, .647)
    axes[1].set_xlabel('Mean log loss  |  lower is better')
    for i, row in view.iterrows():
        axes[1].text(row.mean_log_loss+.0008, i, f'{row.mean_log_loss:.5f}', va='center', fontsize=9)
    for ax in axes:
        ax.grid(axis='x', alpha=.16)
        ax.set_axisbelow(True)
        ax.spines[['top', 'right']].set_visible(False)
    fig.suptitle('Round 7: all 2,588 matches, unchanged 2023 cohort', x=.5, y=.97, fontsize=16, weight='bold')
    fig.text(.5, .915, 'Frozen branch choices and retained references; hatched row is an unselected observation.', ha='center', fontsize=10)
    fig.legend(handles=[Patch(color='#286a91', label='Historical inputs / supervision'),
                        Patch(color='#b77924', label='Historical-market supervision / matched control'),
                        Patch(color='#38846b', label='Later own-round information')],
               loc='lower center', ncol=3, frameon=False, fontsize=9, bbox_to_anchor=(.5, .015))
    fig.subplots_adjust(left=.30, right=.98, top=.84, bottom=.13, wspace=.12)
    fig.savefig(OUT / 'round7_comparison.png', dpi=170, facecolor='white')
    plt.close(fig)


def write_overview(data, table):
    selected = table[table.display]
    distill = data['distillation']
    matched = distill['summary']['comparisons']['hard_labels']['market_lambda1']['log_loss']
    ci = matched['ci95']
    rows = []
    for branch in BRANCHES:
        frozen = data[branch]['frozen_selection']
        gate = frozen['procedure_gate']
        label = {'foundation': 'TabICLv2', 'margin': '得分监督', 'distillation': '市场软标签', 'event': '本人早轮次'}[branch]
        early = '2021 下半年' if branch == 'distillation' else '2020–2021'
        loss_change = gate['candidate_2022_loss'] - gate['reference_2022_loss']
        rows.append(f"| {label} | {early} | {round(2528*gate['accuracy_improvement']):+d} / 2,528 | "
            f"{100*gate['accuracy_improvement']:+.3f} | {loss_change:+.6f} | `{gate['procedure']}` |")
    body = f'''# 第七轮总览：概率误差改善，70% 尚未达到

四条路线完成 **35 个拟合配置**：TabICLv2/匹配控制 6、得分监督 10、市场软标签 7、本人早轮次 12。
另有 foundation 6 个、margin 10 个固定混合对照，以及 event 预设的有限非负混合网格；混合数不计作新增拟合配置。
所有结果覆盖同一 **2,588 场 2023 比赛**，决策阈值始终为 0.5。70% 至少需要 **1,812 场正确**。
2024–2025 最终测试仍封存；由于 2023 已反复查看，本轮全部结果都是探索，不是 SOTA 或最终测试证明。

最有价值的新候选是使用历史市场软标签训练的 `market_lambda1`：**1,712 / 2,588 = 66.15%，log loss 0.604599**。
它在预测时只需要原 T−21 历史特征；训练额外用到 2021–2022 开盘概率，因此不能把收益归因于纯历史比赛建模。
原 R4 纯历史概率推荐继续作为独立信息条件保留。各分支的门槛和冻结选择照实保留，不能用 2023 重新替换。

## 原 T−21 历史条件与既有基准

{markdown_table(selected[selected.branch.isin(['baseline', 'foundation', 'margin'])])}

原整体 Elo 为开发期选出的 `raw_elo_k32_diff`，来自最初同队列正式基准，使用相同 T−21 截止；其概率未经新一轮事后校准。
排名一行是原排名 logistic 概率基准，分类与原排名规则一致。
TabICLv2 的分支冻结选择是 0.75 基础模型 / 0.25 R4 概率混合；虽通过 2022 门槛，2023 却比 R4 少对 10 场、loss 更高。
保留这一失败，不把较好的另一混合追认为当时选择。基础模型只在合成表格上预训练；论文中的一般表格优势并不保证网球时间外推收益。
得分监督的 pooled 探索选择为 0.5 点数监督 / 0.5 R4；其门槛未通过。该分支未选中的 points_C0.1 为 1,714 场正确，仍未超过旧 R5 的 1,715。

## 推断仍为 T−21，但训练含历史市场监督

{markdown_table(selected[selected.branch.eq('distillation')])}

全部 33,833 条训练目标保留，其中 **4,854** 条可用 2021–2022 历史开盘概率。
没有加载 2023 市场报价；部署时也不需要价格。缺失教师概率的训练行保留硬标签，均匀平滑控制修改完全相同的行。
与相同结构、相同数据的 hard-label 控制（也是旧 R5 最低观察 loss 模型）相比，冻结学生多对 6 场，
loss 降低 **{matched['mean_log_loss_reduction']:.6f}**，条件日期块区间 **[{ci[0]:+.6f}, {ci[1]:+.6f}]** 跨过零。
这是有希望的概率结果，尚不足以确认泛化提升。原始市场价格本身的 2023 探索比较结果属于更晚信息条件，不纳入这里的排行。

未选中的 `uniform_lambda1` 确实以 **1,716 场正确**比旧历史模型最高观察多对 1 场，不能说所有新配置都未刷新准确率；
但它的 loss **0.622271** 明显更差，而且未通过开发期选择，不能当作推荐升级。
它距 70% 仍差 **96 场**；冻结市场学生差 **100 场**。本轮预先冻结的新分支选择均未超过旧 R5 的 66.27% 准确率。

## 加入本届本人早轮次：更晚的信息条件

{markdown_table(selected[selected.branch.eq('event')])}

仅使用本人严格更早的淘汰赛轮次，**1,445** 场有合格早轮历史，其余 **1,143** 场严格回退到 R4。
2022 门槛只多对 7 场，未升级；pooled 冻结组合在 2023 为 1,693 场正确。
本分支最高新观察是两个 load 模型的 1,702 场正确，也未超过旧 R5。完整 12 配置及 40 行匹配信息对照见[赛事进程报告](event/report_summary.md)。
轮次信息可能较晚获得，不能把它与原 T−21 条件的模型收益相加解释。

## 冻结规则与门槛

先在早期开发折按 accuracy 选择，要求 log loss 不比 R4 高超过 0.005；再看 2022 至少 +0.5 个百分点的 accuracy 门槛及相同 loss 护栏。
pooled 开发折选择在各分支 2023 评分之前保存。市场监督由于可用报价年份限制，以 2021 下半年为早期选择折；其 pooled 队列与其他分支不同，不直接横比 CV 值。

| 分支 | 早期选择期 | 2022 净增正确 | Accuracy 变化 / pp | Log loss 变化 ↓ | 记录的分支决策 |
| --- | --- | ---: | ---: | ---: | --- |
{chr(10).join(rows)}

R4 本身和 2022 门槛已在此前轮次使用，因此该门槛是探索流程约束，不是独立验证或显著性检验。
日期块区间也没有覆盖完整搜索不确定性和跨日期球员依赖。没有把某一分支门槛通过解释为全局新冠军。
下一位评估者应在打开最终测试前固定比较清单，保留 R4、既有 R5 对照及各分支冻结候选的不同角色。

![同一完整队列的准确率与概率误差](round7_comparison.png)

## 复跑与交付

先按[建模指南](../../docs/modeling.md)复现之前各轮，包括第三轮目标、第五轮模型和第六轮迁移特征矩阵。常规三分支与报告使用原 `.venv`：

```bash
.venv/bin/python -m scripts.train_margin_round7
.venv/bin/python -m scripts.train_distillation_round7
.venv/bin/python -m scripts.train_event_round7
```

基础模型使用隔离环境，禁止升级原建模环境。固定依赖快照见[requirements-foundation.txt](../../requirements-foundation.txt)，在 Apple M4 Pro / MPS、Python 3.12 实测：

```bash
python3.12 -m venv .venv-foundation
.venv-foundation/bin/python -m pip install -r requirements-foundation.txt
.venv-foundation/bin/python -m scripts.setup_foundation_round7
.venv-foundation/bin/python -m scripts.train_foundation_round7
.venv/bin/python -m scripts.report_round7
```

下载器仅获取官方公共 `jingang/TabICL` 的固定 revision `4dcd344ece2c00be9e831fdd35bed57b5ad83e19`，
核对 110,368,038 字节 checkpoint 的 SHA-256 `bdc7dbd5e4ff21f8f0456fcf90c6b7cdf72dbea960f2d05b19bec19f9b3d4ed0`。
完整配置固定为 64 列、原始上下文 2,048/4,096 行（交换后 4,096/8,192）、4 个 ensemble 成员、fp32；CPU/MPS 内存调度变更已单列记录。
该分支本地上下文模型约 2.5 GB，权重及大中间文件不提交 Git，按固定源重建；CPU 或其他硬件未据此证明有相同速度或逐位结果。
报告器只读取冻结产物，复算各行指标、核验目标 ID/标签一致性，然后生成 CSV、图、已执行 notebook 和 HTML；不会重训或重新挑选模型。

[完整比较 CSV](overview_comparison.csv) · [已执行 notebook](../../round7.ipynb) · [HTML](../../round7.html) ·
[本轮协议](../../docs/research/round7_protocol.md) · [基础模型报告](foundation/report_summary.md) ·
[得分监督报告](margin/report_summary.md) · [市场学生报告](distillation/report_summary.md) · [赛事进程报告](event/report_summary.md)。
四分支的重载、对称性、时间选择与来源检查见各自的独立审计：
[foundation](foundation/artifact_audit.json)、[margin](margin/artifact_audit.json)、[distillation](distillation/artifact_audit.json)、[event](event/artifact_audit.json)。
'''
    (OUT / 'overview.md').write_text(body)
    return body


def rebase_links(markdown, source_directory, destination_directory):
    def replace(match):
        label, target = match.group(1), match.group(2)
        if '://' in target or target.startswith('#'):
            return match.group(0)
        path, sep, anchor = target.partition('#')
        relative = os.path.relpath((source_directory / path).resolve(), destination_directory)
        return f'{label}({relative}{sep}{anchor})'
    return re.sub(r'(!?\[[^\]]*\])\(([^)]+)\)', replace, markdown)


def write_notebook(body):
    # Relative links need a different base in the root notebook/HTML.
    notebook_body = rebase_links(body, OUT, ROOT)
    notebook_body = re.sub(r'!\[同一完整队列的准确率与概率误差\]\([^)]*\)', '', notebook_body)
    nb = nbformat.v4.new_notebook()
    nb.metadata.kernelspec = dict(display_name='Python 3', language='python', name='python3')
    nb.cells = [
        nbformat.v4.new_markdown_cell(notebook_body),
        nbformat.v4.new_code_cell("from pathlib import Path\nimport json\nimport pandas as pd\nfrom IPython.display import display, Image\nroot = Path.cwd()\nout = root / 'results/round7'\ncomparison = pd.read_csv(out / 'overview_comparison.csv')\nassert comparison.matches.eq(2588).all()\ndisplay(comparison.loc[comparison.display, ['branch','model','role','correct','accuracy','mean_log_loss']])\ndisplay(Image(filename=str(out / 'round7_comparison.png')))"),
        nbformat.v4.new_markdown_cell('## 全部已保存结果\n\n此表保留未选中的控制和各分支重复的旧参考；不按 2023 重新选模型。'),
        nbformat.v4.new_code_cell("with pd.option_context('display.max_rows', 100, 'display.max_colwidth', 70):\n    display(comparison[['branch','model','role','correct','accuracy','mean_log_loss']])"),
        nbformat.v4.new_markdown_cell('## 冻结选择与门槛\n\n市场学生的早期折为 2021 下半年；其他分支为 2020–2021，不能混用 pooled CV 队列。'),
        nbformat.v4.new_code_cell("choices = []\nfor branch in ('foundation','margin','distillation','event'):\n    frozen = json.loads((out / branch / 'frozen_selection.json').read_text())\n    gate = frozen['procedure_gate']\n    choices.append(dict(branch=branch, recommendation=frozen['recommendation'], pooled_choice=frozen['pooled_selection']['choice'], procedure=gate['procedure'], gate_accuracy_gain=gate['accuracy_improvement'], gate_loss_change=gate['candidate_2022_loss']-gate['reference_2022_loss']))\ndisplay(pd.DataFrame(choices))"),
        nbformat.v4.new_markdown_cell('## 本人早轮次匹配信息对照\n\n每行同学习器、同回退策略；CV 与 2023 分开。'),
        nbformat.v4.new_code_cell("matched = pd.read_csv(out / 'event/matched_information_controls.csv')\nwith pd.option_context('display.max_rows', 50):\n    display(matched)"),
    ]
    NotebookClient(nb, timeout=180, kernel_name='python3', resources={'metadata': {'path': str(ROOT)}}).execute()
    nbformat.write(nb, ROOT / 'round7.ipynb')
    html, _ = HTMLExporter().from_notebook_node(nb)
    (ROOT / 'round7.html').write_text(html)


def main():
    targets = pd.read_csv(ROOT / 'results/round3/development_targets.csv')
    identity = targets.loc[targets.season.eq(2023), ['match_id', 'a_won']]
    data = {branch: load_branch(branch, identity) for branch in BRANCHES}
    counts = [data['foundation']['summary']['fit_candidates'], data['margin']['summary']['fit_candidates'],
              data['distillation']['summary']['new_candidates'], data['event']['summary']['new_candidates']]
    if counts != [6, 10, 7, 12]:
        raise ValueError('Frozen fitted-configuration counts changed')
    controls = write_event_report(data['event'])
    table = build_comparison(data, identity)
    table.to_csv(OUT / 'overview_comparison.csv', index=False)
    plot_comparison(table)
    body = write_overview(data, table)
    write_notebook(body)
    print(f'Rendered {len(table)} saved comparison rows, {len(controls)} event matched controls, and executed round7.ipynb/html.')


if __name__ == '__main__':
    main()
