"""Verify and render frozen Round 8 results; never fit or select using 2023."""
from __future__ import annotations

import json
from pathlib import Path
import re

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import nbformat
from nbclient import NotebookClient
from nbconvert import HTMLExporter
import numpy as np
import pandas as pd

from scripts.train_models import ROOT
from scripts.report_round7 import verify_scores, rebase_links

OUT = ROOT / 'results/round8'
BRANCHES = ('joint_points', 'venue', 'market_graph',
            'market_residual/opening', 'market_residual/latest')
REGIMES = {
    'baseline': '原历史输入 / 旧基准',
    'student': '历史输入；训练期市场监督',
    'joint_points': 'T−21历史点统计',
    'venue': 'T−21历史及往年赛事环境',
    'market_graph': 'T−21历史市场输入',
    'market_residual/opening': '当前开盘价；部分候选另含本人早轮次',
    'market_residual/latest': '当前末盘价/缺失用开盘；部分候选另含本人早轮次',
}
LABELS = {'joint_points':'Joint point exploration', 'venue':'Venue exploration',
          'market_graph':'Historical market graph',
          'market_residual/opening':'Opening residual exploration',
          'market_residual/latest':'Latest residual exploration'}


def read_json(path):
    return json.loads(Path(path).read_text())


def load_all():
    identity = pd.read_csv(ROOT/'results/round3/development_targets.csv')
    identity = identity[identity.season.eq(2023)]
    records, data = [], {}
    def add(branch, row, role, label='', display=False):
        records.append(dict(branch=branch, information_regime=REGIMES[branch], model=row.model,
            role=role, plot_label=label, display=display, matches=int(row.matches),
            correct=int(row.correct), accuracy=float(row.accuracy),
            mean_log_loss=float(row.mean_log_loss), brier=float(row.brier),
            additional_correct_for_70=1812-int(row.correct)))
    previous = pd.read_csv(ROOT/'results/round7/overview_comparison.csv')
    for row in previous[previous.branch.eq('baseline')].itertuples():
        add('baseline', row, row.role, row.plot_label, row.model!='previous_round5_best_loss')
    student = pd.read_csv(ROOT/'results/round7/distillation/validation_comparison.csv')
    verify_scores(student, pd.read_csv(ROOT/'results/round7/distillation/validation_predictions.csv'), identity)
    add('student', student[student.model.eq('market_lambda1')].iloc[0], 'R7：市场监督学生',
        'R7 market-supervised student', True)
    for branch in BRANCHES:
        path = OUT/branch
        table = pd.read_csv(path/'validation_comparison.csv')
        pred = pd.read_csv(path/'validation_predictions.csv')
        verify_scores(table, pred, identity)
        frozen = read_json(path/'frozen_selection.json')
        summary = read_json(path/'summary.json')
        if frozen['final_test_evaluated'] or summary['final_test_evaluated']:
            raise ValueError('Final test must remain sealed')
        pooled, retained = frozen['pooled_selection']['choice'], frozen['recommendation']
        data[branch] = dict(table=table, frozen=frozen, summary=summary)
        for row in table.itertuples():
            role = '固定对照 / 既有模型'
            if row.model.startswith('raw_') and branch == 'joint_points': role = '未校准诊断，禁用于选择'
            if row.model == 'selected_graph_locked_pre2023': role = '冻结选择的历史更新诊断，禁用于选择'
            if row.model == pooled: role = '开发期 pooled 冻结探索'
            if row.model == retained: role += '；分支保留推荐'
            label, display = LABELS[branch] if row.model==pooled else '', row.model==pooled
            if branch.startswith('market_residual') and row.model=='raw_market':
                label = 'Raw opening price' if branch.endswith('opening') else 'Raw latest available price'
                display = True
            if branch.endswith('opening') and row.model=='history_linear_C0.03_alpha1':
                role = '2023最高观察准确率，未选中，loss更差'
                label, display = 'Opening history (unselected)', True
            add(branch, row, role, label, display)
    table = pd.DataFrame(records)
    if not table.matches.eq(2588).all(): raise ValueError('Cohort changed')
    table.to_csv(OUT/'overview_comparison.csv',index=False)
    return data, table


def markdown_table(table):
    lines = ['| 信息条件 / 比较及角色 | 正确数 / 2,588 | 准确率 | Log loss ↓ | 距70%正确数 |',
             '| --- | ---: | ---: | ---: | ---: |']
    for row in table.itertuples():
        lines.append(f'| {row.information_regime}：{row.role}（`{row.model}`） | {row.correct:,} | '
                     f'{row.accuracy:.2%} | {row.mean_log_loss:.6f} | {row.additional_correct_for_70} |')
    return '\n'.join(lines)


def plot(table):
    view = table[table.display].reset_index(drop=True)
    palette = {'baseline':'#68778b','student':'#b9812c','joint_points':'#36768d',
               'venue':'#36768d','market_graph':'#b9812c',
               'market_residual/opening':'#34866e','market_residual/latest':'#34866e'}
    colors = [palette[b] for b in view.branch]
    fig, axes = plt.subplots(1,2,figsize=(16,9),sharey=True,gridspec_kw={'width_ratios':[1.3,1]})
    y = np.arange(len(view))
    bars = axes[0].barh(y,100*view.accuracy,color=colors,height=.64)
    for i,r in view.iterrows():
        if r.plot_label.endswith('(unselected)'): bars[i].set_hatch('///')
        axes[0].text(100*r.accuracy+.1,i,f'{r.accuracy:.2%} ({r.correct})',va='center',fontsize=9)
        axes[1].text(r.mean_log_loss+.0008,i,f'{r.mean_log_loss:.5f}',va='center',fontsize=9)
    axes[0].set_yticks(y,view.plot_label)
    axes[0].invert_yaxis()
    axes[0].set_xlim(min(61.8,100*view.accuracy.min()-1),71.5)
    axes[0].axvline(70,color='#a94141',linestyle='--',linewidth=1)
    axes[0].set_xlabel('Accuracy (%)  |  higher is better')
    axes[1].scatter(view.mean_log_loss,y,c=colors,s=60)
    axes[1].set_xlim(view.mean_log_loss.min()-.004,view.mean_log_loss.max()+.012)
    axes[1].set_xlabel('Mean log loss  |  lower is better')
    for ax in axes:
        ax.grid(axis='x',alpha=.15); ax.set_axisbelow(True)
        ax.spines[['top','right']].set_visible(False)
    fig.suptitle('Round 8: unchanged 2,588-match 2023 cohort',fontsize=17,weight='bold',y=.975)
    fig.text(.5,.934,'Different information conditions; frozen choices and controls. Hatched = unselected observation.',ha='center',fontsize=10)
    fig.text(.5,.025,'All 2023 results are exploratory. Threshold = 0.5. Final 2024–2025 evaluation remains sealed.',ha='center',fontsize=10)
    fig.subplots_adjust(left=.30,right=.97,top=.89,bottom=.10,wspace=.14)
    fig.savefig(OUT/'round8_comparison.png',dpi=170,facecolor='white')
    plt.close(fig)


def overview(data, table):
    gates = ['| 分支 | 2022净正确数 / 2,528 | 准确率变化 | 门槛结果 | 保留推荐 |',
             '| --- | ---: | ---: | --- | --- |']
    for branch,d in data.items():
        gate,frozen = d['frozen']['procedure_gate'],d['frozen']
        gates.append(f"| {branch} | {round(2528*gate['accuracy_improvement']):+d} | "
            f"{100*gate['accuracy_improvement']:+.3f} pp | `{gate['procedure']}` | `{frozen['recommendation']}` |")
    body = f'''# 第八轮：结构性新方法仍未达到70%

**goal继续保持 active；本轮没有宣告成功。** 70%要求同一完整2,588场至少判对1,812场。
当前市场修正菜单中的最高观察行是 **1,760/2,588 = 68.01%**，还差52场；它未被历史选择规则选中，
log loss为0.596723，差于原始开盘价约0.59158，不能把它改写成冻结推荐或SOTA。
2023已经反复用于探索；2022门槛也已复用。2024–2025最终测试继续封存。

## 同一队列的对比

{markdown_table(table[table.display])}

![第八轮同队列对比](round8_comparison.png)

蓝色历史方法、历史市场输入/监督和绿色当前市场报价使用不同信息。
“pooled冻结探索”是在查看本轮2023分数之前用开发期选择的行，未必通过升级门槛。
原始末盘优先条件在缺末盘时使用开盘、均无时精确回退R4，所以与R5单纯末盘条件的覆盖略有不同。
开盘指所用报价字段，不是已验证的决策时点；本人先前轮次可能晚于开盘采集，因此相关组合只称为更晚赛前的开盘价格条件。

## 本轮实际检验了什么

- **联合发接发模型：** 五个结构共同拟合过去四年的binomial点统计，检验接发对手校正、时间衰减、场地偏好和赛事效应；
  每个T−21日期重估过去状态，校准系数只从更早赛季学习。另有15个预先固定R4混合。
  这是Ingram方法的稀疏MAP改编，不是论文完整贝叶斯模型的复现。
- **赛事环境：** 10个拟合配置比较原326项、场地环境、原始赛事环境、对手校正环境及交互；
  过去三个赛季按3/2/1加权，只读取到2022的原始点统计。256场缺赛事历史也保留在完整队列。
- **历史赔率图：** 六个固定图设置，原始概率和半权R4混合共12候选；仅过去开盘赔率形成实力差，
  不读当前比赛赔率，未知或不同连通分量精确回退R4。另报告禁止2023历史更新的诊断，不参与选择。
- **当前报价残差：** 开盘、末盘优先两个独立条件，每个9次拟合×3固定残差权重，比较报价上下文、历史、本人更早轮次。
  所有拟合仅使用符合purge条件的2021–2022记录。没有报价的101场精确回退R4，未删目标。

这是**39个结构/参数设置**：5点模型+10环境学习器+6图设置+18报价残差学习器。
不同日期和折重复拟合、固定混合、旧参考与未校准诊断不再被混称为独立模型试验。
全部冻结概率列和匹配对照都保存，未隐藏负面结果。

## 预设的开发期门槛

{chr(10).join(gates)}

联合点模型与赛事环境早选2020–2021、gate2022；市场图和报价残差早选2021下半年、gate2022。
不同分支的pooled CV不能直接混为相同样本。门槛要求准确率至少+0.5个百分点且log loss恶化不超过0.005。
分支的原始比较基准不同，不能用“通过/未通过”判定跨条件优劣。

## 推荐与复跑

保留原纯历史概率推荐R4、R5冻结准确率探索、R7历史市场监督学生及本轮各分支冻结候选的独立角色。
本轮未给出稳定、可跨条件取代这些选择的新证据。不要依据最高的单行2023准确率重新选择模型。

在前序轮次本地产物就绪后：

```bash
.venv/bin/python -m scripts.train_joint_points_round8
.venv/bin/python -m scripts.train_venue_round8
.venv/bin/python -m scripts.train_market_graph_round8
.venv/bin/python -m scripts.train_market_residual_round8
.venv/bin/python -m scripts.report_market_residual_round8
.venv/bin/python -m scripts.audit_joint_points_round8
.venv/bin/python -m scripts.audit_market_graph_round8
.venv/bin/python -m scripts.audit_market_residual_round8
.venv/bin/python -m scripts.report_round8
```

报告器只核验/展示已保存概率，不重训或改变选择。模型bundle、训练矩阵与OOF大文件本地保留且可复跑。
[完整命令与前置产物](../../docs/modeling.md) · [执行版notebook](../../round8.ipynb) · [HTML](../../round8.html) ·
[全部概率比较](overview_comparison.csv) · [决策/论文协议](../../docs/research/round8_protocol.md)

分支证据：[联合点统计](joint_points/report_summary.md)、[赛事环境](venue/report_summary.md)、
[历史市场图](market_graph/report_summary.md)、[开盘修正](market_residual/opening/report_summary.md)、
[末盘优先修正](market_residual/latest/report_summary.md)。独立审计覆盖各分支完整输出；区间只描述给定预测，
没有校正多轮搜索和全部球员依赖。本轮方法及未来探索均不打开最终测试，也不保证能达到70%。

验证共282项测试、97个子测试。独立审计核验116列×2,588场的重载概率，重载误差为0，最大交换补概率误差7.78e−15。
联合点统计独立重建全部175个验证状态的查询概率，并抽查15个状态的梯度、复核20次校准；并未独立重训全部OOF模型。
赛事环境与市场图的新特征均已全量重建。各audit记录实际覆盖及限制。
'''
    (OUT/'overview.md').write_text(body)
    return body


def notebook(body):
    text = rebase_links(body,OUT,ROOT)
    text = re.sub(r'!\[第八轮同队列对比\]\([^)]*\)','',text)
    nb = nbformat.v4.new_notebook()
    nb.metadata.kernelspec = dict(display_name='Python 3',language='python',name='python3')
    nb.cells = [nbformat.v4.new_markdown_cell(text),nbformat.v4.new_code_cell(
        "from pathlib import Path\nimport json\nimport pandas as pd\nfrom IPython.display import display, Image\nout = Path.cwd()/'results/round8'\ntable = pd.read_csv(out/'overview_comparison.csv')\nassert table.matches.eq(2588).all()\ndisplay(table.loc[table.display,['branch','model','role','correct','accuracy','mean_log_loss']])\ndisplay(Image(filename=str(out/'round8_comparison.png')))"),
        nbformat.v4.new_markdown_cell('## 全部冻结输出\n\n包括未选中控制与旧参考；不根据2023重新选择。'),
        nbformat.v4.new_code_cell("with pd.option_context('display.max_rows',200,'display.max_colwidth',60):\n    display(table[['branch','model','role','correct','accuracy','mean_log_loss']])"),
        nbformat.v4.new_markdown_cell('## 同一学习器/结构的对照\n\n不同信息条件、CV与2023分开；这些相关比较不是独立重复实验。'),
        nbformat.v4.new_code_cell("for branch,filename in [('joint_points','matched_controls.csv'),('venue','matched_controls.csv'),('market_residual/opening','matched_information_controls.csv'),('market_residual/latest','matched_information_controls.csv')]:\n    print(branch)\n    controls = pd.read_csv(out/branch/filename)\n    with pd.option_context('display.max_rows',100):\n        display(controls)"),
    ]
    NotebookClient(nb,timeout=180,kernel_name='python3',resources={'metadata':{'path':str(ROOT)}}).execute()
    nbformat.write(nb,ROOT/'round8.ipynb')
    html,_ = HTMLExporter().from_notebook_node(nb)
    (ROOT/'round8.html').write_text(html)


def main():
    data,table = load_all()
    plot(table)
    notebook(overview(data,table))
    print(f'Verified and rendered {len(table)} comparison rows over the identical 2,588 targets.')


if __name__ == '__main__':
    main()
