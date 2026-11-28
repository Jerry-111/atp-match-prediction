"""Render audited Round 10 outputs; never refit, choose or change a threshold."""
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

OUT = ROOT/'results/round10'
LABELS = dict(equal_mix='Equal expert mixture', global_mix='Global expert mixture',
              conditional_mix='Conditional expert mixture', ret_I_spline='Ordinary inactivity control',
              ret_R_spline='Inactivity + recorded RET', ret_RC_spline='Inactivity + RET return state')


def read_json(path):
    return json.loads(Path(path).read_text())


def collect():
    rows = pd.read_csv(ROOT/'results/round3/development_targets.csv')
    rows = rows[rows.season.eq(2023)]
    prior = pd.read_csv(ROOT/'results/round9/overview_comparison.csv')
    prior = prior[prior.branch.eq('prior')].copy()
    records = prior.to_dict('records')
    branches = {}
    for branch in ('gating','retirement'):
        path = OUT/branch
        table = pd.read_csv(path/'validation_comparison.csv')
        pred = pd.read_csv(path/'validation_predictions.csv')
        verify_scores(table,pred,rows)
        frozen, summary, audit = [read_json(path/(n+'.json')) for n in
                                 ('frozen_selection','summary','artifact_audit')]
        assert audit['status']=='passed'
        assert not frozen['final_test_evaluated'] and not summary['final_test_evaluated']
        choice = frozen['selected_procedure'] if branch=='gating' else frozen['pooled_selection']['choice']
        branches[branch] = dict(table=table,frozen=frozen,summary=summary,audit=audit,choice=choice)
        for row in table.itertuples():
            is_new = row.model in LABELS
            role = '预设新对照' if is_new else '既有参考'
            if row.model==choice:
                role += '；2021冻结选择' if branch=='gating' else '；pooled开发期冻结选择'
            if row.model==frozen['recommendation']:
                role += '；分支保留/采用'
            records.append(dict(branch=branch,model=row.model,role=role,
                information_regime='T−21纯历史：固定专家组合' if branch=='gating'
                    else 'T−21历史；含提供方RET状态，来源部分核验' if is_new and row.model!='ret_I_spline'
                    else 'T−21纯历史对照/旧参考',
                matches=int(row.matches),correct=int(row.correct),accuracy=row.accuracy,
                mean_log_loss=row.mean_log_loss,brier=row.brier,display=is_new,
                plot_label=LABELS.get(row.model,''),selected_current_round=row.model==choice))
    result = pd.DataFrame(records)
    assert result.matches.eq(2588).all()
    result['additional_correct_for_70'] = 1812-result.correct
    result.to_csv(OUT/'overview_comparison.csv',index=False)
    return branches,result


def md_table(table):
    lines = ['| 条件与角色 | 正确 / 2,588 | Accuracy | Log loss ↓ | 距70%正确数 |',
             '| --- | ---: | ---: | ---: | ---: |']
    for r in table.itertuples():
        lines.append(f'| {r.information_regime}：{r.role}（`{r.model}`） | {r.correct} | '
                     f'{r.accuracy:.2%} | {r.mean_log_loss:.6f} | {r.additional_correct_for_70} |')
    return '\n'.join(lines)


def plot(table):
    view=table[table.display].reset_index(drop=True)
    colors=[{'prior':'#68778b','gating':'#367d96','retirement':'#52865b'}[b] for b in view.branch]
    y=np.arange(len(view))
    fig,axes=plt.subplots(1,2,figsize=(16,9),sharey=True,gridspec_kw={'width_ratios':[1.3,1]})
    bars=axes[0].barh(y,100*view.accuracy,color=colors,height=.65)
    for i,r in view.iterrows():
        if '(unselected)' in r.plot_label:bars[i].set_hatch('///')
        axes[0].text(100*r.accuracy+.1,i,f'{r.accuracy:.2%} ({r.correct})',va='center',fontsize=9)
        axes[1].text(r.mean_log_loss+.0008,i,f'{r.mean_log_loss:.5f}',va='center',fontsize=9)
    axes[0].set_yticks(y,view.plot_label);axes[0].invert_yaxis()
    axes[0].set_xlim(min(61.8,100*view.accuracy.min()-1),max(71.5,100*view.accuracy.max()+2))
    axes[0].axvline(70,color='#a94141',linestyle='--',linewidth=1)
    axes[0].set_xlabel('Accuracy (%) | higher is better')
    axes[1].scatter(view.mean_log_loss,y,c=colors,s=65)
    axes[1].set_xlim(view.mean_log_loss.min()-.004,view.mean_log_loss.max()+.011)
    axes[1].set_xlabel('Mean log loss | lower is better')
    for ax in axes:
        ax.grid(axis='x',alpha=.15);ax.set_axisbelow(True);ax.spines[['top','right']].set_visible(False)
    fig.suptitle('Round 10: conditional mixtures and historical RET status',fontsize=16,weight='bold',y=.97)
    fig.text(.5,.928,'Same 2,588 original 2023 targets. Market references use different information.',ha='center',fontsize=10)
    fig.text(.5,.024,'RET is a provider-recorded status, not a diagnosis. Reused validation is exploratory; final years remain sealed.',ha='center',fontsize=9)
    fig.subplots_adjust(left=.30,right=.97,top=.885,bottom=.10,wspace=.14)
    fig.savefig(OUT/'round10_comparison.png',dpi=170,facecolor='white');plt.close(fig)


def report(data,table):
    g,r = data['gating'],data['retirement']
    gt,rt = g['table'].set_index('model'),r['table'].set_index('model')
    chosen_g,chosen_r = gt.loc[g['choice']],rt.loc[r['choice']]
    best = table.sort_values(['accuracy','mean_log_loss'],ascending=[False,True]).iloc[0]
    new = table[table.model.isin(LABELS)]
    new_best = new.sort_values(['accuracy','mean_log_loss'],ascending=[False,True]).iloc[0]
    completion=read_json(OUT/'completion/source_feasibility.json')
    verification=read_json(OUT/'verification.json')
    gate_rows=['| 分支 | 较早数据冻结选择 | 2022流程决定 | 2023冻结探索正确数 |',
               '| --- | --- | --- | ---: |']
    for b,d,c in [('gating',g,chosen_g),('retirement',r,chosen_r)]:
        gate_rows.append(f"| {b} | `{d['choice']}` | `{d['frozen']['procedure_gate']['procedure']}` | {int(c.correct)} |")
    body=f'''# 第十轮：条件混合与历史 RET 状态

本轮所有比较保持完整2,588场、原标签和0.5阈值。新对照最高观察为
**{int(new_best.correct)}/2,588 = {new_best.accuracy:.2%}**（`{new_best.model}`）；
这不改变较早数据冻结的选择。连同旧条件参考，表中最高仍是
**{int(best.correct)}/2,588 = {best.accuracy:.2%}**（{best.role}），
距1,812场目标差**{max(0,1812-int(best.correct))}**场。
2023与2022门槛已重复使用，结果均为探索性；2024–2025继续封存，goal保持进行中。

## 同一批比赛的对比

{md_table(table[table.display])}

![第十轮对比](round10_comparison.png)

当前市场参考使用较晚信息，历史市场监督学生使用额外训练监督，分别保留原有条件。
纯历史输入、历史RET状态与当前报价不能合并成同一信息条件下的升级宣称。

## 条件权重是否有用

四个冻结原始专家的概率完全相同，比较等权、全局权重和五项赛前上下文控制的权重。
先用2020 OOF拟合、2021选方法，再检查2022门槛；没有伪造2020的元模型测试，
没有pooled重选。真实前向比较4,916场，最终拟合用6,130个原始OOF预测。
三阶段各一套全局与条件优化，共**6次优化、零次基础模型重训**。

2021冻结选择为`{g['choice']}`。2023全局混合正确{int(gt.loc['global_mix','correct'])}场、
loss{gt.loc['global_mix','mean_log_loss']:.6f}；条件混合正确{int(gt.loc['conditional_mix','correct'])}场、
loss{gt.loc['conditional_mix','mean_log_loss']:.6f}。
应以条件与全局的配对差异判断路由机制，不能只拿它与更弱的单个专家比较。
[门控报告](gating/report_summary.md)保存三阶段权重变化、配对区间及全部失败结果。

## 历史 RET 是否超出普通休赛信息

I/R/RC恰好三个固定结构，三年前向fold加最终拟合，共**12次拟合**；无事后混合。
三组共享326旧字段和6个普通inactivity字段。R与RC各增4个RET字段，输入维数
332/336/336；I是嵌套基线，只有R与RC同容量。
RET只表示已记录的过去非完赛，不是伤病诊断或恢复概率；不使用当前目标的RET，
也不把其未完成技术统计或额外败局加进普通历史。

2023的I/R/RC分别判对{int(rt.loc['ret_I_spline','correct'])}/
{int(rt.loc['ret_R_spline','correct'])}/{int(rt.loc['ret_RC_spline','correct'])}场。
较早2020–2021选择R，但2022仅比R4多
{round(2528*r['frozen']['procedure_gate']['accuracy_improvement'])}场，没有达到升级门槛；
pooled开发期选择的是I。加入RET在2023有小幅loss和准确率改善，但pooled前向准确率
并未超过I，不能据此断言RET有稳定增益，也不事后改选RC。

30条预先冻结来源样本中，**10条官方核实，20条仍未核实**，且已核实部分偏早期。
未核实记录既不假称通过，也不按官方文档是否易得而筛选。协调者在看新分数前
授权了带此限制的有限探索，来源全面ready标志仍为false。
完整候选、差异和冻结选择见[RET报告](retirement/report_summary.md)；
来源证据与边界见[协议](../../docs/research/round10_retirement_protocol.md)。

## 冻结选择与建议

{chr(10).join(gate_rows)}

门控的2021方法选择与RET的pooled开发期探索不是相同选择程序；RET的实际升级
仍由2020–2021早选后在2022检查。各分支保留/采用的是
`{g['frozen']['recommendation']}`与`{r['frozen']['recommendation']}`。
维持所有既有信息条件及排名/Elo基准，不以2023最优观察覆盖冻结决定。

## 更近历史的来源试点

六组预先指定的2021–2022赛事衔接中，{completion['date_bound_qualified']}组通过保守日期界限，
{completion['distinct_new_source_editions']}个来源赛事能新增历史，直接覆盖所列
{completion['target_matches_in_pilot']}场中的{completion['new_direct_participant_coverage']}场。
两组相邻日期因为保守时区界限重叠被拒绝，一组虽已完赛但原T−21就已可用。
这是元数据可行性，不是预测提升；网页历史发布日期仍有证据层级限制。
没有因此缩短全局滞后或训练新模型。
[逐条证据和计数](../../docs/research/round10_completion_feasibility.md)。
独立复核保存了下一步完整来源工作清单：2021–2022共127个目标赛事、504个候选转移，
包括没有候选来源的赛事；2023的259个候选转移单列，全部尚未获准用于预测。
[独立复核与采样框](../../docs/research/round10_completion_review.md)。

## 核验与复跑

基础测试{verification['base_passed']}通过，{verification['base_skipped']}个可选Torch跳过，
{verification['base_subtests']}子测试通过。本轮未修改神经网络训练或重跑其旧实验。
两个独立模型审计分别保存在[gating](gating/artifact_audit.json)和
[retirement](retirement/artifact_audit.json)。门控审计独立检查全部4,916 OOF与
2,588验证行、六个优化结果、训练期标准化和前向计算；原R4/R5旧fold模型未保存的
限制继续保留。RET独立从原始日期/状态重建全部36,421×12个新值，437,052个数值
逐项误差为0，原326列也不变。其9个OOF与3个最终模型、训练期预处理及7列完整
验证概率均通过；两分支共11列验证概率重载误差为0。未重播优化过程，也没有把
来源PDF再判读一遍，不能将10条人工来源抽样称为全面核验。

```bash
.venv/bin/python -m scripts.train_gating_round10
.venv/bin/python -m scripts.audit_gating_round10
.venv/bin/python -m scripts.train_retirement_round10
.venv/bin/python -m scripts.audit_retirement_round10
.venv/bin/python -m scripts.audit_completion_round10
.venv/bin/python -m scripts.report_round10
```

先重建前序轮次的固定输入，尤其R4/R5与R9原始OOF。模型和大型矩阵本地保存并可复跑；
不要把最终refit对旧训练行的预测当作OOF。门控重载使用`--predict-inputs`；RET使用
`--predict-features`。来源审计可复核本地文件与哈希，不能自动替代人工阅读官方签表。
本报告器只校验并渲染保存概率，不重新训练或选模型。

[全部对比CSV](overview_comparison.csv) · [决策记录](../../docs/research/round10_protocol.md) ·
[执行notebook](../../round10.ipynb) · [HTML](../../round10.html)
'''
    (OUT/'overview.md').write_text(body)
    return body


def notebook(body):
    text=rebase_links(body,OUT,ROOT)
    text=re.sub(r'!\[第十轮对比\]\([^)]*\)','',text)
    nb=nbformat.v4.new_notebook()
    nb.metadata.kernelspec=dict(display_name='Python 3',language='python',name='python3')
    nb.cells=[nbformat.v4.new_markdown_cell(text),nbformat.v4.new_code_cell(
        "from pathlib import Path\nimport pandas as pd\nfrom IPython.display import display, Image\nout=Path.cwd()/'results/round10'\ntable=pd.read_csv(out/'overview_comparison.csv')\nassert table.matches.eq(2588).all()\ndisplay(table.loc[table.display,['branch','model','role','correct','accuracy','mean_log_loss']])\ndisplay(Image(filename=str(out/'round10_comparison.png')))"),
        nbformat.v4.new_markdown_cell('## 全部冻结比较\n包含旧参考与所有新对照，不按2023重选。'),
        nbformat.v4.new_code_cell("with pd.option_context('display.max_rows',100,'display.max_colwidth',60):\n    display(table[['branch','model','role','correct','accuracy','mean_log_loss']])"),
        nbformat.v4.new_markdown_cell('## 匹配对照与日期证据\n来源可行性计数不代表预测提升。'),
        nbformat.v4.new_code_cell("import json\ndisplay(pd.read_csv(out/'gating/matched_routing.csv'))\ndisplay(pd.read_csv(out/'retirement/matched_controls.csv'))\nledger=json.loads((out/'completion/source_feasibility.json').read_text())\ndisplay(pd.DataFrame(ledger['cases'])[['source','target','date_bound_qualified','already_available_under_T_minus_21','target_matches','newly_covered_either_player']])")]
    NotebookClient(nb,timeout=180,kernel_name='python3',resources={'metadata':{'path':str(ROOT)}}).execute()
    nbformat.write(nb,ROOT/'round10.ipynb')
    html,_=HTMLExporter().from_notebook_node(nb)
    (ROOT/'round10.html').write_text(html)


def main():
    data,table=collect()
    plot(table)
    notebook(report(data,table))
    print(f'Rendered {len(table)} comparisons; no fits or selection changes.')


if __name__=='__main__':
    main()
