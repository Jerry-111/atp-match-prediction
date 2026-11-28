"""Verify and render Round 9 frozen outputs without fitting or changing choices."""
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
from scripts.report_round7 import verify_scores,rebase_links

OUT=ROOT/'results/round9'
BRANCHES=('matchup','sequence','neural')
LABELS={'matchup':'Style matchup, frozen','sequence':'Sequence, frozen','neural':'Tabular neural, frozen'}


def read_json(path):return json.loads(Path(path).read_text())


def load_results():
    identity=pd.read_csv(ROOT/'results/round3/development_targets.csv')
    identity=identity[identity.season.eq(2023)]
    prior=pd.read_csv(ROOT/'results/round8/overview_comparison.csv')
    keep=prior.branch.isin(['baseline','student']) | prior.plot_label.isin(['Raw latest available price','Opening history (unselected)'])
    records=[];data={}
    for row in prior[keep].itertuples():
        regime=row.information_regime
        if row.model=='raw_market':regime='当前末盘价优先；缺失用开盘或R4回退'
        if row.model=='history_linear_C0.03_alpha1':regime='当前开盘价＋T−21历史修正；缺失回退R4'
        records.append(dict(branch='prior',model=row.model,role=row.role,information_regime=regime,
            matches=int(row.matches),correct=int(row.correct),accuracy=row.accuracy,mean_log_loss=row.mean_log_loss,brier=row.brier,
            display=row.model!='previous_round5_best_loss',plot_label=row.plot_label,
            selected_current_round=False))
    for branch in BRANCHES:
        path=OUT/branch
        table=pd.read_csv(path/'validation_comparison.csv');pred=pd.read_csv(path/'validation_predictions.csv')
        verify_scores(table,pred,identity)
        frozen,summary,audit=[read_json(path/(name+'.json')) for name in ('frozen_selection','summary','artifact_audit')]
        assert not frozen['final_test_evaluated'] and not summary['final_test_evaluated'] and audit['status']=='passed'
        choice,retained=frozen['pooled_selection']['choice'],frozen['recommendation']
        data[branch]=dict(table=table,frozen=frozen,summary=summary,audit=audit)
        for row in table.itertuples():
            role='固定对照 / 既有参考';display=False;label=''
            if row.model==choice:role='pooled开发期冻结探索';display=True;label=LABELS[branch]
            if row.model==retained:role+='；分支保留/采用'
            if branch=='sequence' and row.model=='sequence_mean':role+='；相同序列输入的池化对照';display=True;label='Mean-pooled history control'
            records.append(dict(branch=branch,model=row.model,role=role,
                information_regime='T−21纯历史；原始数据/表征适配' if not row.model.startswith('previous_round7') else '历史输入＋训练期市场监督参考',
                matches=int(row.matches),correct=int(row.correct),accuracy=row.accuracy,mean_log_loss=row.mean_log_loss,brier=row.brier,
                display=display,plot_label=label,selected_current_round=row.model==choice))
    table=pd.DataFrame(records)
    new=table[table.branch.ne('prior') & ~table.model.str.startswith(('previous_','round4_','round5_','rank_'))]
    lowest=new.sort_values(['mean_log_loss','model']).iloc[0]
    index=table.index[(table.branch==lowest.branch)&(table.model==lowest.model)][0]
    if not table.loc[index,'display']:
        table.loc[index,['display','plot_label','role']]=[True,'Lowest new observed loss','最低新观察loss，未选中']
    table['additional_correct_for_70']=1812-table.correct
    assert table.matches.eq(2588).all()
    table.to_csv(OUT/'overview_comparison.csv',index=False)
    return data,table


def md_table(table):
    lines=['| 条件 / 比较角色 | 正确数 / 2,588 | Accuracy | Log loss ↓ | 距70%正确数 |',
           '| --- | ---: | ---: | ---: | ---: |']
    for r in table.itertuples():
        lines.append(f'| {r.information_regime}：{r.role}（`{r.model}`） | {r.correct} | {r.accuracy:.2%} | {r.mean_log_loss:.6f} | {r.additional_correct_for_70} |')
    return '\n'.join(lines)


def plot(table):
    view=table[table.display].reset_index(drop=True)
    palette={'prior':'#68778b','matchup':'#3c788b','sequence':'#34866e','neural':'#a7762f'}
    colors=[palette[b] for b in view.branch];y=np.arange(len(view))
    fig,axes=plt.subplots(1,2,figsize=(16,8.5),sharey=True,gridspec_kw={'width_ratios':[1.3,1]})
    bars=axes[0].barh(y,100*view.accuracy,color=colors,height=.65)
    for i,r in view.iterrows():
        if '(unselected)' in r.plot_label or 'observed' in r.plot_label:bars[i].set_hatch('///')
        axes[0].text(100*r.accuracy+.10,i,f'{r.accuracy:.2%} ({r.correct})',va='center',fontsize=9)
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
    fig.suptitle('Round 9: frozen choices and matched controls',fontsize=17,weight='bold',y=.97)
    fig.text(.5,.925,'All 2,588 original 2023 matches. Earlier market references have different information.',ha='center',fontsize=10)
    fig.text(.5,.025,'Hatched = unselected observation. Reused validation is exploratory; 2024–2025 remain sealed.',ha='center',fontsize=10)
    fig.subplots_adjust(left=.30,right=.97,top=.88,bottom=.10,wspace=.14)
    fig.savefig(OUT/'round9_comparison.png',dpi=170,facecolor='white');plt.close(fig)


def overview(data,table):
    selected=table[table.selected_current_round]
    maximum=table.sort_values(['accuracy','mean_log_loss'],ascending=[False,True]).iloc[0]
    title='本轮仍未达到70%' if maximum.correct<1812 else '达到70%的观察必须按冻结选择及审计区分'
    gates=['| 分支 | 2022净正确数 / 2,528 | 准确率变化 | 预设流程决定 |', '| --- | ---: | ---: | --- |']
    for branch,d in data.items():
        g=d['frozen']['procedure_gate']
        gates.append(f"| {branch} | {round(2528*g['accuracy_improvement']):+d} | {100*g['accuracy_improvement']:+.3f}pp | `{g['procedure']}` |")
    sequence=data['sequence']['table'].set_index('model')
    gru,mean=sequence.loc['sequence_gru'],sequence.loc['sequence_mean']
    neural=data['neural']['table'].set_index('model')
    real,plain=neural.loc['realmlp_td_s'],neural.loc['plain_mlp']
    verification=read_json(OUT/'verification.json')
    body=f'''# 第九轮：{title}

目标仍要求同一完整2,588场至少判对1,812场、固定0.5阈值及合法赛前信息。
本表最高观察为 **{int(maximum.correct)}/2,588 = {maximum.accuracy:.2%}**（`{maximum.model}`，{maximum.role}），
距目标还差 **{max(0,1812-int(maximum.correct))}** 场。这个事后观察不改变任何分支的开发期冻结选择。
本轮冻结探索候选最高为 **{selected.correct.max()}/2,588**。2023及2022升级门槛均已重复使用；
不能把任何观察称为独立最终测试或SOTA。2024–2025继续封存，goal保持进行中。

## 一张完整条件对比表

{md_table(table[table.display])}

![第九轮对比](round9_comparison.png)

此前原始末盘优先报价和68.01%的未选开盘修正使用更晚市场信息，单列作背景；不归入纯历史模型。
原R4推荐、R5冻结准确率探索及R7市场监督学生分别保留原有角色。
补充时间审计发现历史报价源有不同轮次共享时间戳，不能把其UTC字段当作经核实的实际完赛时间。
核验全部历史图边后，额外要求已纠正赛事日期也满足T−21不会删除任何已用边；原数值结果完全不变。
这是对时间证据的限定，不证明更短滞后可用。[来源限制与可复核计数](../../docs/research/round9_freshness_feasibility.md)

## 这次改变了什么，结果支持什么

- **风格对位：** 同一188项主效应上，比较无新增交互、21项共享反对称交互及63项场地偏差。
  已有字段能够恢复双方风格水平，所以这是交互结构变化，不是新增信息。
  完整匹配结果见[分支报告](matchup/report_summary.md)。
- **赛事序列：** 每人最近16个不同历史日期组，先合并同日比赛，再在四年、T−21窗口内编码22项统计。
  原始GRU判对{int(gru.correct)}场，池化对照判对{int(mean.correct)}场；GRU相对池化的loss变化为
  {gru.mean_log_loss-mean.mean_log_loss:+.6f}。池化也看到时间字段，固定shuffle也可能被学会逆转；
  这些控制限定解释为表示/容量/时间对齐的比较。[序列报告](sequence/report_summary.md)
- **全量表格神经网络：** 两个配方均使用全部326旧字段和33,833训练目标。
  TD-S的三种子平均为{int(real.correct)}场、loss{real.mean_log_loss:.6f}，普通MLP为{int(plain.correct)}场、
  loss{plain.mean_log_loss:.6f}。这是包含激活、优化和标签平滑差异的配方比较，不是单一架构因素实验。
  每个种子仅由更早内层年份选择轮数，再重新训练完整外层样本。[神经网络报告](neural/report_summary.md)

本轮总计**8种拟合结构/配方、15个固定混合**。两条神经路线各24个种子/外层或最终模型，
另各24次内层轮数选择；加上12次线性拟合，共108个优化阶段，不把这些当108个独立结构选择。
种子始终平均，不选最好种子。所有目标、零历史球员及技术统计缺失记录都按协议保留。

## 冻结规则与建议

{chr(10).join(gates)}

早选2020–2021，2022要求准确率至少+0.5个百分点且loss恶化不超过0.005；pooled2020–2022选择在2023之前冻结。
分支保留/采用与pooled探索角色不同，表中未选的最低loss也不能事后升级。
继续保留各信息条件及原始排名/Elo基准；本轮是否值得进入最终比较，必须在打开最终年份之前固定。

## 核验和复跑

基础环境测试：{verification['base_passed']}通过、{verification['base_skipped']}个可选Torch跳过、
{verification['base_subtests']}子测试通过。隔离Torch环境另有{verification['torch_tests']}项聚焦测试通过；
其中7项特征测试与基础环境重叠，不重复计数。
三个独立模型审计覆盖36列完整预测，另核验神经种子平均、保存的OOF权重、时间分割和预处理。
序列的原始来源审计独立重建55个目标中的1,518个token、2,747场不同历史比赛；没有假称所有token均由独立算法重建。
无原始记录恰好落在四年边界的真实例子，也没有Carpet token；这些边界以合成测试覆盖。

```bash
.venv/bin/python -m scripts.train_matchup_round9
.venv-foundation/bin/python -m scripts.train_sequence_round9 --device cpu
.venv-foundation/bin/python -m scripts.train_neural_round9
.venv/bin/python -m scripts.audit_matchup_round9
.venv-foundation/bin/python -m scripts.audit_sequence_source_round9
.venv-foundation/bin/python -m scripts.audit_sequence_round9
.venv-foundation/bin/python -m scripts.audit_neural_round9
.venv/bin/python -m scripts.report_round9
```

需先重建前序轮次的已审计输入；大型矩阵、序列归档及权重本地保存并可按命令复跑。
报告器仅核验保存概率并生成展示，不训练或重选。神经审计不重播训练梯度轨迹；TD-S内层probe scaler没有原样保存，
只能按已记录内层ID和源码确定性重建，无法与未保存对象逐项比对。完整范围及限制见各audit文件。

[全部结果](overview_comparison.csv) · [研究协议](../../docs/research/round9_protocol.md) ·
[方法和反例](../../docs/research/round9_ideas.md) · [执行版notebook](../../round9.ipynb) · [HTML](../../round9.html)
'''
    (OUT/'overview.md').write_text(body);return body


def notebook(body):
    text=rebase_links(body,OUT,ROOT);text=re.sub(r'!\[第九轮对比\]\([^)]*\)','',text)
    nb=nbformat.v4.new_notebook();nb.metadata.kernelspec=dict(display_name='Python 3',language='python',name='python3')
    nb.cells=[nbformat.v4.new_markdown_cell(text),nbformat.v4.new_code_cell(
        "from pathlib import Path\nimport pandas as pd\nfrom IPython.display import display, Image\nout=Path.cwd()/'results/round9'\ntable=pd.read_csv(out/'overview_comparison.csv')\nassert table.matches.eq(2588).all()\ndisplay(table.loc[table.display,['branch','model','role','correct','accuracy','mean_log_loss']])\ndisplay(Image(filename=str(out/'round9_comparison.png')))"),
        nbformat.v4.new_markdown_cell('## 全部冻结比较\n\n包含未选对照及重复的旧参考，未按2023重新选择。'),
        nbformat.v4.new_code_cell("with pd.option_context('display.max_rows',100,'display.max_colwidth',60):\n    display(table[['branch','model','role','correct','accuracy','mean_log_loss']])"),
        nbformat.v4.new_markdown_cell('## 匹配对照\n\n不同阶段和信息条件分别保留。'),
        nbformat.v4.new_code_cell("for branch in ('matchup','sequence','neural'):\n    print(branch)\n    display(pd.read_csv(out/branch/'matched_controls.csv'))")]
    NotebookClient(nb,timeout=180,kernel_name='python3',resources={'metadata':{'path':str(ROOT)}}).execute()
    nbformat.write(nb,ROOT/'round9.ipynb');html,_=HTMLExporter().from_notebook_node(nb)
    (ROOT/'round9.html').write_text(html)


def main():
    data,table=load_results();plot(table);notebook(overview(data,table))
    print(f'Verified and rendered {len(table)} full-cohort comparison rows.')


if __name__=='__main__':main()
