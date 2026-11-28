"""Report frozen market-residual comparisons, never fit or select models."""
import json
import numpy as np
import pandas as pd
from scripts.train_models import ROOT, mean_loss
from scripts.train_round5 import accuracy

OUT=ROOT/'results/round8/market_residual'


def main():
    overview=[]
    for condition in ('opening','latest'):
        out=OUT/condition
        frozen=json.loads((out/'frozen_selection.json').read_text())
        summary=json.loads((out/'summary.json').read_text())
        comparison=pd.read_csv(out/'validation_comparison.csv')
        oof=pd.read_csv(out/'oof_predictions.csv')
        val=pd.read_csv(out/'validation_predictions.csv')
        controls=[]
        for old,new in (('price','history'),('history','event')):
            for learner in ('linear_C0.003','linear_C0.03','cat_d3'):
                for alpha in (.25,.5,1.):
                    base=f'{old}_{learner}_alpha{alpha:g}';candidate=f'{new}_{learner}_alpha{alpha:g}'
                    for period in ('2021H2','2022','pooled','2023_exploratory'):
                        data=val if period=='2023_exploratory' else oof
                        if period in ('2021H2','2022'): data=data[data.season.eq(2021 if period=='2021H2' else 2022)]
                        y=data.a_won.to_numpy();a=data[base].to_numpy();b=data[candidate].to_numpy()
                        controls.append(dict(condition=condition,information_change=f'{old}_to_{new}',learner=learner,
                            alpha=alpha,period=period,matches=len(data),base=base,candidate=candidate,
                            accuracy_gain=accuracy(y,b)-accuracy(y,a),log_loss_reduction=mean_loss(y,a)-mean_loss(y,b)))
        pd.DataFrame(controls).to_csv(out/'matched_information_controls.csv',index=False)
        ranked=comparison.sort_values(['accuracy','mean_log_loss'],ascending=[False,True])
        selected=comparison.set_index('model').loc[frozen['pooled_selection']['choice']]
        best=ranked.iloc[0];gate=frozen['procedure_gate']
        rows=['| 模型 | 正确 / 2588 | Accuracy | Log loss | 角色 |','| --- | ---: | ---: | ---: | --- |']
        for r in ranked.itertuples():
            role='分支保留' if r.model==frozen['recommendation'] else 'pooled冻结探索' if r.model==frozen['pooled_selection']['choice'] else '预设未选控制'
            rows.append(f'| `{r.model}` | {r.correct} | {r.accuracy:.3%} | {r.mean_log_loss:.6f} | {role} |')
        interval=summary['comparisons_vs_raw_market'][frozen['pooled_selection']['choice']]
        text=f'''# 第八轮市场残差：{condition}

9个拟合配置，每个3个固定残差强度；另保留原价格先验，共28列预测。
原始价格是Bernoulli logit offset，再学习价格情境、326项历史、加27项本人前轮信息的有界修正。
这些是比原T−21更晚的赛前信息条件。2023已重复使用，以下全部为探索结果；2024–2025仍封存。

2021H2早选的2022增量为{round(2528*gate['accuracy_improvement']):+d}/2528，即{100*gate['accuracy_improvement']:+.3f}个百分点；
所需门槛+0.5个百分点未通过。分支保留`{frozen['recommendation']}`。
pooled冻结探索`{frozen['pooled_selection']['choice']}`得到**{int(selected.correct)}/2588={selected.accuracy:.3%}**、loss**{selected.mean_log_loss:.6f}**。
最高观察`{best.model}`为**{int(best.correct)}/2588={best.accuracy:.3%}**、loss**{best.mean_log_loss:.6f}**，距70%还差**{1812-int(best.correct)}场**。
观察排序没有替代冻结选择，也不能据此声称独立达到泛化性能。

{chr(10).join(rows)}

拟合使用{summary['training_rows']}条有合法价格且早于训练截止的2021–2022目标；每个评估模型都预测全部2588场。
2023价格有效{summary['available_quotes']}场，缺失{summary['fallback_rows']}场精确回退原R4，不能用可报价子集充当全量准确率。
latest条件优先closing，否则opening；2021closing全部缺失，因此训练与验证的报价信息质量随年代改变，不能当固定临场采集时点。
opening条件的输入严格不含closing或movement。所有来源都缺逐笔报价采集时间，prematch依据供应方描述。
“opening”指使用的价格字段，不是已核实的预测决策时点。event配置的本人前轮结果在目标比赛前已知，
但不能证明早于该场opening报价采集。因此应称“使用开盘价格的较晚赛前预测”，不能声称在开盘发布瞬间已可做出。

pooled探索相对相同quote条件原价格的准确率变化为{100*interval['accuracy']['accuracy_improvement']:+.3f}个百分点，
条件日期块区间[{100*interval['accuracy']['ci95'][0]:+.3f},{100*interval['accuracy']['ci95'][1]:+.3f}]。
区间没有覆盖全部搜索或跨日期球员依赖。每次信息增加均与相同learner、C和alpha比较，
完整72行逐期对照见[matched_information_controls.csv](matched_information_controls.csv)。

复跑：`.venv/bin/python -m scripts.train_market_residual_round8`，然后
`.venv/bin/python -m scripts.report_market_residual_round8`。仅重载：
`.venv/bin/python -m scripts.train_market_residual_round8 --condition {condition} --predict-features path/to/prepared_inputs.csv`。
输入构造函数保存准确R4 fallback及所需quote字段，CLI使用冻结schema；不含未来结果。
[协议](../../../../docs/research/round8_market_residual.md) · [冻结选择](frozen_selection.json) ·
[搜索记录](search_registry.json) · [独立审计](artifact_audit.json)
'''
        (out/'report_summary.md').write_text(text)
        for n in dict.fromkeys(['raw_market',frozen['pooled_selection']['choice'],best.model]):
            r=comparison.set_index('model').loc[n]
            overview.append(dict(condition=condition,model=n,correct=int(r.correct),accuracy=r.accuracy,mean_log_loss=r.mean_log_loss,
                role='raw prior' if n=='raw_market' else 'pooled frozen' if n==frozen['pooled_selection']['choice'] else 'best observed, unselected'))
    pd.DataFrame(overview).to_csv(OUT/'overview_comparison.csv',index=False)

if __name__=='__main__': main()
