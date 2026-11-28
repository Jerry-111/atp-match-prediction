"""Explicitly adaptive closing-price blending controls; no recommendation upgrade.

This branch follows inspection of Round5 main/opening2023 results. Weights are
selected on2022 only, but neither year is an independent validation sample.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from scripts.metrics import score_probabilities
from scripts.train_models import ROOT, mean_loss, cluster_interval, sha256
from scripts.train_round5 import accuracy, accuracy_interval, selection_probability, LOSS_GUARD
from scripts.train_market_round5 import MarketModel

OUT = ROOT / 'results/round5/closing_blend'


def specs():
    result = []
    for base in ('round4', 'round5_exploratory'):
        result += [dict(name=base + '_history', kind='history', base=base),
                   dict(name=base + '_closing', kind='opening', base=base)]
        for mode in ('probability', 'logit'):
            for weight in (.25, .5, .75):
                result.append(dict(name=f'{base}_{mode}_closing{weight:g}', kind='blend',
                                   base=base, mode=mode, weight=weight))
    return result


def predict_closing_controls(bases, market, candidates=None):
    required = ['market_close_logit_diff', 'market_close_available', 'market_close_overround']
    if market.columns.duplicated().any() or not set(required).issubset(market):
        raise ValueError('Unique closing-price features required')
    closing = market[required].rename(columns={c: c.replace('_close_', '_open_') for c in required})
    candidates = specs() if candidates is None else candidates
    return pd.DataFrame({s['name']: MarketModel(s).predict_proba(bases[s['base']], closing)
                         for s in candidates}, index=market.index)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    market_path = ROOT / 'results/round5/market/features.csv'
    rows = pd.read_csv(ROOT / 'results/round3/development_targets.csv')
    rows = rows[rows.season.isin([2022, 2023])].reset_index(drop=True)
    market = pd.read_csv(market_path).set_index('match_id')
    if not set(rows.match_id).issubset(market.index) or market.index.duplicated().any():
        raise ValueError('Closing-price target alignment mismatch')
    market = market.loc[rows.match_id].reset_index(drop=True)
    frozen4 = json.loads((ROOT / 'results/round4/frozen_selection.json').read_text())
    frozen5 = json.loads((ROOT / 'results/round5/frozen_selection.json').read_text())
    frame4 = pd.concat([pd.read_csv(ROOT / 'results/round4/oof_predictions.csv'),
                        pd.read_csv(ROOT / 'results/round4/validation_predictions.csv')])
    frame5 = pd.concat([pd.read_csv(ROOT / 'results/round5/oof_predictions.csv'),
                        pd.read_csv(ROOT / 'results/round5/validation_predictions.csv')])
    bases = {}
    for label, frame in [('round4', frame4), ('round5_exploratory', frame5)]:
        frame = frame.set_index('match_id').loc[rows.match_id].reset_index()
        if not frame[['match_id', 'season', 'a_won']].equals(rows[['match_id', 'season', 'a_won']]):
            raise ValueError('Historical probability/label identity mismatch')
        if label == 'round4':
            bases[label] = frame[frozen4['recommendation']].to_numpy()
        else:
            chosen = frozen5['pooled_selection']
            bases[label] = selection_probability(chosen, {c: frame[c].to_numpy() for c in chosen['components']})
    candidates = specs()
    registry = dict(candidates=candidates, selection_season=2022, scoring_season=2023,
        source_features_sha256=sha256(market_path),
        adaptive_after_round5_2023=True, independent_validation=False, final_test_evaluated=False,
        information_regime='Provider-described closing prematch prices plus fixed historical model',
        no_parameter_fitting=True, no_recommendation_upgrade=True,
        caveat='The direction follows observed2023 results; weights use2022, already used in earlier searches.')
    (OUT / 'search_registry.json').write_text(json.dumps(registry, indent=2) + '\n')
    train, valid = rows.season.eq(2022).to_numpy(), rows.season.eq(2023).to_numpy()
    train_bases = {n: p[train] for n, p in bases.items()}
    p_train = predict_closing_controls(train_bases, market[train], candidates)
    y = rows.a_won.to_numpy()
    limit = mean_loss(y[train], train_bases['round4']) + LOSS_GUARD
    eligible = [c for c in p_train if mean_loss(y[train], p_train[c].to_numpy()) <= limit]
    selected = min(eligible, key=lambda c: (-accuracy(y[train], p_train[c]), mean_loss(y[train], p_train[c].to_numpy()), c))
    frozen = dict(selected_control=selected, decision_basis='2022accuracy, .005loss guard, then lower loss/name',
        selection_matches=int(train.sum()), no_recommendation_upgrade=True,
        adaptive_after_round5_2023=True, independent_validation=False, final_test_evaluated=False)
    (OUT / 'frozen_selection.json').write_text(json.dumps(frozen, indent=2) + '\n')
    p = predict_closing_controls({n: v[valid] for n, v in bases.items()}, market[valid], candidates)
    vr = rows[valid]
    scores = []
    for c in p:
        score = score_probabilities(zip(vr.match_id, vr.a_won), zip(vr.match_id, p[c]))
        scores.append(dict(model=c, selected_control=c == selected, **score,
                          brier=float(np.mean((vr.a_won.to_numpy() - p[c].to_numpy()) ** 2))))
    table = pd.DataFrame(scores).sort_values(['accuracy', 'mean_log_loss'], ascending=[False, True])
    table.to_csv(OUT / 'validation_comparison.csv', index=False)
    pd.concat([vr.reset_index(drop=True), p.reset_index(drop=True)], axis=1).to_csv(OUT / 'validation_predictions.csv', index=False)
    pd.DataFrame([dict(model=c, accuracy=accuracy(y[train], p_train[c]),
                       mean_log_loss=mean_loss(y[train], p_train[c].to_numpy())) for c in p_train]).to_csv(OUT / 'selection_2022.csv', index=False)
    reference = p.round4_closing.to_numpy()
    summary = dict(selected_control=selected, comparisons_vs_closing={c: dict(
        accuracy=accuracy_interval(vr, reference, p[c].to_numpy()),
        log_loss=cluster_interval(vr, reference, p[c].to_numpy())) for c in p},
        closing_available=int(market.loc[valid, 'market_close_available'].sum()), validation_matches=len(vr),
        required_correct_for_70=int(np.ceil(.7 * len(vr))), adaptive_after_round5_2023=True,
        no_recommendation_upgrade=True, final_test_evaluated=False)
    (OUT / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    row = table.set_index('model').loc[selected]
    best = table.iloc[0]
    body = f'''# 追加探索：固定收盘报价融合

在看到第五轮纯历史与开盘分支的2023结果后，追加16个固定控制。只用2022选择权重，但2022/2023都已反复使用，**没有独立验证，也不升级现有推荐**。使用实际赛前最后报价是更晚的信息条件，不属于原T−21模型的纯算法提升。

历史底模预先限于R4推荐与R5 pooled探索性组合。概率/logit融合仅取市场权重0.25/0.5/0.75，另保留每个底模及纯市场基准；没有重新拟合参数或改变0.5阈值。缺报价行精确使用相应历史底模。完整2588场中，2477场有closing报价，111场回退。

2022选中的控制为`{selected}`。2023准确率 **{row.accuracy:.2%}**（**{int(row.correct)}/2588**），log loss **{row.mean_log_loss:.5f}**；达到70%还差 **{max(0, 1812-int(row.correct))}** 场。全部控制的最高观察准确率为`{best.model}`：**{best.accuracy:.2%}**，不据此更改冻结选择。

| 控制 | 正确数 | 准确率 | Log loss |
| --- | ---: | ---: | ---: |
'''
    for result in table.itertuples():
        body += f'| `{result.model}` | {result.correct} | {result.accuracy:.2%} | {result.mean_log_loss:.5f} |\n'
    body += '''
来源：Valuebetennis，按[CC BY4.0公开数据](https://www.valuebetennis.com/en/donnees.htm)及其赛前报价定义使用；缺少逐条报价采集时间的限制不变。该对照不能证明优于市场或达到独立测试的70%。配对区间见`summary.json`，不覆盖反复搜索偏差。

复跑：`python -m scripts.closing_blend_round5`。需先完成第五轮主实验和开盘市场实验；此前产物不会覆盖。
'''
    (OUT / 'report_summary.md').write_text(body)
    print('Frozen exploratory control:', selected)
    print(table.to_string(index=False))


if __name__ == '__main__':
    main()
