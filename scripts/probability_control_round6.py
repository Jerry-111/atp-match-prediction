"""Explicitly adaptive single minimum-CV-loss comparator, no new grid search."""
import json

import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from scripts.train_models import ROOT, temporal_masks
from scripts.train_round4 import sha256, cluster_interval
from scripts.train_round5 import fit_model, accuracy_interval
from scripts.metrics import score_probabilities


def main():
    parent, out = ROOT / 'results/round6', ROOT / 'results/round6/probability_control'
    out.mkdir(parents=True, exist_ok=True)
    registry = json.loads((parent / 'search_registry.json').read_text())
    cv = pd.read_csv(parent / 'temporal_cv.csv')
    specs = {s['name']: s for s in registry['candidates'] if s['kind'] != 'previous'}
    selected = cv[cv.name.isin(specs)].sort_values(['mean_log_loss', 'name']).iloc[0]
    frozen = dict(selected=str(selected['name']), rule='Minimum pooled2020--2022 log loss among34 fixed new configurations',
        cv_log_loss=float(selected.mean_log_loss), cv_accuracy=float(selected.accuracy),
        adaptive_after_round6_accuracy_results=True, recommendation_upgraded=False,
        final_test_evaluated=False, independent_validation=False,
        registry_sha256=sha256(parent / 'search_registry.json'), cv_sha256=sha256(parent / 'temporal_cv.csv'))
    (out / 'frozen_selection.json').write_text(json.dumps(frozen, indent=2) + '\n')
    rows = pd.read_csv(ROOT / 'results/round3/development_targets.csv')
    x = pd.read_csv(parent / 'development_features.csv')
    train, valid = temporal_masks(rows, 2023)
    bundle = joblib.load(parent / 'frozen_models.joblib')
    name = frozen['selected']
    model = bundle['models'].get(name)
    if model is None:
        model = fit_model(specs[name], x[train], rows[train])
        fitted = True
    else:
        fitted = False
    p = model.predict_proba(x[valid])
    prior = pd.read_csv(parent / 'validation_predictions.csv')
    vr = rows[valid].reset_index(drop=True)
    if not vr[['match_id', 'a_won']].equals(prior[['match_id', 'a_won']]):
        raise ValueError('Prior validation identities changed')
    predictions = {'minimum_cv_loss_control': p,
        **{n: prior[n].to_numpy() for n in ('previous_probability_recommendation',
            'previous_round5_exploration', 'previous_round5_best_loss')}}
    scores = []
    for model_name, values in predictions.items():
        scores.append(dict(model=model_name, **score_probabilities(zip(vr.match_id, vr.a_won), zip(vr.match_id, values)),
            brier=float(np.mean((vr.a_won.to_numpy()-values)**2))))
    table = pd.DataFrame(scores)
    table.to_csv(out / 'validation_comparison.csv', index=False)
    pd.concat([vr[['match_id', 'season', 'tourney_date', 'a_won']], pd.DataFrame(predictions)], axis=1).to_csv(out / 'validation_predictions.csv', index=False)
    comparisons = {n: dict(accuracy=accuracy_interval(vr, v, p), log_loss=cluster_interval(vr, v, p))
        for n, v in predictions.items() if n != 'minimum_cv_loss_control'}
    summary = dict(**frozen, selected_spec=specs[name], extra_final_fit=fitted,
        validation_matches=len(vr), comparisons=comparisons,
        feature_sha256=sha256(parent / 'development_features.csv'))
    (out / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    joblib.dump(dict(model=model, feature_names=specs[name]['columns'], frozen=frozen), out / 'frozen_model.joblib')
    result = table.iloc[0]
    text = f'''# 追加概率目标对照

本方向在看到第六轮accuracy选择结果后提出，明确为自适应探索；不升级正式推荐，不是独立验证。在既有34项配置中按pooled2020–2022最低log loss选出`{name}`，CV loss={selected.mean_log_loss:.6f}，没有新增参数搜索。

该模型2023正确{int(result.correct)}/2588，准确率**{result.accuracy:.2%}**，log loss **{result.mean_log_loss:.6f}**，Brier **{result.brier:.6f}**。70%仍需额外{max(0,1812-int(result.correct))}场正确。概率改善与分类准确率分开判断；配对区间见`summary.json`，不涵盖全部选择不确定性。

| 比较 | 正确数 | Accuracy | Log loss | Brier |
| --- | ---: | ---: | ---: | ---: |
'''
    for r in table.itertuples():
        text += f'| {r.model} | {r.correct} | {r.accuracy:.2%} | {r.mean_log_loss:.6f} | {r.brier:.6f} |\n'
    text += '\n复跑：`python -m scripts.probability_control_round6`，需先完成主实验。2024–2025最终测试未打开。\n'
    (out / 'report_summary.md').write_text(text)
    print(text)


if __name__ == '__main__':
    with threadpool_limits(limits=4):
        main()
