"""Prespecified graph of historical opening odds; no current-match odds input."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import joblib
import numpy as np
import pandas as pd
from scipy.special import expit
from threadpoolctl import threadpool_limits

from scripts.market_graph_features import (build_market_graph_features, graph_specs,
    load_opening_history, HISTORY_COLUMNS)
from scripts.metrics import score_probabilities
from scripts.train_models import ROOT, mean_loss, sha256, cluster_interval
from scripts.train_round5 import REFERENCE, accuracy, accuracy_interval, predict_round5, selection_probability
from scripts.train_distillation_round7 import (R5_COMPARATOR, R5_LOSS, validate_rows,
    fold_masks, select_candidate, choose_procedure, predict_distillation)

STUDENT = 'previous_round7_market_student'
LOCKED = 'selected_graph_locked_pre2023'


def candidate_specs():
    return [dict(name=s['name']+('_r4blend0.5' if w == .5 else ''), graph=s['name'], weight=w)
            for s in graph_specs() for w in (1., .5)]


def graph_predictions(reference, graph, specs=None):
    specs = candidate_specs() if specs is None else specs
    reference = np.asarray(reference, dtype=float)
    if (reference.shape != (len(graph),) or not np.isfinite(reference).all()
            or np.any((reference < 0) | (reference > 1)) or graph.columns.duplicated().any()
            or not graph.market_graph_available.isin([0, 1]).all()):
        raise ValueError('Aligned valid reference and graph availability required')
    available = graph.market_graph_available.to_numpy(dtype=bool)
    output = {}
    for s in specs:
        if not 0 < s['weight'] <= 1:
            raise ValueError('Fixed graph blend weight must be in (0,1]')
        z = graph[s['graph']+'_diff'].to_numpy(dtype=float)
        if not np.isfinite(z).all():
            raise ValueError('Finite graph logits required')
        p = reference.copy()
        p[available] = (1-s['weight'])*reference[available]+s['weight']*expit(z[available])
        output[s['name']] = p
    return pd.DataFrame(output, index=graph.index)


def predict_market_graph(bundle, features):
    required = bundle['feature_names']
    if (features.empty or features.columns.duplicated().any() or not set(required).issubset(features)
            or not np.isfinite(features[required].to_numpy(dtype=float)).all()):
        raise ValueError('Finite complete frozen inference features required')
    x = features[bundle['previous_features']]
    old = predict_round5(bundle['previous_bundle'], x)
    output = graph_predictions(old[REFERENCE].to_numpy(), features, bundle['specs'])
    for alias, name in bundle['previous_aliases'].items():
        output[alias] = old[name].to_numpy()
    student = predict_distillation(bundle['student_bundle'], x)
    output[STUDENT] = student[bundle['student_name']].to_numpy()
    # This fixed diagnostic is never a selection candidate.
    chosen = bundle['frozen']['pooled_selection']['choice']
    if chosen == REFERENCE:
        output[LOCKED] = output[REFERENCE].to_numpy()
    else:
        spec = next(s for s in bundle['specs'] if s['name'] == chosen)
        locked = features[[c for c in features if c.startswith('locked_')]].rename(columns=lambda c:c[7:])
        output[LOCKED] = graph_predictions(output[REFERENCE].to_numpy(), locked, [spec])[chosen].to_numpy()
    return output


def write_report(out, summary, frozen, table, cv):
    chosen = frozen['pooled_selection']['choice']
    gate = frozen['procedure_gate']
    chosen_row = table.set_index('model').loc[chosen]
    lines = ['# Round 8：历史开盘赔率的动态实力图', '',
        '这是独立的“历史市场输入、T−21”条件：每场按供应方所报比赛UTC≤T−21筛选历史开盘赔率，'
        '不使用当前比赛赔率。2023 年较早比赛的历史报价允许更新图；R7 学生只使用 2021–2022 '
        '市场监督且推理不读市场，两者信息条件不同。全部结果仍为反复使用 2023 后的探索，2024–2025 未打开。', '',
        f"本分支推荐：`{frozen['recommendation']}`。2021H2 早选 `{gate['early_selection']['choice']}`；"
        f"2022 正确率增量 {100*gate['accuracy_improvement']:+.3f} 个百分点（门槛 +0.5），"
        f"log loss 增量 {gate['loss_increase']:+.6f}（上限 +0.005）。", '',
        f"早期+2022 pooled 冻结候选 `{chosen}`：{int(chosen_row.correct)}/2588，"
        f"准确率 {chosen_row.accuracy:.3%}，loss {chosen_row.mean_log_loss:.6f}。"
        f"距离 70% 所需 1812 场还差 {max(0,1812-int(chosen_row.correct))} 场。", '',
        '6 个图设置 × 原始概率/固定半权 R4 混合 = 12 个候选，另加可选 R4。'
        '半衰期 90/365/无限与是否采用同场地1、异场地0.25交叉。无限表示历史等权而非冻结更新。'
        '独立报告的 locked-pre2023 行只给 pooled 选择禁用2023历史边，不能参与选择。'
        'R5和R7旧模型仅为对照。这里没有新监督分类器拟合，也没有从2023选择参数。', '',
        '| 模型 | 正确数 | 准确率 | Log loss |', '| --- | ---: | ---: | ---: |']
    for r in table.itertuples():
        lines.append(f'| `{r.model}` | {r.correct} | {r.accuracy:.3%} | {r.mean_log_loss:.6f} |')
    lines += ['', f"2023 有 {summary['graph_available_2023']} 场具备同一过去图分量中的两名球员；"
        f"其余 {summary['exact_fallback_rows_2023']} 场精确回退 R4。历史边共 {summary['historical_edges']} 条，"
        '仅复用已审计的 outcome-blind Tour 身份链接。未删除任何目标或改动阈值。', '',
        '| 设置 | H2+2022 准确率 | H2+2022 loss |', '| --- | ---: | ---: |']
    for r in cv[cv.fold.eq('pooled')].itertuples():
        lines.append(f'| `{r.model}` | {r.accuracy:.3%} | {r.mean_log_loss:.6f} |')
    comparison = summary['comparisons_vs_r4'][chosen]
    ci = comparison['log_loss']['ci95']
    lines += ['', f"冻结候选相对R4的2023净正确数 {comparison['accuracy']['net_correct']:+d}；"
        f"loss降低 {comparison['log_loss']['mean_log_loss_reduction']:.6f}，配对日期块95%区间"
        f" [{ci[0]:.6f}, {ci[1]:.6f}]。区间未计入多轮搜索、重复门槛或全部球员相关性。", '',
        '方法改编自 Penn/Michael/Bhatt 的赔率加权图：统一BO3尺度，固定0.001正则的稀疏最小二乘，'
        '按目标局数还原。数据来自获准Valuebetennis/Pinnacle历史开盘单一来源；不复现论文原始数据、'
        '超参数或剔除未知球员的评价。提供方无逐条采集时间；其比赛UTC不是独立核实的实际开赛、完成或报价可得时间。补充审计发现轮次时间矛盾，但所有已用边同时满足已纠正赛事日期≤T−21；此次仅修正文案，预测和选择保持冻结。', '',
        '比较表的2023排序不改变冻结选择；此结果不能证明SOTA或独立测试达到70%。', '',
        '复跑：`.venv/bin/python -m scripts.train_market_graph_round8`。',
        '仅重载：同一命令追加 `--predict-features results/round8/market_graph/inference_features.csv`。',
        '研究协议：[round8_market_graph.md](../../../docs/research/round8_market_graph.md)。',
        '时间审计与限制：[freshness feasibility](../../../docs/research/round9_freshness_feasibility.md)。', '']
    (out/'report_summary.md').write_text('\n'.join(lines))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT/'results/round8/market_graph')
    parser.add_argument('--predict-features', type=Path)
    args = parser.parse_args()
    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    if args.predict_features:
        frame = pd.read_csv(args.predict_features)
        predict_market_graph(joblib.load(out/'frozen_models.joblib'), frame).to_csv(out/'inference_predictions.csv', index=False)
        return
    started = time.perf_counter()
    r5, d7 = ROOT/'results/round5', ROOT/'results/round7/distillation'
    audit = json.loads((r5/'artifact_audit.json').read_text())
    for name in ('development_features.csv', 'frozen_models.joblib', 'oof_predictions.csv', 'frozen_selection.json'):
        if sha256(r5/name) != audit['artifact_sha256'][name]:
            raise ValueError('Audited historical artifact changed: '+name)
    rows_path = ROOT/'results/round3/development_targets.csv'
    rows = pd.read_csv(rows_path)
    validate_rows(rows)
    x = pd.read_csv(r5/'development_features.csv')
    if len(rows) != len(x) or not x.format_best_of_five.isin([0, 1]).all():
        raise ValueError('Frozen format alignment changed')
    old_registry = json.loads((r5/'search_registry.json').read_text())
    # Round 7's audited target checksum independently binds these positional features.
    d7registry = json.loads((d7/'search_registry.json').read_text())
    if sha256(rows_path) != d7registry['target_sha256']:
        raise ValueError('Audited target order changed')
    targets = rows.drop(columns=['a_won']).assign(best_of=np.where(x.format_best_of_five, 5, 3))
    specs = candidate_specs()
    protocol = ROOT/'docs/research/round8_market_graph.md'
    registry = dict(round=8, experiment='historical_opening_market_graph', candidates=specs,
        graph_settings=graph_specs(), selectable_count=13, graph_setting_count=6,
        fixed_blends=6, new_binary_classifier_fits=0, source_years_read=[2021,2022,2023],
        information_regime='Past opening-market inputs filtered by provider-reported UTC at T-21',
        protocol_sha256=sha256(protocol), source_code_sha256=sha256(Path(__file__)),
        feature_code_sha256=sha256(ROOT/'scripts/market_graph_features.py'),
        target_sha256=sha256(rows_path), previous_artifact_sha256=audit['artifact_sha256'],
        source_manifest_sha256=sha256(ROOT/'data/market-manifest.json'),
        linkage_sha256=sha256(ROOT/'results/round5/market/linked_matches.csv'),
        report_only_comparators=[R5_COMPARATOR,R5_LOSS,STUDENT,LOCKED],
        source_outcomes_read=False, current_target_odds_used=False,
        cutoff='source UTC <= tournament UTC midnight - 21 days',
        decision_threshold=.5, exploratory_after_2023=True, final_test_evaluated=False)
    (out/'search_registry.json').write_text(json.dumps(registry,indent=2)+'\n')
    history = load_opening_history(targets)
    history.to_csv(out/'opening_history.csv',index=False)
    graph = build_market_graph_features(targets, history)
    locked = build_market_graph_features(targets, history, locked_before='2023-01-01')
    pd.concat([rows[['match_id']], graph],axis=1).to_csv(out/'graph_features.csv',index=False)
    features = pd.concat([x, graph, locked.add_prefix('locked_')],axis=1)
    features.to_csv(out/'inference_features.csv',index=False)
    print(f'Built {len(history)} historical edges and {len(rows)} feature rows in {time.perf_counter()-started:.2f}s',flush=True)
    masks = {f:fold_masks(rows,f)[1] for f in ('2021H2','2022','2023')}
    omask = masks['2021H2'] | masks['2022']
    orows = rows[omask]
    fold = np.where(orows.season.eq(2021),'2021H2','2022')
    y = orows.a_won.to_numpy()
    old_oof = pd.read_csv(r5/'oof_predictions.csv').set_index('match_id').loc[orows.match_id]
    if not np.array_equal(old_oof.a_won.to_numpy(),y) or not np.array_equal(old_oof.season,orows.season):
        raise ValueError('Honest OOF target labels/order mismatch')
    old_frozen = json.loads((r5/'frozen_selection.json').read_text())
    predictions = graph_predictions(old_oof[REFERENCE].to_numpy(), graph[omask], specs)
    predictions[REFERENCE] = old_oof[REFERENCE].to_numpy()
    predictions[R5_COMPARATOR] = selection_probability(old_frozen['pooled_selection'],
        {c:old_oof[c].to_numpy() for c in old_frozen['pooled_selection']['components']})
    predictions[R5_LOSS] = old_oof['all_inputs_spline_C0.01'].to_numpy()
    student_frozen = json.loads((d7/'frozen_selection.json').read_text())
    student_name = student_frozen['recommendation']
    student_oof = pd.read_csv(d7/'oof_predictions.csv').set_index('match_id').loc[orows.match_id]
    if not np.array_equal(student_oof.a_won.to_numpy(),y):
        raise ValueError('Student comparator OOF mismatch')
    predictions[STUDENT] = student_oof[student_name].to_numpy()
    values = {c:predictions[c].to_numpy() for c in predictions}
    cv = []
    for name,p in values.items():
        for f in ('2021H2','2022','pooled'):
            mask = np.ones(len(y),dtype=bool) if f == 'pooled' else fold == f
            cv.append(dict(model=name,fold=f,n=int(mask.sum()),accuracy=accuracy(y[mask],p[mask]),mean_log_loss=mean_loss(y[mask],p[mask])))
    cv = pd.DataFrame(cv)
    cv.to_csv(out/'temporal_cv.csv',index=False)
    gate = choose_procedure(y, values, specs, fold)
    gate['procedure'] = 'adopt_historical_market_graph' if gate['procedure'] == 'adopt_distillation' else 'retain_round4'
    pooled = select_candidate(y,values,specs,np.ones(len(y),dtype=bool))
    chosen = pooled['choice'] if gate['procedure'] == 'adopt_historical_market_graph' else REFERENCE
    frozen = dict(recommendation=chosen,pooled_selection=pooled,procedure_gate=gate,
        history_lag_days=21, decision_threshold=.5, binary_selection_end_season=2022,
        information_regime=registry['information_regime'], graph_updates_past_2023_prices=True,
        exploratory_after_2023=True, final_test_evaluated=False)
    # Freeze selection before any new 2023 metric or comparator probability is read.
    (out/'frozen_selection.json').write_text(json.dumps(frozen,indent=2)+'\n')
    print('Frozen:',json.dumps(frozen),flush=True)
    bundle = dict(specs=specs,graph_settings=graph_specs(),frozen=frozen,
        feature_names=list(features),previous_features=list(x),
        previous_bundle=joblib.load(r5/'frozen_models.joblib'),
        previous_aliases={REFERENCE:REFERENCE,R5_COMPARATOR:old_frozen['pooled_selection']['choice'],R5_LOSS:'all_inputs_spline_C0.01'},
        student_bundle=joblib.load(d7/'frozen_models.joblib'),student_name=student_name,
        history=history,source_registry=registry)
    joblib.dump(bundle,out/'frozen_models.joblib')
    vr = rows[masks['2023']]
    p = predict_market_graph(bundle,features.loc[vr.index])
    scores = []
    for name in p:
        score = score_probabilities(zip(vr.match_id,vr.a_won),zip(vr.match_id,p[name]))
        scores.append(dict(model=name,recommended=name==chosen,**score,
            brier=float(np.mean((vr.a_won.to_numpy()-p[name].to_numpy())**2))))
    table = pd.DataFrame(scores).sort_values(['accuracy','mean_log_loss'],ascending=[False,True])
    table.to_csv(out/'validation_comparison.csv',index=False)
    pd.concat([vr.reset_index(drop=True),p.reset_index(drop=True)],axis=1).to_csv(out/'validation_predictions.csv',index=False)
    pd.concat([orows[['match_id','season','a_won']].reset_index(drop=True).assign(fold=fold),predictions.reset_index(drop=True)],axis=1).to_csv(out/'oof_predictions.csv',index=False)
    coverage = rows[['season','surface']].assign(available=graph.market_graph_available).groupby(['season','surface']).agg(n=('available','size'),available=('available','sum')).reset_index()
    coverage.to_csv(out/'coverage.csv',index=False)
    summary = dict(recommendation=chosen,pooled_choice=pooled['choice'],validation_matches=len(vr),
        correct_required_for_70=1812,historical_edges=len(history),
        source_edges_by_year={str(y):int((history.source_season==y).sum()) for y in (2021,2022,2023)},
        graph_available_2023=int(graph.loc[vr.index,'market_graph_available'].sum()),
        exact_fallback_rows_2023=int((graph.loc[vr.index,'market_graph_available']==0).sum()),
        comparisons_vs_r4={c:dict(accuracy=accuracy_interval(vr,p[REFERENCE].to_numpy(),p[c].to_numpy()),
            log_loss=cluster_interval(vr,p[REFERENCE].to_numpy(),p[c].to_numpy())) for c in p},
        elapsed_seconds=time.perf_counter()-started,new_candidate_count=12,selectable_entries=13,
        prediction_columns=len(p.columns),graph_feature_columns=len(graph.columns),
        source_outcomes_read=False,source_years_read=[2021,2022,2023],
        exploratory_after_2023=True,final_test_evaluated=False)
    (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    write_report(out,summary,frozen,table,cv)
    print(table.to_string(index=False),flush=True)


if __name__ == '__main__':
    from scripts.train_market_graph_round8 import main as canonical_main
    with threadpool_limits(limits=4):
        canonical_main()
