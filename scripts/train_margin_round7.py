"""Performance-supervised soft targets with matched binary/smoothing controls.

10 fixed linear configurations; training-only performance labels; honest inner
temporal calibration; identical T-21 inference features and full Tour cohort.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn.linear_model import LogisticRegression
from threadpoolctl import threadpool_limits

from scripts.advanced_features import _date
from scripts.margin_targets import build_performance_targets
from scripts.metrics import score_probabilities
from scripts.multilevel_features import _season_date
from scripts.round5_learning import calibration_masks, positive_calibration_slope
from scripts.train_models import ROOT, FOLDS, mean_loss, sha256, temporal_masks
from scripts.train_round5 import accuracy, accuracy_interval, predict_round5
from scripts.train_transfer_round6 import verify_tour_targets

REFERENCE = 'round4_reference'
R5_COMPARATOR = 'round5_pooled_comparator'
ALPHA = .5
KINDS = ('binary', 'uniform_points', 'uniform_games', 'points', 'games')


def candidate_specs(columns):
    odd = [c for c in columns if c.endswith('_diff')]
    if not odd or len(odd) != len(set(odd)):
        raise ValueError('Distinct odd inference features required')
    return [dict(name=f'margin_{kind}_C{c:g}', kind=kind, C=c, alpha=0. if kind == 'binary' else ALPHA,
                 half_life=3., columns=odd) for c in (.01, .1) for kind in KINDS]


def soft_targets(y, performance, kind, alpha=ALPHA):
    y = np.asarray(y, dtype=float)
    if kind not in KINDS or not 0 <= alpha <= 1 or y.shape != (len(performance),) or not np.isin(y, (0, 1)).all():
        raise ValueError('Known objective and aligned binary target required')
    if kind == 'binary':
        return y.copy()
    source = 'point' if kind in ('points', 'uniform_points') else 'game'
    available = performance[f'{source}_available'].to_numpy()
    if not np.isin(available, (0, 1)).all():
        raise ValueError('Availability must be binary')
    observed = performance[f'{source}_share'].to_numpy(dtype=float)
    if not np.isfinite(observed[available == 1]).all() or np.any((observed[available == 1] < 0)|(observed[available == 1] > 1)):
        raise ValueError('Available performance targets must be valid fractions')
    anchor = np.full(len(y), .5) if kind.startswith('uniform_') else observed
    # Every configuration retains every row. Its matched smoothing control has
    # exactly the same hard-label fallback mask as its performance counterpart.
    return np.where(available == 1, (1-alpha)*y+alpha*anchor, y)


class CalibratedPerformanceClassifier:
    """Identical antisymmetric logistic heads, differing only in soft target."""
    def __init__(self, spec):
        self.spec = spec

    def _input(self, x):
        names = self.spec['columns']
        if x.empty or x.columns.duplicated().any() or not set(names).issubset(x):
            raise ValueError('Missing, duplicate or empty features')
        if any(not n.endswith('_diff') for n in names):
            raise ValueError('No-intercept learner accepts only odd difference features')
        values = x[names].to_numpy(dtype=float)
        if not np.isfinite(values).all():
            raise ValueError('Finite inference features required')
        return values

    def _fit_head(self, x, rows, performance):
        values = self._input(x)
        seasons = rows.season.to_numpy(dtype=float)
        weights = 2.**(-(seasons.max()-seasons)/self.spec['half_life'])
        weights *= len(weights)/weights.sum()
        scale = np.sqrt(np.average(values*values, weights=weights, axis=0))
        scale[scale < 1e-10] = 1.
        scaled = np.clip(values/scale, -8., 8.)
        q = soft_targets(rows.a_won, performance, self.spec['kind'], self.spec['alpha'])
        # Two weighted Bernoulli observations realize the exact fractional
        # cross entropy. Total weight is invariant to the target construction.
        xx = np.concatenate([scaled, scaled])
        yy = np.concatenate([np.ones(len(q)), np.zeros(len(q))])
        ww = np.concatenate([weights*q, weights*(1-q)])
        estimator = LogisticRegression(C=self.spec['C'], fit_intercept=False, solver='lbfgs',
                                       tol=1e-8, max_iter=2000, random_state=158)
        estimator.fit(xx, yy, sample_weight=ww)
        if estimator.n_iter_[0] >= estimator.max_iter:
            raise ValueError('Performance logistic optimizer did not converge')
        return scale, estimator

    @staticmethod
    def _margin(values, scale, estimator):
        return estimator.decision_function(np.clip(values/scale, -8., 8.))

    def fit(self, x, rows, performance):
        if not x.index.equals(rows.index) or not x.index.equals(performance.index):
            raise ValueError('Fit features, outcomes and performance labels must align')
        if rows.empty or not rows.season.isin(range(2009, 2023)).all() or not rows.a_won.isin((0, 1)).all():
            raise ValueError('Model fitting requires training-only 2009--2022 outcomes')
        for day, season in zip(rows.tourney_date, rows.season):
            day = _date(day)
            _season_date(day, int(season))
            if day.year > 2022:
                raise ValueError('Training dates must precede 2023')
        early, later, metadata = calibration_masks(rows.season.to_numpy(), rows.tourney_date.to_numpy())
        scale, estimator = self._fit_head(x.iloc[early], rows.iloc[early], performance.iloc[early])
        logits = self._margin(self._input(x.iloc[later]), scale, estimator)
        self.calibration_slope_ = positive_calibration_slope(logits, rows.iloc[later].a_won.to_numpy())
        self.calibration_metadata_ = dict(metadata, slope=self.calibration_slope_,
            source='Earlier-only model predicts last outer-training season; no in-sample calibration',
            final_refit='Refit all outer training rows; calibration transport is an assumption')
        self.scale_, self.estimator_ = self._fit_head(x, rows, performance)
        self.training_metadata_ = dict(rows=len(rows), last_season=int(rows.season.max()),
            first_date=str(rows.tourney_date.min()), last_date=str(rows.tourney_date.max()),
            total_weight=float(len(rows)), point_missing=int(performance.point_available.eq(0).sum()),
            game_missing=int(performance.game_available.eq(0).sum()))
        return self

    def predict_proba(self, x):
        logits = self._margin(self._input(x), self.scale_, self.estimator_)
        return np.clip(expit(self.calibration_slope_*logits), 1e-8, 1-1e-8)


def add_blends(predictions, specs):
    out = {n: np.asarray(v) for n, v in predictions.items()}
    for spec in specs:
        name = spec['name']
        out[name+'_r4blend0.5'] = .5*out[name]+.5*out[REFERENCE]
    return out


def select_candidate(y, predictions, mask):
    y, mask = np.asarray(y), np.asarray(mask, dtype=bool)
    if y.shape != mask.shape or not mask.any() or REFERENCE not in predictions:
        raise ValueError('Aligned nonempty historical selection required')
    limit = mean_loss(y[mask], predictions[REFERENCE][mask])+.005
    allowed = [n for n, p in predictions.items() if n != R5_COMPARATOR and mean_loss(y[mask], p[mask]) <= limit+1e-12]
    name = min(allowed, key=lambda n: (-accuracy(y[mask], predictions[n][mask]),
                                     mean_loss(y[mask], predictions[n][mask]), n))
    return dict(choice=name, accuracy=accuracy(y[mask], predictions[name][mask]),
                mean_log_loss=mean_loss(y[mask], predictions[name][mask]),
                selection_rows=int(mask.sum()), eligible_models=len(allowed), loss_limit=limit)


def choose_procedure(y, predictions, seasons):
    y, seasons = np.asarray(y), np.asarray(seasons)
    if not np.isin(seasons, FOLDS).all() or not set(FOLDS).issubset(seasons):
        raise ValueError('Only 2020--2022 OOF decisions allowed')
    early = select_candidate(y, predictions, np.isin(seasons, (2020, 2021)))
    mask = seasons == 2022
    ref, candidate = predictions[REFERENCE][mask], predictions[early['choice']][mask]
    gain = accuracy(y[mask], candidate)-accuracy(y[mask], ref)
    loss_change = mean_loss(y[mask], candidate)-mean_loss(y[mask], ref)
    passed = gain >= .005-1e-12 and loss_change <= .005+1e-12
    return dict(procedure='adopt_margin' if passed else 'retain_round4', early_selection=early,
                reference=REFERENCE, selection_seasons=[2020, 2021], gate_season=2022,
                accuracy_improvement=gain, loss_increase=loss_change,
                reference_2022_accuracy=accuracy(y[mask], ref), candidate_2022_accuracy=accuracy(y[mask], candidate),
                reference_2022_loss=mean_loss(y[mask], ref), candidate_2022_loss=mean_loss(y[mask], candidate),
                required_accuracy_improvement=.005, allowed_loss_increase=.005)


def predict_margin(bundle, features):
    if features.columns.duplicated().any() or not set(bundle['feature_names']).issubset(features):
        raise ValueError('Frozen inference schema mismatch')
    x = features[bundle['feature_names']]
    old = predict_round5(bundle['previous_bundle'], x)
    p = {REFERENCE: old.previous_probability_recommendation.to_numpy(),
         R5_COMPARATOR: old[bundle['previous_pooled_choice']].to_numpy()}
    p.update({name: model.predict_proba(x) for name, model in bundle['models'].items()})
    p = add_blends(p, bundle['specs'])
    if any(v.shape != (len(x),) or not np.isfinite(v).all() or np.any((v < 0)|(v > 1)) for v in p.values()):
        raise ValueError('Invalid frozen margin probability')
    return pd.DataFrame(p, index=features.index)


def score_table(rows, predictions):
    records = []
    for name, p in predictions.items():
        records.append(dict(model=name, **score_probabilities(zip(rows.match_id, rows.a_won), zip(rows.match_id, p)),
                            brier=float(np.mean((rows.a_won.to_numpy()-np.asarray(p))**2))))
    return pd.DataFrame(records).sort_values(['accuracy', 'mean_log_loss'], ascending=[False, True])


def write_report(out, frozen, summary, comparison, matched):
    table = comparison.set_index('model')
    lines = ['# 第七轮：把点／局表现用作监督目标', '',
        '本分支沿用同一批 2,588 场 2023 比赛及 T−21 输入。2023 已反复查看，所有结果均为探索性；2024–2025 最终测试未打开。', '',
        f"正式推荐：`{frozen['recommendation']}`。2020–2021 选择的方法在 2022 的准确率增量为 "
        f"{100*frozen['procedure_gate']['accuracy_improvement']:.3f} 个百分点，门槛为 0.5 个百分点。", '',
        '所有模型使用相同 188 个历史差值输入和线性容量。混合目标为一半胜负、一半真实点／局比例；'
        '均匀平滑对照把后一项替换为 0.5，并保持对应缺失回退完全相同。每种方法都用训练期内向前预测拟合正斜率校准。', '',
        '| 固定比较 | 正确 / 2588 | 准确率 | Log loss |', '| --- | ---: | ---: | ---: |']
    names = list(dict.fromkeys([REFERENCE, R5_COMPARATOR, frozen['pooled_selection']['choice']]+
        [s['name'] for s in summary['candidate_specs']]))
    for name in names:
        r = table.loc[name]
        lines.append(f'| {name} | {int(r.correct)} | {r.accuracy:.3%} | {r.mean_log_loss:.6f} |')
    lines += ['', f"pooled 2020–2022 冻结选择为 `{frozen['pooled_selection']['choice']}`，在新 2023 评分前写入文件；"
              '若升级门槛未通过，该选择只作探索性比较，不替换正式推荐。', '',
        f"训练监督共 {summary['training_rows']} 行；点统计有效 {summary['target_metadata']['point_valid']}，"
        f"局比例有效 {summary['target_metadata']['game_valid']}。缺失点/局统计回退胜负标签，任何模型都不删训练行；"
        '从未读取 2023 目标比分或点统计。', '',
        '匹配向前折对照（负值也保留）：', '',
        '| 表现目标 | 对照 | Pooled accuracy 增量 pp | Pooled log-loss 改善 |',
        '| --- | --- | ---: | ---: |']
    for r in matched[matched.fold.astype(str).eq('pooled')].itertuples():
        lines.append(f'| {r.model} | {r.control} | {100*r.accuracy_improvement:+.3f} | {r.log_loss_improvement:+.6f} |')
    lines += ['', '对比 binary 衡量整体目标改变；对比各自 uniform 控制，才更接近区分真实点/局信息与单纯平滑。'
        '两者都不是因果效应；统计相关、校准从较小内层模型迁移到最终拟合、固定线性容量均可能影响结果。'
        '正校准斜率不会改变单模型 0.5 分类，能影响混合概率及 loss。', '',
        '论文依据及明确改编见 `docs/research/round7_margin.md`。既有文献不证明本实现能达到 70%，不声称 SOTA。'
        '日期分块区间仅描述这些固定预测的条件不确定性，不涵盖反复搜索。', '',
        '复跑：`.venv/bin/python -m scripts.train_margin_round7`。仅重载：'
        '`.venv/bin/python -m scripts.train_margin_round7 --predict-features results/round5/development_features.csv`。', '']
    (out/'report_summary.md').write_text('\n'.join(lines))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT/'results/round7/margin')
    parser.add_argument('--predict-features', type=Path)
    args = parser.parse_args()
    out, previous = args.output, ROOT/'results/round5'
    out.mkdir(parents=True, exist_ok=True)
    if args.predict_features:
        x = pd.read_csv(args.predict_features)
        p = predict_margin(joblib.load(out/'frozen_models.joblib'), x)
        if 'match_id' in x:
            p.insert(0, 'match_id', x.match_id.to_numpy())
        p.to_csv(out/'inference_predictions.csv', index=False)
        return
    audit = json.loads((previous/'artifact_audit.json').read_text())
    for name in ('development_features.csv', 'frozen_models.joblib', 'oof_predictions.csv', 'search_registry.json'):
        if sha256(previous/name) != audit['artifact_sha256'][name]:
            raise ValueError(f'Previous audited artifact changed: {name}')
    old_registry = json.loads((previous/'search_registry.json').read_text())
    inventory_path = ROOT/'data/processed/mens_singles_2006_2025.csv'
    if sha256(inventory_path) != old_registry['inventory_sha256']:
        raise ValueError('Pinned inventory changed')
    rows = pd.read_csv(ROOT/'results/round3/development_targets.csv')
    inventory = pd.read_csv(inventory_path, dtype={'a_won': 'string'})
    inventory = inventory[inventory.season.le(2023)].copy()
    verify_tour_targets(rows, inventory)
    x = pd.read_csv(previous/'development_features.csv')
    if x.shape != (len(rows), 326):
        raise ValueError('Audited inference feature matrix changed')
    training = rows.season.le(2022)
    performance = build_performance_targets(rows[training])
    specs = candidate_specs(x.columns)
    old_oof = pd.read_csv(previous/'oof_predictions.csv')
    orows = rows[rows.season.isin(FOLDS)]
    if not orows[['match_id', 'season', 'a_won']].reset_index(drop=True).equals(old_oof[['match_id', 'season', 'a_won']]):
        raise ValueError('Prior OOF identity changed')
    old_frozen = json.loads((previous/'frozen_selection.json').read_text())
    old_selection = old_frozen['pooled_selection']
    oof = {REFERENCE: old_oof.previous_probability_recommendation.to_numpy(),
           R5_COMPARATOR: sum(w*old_oof[n].to_numpy() for n, w in zip(old_selection['components'], old_selection['weights']))}
    folds = {}
    for year in (*FOLDS, 2023):
        tr, va = temporal_masks(rows, year)
        early, later, meta = calibration_masks(rows.loc[tr, 'season'].to_numpy(), rows.loc[tr, 'tourney_date'].to_numpy())
        folds[str(year)] = dict(training_rows=int(tr.sum()), validation_rows=int(va.sum()),
            latest_training_date=str(rows.loc[tr, 'tourney_date'].max()), first_validation_date=str(rows.loc[va, 'tourney_date'].min()),
            inner_calibration=meta)
    registry = dict(round=7, branch='performance_targets', fit_candidates=10, candidates=specs,
        selection_entries=21, deterministic_blends=dict(model_fraction=.5, reference=REFERENCE),
        feature_names=list(x), inference_features=326, fitted_odd_features=len(specs[0]['columns']),
        feature_sha256=sha256(previous/'development_features.csv'), inventory_sha256=sha256(inventory_path),
        target_sha256=sha256(ROOT/'results/round3/development_targets.csv'),
        protocol_sha256=sha256(ROOT/'docs/research/round7_margin.md'),
        source_code_sha256=sha256(Path(__file__)), target_loader_sha256=sha256(ROOT/'scripts/margin_targets.py'),
        target_metadata=performance.attrs, folds=folds, decision_threshold=.5,
        final_test_evaluated=False, read_2023_target_stats=False)
    (out/'search_registry.json').write_text(json.dumps(registry, indent=2)+'\n')
    pd.concat([rows.loc[training, ['match_id', 'season', 'a_won']], performance], axis=1).to_csv(out/'training_targets.csv', index=False)
    y, calibration_records = orows.a_won.to_numpy(), []
    for i, spec in enumerate(specs, 1):
        values = pd.Series(index=orows.index, dtype=float)
        for year in FOLDS:
            tr, va = temporal_masks(rows, year)
            model = CalibratedPerformanceClassifier(spec).fit(x[tr], rows[tr], performance.loc[rows.index[tr]])
            values.loc[rows.index[va]] = model.predict_proba(x[va])
            calibration_records.append(dict(model=spec['name'], outer_year=year, **model.calibration_metadata_))
        oof[spec['name']] = values.to_numpy()
        print(f'[{i}/10] {spec["name"]}: accuracy={accuracy(y, values.to_numpy()):.5f}, loss={mean_loss(y, values.to_numpy()):.6f}', flush=True)
    oof = add_blends(oof, specs)
    gate = choose_procedure(y, oof, orows.season)
    pooled = select_candidate(y, oof, np.ones(len(y), dtype=bool))
    recommendation = pooled['choice'] if gate['procedure'] == 'adopt_margin' else REFERENCE
    frozen = dict(round=7, branch='performance_targets', recommendation=recommendation,
        procedure_gate=gate, pooled_selection=pooled, decision_threshold=.5, history_lag_days=21,
        fit_end_season=2022, exploratory_after_2023=True, final_test_evaluated=False, read_2023_target_stats=False)
    (out/'frozen_selection.json').write_text(json.dumps(frozen, indent=2)+'\n')
    print('Frozen selection:', json.dumps(frozen), flush=True)
    cv = score_table(orows, oof)
    for year in FOLDS:
        mask = orows.season.eq(year).to_numpy()
        cv[f'{year}_accuracy'] = cv.model.map({n: accuracy(y[mask], p[mask]) for n, p in oof.items()})
        cv[f'{year}_log_loss'] = cv.model.map({n: mean_loss(y[mask], p[mask]) for n, p in oof.items()})
    cv.to_csv(out/'temporal_cv.csv', index=False)
    pd.concat([orows[['match_id', 'season', 'a_won']].reset_index(drop=True), pd.DataFrame(oof)], axis=1).to_csv(out/'oof_predictions.csv', index=False)
    matched = []
    for spec in specs:
        if spec['kind'] not in ('points', 'games'):
            continue
        for control_kind in ('binary', 'uniform_'+spec['kind']):
            control = next(s['name'] for s in specs if s['kind'] == control_kind and s['C'] == spec['C'])
            for year in (*FOLDS, 'pooled'):
                mask = np.ones(len(y), dtype=bool) if year == 'pooled' else orows.season.eq(year).to_numpy()
                matched.append(dict(model=spec['name'], control=control, fold=year,
                    accuracy_improvement=accuracy(y[mask], oof[spec['name']][mask])-accuracy(y[mask], oof[control][mask]),
                    log_loss_improvement=mean_loss(y[mask], oof[control][mask])-mean_loss(y[mask], oof[spec['name']][mask])))
    matched = pd.DataFrame(matched)
    matched.to_csv(out/'matched_objective_controls.csv', index=False)
    tr, va = temporal_masks(rows, 2023)
    models = {}
    for spec in specs:
        model = CalibratedPerformanceClassifier(spec).fit(x[tr], rows[tr], performance.loc[rows.index[tr]])
        models[spec['name']] = model
        calibration_records.append(dict(model=spec['name'], outer_year=2023, **model.calibration_metadata_))
    (out/'calibration_metadata.json').write_text(json.dumps(calibration_records, indent=2)+'\n')
    bundle = dict(models=models, feature_names=list(x), specs=specs, frozen=frozen,
                  previous_bundle=joblib.load(previous/'frozen_models.joblib'), previous_pooled_choice=old_selection['choice'])
    joblib.dump(bundle, out/'frozen_models.joblib')
    vr = rows[va]
    p = predict_margin(bundle, x[va])
    table = score_table(vr, {n: p[n].to_numpy() for n in p})
    table['recommended'] = table.model.eq(recommendation)
    table.to_csv(out/'validation_comparison.csv', index=False)
    metadata_columns = ['match_id', 'season', 'tourney_date', 'surface', 'player_a_id', 'player_b_id', 'rank_a', 'rank_b', 'a_won']
    pd.concat([vr[metadata_columns].reset_index(drop=True), p.reset_index(drop=True)], axis=1).to_csv(out/'validation_predictions.csv', index=False)
    summary = dict(round=7, recommendation=recommendation, pooled_choice=pooled['choice'],
        fit_candidates=10, selection_entries=21, candidate_specs=specs,
        training_rows=int(tr.sum()), validation_matches=len(vr), correct_required_for_70=int(np.ceil(.7*len(vr))),
        target_metadata=performance.attrs, feature_matrix='results/round5/development_features.csv',
        comparison_vs_round4=accuracy_interval(vr, p[REFERENCE].to_numpy(), p[pooled['choice']].to_numpy()),
        exploratory_after_2023=True, final_test_evaluated=False, read_2023_target_stats=False)
    (out/'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    write_report(out, frozen, summary, table, matched)
    print(table.to_string(index=False), flush=True)


if __name__ == '__main__':
    from scripts.train_margin_round7 import main as canonical_main
    with threadpool_limits(limits=4):
        canonical_main()
