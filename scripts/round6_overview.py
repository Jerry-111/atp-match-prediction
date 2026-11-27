"""Read-only unified Round 6 summary; preserves distinct information regimes."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.train_models import ROOT


def read_json(path):
    return json.loads(path.read_text())


def main():
    out = ROOT / 'results/round6'
    main_frozen = read_json(out / 'frozen_selection.json')
    transfer_frozen = read_json(out / 'transfer/frozen_selection.json')
    main_summary = read_json(out / 'summary.json')
    transfer_summary = read_json(out / 'transfer/summary.json')
    probability = read_json(out / 'probability_control/summary.json')
    source = read_json(out / 'source_audit.json')
    registry = read_json(out / 'transfer/search_registry.json')
    main_table = pd.read_csv(out / 'validation_comparison.csv').set_index('model')
    transfer_table = pd.read_csv(out / 'transfer/validation_comparison.csv').set_index('model')
    probability_table = pd.read_csv(out / 'probability_control/validation_comparison.csv').set_index('model')
    records = []

    def add(table, model, role, selection, regime='原 T−21 历史信息'):
        row = table.loc[model]
        if int(row.matches) != 2588 or not np.isclose(row.accuracy, row.correct / row.matches):
            raise ValueError('Comparison changed cohort or metric denominator')
        records.append(dict(information_regime=regime, role=role, model=model, selection=selection,
            matches=int(row.matches), correct=int(row.correct), accuracy=float(row.accuracy),
            mean_log_loss=float(row.mean_log_loss), brier=float(row.brier),
            additional_correct_for_70=int(np.ceil(.7 * row.matches) - row.correct)))

    add(main_table, 'previous_probability_recommendation', '第四轮正式推荐；继续保留', 'formal')
    add(main_table, 'previous_round5_exploration', '第五轮冻结探索组合', 'previous_pooled_cv')
    add(main_table, 'previous_round5_best_loss', '第五轮最低观察 loss 对照', 'previous_observed_loss')
    add(main_table, main_frozen['pooled_selection']['choice'], '第六轮主实验探索组合', 'pooled_cv_accuracy')
    add(transfer_table, transfer_frozen['pooled_selection']['choice'], '第六轮迁移分支探索选择', 'pooled_cv_accuracy')
    add(probability_table, 'minimum_cv_loss_control', '第六轮追加概率目标对照', 'adaptive_pooled_cv_loss')
    historical_count = len(records)
    old_overview = pd.read_csv(ROOT / 'results/round5/overview.csv').set_index('model')
    for name, role, regime in (
        ('opening_market', '既有原始开盘市场＋缺失回退', '另加赛前开盘报价'),
        ('context_calibrated_C0.01', '既有开盘融合模型', '另加赛前开盘报价'),
        ('round5_exploratory_logit_closing0.5', '既有追加自适应收盘融合', '另加赛前收盘报价'),
        ('closing_market_descriptive', '既有原始收盘市场＋缺失回退', '另加赛前收盘报价'),
    ):
        add(old_overview, name, role, 'previous_later_information', regime)
    table = pd.DataFrame(records)
    table.to_csv(out / 'overview.csv', index=False)
    if main_summary['new_candidates'] != 34 or transfer_summary['fit_candidates'] != 12 or len(registry['blends']) != 36:
        raise ValueError('Recorded Round 6 search size changed')
    if main_frozen['procedure_gate']['procedure'] != 'retain_round4' or transfer_frozen['procedure_gate']['procedure'] != 'retain_round4':
        raise ValueError('Overview conclusion requires both recorded gates to retain Round 4')
    # Identity/labels must match across the separate reports, beyond equal counts.
    identity = None
    for directory in (out, out / 'transfer', out / 'probability_control'):
        rows = pd.read_csv(directory / 'validation_predictions.csv', usecols=['match_id', 'a_won'])
        if rows.match_id.duplicated().any() or len(rows) != 2588:
            raise ValueError('Comparison contains duplicate or missing targets')
        if identity is not None and not identity.equals(rows):
            raise ValueError('Comparison target order/labels differ')
        identity = rows
    matched = pd.read_csv(out / 'transfer/matched_transfer_controls.csv')
    pooled = matched[matched.fold.eq('pooled')]
    good_accuracy = int(pooled.accuracy_improvement.gt(0).sum())
    worse_loss = int(pooled.log_loss_improvement.lt(0).sum())
    tci = transfer_summary['pooled_comparison_vs_round4']['ci95']
    gate_main, gate_transfer = main_frozen['procedure_gate'], transfer_frozen['procedure_gate']
    source_2023 = next(s for s in source['seasons'] if s['season'] == 2023)
    lines = [
        '# 第六轮总览：70% 尚未达到，保留原推荐', '',
        '本轮没有超越第五轮已保存的最高探索准确率 **1,715 / 2,588 = 66.27%**，'
        '也没有超越其最低观察 log loss **0.60645**。两条预设升级流程均未过门槛，'
        '正式推荐仍为第四轮年龄增强模型。2024–2025 最终测试继续封存；所有新增 2023 结果属于探索。', '',
        '新增工作包括补充被旧历史遗漏的合格比赛、历史排名轨迹、破发点/保发残差，以及跨级别监督迁移。'
        '主实验 **34** 个拟合配置，迁移分支 **12** 个拟合配置和 **36** 个固定融合对照，'
        '合计 **82 项显式新配置/对照**；主实验另有预设的有限非负混合选权。'
        '后续概率对照只额外最终拟合已有配置，不算第 83 个新配置。', '',
        '| 同一完整 2023 队列 | 正确数 / 2,588 | Accuracy | Log loss ↓ | 距 70% 所需额外正确数 |',
        '| --- | ---: | ---: | ---: | ---: |',
    ]
    for row in table.iloc[:historical_count].itertuples():
        lines.append(f'| {row.role} | {row.correct:,} | {100*row.accuracy:.2f}% | {row.mean_log_loss:.5f} | {row.additional_correct_for_70} |')
    lines += ['',
        f"主实验在 2020–2021 选定流程后，2022 比原参考多判对 **10 / 2,528** 场，"
        f"即 **{100*gate_main['accuracy_improvement']:.3f} 个百分点**，未达 0.5 个百分点门槛。"
        '其 pooled 2020–2022 探索组合在 2023 为 1,693 场正确，比第五轮探索组合少 22 场。'
        '排名轨迹在 6 个匹配历史对照中有 4 个提升准确率，是本轮较有希望的信息；'
        '扩充历史和关键分残差没有稳定准确率收益。', '',
        f"迁移分支加入 **75,261** 条 Challenger/资格赛监督目标，使非零辅助权重配置的合格拟合池达到 **109,094** 行；"
        '冻结选择的辅助权重为 0，实际仍仅用 **33,833** 行主巡回赛目标。'
        '实验使用 48 项相同特征、相同正则化的仅主巡回赛模型作对照。'
        f'9 个非零辅助权重配置中，仅 {good_accuracy} 个提高 pooled CV 准确率，{worse_loss} 个都恶化 log loss。'
        '冻结的早期选择和 pooled 选择均采用辅助权重 0；数据量增加本身没有得到支持。'
        f"其 2022 门槛增益为 **{100*gate_transfer['accuracy_improvement']:.3f} 个百分点**；"
        '2023 pooled 探索选择净多对 6 场，条件日期块增益区间 '
        f'**[{100*tci[0]:+.3f}, {100*tci[1]:+.3f}] 个百分点**，跨过零。', '',
        f"追加概率目标对照选择 `{probability['selected']}`，规则是已有 34 项配置中 pooled CV log loss 最低。"
        '这项选择是在看完本轮准确率结果后新增的，明确标为自适应探索，不升级正式推荐。'
        '其 1,710 场正确比旧准确率探索组合少 5 场；loss 0.60755 也高于旧概率对照 0.60645。', '',
        '达到 70% 至少需要 1,812 场正确；当前最高历史探索结果仍差 **97 场**。'
        '该差距以固定标签、完整队列和相同 T−21 信息条件计算。', '',
        '## 既有市场分支：较晚信息条件，单列保留', '',
        '下表是第五轮既有结果，本轮没有重选市场模型。报价时间条件不同，不能并入上表解释历史特征收益。'
        '缺失价格使用原模型回退，因此每行仍覆盖 2,588 场。', '',
        '| 既有对照 | 正确数 / 2,588 | Accuracy | Log loss ↓ |',
        '| --- | ---: | ---: | ---: |',
    ]
    for row in table.iloc[historical_count:].itertuples():
        lines.append(f'| {row.role} | {row.correct:,} | {100*row.accuracy:.2f}% | {row.mean_log_loss:.5f} |')
    lines += ['',
        '## 数据核验与复跑', '',
        f"独立来源审计匹配了 2023 年 **{source_2023['matched']} / 2,588** 场，其胜者记录全部一致；"
        f"剩余 **{source_2023['unmatched']}** 场无法匹配，并非已被证明无误。2022 的一项来源分歧经官方签表核验支持原标签。"
        '没有更改标签、匹配关系或队列。来源一致也不是模型的独立验证，仍可能存在共同上游误差。', '',
        '复跑前五轮后，按顺序执行：', '',
        '```bash',
        'python -m scripts.train_round6',
        'python -m scripts.report_round6',
        'python -m scripts.train_transfer_round6',
        'python -m scripts.probability_control_round6',
        'python -m scripts.round6_overview',
        '```', '',
        '这些结果来自反复使用的 2023 队列和已复用的 2022 门槛。日期块区间没有覆盖完整搜索不确定性或所有球员依赖，'
        '因此不构成最终测试优越性或 SOTA 证据。保留负结果、冻结模型和原推荐，下一步仍应由最终评估者在封存季检验。', '',
        '[主报告](report_summary.md) · [迁移报告](transfer/report_summary.md) · '
        '[追加概率对照](probability_control/summary.json) · [来源审计](source_audit.md) · '
        '[主实验重载审计](artifact_audit.json) · [实验协议](../../docs/research/round6_protocol.md) · '
        '[迁移协议与论文证据](../../docs/research/round6_transfer.md)', '',
    ]
    (out / 'overview.md').write_text('\n'.join(lines))
    print(table.to_string(index=False))


if __name__ == '__main__':
    main()
