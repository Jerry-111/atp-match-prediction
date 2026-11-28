# 第六轮主实验：遗漏历史、排名轨迹与关键分

34项新配置，383项输入；仍用33,833条拟合目标与完整2,588场2023验证目标。所有2023结果属于反复使用样本上的探索，2024–2025最终测试未打开。没有市场价格输入，也没有放宽T−21历史截止。

## 开发期选择及2023结果

- 2022流程门槛：原参考准确率67.1282%，早期选中流程67.5237%，增幅+0.396个百分点，要求至少0.5个百分点且loss护栏通过。结论：`retain_round4`。
- 正式推荐：`previous_probability_recommendation`。pooled开发折选择的探索比较：`accuracy_grid_blend`，2023正确**1693/2588 = 65.42%**，log loss **0.60774**。
- 相对第五轮冻结探索组合：纠正64场，同时新错86场，净增-22场；准确率差的条件日期块区间为[-1.679, -0.040]个百分点。该区间未覆盖全部搜索不确定性。
- 本表最高观察准确率`previous_round5_exploration`为66.27%；最低观察loss`previous_round5_best_loss`为0.60645。这些事后排序不改写冻结选择。
- 70%需要正确1,812场，探索比较仍差119场。

| 模型 | 正确数/2588 | Accuracy | Log loss ↓ | Brier ↓ |
| --- | ---: | ---: | ---: | ---: |
| `previous_round5_exploration` | 1715 | 66.27% | 0.60987 | 0.21166 |
| `previous_round5_best_loss` | 1706 | 65.92% | 0.60645 | 0.21032 |
| `r6_base_spline_C0.01` | 1704 | 65.84% | 0.60751 | 0.21077 |
| `r6_all_spline_C0.01` | 1702 | 65.77% | 0.60800 | 0.21103 |
| `r6_trajectory_linear_C0.003` | 1698 | 65.61% | 0.60745 | 0.21089 |
| `r6_all_linear_C0.01` | 1698 | 65.61% | 0.60797 | 0.21108 |
| `r6_trajectory_spline_C0.003` | 1697 | 65.57% | 0.60724 | 0.21075 |
| `previous_probability_recommendation` | 1696 | 65.53% | 0.60941 | 0.21149 |
| `accuracy_grid_blend` | 1693 | 65.42% | 0.60774 | 0.21094 |
| `r6_combined_spline_C0.003` | 1692 | 65.38% | 0.60844 | 0.21126 |
| `r6_base_linear_C0.003` | 1691 | 65.34% | 0.60758 | 0.21088 |
| `r6_combined_cat_d6` | 1690 | 65.30% | 0.61319 | 0.21313 |
| `r6_combined_linear_C0.003` | 1689 | 65.26% | 0.60863 | 0.21139 |
| `r6_extended_linear_C0.003` | 1688 | 65.22% | 0.60794 | 0.21101 |
| `r6_extended_spline_C0.003` | 1687 | 65.19% | 0.60777 | 0.21090 |
| `r6_breakpoint_spline_C0.003` | 1684 | 65.07% | 0.60828 | 0.21116 |
| `r6_breakpoint_linear_C0.003` | 1683 | 65.03% | 0.60845 | 0.21127 |
| `rank_reference` | 1640 | 63.37% | 0.62980 | 0.22016 |

## 匹配信息对照

每块在相同学习器、C及half-life下与275项旧核心特征比较，计6个匹配对照。下表是pooled2020–2022结果的描述均值，不能当独立样本做显著性检验。完整各年差值在`matched_information_controls.csv`。

| 加入信息 | Accuracy改善的对照数 | 平均Accuracy变化（百分点） | 平均loss降低 |
| --- | ---: | ---: | ---: |
| breakpoint | 1/6 | -0.057 | -0.00023 |
| combined | 1/6 | -0.073 | +0.00050 |
| extended | 1/6 | -0.185 | -0.00032 |
| trajectory | 4/6 | +0.125 | +0.00099 |

新增历史评级包含相同算法的旧历史控制；允许赛事与排除项详见来源审计。排名轨迹只用2006以来、每场T−21以前的可见记录，同日冲突快照不使用，不能称官方生涯最高排名。BP/保发残差固定收缩，并不被解释成心理能力或因果效应。

## 复跑与限制

先复现前五轮，然后运行`python -m scripts.train_round6`及`python -m scripts.report_round6`。本报告只读取已保存结果，不改变模型。使用`python -m scripts.train_round6 --predict-features path/to/features.csv`重载冻结模型，需要全部383项冻结输入。大特征表和本地joblib按固定来源重建，不提交Git。

跨级别监督迁移是单独的匹配实验，见[迁移报告](transfer/report_summary.md)；跨来源标签审计见[独立记录](source_audit.md)。既有市场结果来自更晚信息条件，不与本轮历史算法增益混为一谈。[实验协议](../../docs/research/round6_protocol.md) · [机器选择记录](frozen_selection.json) · [独立重载审计](artifact_audit.json)。
