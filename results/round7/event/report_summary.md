# 第七轮赛事进程分支：门槛未通过

这一分支允许加入两名球员在**本届赛事已完成、更早的本人淘汰赛轮次**，属于比原 T−21 更晚的信息条件。
27 项新输入与原历史特征组合，共 12 个拟合配置；线性和样条分别有 base/context/load/performance/combined 五组，另有两个 CatBoost 深度。
所有模型覆盖相同 2,588 场；其中 **1,445** 场有合格本人早轮历史，**1,143** 场严格回退到原 R4 概率。
同一回退也应用于不含新增信息的 base 控制，因此不把回退差异误当信息效果。

2020–2021 选择的流程在 2022 比 R4 多判对 **7 / 2,528** 场，
准确率变化 **+0.277 个百分点**，未达 +0.5 个百分点门槛；loss 护栏通过。
分支继续保留 `previous_probability_recommendation`。pooled 2020–2022 冻结探索组合 `accuracy_grid_blend` 在 2023 为
**1693 / 2,588 = 65.42%**、loss **0.607325**。
新单模型的最高观察准确率是 `event_load_linear`（与 load_spline 同正确数）：**1702 / 2,588**、loss **0.606564**。
这个事后排序不改变预先冻结选择；也没有超过旧 R5 探索组合的 1,715 场正确。

## 全部同队列结果

| 比较及角色 | 正确数 / 2,588 | Accuracy | Log loss ↓ | 距 70% 额外正确数 |
| --- | ---: | ---: | ---: | ---: |
| `previous_round5_exploration`（对照/观察） | 1,715 | 66.27% | 0.609871 | 97 |
| `previous_round5_best_loss`（对照/观察） | 1,706 | 65.92% | 0.606453 | 106 |
| `event_load_linear`（对照/观察） | 1,702 | 65.77% | 0.606564 | 110 |
| `event_load_spline`（对照/观察） | 1,702 | 65.77% | 0.607061 | 110 |
| `event_combined_spline`（对照/观察） | 1,698 | 65.61% | 0.607288 | 114 |
| `event_performance_spline`（对照/观察） | 1,697 | 65.57% | 0.607541 | 115 |
| `event_base_spline`（对照/观察） | 1,696 | 65.53% | 0.608863 | 116 |
| `previous_probability_recommendation`（分支保留） | 1,696 | 65.53% | 0.609407 | 116 |
| `event_base_linear`（对照/观察） | 1,695 | 65.49% | 0.608341 | 117 |
| `event_performance_linear`（对照/观察） | 1,693 | 65.42% | 0.607113 | 119 |
| `accuracy_grid_blend`（pooled 冻结探索） | 1,693 | 65.42% | 0.607325 | 119 |
| `event_combined_linear`（对照/观察） | 1,692 | 65.38% | 0.606860 | 120 |
| `event_context_linear`（对照/观察） | 1,691 | 65.34% | 0.608370 | 121 |
| `event_context_spline`（对照/观察） | 1,690 | 65.30% | 0.608827 | 122 |
| `event_all_cat_d4`（对照/观察） | 1,687 | 65.19% | 0.610083 | 125 |
| `event_all_cat_d6`（对照/观察） | 1,684 | 65.07% | 0.609589 | 128 |
| `rank_reference`（对照/观察） | 1,640 | 63.37% | 0.629796 | 172 |

## 匹配信息对照

context 包含已完成轮次/轮次情境；load 在它之上加入局数、盘数、时长等负荷；performance 加入本人更早轮次的表现；combined 同时加入两者。
下表每格是两种固定学习器的描述，不能把两个高度相关对照当独立试验。开发期和 2023 观察分开。

| 相同学习器的加入信息对照 | pooled CV 改善学习器数 | 开发折平均准确率变化 / pp | 开发折平均 loss 降低 | 2023 改善学习器数 | 2023 平均准确率变化 / pp |
| --- | ---: | ---: | ---: | ---: | ---: |
| `base_to_context` | 1/2 | -0.008 | -0.000425 | 0/2 | -0.193 |
| `context_to_load` | 2/2 | +0.204 | +0.000139 | 2/2 | +0.444 |
| `context_to_performance` | 2/2 | +0.196 | +0.000431 | 2/2 | +0.174 |
| `context_to_combined` | 2/2 | +0.098 | -0.000001 | 2/2 | +0.174 |

[完整逐年与 pooled 对照](matched_information_controls.csv)保留 40 行；positive accuracy_gain / log_loss_reduction 表示新信息更好。
CatBoost 同时改变学习器和输入，不能单独解释为赛事进程信息的增益。

## 复跑、来源与限制

在前序轮次就绪后运行 `.venv/bin/python -m scripts.train_event_round7`，再运行
`.venv/bin/python -m scripts.report_round7` 生成报告；报告器只读已冻结概率，不训练或改选模型。
原始历史的既有重复在拟合前按稳定键和原行内容核验处理，保留目标和标签；本人早轮次来源可逐项追溯。
不能从赛事开始日期推断未知实际比赛时间；轮次顺序仅支持严格更早的本人淘汰赛记录，排除同轮、未来轮及循环赛。
所有 2023 结果属于复用样本上的探索，2022 门槛也已复用，不是独立确认。2024–2025 继续封存。

[冻结选择](frozen_selection.json) · [独立审计](artifact_audit.json) ·
[协议](../../../docs/research/round7_event.md) · [来源审计](../../../docs/research/round7_event_source_audit.md) ·
[整轮总览](../overview.md)
