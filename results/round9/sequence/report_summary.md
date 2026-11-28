# Round 9：球员历史赛事序列

全2588场2023、固定0.5阈值和原T−21历史信息规则保持不变；2024–2025未读取。这是反复使用2023后的探索结果，不能证明SOTA。每人最近16个不同历史赛事日期组，同日比赛先聚合，四年窗口内共22字段；技术统计缺失和零历史均保留。

本分支推荐 `previous_probability_recommendation`。2020–2021早选 `sequence_gru_r4blend0.75`；2022准确率变化 -0.158 个百分点（门槛+0.5），loss变化 -0.001594（最多+0.005）。

pooled冻结候选 `sequence_gru`：1703/2588，65.804%，loss 0.608496。距离70%所需1812场还差 109 场。

3个结构（mean、GRU、固定打乱GRU）×2固定种子，按概率平均；从不选择较好的种子。每个结构另有3个预先固定R4混合：3raw+9mix+R4=13可选项，混合不增加训练。共24个返回的seed模型（18个OOF、6个最终），各自先进行一次严格内层时间尾段epoch选择，合计48个优化阶段。所有结构接收相同326静态字段、时间字段和mask。

| 模型 | 正确数 | 准确率 | Log loss |
| --- | ---: | ---: | ---: |
| `previous_round5_exploration` | 1715 | 66.267% | 0.609871 |
| `previous_round7_market_student` | 1712 | 66.151% | 0.604599 |
| `previous_round5_best_loss` | 1706 | 65.920% | 0.606453 |
| `sequence_mean` | 1704 | 65.842% | 0.606122 |
| `sequence_gru` | 1703 | 65.804% | 0.608496 |
| `sequence_mean_r4blend0.75` | 1696 | 65.533% | 0.605994 |
| `previous_probability_recommendation` | 1696 | 65.533% | 0.609407 |
| `sequence_gru_r4blend0.75` | 1695 | 65.495% | 0.607430 |
| `sequence_mean_r4blend0.25` | 1695 | 65.495% | 0.607622 |
| `sequence_gru_r4blend0.5` | 1692 | 65.379% | 0.607258 |
| `sequence_shuffled_gru` | 1691 | 65.340% | 0.607870 |
| `sequence_mean_r4blend0.5` | 1687 | 65.185% | 0.606492 |
| `sequence_shuffled_gru_r4blend0.25` | 1687 | 65.185% | 0.607927 |
| `sequence_shuffled_gru_r4blend0.5` | 1686 | 65.147% | 0.607161 |
| `sequence_gru_r4blend0.25` | 1686 | 65.147% | 0.607926 |
| `sequence_shuffled_gru_r4blend0.75` | 1685 | 65.108% | 0.607128 |
| `rank_reference` | 1640 | 63.369% | 0.629796 |

| 原始编码器对照 | Pooled CV准确率变化 | Pooled CV loss变化 | 2023准确率变化 | 2023 loss变化 |
| --- | ---: | ---: | ---: | ---: |
| sequence_gru − sequence_mean | +0.065pp | -0.000864 | -0.039pp | +0.002374 |
| sequence_gru − sequence_shuffled_gru | +0.522pp | -0.002097 | +0.464pp | +0.000627 |
| sequence_shuffled_gru − sequence_mean | -0.457pp | +0.001233 | -0.502pp | +0.001747 |

冻结候选比R4净多判对 +7 场；loss降低 0.000911，条件日期块95%区间 [-0.002932, 0.004905]。区间不覆盖反复搜索、重复门槛或全部球员相关性。

2023共有 98 次球员出场无可用序列，编码为零向量并继续使用同一静态输入，没有删除目标。技术统计不完整的历史比赛仍保留其胜负、排名和场地。

mean池化同样看到时间字段，因此GRU差异涉及归纳偏置与容量；固定shuffle可被足够强的模型学会逆转，它不等于彻底删除时间信息。公开通用序列论文或其他筛选样本的网球准确率不能替代本实验结果。

复跑：`.venv-foundation/bin/python -m scripts.train_sequence_round9 --device cpu`。
重载：同一命令追加 `--predict-features results/round5/development_features.csv --predict-sequences results/round9/sequence/sequence_features.npz`。这些全开发集推理输出不是早年OOF；真正OOF及18个相应fold模型单独保存。
协议：[round9_sequence.md](../../../docs/research/round9_sequence.md)。
