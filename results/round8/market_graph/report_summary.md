# Round 8：历史开盘赔率的动态实力图

这是独立的“历史市场输入、T−21”条件：每场只使用截止日前已发生比赛的开盘赔率，不使用当前比赛赔率。2023 年较早比赛的历史报价允许更新图；R7 学生只使用 2021–2022 市场监督且推理不读市场，两者信息条件不同。全部结果仍为反复使用 2023 后的探索，2024–2025 未打开。

本分支推荐：`previous_probability_recommendation`。2021H2 早选 `market_graph_h90_surface1_r4blend0.5`；2022 正确率增量 +0.237 个百分点（门槛 +0.5），log loss 增量 +0.001088（上限 +0.005）。

早期+2022 pooled 冻结候选 `market_graph_h90_surface1_r4blend0.5`：1715/2588，准确率 66.267%，loss 0.607183。距离 70% 所需 1812 场还差 97 场。

6 个图设置 × 原始概率/固定半权 R4 混合 = 12 个候选，另加可选 R4。半衰期 90/365/无限与是否采用同场地1、异场地0.25交叉。无限表示历史等权而非冻结更新。独立报告的 locked-pre2023 行只给 pooled 选择禁用2023历史边，不能参与选择。R5和R7旧模型仅为对照。这里没有新监督分类器拟合，也没有从2023选择参数。

| 模型 | 正确数 | 准确率 | Log loss |
| --- | ---: | ---: | ---: |
| `market_graph_h90_surface0_r4blend0.5` | 1726 | 66.692% | 0.608426 |
| `market_graph_hinf_surface0_r4blend0.5` | 1724 | 66.615% | 0.609873 |
| `market_graph_h365_surface0_r4blend0.5` | 1723 | 66.577% | 0.608960 |
| `market_graph_h365_surface1_r4blend0.5` | 1717 | 66.345% | 0.607834 |
| `selected_graph_locked_pre2023` | 1716 | 66.306% | 0.611554 |
| `market_graph_h90_surface1_r4blend0.5` | 1715 | 66.267% | 0.607183 |
| `market_graph_hinf_surface1_r4blend0.5` | 1715 | 66.267% | 0.608729 |
| `previous_round5_exploration` | 1715 | 66.267% | 0.609871 |
| `previous_round7_market_student` | 1712 | 66.151% | 0.604599 |
| `previous_round5_best_loss` | 1706 | 65.920% | 0.606453 |
| `market_graph_h365_surface1` | 1701 | 65.726% | 0.616606 |
| `previous_probability_recommendation` | 1696 | 65.533% | 0.609407 |
| `market_graph_h90_surface1` | 1689 | 65.263% | 0.613323 |
| `market_graph_h365_surface0` | 1689 | 65.263% | 0.619983 |
| `market_graph_hinf_surface1` | 1680 | 64.915% | 0.620346 |
| `market_graph_h90_surface0` | 1671 | 64.567% | 0.617280 |
| `market_graph_hinf_surface0` | 1663 | 64.258% | 0.623669 |

2023 有 2412 场具备同一过去图分量中的两名球员；其余 176 场精确回退 R4。历史边共 7341 条，仅复用已审计的 outcome-blind Tour 身份链接。未删除任何目标或改动阈值。

| 设置 | H2+2022 准确率 | H2+2022 loss |
| --- | ---: | ---: |
| `market_graph_h90_surface0` | 65.455% | 0.613812 |
| `market_graph_h90_surface0_r4blend0.5` | 66.735% | 0.602655 |
| `market_graph_h90_surface1` | 65.659% | 0.610545 |
| `market_graph_h90_surface1_r4blend0.5` | 66.880% | 0.601571 |
| `market_graph_h365_surface0` | 65.252% | 0.616562 |
| `market_graph_h365_surface0_r4blend0.5` | 66.531% | 0.603797 |
| `market_graph_h365_surface1` | 65.804% | 0.612401 |
| `market_graph_h365_surface1_r4blend0.5` | 66.444% | 0.602282 |
| `market_graph_hinf_surface0` | 64.786% | 0.618441 |
| `market_graph_hinf_surface0_r4blend0.5` | 66.386% | 0.604371 |
| `market_graph_hinf_surface1` | 65.775% | 0.614081 |
| `market_graph_hinf_surface1_r4blend0.5` | 66.386% | 0.602725 |
| `previous_probability_recommendation` | 66.386% | 0.600284 |
| `previous_round5_exploration` | 67.083% | 0.595721 |
| `previous_round5_best_loss` | 66.676% | 0.595978 |
| `previous_round7_market_student` | 67.113% | 0.595979 |

冻结候选相对R4的2023净正确数 +19；loss降低 0.002224，配对日期块95%区间 [-0.000522, 0.004891]。区间未计入多轮搜索、重复门槛或全部球员相关性。

方法改编自 Penn/Michael/Bhatt 的赔率加权图：统一BO3尺度，固定0.001正则的稀疏最小二乘，按目标局数还原。数据来自获准Valuebetennis/Pinnacle历史开盘单一来源；不复现论文原始数据、超参数或剔除未知球员的评价。提供方无逐条采集时间，实际比赛UTC只是在其赛前声明下的保守上界。

比较表的2023排序不改变冻结选择；此结果不能证明SOTA或独立测试达到70%。

复跑：`.venv/bin/python -m scripts.train_market_graph_round8`。
仅重载：同一命令追加 `--predict-features results/round8/market_graph/inference_features.csv`。
研究协议：[round8_market_graph.md](../../../docs/research/round8_market_graph.md)。
