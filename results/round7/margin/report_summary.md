# 第七轮：把点／局表现用作监督目标

本分支沿用同一批 2,588 场 2023 比赛及 T−21 输入。2023 已反复查看，所有结果均为探索性；2024–2025 最终测试未打开。

正式推荐：`round4_reference`。2020–2021 选择的方法在 2022 的准确率增量为 0.119 个百分点，门槛为 0.5 个百分点。

所有模型使用相同 188 个历史差值输入和线性容量。混合目标为一半胜负、一半真实点／局比例；均匀平滑对照把后一项替换为 0.5，并保持对应缺失回退完全相同。每种方法都用训练期内向前预测拟合正斜率校准。

| 固定比较 | 正确 / 2588 | 准确率 | Log loss |
| --- | ---: | ---: | ---: |
| round4_reference | 1696 | 65.533% | 0.609407 |
| round5_pooled_comparator | 1715 | 66.267% | 0.609871 |
| margin_points_C0.01_r4blend0.5 | 1693 | 65.417% | 0.607165 |
| margin_binary_C0.01 | 1701 | 65.726% | 0.607128 |
| margin_uniform_points_C0.01 | 1706 | 65.920% | 0.608432 |
| margin_uniform_games_C0.01 | 1705 | 65.881% | 0.608435 |
| margin_points_C0.01 | 1702 | 65.765% | 0.607718 |
| margin_games_C0.01 | 1701 | 65.726% | 0.607161 |
| margin_binary_C0.1 | 1708 | 65.997% | 0.607313 |
| margin_uniform_points_C0.1 | 1709 | 66.036% | 0.608309 |
| margin_uniform_games_C0.1 | 1711 | 66.113% | 0.608292 |
| margin_points_C0.1 | 1714 | 66.229% | 0.607572 |
| margin_games_C0.1 | 1712 | 66.151% | 0.607001 |

pooled 2020–2022 冻结选择为 `margin_points_C0.01_r4blend0.5`，在新 2023 评分前写入文件；若升级门槛未通过，该选择只作探索性比较，不替换正式推荐。

训练监督共 33833 行；点统计有效 33796，局比例有效 33833。缺失点/局统计回退胜负标签，任何模型都不删训练行；从未读取 2023 目标比分或点统计。

匹配向前折对照（负值也保留）：

| 表现目标 | 对照 | Pooled accuracy 增量 pp | Pooled log-loss 改善 |
| --- | --- | ---: | ---: |
| margin_points_C0.01 | margin_binary_C0.01 | +0.049 | -0.000296 |
| margin_points_C0.01 | margin_uniform_points_C0.01 | +0.245 | +0.000353 |
| margin_games_C0.01 | margin_binary_C0.01 | +0.033 | -0.000124 |
| margin_games_C0.01 | margin_uniform_games_C0.01 | +0.147 | +0.000503 |
| margin_points_C0.1 | margin_binary_C0.1 | +0.179 | -0.000613 |
| margin_points_C0.1 | margin_uniform_points_C0.1 | +0.245 | +0.000457 |
| margin_games_C0.1 | margin_binary_C0.1 | +0.261 | -0.000330 |
| margin_games_C0.1 | margin_uniform_games_C0.1 | +0.277 | +0.000720 |

对比 binary 衡量整体目标改变；对比各自 uniform 控制，才更接近区分真实点/局信息与单纯平滑。两者都不是因果效应；统计相关、校准从较小内层模型迁移到最终拟合、固定线性容量均可能影响结果。正校准斜率不会改变单模型 0.5 分类，能影响混合概率及 loss。

论文依据及明确改编见 `docs/research/round7_margin.md`。既有文献不证明本实现能达到 70%，不声称 SOTA。日期分块区间仅描述这些固定预测的条件不确定性，不涵盖反复搜索。

复跑：`.venv/bin/python -m scripts.train_margin_round7`。仅重载：`.venv/bin/python -m scripts.train_margin_round7 --predict-features results/round5/development_features.csv`。
