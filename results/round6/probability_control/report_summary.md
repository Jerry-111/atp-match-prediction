# 追加概率目标对照

本方向在看到第六轮accuracy选择结果后提出，明确为自适应探索；不升级正式推荐，不是独立验证。在既有34项配置中按pooled2020–2022最低log loss选出`r6_trajectory_spline_C0.01`，CV loss=0.597430，没有新增参数搜索。

该模型2023正确1710/2588，准确率**66.07%**，log loss **0.607552**，Brier **0.210844**。70%仍需额外102场正确。概率改善与分类准确率分开判断；配对区间见`summary.json`，不涵盖全部选择不确定性。

| 比较 | 正确数 | Accuracy | Log loss | Brier |
| --- | ---: | ---: | ---: | ---: |
| minimum_cv_loss_control | 1710 | 66.07% | 0.607552 | 0.210844 |
| previous_probability_recommendation | 1696 | 65.53% | 0.609407 | 0.211493 |
| previous_round5_exploration | 1715 | 66.27% | 0.609871 | 0.211658 |
| previous_round5_best_loss | 1706 | 65.92% | 0.606453 | 0.210323 |

复跑：`python -m scripts.probability_control_round6`，需先完成主实验。2024–2025最终测试未打开。
