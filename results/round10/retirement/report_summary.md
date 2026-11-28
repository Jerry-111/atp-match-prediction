# Round 10：历史 RET 状态与普通停赛对照

来源仍是部分核验：30 条预先固定样本中 10 条官方确认、20 条未确认；已确认中
未发现 RET 方向矛盾，但年代偏早。协调者接受此限制，授权三个有限配置。
研究对象是提供方记录的历史非完赛状态，不是全面核验的伤病数据；没有按官方
资料可得性筛选历史记录。W/O 和 DEF 分开，不加入 RET，也不增加训练败局。

同一完整 2023 年 2,588 场，阈值 0.5、T−21 历史不变。2023 已反复查看，以下
比较均为探索性；2024–2025 未解封，不宣称 SOTA 或预设能达到 70%。

本分支冻结推荐 **previous_probability_recommendation**：1696/2,588，
accuracy 65.5332%，log loss 0.609407。
2022 gate：**retain_round4**，accuracy 变化 +0.1978%，
loss 变化 -0.004451；门槛为
accuracy ≥+0.5pp 且 loss ≤+0.005。
2020–2022 pooled 候选 **ret_I_spline**：
1704/2,588、65.8423%、loss 0.606294。
2023 表中事后最高行 **previous_round5_exploration**：1715/2,588；
这不改变冻结推荐。70% 需要 1,812 场，最高行仍差 97 场。

三个固定 spline 配置、每个三次 forward CV 和一次最终 refit，共 12 fits；
没有混合或阈值搜索。I 的六个普通 inactivity 字段在三组都保留；R 加四个
历史 RET recency/presence 字段，RC 加四个 recency/后续出场衰减字段。
候选矩阵维数分别 332/336/336；I 是嵌套对照，R/RC 同容量。联合存储矩阵338列。
新增统计均以日期分组，不虚构同日顺序。零历史行保留；重拟合模型在零新状态
时不保证与 R4 概率完全相同。所有比例/缩放只能在当期训练数据拟合。

## 同批 2023 比较

| model | accuracy | mean_log_loss | recommended |
| --- | --- | --- | --- |
| previous_round5_exploration | 0.662674 | 0.609871 | False |
| ret_RC_spline | 0.659969 | 0.605707 | False |
| ret_R_spline | 0.659583 | 0.605856 | False |
| previous_round5_best_loss | 0.659196 | 0.606453 | False |
| ret_I_spline | 0.658423 | 0.606294 | False |
| previous_probability_recommendation | 0.655332 | 0.609407 | True |
| rank_reference | 0.633694 | 0.629796 | False |

## 配对机制对照

差值为候选减对照；accuracy 正、loss 负表示更好。R−I/RC−I 包含新增维数，
R−RC 才是同容量机制对照，均不解释为 RET 的因果效应。

| candidate | control | oof_accuracy_change | oof_loss_change | validation_accuracy_change | validation_loss_change |
| --- | --- | --- | --- | --- | --- |
| ret_R_spline | ret_I_spline | -0.000326 | -0.000157 | 0.001159 | -0.000439 |
| ret_RC_spline | ret_I_spline | -0.000163 | -0.000075 | 0.001546 | -0.000587 |
| ret_RC_spline | ret_R_spline | 0.000163 | 0.000083 | 0.000386 | -0.000148 |

完整 CV 在 temporal_cv.csv，来源/覆盖在 search_registry.json 和
coverage_by_season.csv，按赛事日期块的配对区间在 summary.json。区间没有多重
搜索校正；重复使用 2022 gate 也不是独立确认。

[精确协议](../../../docs/research/round10_retirement_protocol.md) ·
[来源审计](source_feasibility.json) · [固定样本](sample_registry.json)
