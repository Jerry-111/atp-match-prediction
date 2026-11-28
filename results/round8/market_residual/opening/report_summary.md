# 第八轮市场残差：opening

9个拟合配置，每个3个固定残差强度；另保留原价格先验，共28列预测。
原始价格是Bernoulli logit offset，再学习价格情境、326项历史、加27项本人前轮信息的有界修正。
这些是比原T−21更晚的赛前信息条件。2023已重复使用，以下全部为探索结果；2024–2025仍封存。

2021H2早选的2022增量为-14/2528，即-0.554个百分点；
所需门槛+0.5个百分点未通过。分支保留`raw_market`。
pooled冻结探索`price_cat_d3_alpha1`得到**1733/2588=66.963%**、loss**0.591239**。
最高观察`history_linear_C0.03_alpha1`为**1760/2588=68.006%**、loss**0.596723**，距70%还差**52场**。
观察排序没有替代冻结选择，也不能据此声称独立达到泛化性能。

| 模型 | 正确 / 2588 | Accuracy | Log loss | 角色 |
| --- | ---: | ---: | ---: | --- |
| `history_linear_C0.03_alpha1` | 1760 | 68.006% | 0.596723 | 预设未选控制 |
| `event_linear_C0.03_alpha1` | 1748 | 67.543% | 0.598128 | 预设未选控制 |
| `history_linear_C0.03_alpha0.5` | 1742 | 67.311% | 0.591694 | 预设未选控制 |
| `history_linear_C0.003_alpha1` | 1738 | 67.156% | 0.592442 | 预设未选控制 |
| `event_linear_C0.03_alpha0.5` | 1737 | 67.117% | 0.592187 | 预设未选控制 |
| `history_linear_C0.003_alpha0.5` | 1736 | 67.079% | 0.591037 | 预设未选控制 |
| `event_linear_C0.003_alpha0.5` | 1734 | 67.002% | 0.591465 | 预设未选控制 |
| `history_linear_C0.03_alpha0.25` | 1733 | 66.963% | 0.590998 | 预设未选控制 |
| `price_cat_d3_alpha1` | 1733 | 66.963% | 0.591239 | pooled冻结探索 |
| `history_linear_C0.003_alpha0.25` | 1732 | 66.924% | 0.591059 | 预设未选控制 |
| `event_cat_d3_alpha0.25` | 1732 | 66.924% | 0.592029 | 预设未选控制 |
| `price_cat_d3_alpha0.25` | 1731 | 66.886% | 0.591303 | 预设未选控制 |
| `event_linear_C0.003_alpha0.25` | 1730 | 66.847% | 0.591251 | 预设未选控制 |
| `event_cat_d3_alpha0.5` | 1730 | 66.847% | 0.592706 | 预设未选控制 |
| `event_linear_C0.003_alpha1` | 1730 | 66.847% | 0.593472 | 预设未选控制 |
| `history_cat_d3_alpha0.25` | 1729 | 66.808% | 0.592069 | 预设未选控制 |
| `event_linear_C0.03_alpha0.25` | 1728 | 66.770% | 0.591189 | 预设未选控制 |
| `price_linear_C0.03_alpha0.5` | 1728 | 66.770% | 0.591484 | 预设未选控制 |
| `price_linear_C0.003_alpha1` | 1728 | 66.770% | 0.591493 | 预设未选控制 |
| `price_linear_C0.003_alpha0.5` | 1728 | 66.770% | 0.591495 | 预设未选控制 |
| `price_linear_C0.03_alpha0.25` | 1728 | 66.770% | 0.591511 | 预设未选控制 |
| `price_linear_C0.003_alpha0.25` | 1728 | 66.770% | 0.591528 | 预设未选控制 |
| `price_linear_C0.03_alpha1` | 1728 | 66.770% | 0.591563 | 预设未选控制 |
| `raw_market` | 1728 | 66.770% | 0.591583 | 分支保留 |
| `history_cat_d3_alpha0.5` | 1728 | 66.770% | 0.592812 | 预设未选控制 |
| `price_cat_d3_alpha0.5` | 1726 | 66.692% | 0.591157 | 预设未选控制 |
| `event_cat_d3_alpha1` | 1725 | 66.654% | 0.594742 | 预设未选控制 |
| `history_cat_d3_alpha1` | 1725 | 66.654% | 0.595055 | 预设未选控制 |

拟合使用4854条有合法价格且早于训练截止的2021–2022目标；每个评估模型都预测全部2588场。
2023价格有效2487场，缺失101场精确回退原R4，不能用可报价子集充当全量准确率。
latest条件优先closing，否则opening；2021closing全部缺失，因此训练与验证的报价信息质量随年代改变，不能当固定临场采集时点。
opening条件的输入严格不含closing或movement。所有来源都缺逐笔报价采集时间，prematch依据供应方描述。
“opening”指使用的价格字段，不是已核实的预测决策时点。event配置的本人前轮结果在目标比赛前已知，
但不能证明早于该场opening报价采集。因此应称“使用开盘价格的较晚赛前预测”，不能声称在开盘发布瞬间已可做出。

pooled探索相对相同quote条件原价格的准确率变化为+0.193个百分点，
条件日期块区间[-0.454,+0.827]。
区间没有覆盖全部搜索或跨日期球员依赖。每次信息增加均与相同learner、C和alpha比较，
完整72行逐期对照见[matched_information_controls.csv](matched_information_controls.csv)。

复跑：`.venv/bin/python -m scripts.train_market_residual_round8`，然后
`.venv/bin/python -m scripts.report_market_residual_round8`。仅重载：
`.venv/bin/python -m scripts.train_market_residual_round8 --condition opening --predict-features path/to/prepared_inputs.csv`。
输入构造函数保存准确R4 fallback及所需quote字段，CLI使用冻结schema；不含未来结果。
[协议](../../../../docs/research/round8_market_residual.md) · [冻结选择](frozen_selection.json) ·
[搜索记录](search_registry.json) · [独立审计](artifact_audit.json)
