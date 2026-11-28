# 第八轮市场残差：latest

9个拟合配置，每个3个固定残差强度；另保留原价格先验，共28列预测。
原始价格是Bernoulli logit offset，再学习价格情境、326项历史、加27项本人前轮信息的有界修正。
这些是比原T−21更晚的赛前信息条件。2023已重复使用，以下全部为探索结果；2024–2025仍封存。

2021H2早选的2022增量为-3/2528，即-0.119个百分点；
所需门槛+0.5个百分点未通过。分支保留`raw_market`。
pooled冻结探索`price_cat_d3_alpha1`得到**1734/2588=67.002%**、loss**0.590046**。
最高观察`price_linear_C0.03_alpha0.5`为**1758/2588=67.929%**、loss**0.587687**，距70%还差**54场**。
观察排序没有替代冻结选择，也不能据此声称独立达到泛化性能。

| 模型 | 正确 / 2588 | Accuracy | Log loss | 角色 |
| --- | ---: | ---: | ---: | --- |
| `price_linear_C0.03_alpha0.5` | 1758 | 67.929% | 0.587687 | 预设未选控制 |
| `price_linear_C0.003_alpha1` | 1757 | 67.890% | 0.587689 | 预设未选控制 |
| `price_linear_C0.003_alpha0.5` | 1755 | 67.813% | 0.587653 | 预设未选控制 |
| `price_linear_C0.003_alpha0.25` | 1755 | 67.813% | 0.587663 | 预设未选控制 |
| `price_linear_C0.03_alpha0.25` | 1755 | 67.813% | 0.587667 | 预设未选控制 |
| `price_linear_C0.03_alpha1` | 1755 | 67.813% | 0.587853 | 预设未选控制 |
| `history_linear_C0.03_alpha0.25` | 1753 | 67.736% | 0.587184 | 预设未选控制 |
| `price_cat_d3_alpha0.25` | 1753 | 67.736% | 0.587694 | 预设未选控制 |
| `raw_market` | 1752 | 67.697% | 0.587691 | 分支保留 |
| `history_linear_C0.003_alpha1` | 1751 | 67.658% | 0.588572 | 预设未选控制 |
| `history_linear_C0.03_alpha0.5` | 1750 | 67.620% | 0.587891 | 预设未选控制 |
| `event_linear_C0.03_alpha0.5` | 1750 | 67.620% | 0.588407 | 预设未选控制 |
| `event_cat_d3_alpha1` | 1750 | 67.620% | 0.591131 | 预设未选控制 |
| `event_linear_C0.03_alpha0.25` | 1749 | 67.581% | 0.587386 | 预设未选控制 |
| `price_cat_d3_alpha0.5` | 1747 | 67.504% | 0.588094 | 预设未选控制 |
| `history_cat_d3_alpha0.25` | 1747 | 67.504% | 0.588188 | 预设未选控制 |
| `event_cat_d3_alpha0.25` | 1747 | 67.504% | 0.588215 | 预设未选控制 |
| `history_linear_C0.03_alpha1` | 1747 | 67.504% | 0.592763 | 预设未选控制 |
| `history_cat_d3_alpha1` | 1746 | 67.465% | 0.591072 | 预设未选控制 |
| `history_linear_C0.003_alpha0.25` | 1745 | 67.427% | 0.587216 | 预设未选控制 |
| `event_linear_C0.003_alpha0.25` | 1744 | 67.388% | 0.587418 | 预设未选控制 |
| `event_linear_C0.003_alpha1` | 1744 | 67.388% | 0.589662 | 预设未选控制 |
| `history_linear_C0.003_alpha0.5` | 1742 | 67.311% | 0.587213 | 预设未选控制 |
| `event_linear_C0.003_alpha0.5` | 1742 | 67.311% | 0.587664 | 预设未选控制 |
| `event_cat_d3_alpha0.5` | 1740 | 67.233% | 0.588966 | 预设未选控制 |
| `history_cat_d3_alpha0.5` | 1739 | 67.195% | 0.588919 | 预设未选控制 |
| `price_cat_d3_alpha1` | 1734 | 67.002% | 0.590046 | pooled冻结探索 |
| `event_linear_C0.03_alpha1` | 1730 | 66.847% | 0.594228 | 预设未选控制 |

拟合使用4854条有合法价格且早于训练截止的2021–2022目标；每个评估模型都预测全部2588场。
2023价格有效2487场，缺失101场精确回退原R4，不能用可报价子集充当全量准确率。
latest条件优先closing，否则opening；2021closing全部缺失，因此训练与验证的报价信息质量随年代改变，不能当固定临场采集时点。
opening条件的输入严格不含closing或movement。所有来源都缺逐笔报价采集时间，prematch依据供应方描述。
“opening”指使用的价格字段，不是已核实的预测决策时点。event配置的本人前轮结果在目标比赛前已知，
但不能证明早于该场opening报价采集。因此应称“使用开盘价格的较晚赛前预测”，不能声称在开盘发布瞬间已可做出。

pooled探索相对相同quote条件原价格的准确率变化为-0.696个百分点，
条件日期块区间[-1.767,+0.347]。
区间没有覆盖全部搜索或跨日期球员依赖。每次信息增加均与相同learner、C和alpha比较，
完整72行逐期对照见[matched_information_controls.csv](matched_information_controls.csv)。

复跑：`.venv/bin/python -m scripts.train_market_residual_round8`，然后
`.venv/bin/python -m scripts.report_market_residual_round8`。仅重载：
`.venv/bin/python -m scripts.train_market_residual_round8 --condition latest --predict-features path/to/prepared_inputs.csv`。
输入构造函数保存准确R4 fallback及所需quote字段，CLI使用冻结schema；不含未来结果。
[协议](../../../../docs/research/round8_market_residual.md) · [冻结选择](frozen_selection.json) ·
[搜索记录](search_registry.json) · [独立审计](artifact_audit.json)
