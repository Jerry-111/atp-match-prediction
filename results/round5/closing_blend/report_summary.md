# 追加探索：固定收盘报价融合

在看到第五轮纯历史与开盘分支的2023结果后，追加16个固定控制。只用2022选择权重，但2022/2023都已反复使用，**没有独立验证，也不升级现有推荐**。使用实际赛前最后报价是更晚的信息条件，不属于原T−21模型的纯算法提升。

历史底模预先限于R4推荐与R5 pooled探索性组合。概率/logit融合仅取市场权重0.25/0.5/0.75，另保留每个底模及纯市场基准；没有重新拟合参数或改变0.5阈值。缺报价行精确使用相应历史底模。完整2588场中，2477场有closing报价，111场回退。

2022选中的控制为`round5_exploratory_logit_closing0.5`。2023准确率 **67.27%**（**1741/2588**），log loss **0.59464**；达到70%还差 **71** 场。全部控制的最高观察准确率为`round4_closing`：**67.70%**，不据此更改冻结选择。

| 控制 | 正确数 | 准确率 | Log loss |
| --- | ---: | ---: | ---: |
| `round4_closing` | 1752 | 67.70% | 0.58772 |
| `round5_exploratory_closing` | 1750 | 67.62% | 0.58840 |
| `round4_logit_closing0.75` | 1748 | 67.54% | 0.58984 |
| `round4_probability_closing0.75` | 1747 | 67.50% | 0.58992 |
| `round5_exploratory_logit_closing0.75` | 1746 | 67.47% | 0.59039 |
| `round5_exploratory_probability_closing0.75` | 1745 | 67.43% | 0.59037 |
| `round5_exploratory_probability_closing0.5` | 1741 | 67.27% | 0.59450 |
| `round5_exploratory_logit_closing0.5` | 1741 | 67.27% | 0.59464 |
| `round4_logit_closing0.5` | 1734 | 67.00% | 0.59415 |
| `round4_probability_closing0.5` | 1734 | 67.00% | 0.59421 |
| `round4_probability_closing0.25` | 1724 | 66.62% | 0.60064 |
| `round4_logit_closing0.25` | 1724 | 66.62% | 0.60067 |
| `round5_exploratory_probability_closing0.25` | 1719 | 66.42% | 0.60091 |
| `round5_exploratory_logit_closing0.25` | 1719 | 66.42% | 0.60114 |
| `round5_exploratory_history` | 1715 | 66.27% | 0.60987 |
| `round4_history` | 1696 | 65.53% | 0.60941 |

来源：Valuebetennis，按[CC BY4.0公开数据](https://www.valuebetennis.com/en/donnees.htm)及其赛前报价定义使用；缺少逐条报价采集时间的限制不变。该对照不能证明优于市场或达到独立测试的70%。配对区间见`summary.json`，不覆盖反复搜索偏差。

复跑：`python -m scripts.closing_blend_round5`。需先完成第五轮主实验和开盘市场实验；此前产物不会覆盖。
