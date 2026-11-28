# 第八轮：结构性新方法仍未达到70%

**goal继续保持 active；本轮没有宣告成功。** 70%要求同一完整2,588场至少判对1,812场。
当前市场修正菜单中的最高观察行是 **1,760/2,588 = 68.01%**，还差52场；它未被历史选择规则选中，
log loss为0.596723，差于原始开盘价约0.59158，不能把它改写成冻结推荐或SOTA。
2023已经反复用于探索；2022门槛也已复用。2024–2025最终测试继续封存。

## 同一队列的对比

| 信息条件 / 比较及角色 | 正确数 / 2,588 | 准确率 | Log loss ↓ | 距70%正确数 |
| --- | ---: | ---: | ---: | ---: |
| 原历史输入 / 旧基准：排名 logistic 原基准（`rank_reference`） | 1,640 | 63.37% | 0.629796 | 172 |
| 原历史输入 / 旧基准：R4：保留的纯历史概率推荐（`previous_probability_recommendation`） | 1,696 | 65.53% | 0.609407 | 116 |
| 原历史输入 / 旧基准：R5：冻结准确率探索组合（`previous_round5_exploration`） | 1,715 | 66.27% | 0.609871 | 97 |
| 原历史输入 / 旧基准：整体 Elo 原基准（开发期选 K=32）（`raw_elo_k32_diff`） | 1,640 | 63.37% | 0.639146 | 172 |
| 历史输入；训练期市场监督：R7：市场监督学生（`market_lambda1`） | 1,712 | 66.15% | 0.604599 | 100 |
| T−21历史点统计：开发期 pooled 冻结探索（`joint_recent_r4blend0.25`） | 1,689 | 65.26% | 0.609938 | 123 |
| T−21历史及往年赛事环境：开发期 pooled 冻结探索（`venue_strength_linear`） | 1,703 | 65.80% | 0.607656 | 109 |
| T−21历史市场输入：开发期 pooled 冻结探索（`market_graph_h90_surface1_r4blend0.5`） | 1,715 | 66.27% | 0.607183 | 97 |
| 当前开盘价；部分候选另含本人早轮次：固定对照 / 既有模型；分支保留推荐（`raw_market`） | 1,728 | 66.77% | 0.591583 | 84 |
| 当前开盘价；部分候选另含本人早轮次：开发期 pooled 冻结探索（`price_cat_d3_alpha1`） | 1,733 | 66.96% | 0.591239 | 79 |
| 当前开盘价；部分候选另含本人早轮次：2023最高观察准确率，未选中，loss更差（`history_linear_C0.03_alpha1`） | 1,760 | 68.01% | 0.596723 | 52 |
| 当前末盘价/缺失用开盘；部分候选另含本人早轮次：固定对照 / 既有模型；分支保留推荐（`raw_market`） | 1,752 | 67.70% | 0.587691 | 60 |
| 当前末盘价/缺失用开盘；部分候选另含本人早轮次：开发期 pooled 冻结探索（`price_cat_d3_alpha1`） | 1,734 | 67.00% | 0.590046 | 78 |

![第八轮同队列对比](round8_comparison.png)

蓝色历史方法、历史市场输入/监督和绿色当前市场报价使用不同信息。
“pooled冻结探索”是在查看本轮2023分数之前用开发期选择的行，未必通过升级门槛。
原始末盘优先条件在缺末盘时使用开盘、均无时精确回退R4，所以与R5单纯末盘条件的覆盖略有不同。
开盘指所用报价字段，不是已验证的决策时点；本人先前轮次可能晚于开盘采集，因此相关组合只称为更晚赛前的开盘价格条件。

## 本轮实际检验了什么

- **联合发接发模型：** 五个结构共同拟合过去四年的binomial点统计，检验接发对手校正、时间衰减、场地偏好和赛事效应；
  每个T−21日期重估过去状态，校准系数只从更早赛季学习。另有15个预先固定R4混合。
  这是Ingram方法的稀疏MAP改编，不是论文完整贝叶斯模型的复现。
- **赛事环境：** 10个拟合配置比较原326项、场地环境、原始赛事环境、对手校正环境及交互；
  过去三个赛季按3/2/1加权，只读取到2022的原始点统计。256场缺赛事历史也保留在完整队列。
- **历史赔率图：** 六个固定图设置，原始概率和半权R4混合共12候选；仅过去开盘赔率形成实力差，
  不读当前比赛赔率，未知或不同连通分量精确回退R4。另报告禁止2023历史更新的诊断，不参与选择。
- **当前报价残差：** 开盘、末盘优先两个独立条件，每个9次拟合×3固定残差权重，比较报价上下文、历史、本人更早轮次。
  所有拟合仅使用符合purge条件的2021–2022记录。没有报价的101场精确回退R4，未删目标。

这是**39个结构/参数设置**：5点模型+10环境学习器+6图设置+18报价残差学习器。
不同日期和折重复拟合、固定混合、旧参考与未校准诊断不再被混称为独立模型试验。
全部冻结概率列和匹配对照都保存，未隐藏负面结果。

## 预设的开发期门槛

| 分支 | 2022净正确数 / 2,528 | 准确率变化 | 门槛结果 | 保留推荐 |
| --- | ---: | ---: | --- | --- |
| joint_points | -5 | -0.198 pp | `retain_round4` | `round4_reference` |
| venue | +9 | +0.356 pp | `retain_round4` | `previous_probability_recommendation` |
| market_graph | +6 | +0.237 pp | `retain_round4` | `previous_probability_recommendation` |
| market_residual/opening | -14 | -0.554 pp | `retain_raw_market` | `raw_market` |
| market_residual/latest | -3 | -0.119 pp | `retain_raw_market` | `raw_market` |

联合点模型与赛事环境早选2020–2021、gate2022；市场图和报价残差早选2021下半年、gate2022。
不同分支的pooled CV不能直接混为相同样本。门槛要求准确率至少+0.5个百分点且log loss恶化不超过0.005。
分支的原始比较基准不同，不能用“通过/未通过”判定跨条件优劣。

## 推荐与复跑

保留原纯历史概率推荐R4、R5冻结准确率探索、R7历史市场监督学生及本轮各分支冻结候选的独立角色。
本轮未给出稳定、可跨条件取代这些选择的新证据。不要依据最高的单行2023准确率重新选择模型。

在前序轮次本地产物就绪后：

```bash
.venv/bin/python -m scripts.train_joint_points_round8
.venv/bin/python -m scripts.train_venue_round8
.venv/bin/python -m scripts.train_market_graph_round8
.venv/bin/python -m scripts.train_market_residual_round8
.venv/bin/python -m scripts.report_market_residual_round8
.venv/bin/python -m scripts.audit_joint_points_round8
.venv/bin/python -m scripts.audit_market_graph_round8
.venv/bin/python -m scripts.audit_market_residual_round8
.venv/bin/python -m scripts.report_round8
```

报告器只核验/展示已保存概率，不重训或改变选择。模型bundle、训练矩阵与OOF大文件本地保留且可复跑。
[完整命令与前置产物](../../docs/modeling.md) · [执行版notebook](../../round8.ipynb) · [HTML](../../round8.html) ·
[全部概率比较](overview_comparison.csv) · [决策/论文协议](../../docs/research/round8_protocol.md)

分支证据：[联合点统计](joint_points/report_summary.md)、[赛事环境](venue/report_summary.md)、
[历史市场图](market_graph/report_summary.md)、[开盘修正](market_residual/opening/report_summary.md)、
[末盘优先修正](market_residual/latest/report_summary.md)。独立审计覆盖各分支完整输出；区间只描述给定预测，
没有校正多轮搜索和全部球员依赖。本轮方法及未来探索均不打开最终测试，也不保证能达到70%。

验证共282项测试、97个子测试。独立审计核验116列×2,588场的重载概率，重载误差为0，最大交换补概率误差7.78e−15。
联合点统计独立重建全部175个验证状态的查询概率，并抽查15个状态的梯度、复核20次校准；并未独立重训全部OOF模型。
赛事环境与市场图的新特征均已全量重建。各audit记录实际覆盖及限制。
