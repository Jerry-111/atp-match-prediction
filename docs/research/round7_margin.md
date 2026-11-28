# 第七轮：改变监督目标，学习点／局表现

2026-11-27，用户要求改变方法、参考论文，继续向同一批 2,588 场 2023 比赛的 70% 准确率探索。本协议写于本分支模型拟合之前。2023 与 2022 已在前六轮反复查看；这是有规则的探索，不是新的独立验证。2024–2025 最终测试继续封存。

## 独立提案、文献核对与决定

使用 scientific-brainstorming 的独立生成和随后证据核对流程；协调者预先提到“动态实力／对位／新来源审计”，记录为潜在锚定。features agent 在看其他 agent 提案之前提出 F7-A（改变训练监督目标）与 F7-B（联合点级层次实力估计），随后使用 nature-academic-search 的一手来源核验流程检查可行性。没有人类专家投票；提案与文献都不构成新模型有效性的实证。

F7-A 的假设是：同样的赛前特征除了拟合一位胜者，还能从训练比赛胜负的强弱中学到信息。当前 WElo 使用比分作为历史评级更新的权重，本实验则改变监督标签，直接比较相同模型容量下的 hard outcome 与 performance mixture。反证是向前折无改善，或改善可被不含真实表现的均匀标签平滑完全解释。F7-B 有独立论文和作者实现支持，但联合后验计算较贵，由协调者决定是否另开实现；本分支不冒称它的复现。

检索日期为 2026-11-27；查询包括 tennis point-based Bayesian model、margin of victory Kovalchik、tennis supervised point share target 和 label smoothing。学术 MCP 工具未挂载；Crossref 网页 API 无法通过当前检索工具访问，改用作者 PDF、作者代码、会议主办方论文摘要及 arXiv 原稿核对。不是系统综述，不声称文献空白。

| 一手来源 | 可支持的内容 | 本分支边界 |
| --- | --- | --- |
| Ingram (2019), *A point-based Bayesian hierarchical model to predict the outcome of tennis matches*, [作者完整论文](https://martiningram.github.io/papers/bayes_point_based.pdf)，[DOI](https://doi.org/10.1515/jqas-2018-0008)，[作者代码](https://github.com/martiningram/tennis_bayes_point_based) | 直接用发球分计数估计能力，说明细粒度结果是可用的监督来源；原文也有与 Elo 比较的负面证据 | 本分支不是其 binomial 层次模型，也不使用其 2014 数据或结果估计本任务提升 |
| Kovalchik 的 *Extensions of the Elo Rating System for Margin of Victory*，[MathSport 2019 作者摘要，印刷页 35](https://www.mathsportinternational.com/MathSport2019BookOfAbstract.pdf)，后续期刊 [DOI10.1016/j.ijforecast.2020.01.006](https://doi.org/10.1016/j.ijforecast.2020.01.006) | 将胜负强弱加入评级是已研究的方向；摘要比较多种更新规则 | 未获取该期刊完整方法；本实验不声称复现其四种 MOV 算法，不从二手资料猜公式 |
| Szegedy 等，*Rethinking the Inception Architecture for Computer Vision*，[作者原稿](https://arxiv.org/abs/1512.00567)；Müller、Kornblith、Hinton，*When Does Label Smoothing Help?*，[作者原稿](https://arxiv.org/abs/1906.02629) | hard label 与均匀标签混合是独立的正则化解释；校准也可能改变 | 论文主要是神经网络任务，不证明网球线性模型受益。必须设置平滑对照，不能把所有提升归因于真实点／局信息 |

## 训练标签与禁止的信息

输入固定为已审计的第五轮 326 列，其中 188 个 `_diff` 列进入同一无截距线性模型。输入的历史仍截至各目标 T−21；本分支没有添加比分、点数、比赛后信息作为预测变量。

训练标签仅对 2009–2022 生成。年度清单在打开文件**之前**过滤为 2009–2022；接口拒绝 2023+ 目标行，注入的未来 raw row 连 ID 都不读取。对每个训练 match ID 唯一连接并核对 A/B、胜者、季、日期、场地。2019 Washington 复用已发布纠正。

点比例为 `(A 发球赢分 + B 发球失分)/(A 发球总分 + B 发球总分)`，不是仅 A 的发球胜率。局比例从标准完整比分解析；抢七括号内点数不额外当作局。仅看训练数据的可行性检查：33,833 行均可连接，33,796 行有合法点计数，33,833 行局比例有效。训练赢家中 1,552 场总点不足一半、683 场总局不足一半，说明细粒度优势不能机械替代比赛胜负。

令 binary outcome 为 y，实际 A 点／局比例为 r，固定 `alpha=.5`：

- binary：`q=y`。
- points/games：`q=(1-alpha)*y+alpha*r`。
- uniform_points/uniform_games：`q=(1-alpha)*y+alpha*.5`。

对应 r 缺失时，表现目标及其专属均匀平滑对照都回退 `q=y`。因此没有任何训练行被删除，point 的 37 条缺失也不会制造与平滑对照不同的回退路径。最初提议的单一 uniform 控制在实现前拆分为两个缺失模式匹配的控制；这是新拟合前的对照完善，不是按新分数扩网格。

## 固定模型与诚实时间校准

网格为两种 `C=.01/.1` × 五种上述目标，共 **10 个拟合配置**。不搜索 alpha，不加 boosting、隐藏层、额外特征或学习率。全部使用三年 recency half-life，按训练样本加权 RMS 缩放，固定 clip[-8,8]，无截距 L2 logistic。每行复制为 y=1 和 y=0 的两条记录，权重分别为 `w*q` 与 `w*(1-q)`，精确实现 fractional cross entropy。权重总量归一到训练行数，所有目标构造的有效总质量相同。

q 是表现混合的期望，不自动等于比赛胜率，因此每一种目标（包括 binary）都采用相同校准程序：取外层训练的最后赛季为校准年，只用更早赛季并在首个校准日期前 purge 21 天拟合内层模型；用该模型真正样本外的 logit 对最后训练年的 binary 结果拟合正、零截距校准斜率。随后用全外层训练重拟合 head，保留这个斜率。斜率约束 `[.05,20]`，固定优化，不用验证年选取。该程序假设校准能够从较小内层训练迁移至全量重拟合，局限必须报告。

正斜率不改变单模型 0.5 分类，只改变概率与之后的固定混合。无截距 head 和正斜率严格保持 A/B 概率互补。Scaler 仅由对应实际训练行估计；内层不能先用外层全量拟合 scaler。

## 外层比较与冻结

沿用 2020/2021/2022 向前折、此前赛季和 21 天 purge。当前正式第四轮概率由既有真实 OOF 复用。10 个拟合模型各增加一个固定 50:50 R4 概率混合，加 R4 共 **21 个可选择项**；R5 pooled 混合仅报告，不参与早期或 pooled 选择。没有连续混合权重搜索。

先用 2020–2021 最高准确率选择，要求 loss≤R4+.005；平局按较低 loss、稳定名称。该早期程序在 2022 须准确率提升≥.005且 loss 增加≤.005 才能升级。随后 pooled 2020–2022 重选，写入冻结 JSON 后再拟合最终模型和评分同一 2,588 场 2023 比赛。失败则保留 R4，pooled 选择仅是探索性比较。固定阈值 .5，不改变目标筛选。

保存全 22 列 OOF/2023 概率（含 R5 只报告对照）、全部指标、点／局相对相同 C 的 binary 与专属 uniform 控制的逐折差异、内层校准元数据、哈希及 joblib bundle。所有负结果保留。2022 门槛及 R4 也经过前轮选择，不是全新的确认性检验；最终封存测试才用于后续独立评估。

复跑：`.venv/bin/python -m scripts.train_margin_round7`；测试：`.venv/bin/python -m unittest tests.test_margin_round7 -v`。CPU 限制为 4 线程。原先所有结果和推荐保持原文件，未承诺达到 70%。
