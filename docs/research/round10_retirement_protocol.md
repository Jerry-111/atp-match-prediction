# Round 10 F10-R：历史 RET 状态，来源审计与候选协议

状态：**仅来源可行性审计；未拟合。当前官方核验不完整，不能记为来源全面通过。**
协调者已接受 10/30 官方核验、20 未核验且偏早年的限制，授权以下三个有限配置。
这项授权不代表来源全面通过：`model_fitting_ready=false` 保留原语义，另记
`root_limited_experiment_authorized=true`。2023 的固定
2,588 场、0.5 阈值和 2024–2025 封存状态保持不变。

## 已取得的证据及缺口

[可复跑审计](../../scripts/audit_retirement_source_round10.py) 只读取本地已有、
逐文件 SHA256 校验过的 17 个 TennisMyLife 年度文件（2006–2022）；不下载新
年度 raw，不读取目标胜负。许可沿用[现有来源说明](../data.md)，raw 和官方
PDF 均不随仓库重新分发；官方 PDF 保存在临时缓存，提交 URL、摘要与哈希。

在 source level `250/500/M/G` 中，单独识别出 1,387 个 RET、262 个 W/O 或
WALKOVER、6 个 DEF。RET 涉及 341 位记录中的 loser，没有重复 match ID；
Hard/Clay/Grass/Carpet 分别为 787/426/164/10 条；轮次含 6 条 RR。
这只是状态与身份 census，**RET 不等于伤病诊断，DEF、W/O 不并入 RET**。

在取得官方证据之前冻结[30 条样本](../../results/round10/retirement/sample_registry.json)：
三条缺技术统计的 RET 全纳入，固定哈希选一条 Carpet，其余按年代×场地×轮次
分层，以最少已抽样单元优先和固定 SHA256 打破平局。年代为 2006–2011、
2012–2017、2018–2022；早/中/晚轮次分别为 R128/R64/R32、R16/QF、SF/F。
最终年代数 11/9/10，Hard/Clay/Grass/Carpet 为 10/9/10/1。缺统计指审计中
列出的六个发球分数计数字段至少一个为空，并不泛指所有技术列。
没有依据目标结果、模型误差或证据容易取得与否重新抽样。

冻结样本 SHA256：
`285accb1a21c9313abb8297ad249834e82a4b8bc6352da964a312e3cb5cccbe8`。

[逐条结果](../../results/round10/retirement/source_feasibility.json) 和
[人工判读记录](../../results/round10/retirement/official_reviews.json) 当前为：

| 项目 | 数量 | 能支持的结论 |
| --- | ---: | --- |
| 官方 PDF 签表或赛报已检查 | 10/30 | 姓名、获胜/退赛一方、赛事轮次及 RET 状态一致 |
| 已观察到的官方身份/RET 方向矛盾 | 0/10 | 只适用于已核实的 10 条，不能表述为 30 条全通过 |
| 尚缺完整官方证据 | 20/30 | 保留原样本，未自动通过，也未自动删除 |
| 含缺技术统计的 3 条 | 2 已核实、1 未核实 | 技术统计缺失不能自动推翻 RET 状态 |

已核实年代分布是 9/0/1，明显偏早期；没有充分检验 2012–2017 年的来源
语义稳定性。这个有目的的覆盖样本也不能估计总体错误率。9 张官方签表成功
取得，随后 20 个 canonical URL 返回 429，另一个返回 404；不重复尝试被拒
URL、不使用代理或镜像绕过。补充得到的 ATP 2018 官方赛报使核实数达到 10。
未核实的条目不是矛盾条目，未发现矛盾也不证明未核实条目正确。

三个优先问题的处理：

* **2007-520:8，Youzhny/Hernych，源 `0-0 RET`：**
  [官方法网签表](https://www.protennislive.com/posting/2007/520/mds.pdf) 第 1 页
  15/16 号位连接至 Youzhny，并标 Ret.，确认 Hernych 为 RET 一方；官方未给
  数值比分。不能据此把 `0-0` 当作已确认的实际赛分，也不能改为 W/O。
* **2007-433:27，Blake/del Potro，RR：**
  [官方 Las Vegas 签表](https://www.protennislive.com/posting/2007/433/mds.pdf)
  第 1 页 Group 1 明确记录 Blake 获胜、del Potro 退赛，比分 6–1 3–1。
  循环赛 standings 与后续淘汰签表不同；不能使用“输球后又出现”规则否定
  RR 记录。候选历史按日期整体处理，不从 round 或 match_num 推断同日顺序。
* **2007-8996:27，Massu/Roitman，QF：** canonical 源 ID 的官方 draw URL
  返回 404；[ITF 赛事页](https://www.itftennis.com/en/tournament/vina-del-mar/chi/2007/m-is-chi-01a-2007/draws-and-results/)
  可检索，但重定向结果未能检查。
  [同期 Europa Press 报道](https://www.europapress.es/deportes/tenis-00166/noticia-tenis-massu-horna-clasifican-semifinales-torneo-vina-mar-20070203003134.html)
  支持 Roitman 退赛、Massu 晋级，但它不是官方 draw/赛报，**不计入官方核验数**。
  不使用同一数据提供方的另一份表作为独立证据。

此外，2010 Dubai 的官方 6–3 0–0 Ret 与源 6–3 RET 是比分表示差别；没有
获胜方向矛盾。2018 Montpellier 的
[ATP 官方赛报](https://www.atptour.com/-/media/a9e07c779c724fdd9d14b19facf48c52.pdf)
PDF 第 41 页（印刷页 Day 3, page 4）明确记载 Tsonga 在对 Pouille 的半决赛
退赛，但未给出该场完整比分。审计把“确认状态”与“确认数值比分”分开。
官方 PDF 中的姓名通过 pinned raw 行对应现有球员 ID；PDF 本身没有这些
archive ID，不能声称额外独立核验了数字 ID 分配。

## 实际输入覆盖，不含预测效果

读取原开发目标的 ID、球员、日期与 season 元数据，过滤至 2022 年；历史窗口
两端均含，`[T−90,T−21]` 或 `[T−364,T−21]`，只匹配 RET loser。

| 目标年份 | 全部原目标 | 任一球员 90 日窗口有 RET | 任一球员 364 日窗口有 RET |
| --- | ---: | ---: | ---: |
| 2020 | 1,214 | 40（3.29%） | 405（33.36%） |
| 2021 | 2,388 | 303（12.69%） | 865（36.22%） |
| 2022 | 2,528 | 310（12.26%） | 1,118（44.22%） |

其余 2009–2022 各年计数保存在 JSON。没有读取 2023 年度 raw，没有计算本分支
2023 覆盖或任何预测指标。2020 赛历中断会影响 inactivity，须保留年份分项。

## 机制与反证

候选假设是：已知的历史非完赛状态及之后的出场，可能说明旧能力估计的可靠性；
这不是“受伤一定降低胜率”。竞争解释包括普通停赛、年龄/排名、赛事记录偏差、
回归参赛者选择，以及已完成比赛目标本身排除 RET 的选择效应。

[Cutmore 与 Knottenbelt 2013 原文](https://www.doc.ic.ac.uk/~wjk/publications/cutmore-knottenbelt-mis-2013.pdf)
研究的是赛中退赛风险与结算，其点胜率不受伤病影响的假设恰好限制了外推；
[Oliver 等 2024 原文](https://onlinelibrary.wiley.com/doi/full/10.1002/ejsc.12177)
是退赛及比赛暴露的回顾性分析，不是未来完整比赛胜率的验证。本分支为受启发的
工程假设，既非两篇论文的直接复现，也不引入临床变量。RET 与普通长间隔输入
若没有区别，或提升只来自排除的目标 RET 预测，机制在本任务中不成立。

## 冻结精确定义：共同 inactivity 基线与两个 RET 扩展

以下是内部评审后、拟合前确定的菜单。保留原目标、
原标签与 T−21，不创建“健康子集”，不根据 2023 错误调整窗口。若获准实现，
历史只扩展读取既有 pinned 2006–2023 raw 中的上述合法 Tour 级别 RET；原表
的官方已确认日期修正仍适用，先过滤 source year 再解析日期。2024/2025 拒绝。
Carpet 与 RR 的 RET 可作为过去的状态，不能因目标没有 Carpet 而改写来源。
W/O、DEF、歧义 marker 不进 RET 状态；技术统计缺失不构成删除条件。
不使用 RET 的不完整技术统计或把它作为额外二元训练败局。

对球员 p、目标日期 T，令 C=T−21、L=T−364。所有状态仅由日期 ≤C 的历史计算，
同日期整批处理。completed history 使用现有 `source=tour` 已清洗的完整比赛
inventory，不绕过已有去重、RET 排除或身份规则。一天的多场/多个赛事聚为
一个“出场日期”，不虚构同日顺序。

I 的三个普通 inactivity 标量 `(g,r_I,q_I)` 各输出 A−B 的 `_diff` 和 A+B
的 `_sum`，六列在 **I/R/RC 三组中全部保留**。R 额外加入 RET 的 `r_R` 与
存在标志，RC 额外加入同一 `r_R` 与出场衰减 `q_R`，各再输出四列。于是模型
候选矩阵为 **332/336/336** 列；I 是嵌套基线，只有 R/RC 同容量。所有新标量
的未知锚点 recency/state/presence 为 0；共同 g 的无历史值按下述定义。

* 公共 `g = log1p(min(365,C−d_last))/log(366)`，`d_last` 是最近合格 completed
  出场日期。没有已知 completed 历史时 `g=1`。已有 326 静态字段中的历史量与
  缺失/覆盖信息在三组完全一致，不根据未来首次参赛建立 player roster。
* **I：普通 inactivity 对照。** 在已知相邻 completed 日期间隔 ≥28 日的
  “回来出场日期”中，选 `[L,C]` 内最近的日期 a；只有前一个日期已经观察到才
  能构成间隔，首次出现不算回归。`r=exp(−(C−a)/90)`，
  `q=r*exp(−k/5)`，k 为 `(a,C]` 内 distinct completed 出场日期数。
* **R：历史 RET 状态。** a 为 `[L,C]` 内最近 RET-loser 日期；
  `r=exp(−(C−a)/90)`，`q=1`（存在锚点），否则二者为 0。
* **RC：历史 RET 后出场衰减。** a、r 与 R 相同；
  `q=r*exp(−k/5)`，k 为 `(a,C]` 内 distinct completed 出场日期数。
  同日出现不算“之后”；不使用未来是否返回或未来战绩，`q` 不叫“康复概率”。

I 的普通间隔状态在 R/RC 中不被替换；RC 增加采用相同日历/出场衰减形式的
RET 状态。R 与 RC 比较固定存在标志与已观察出场衰减。它们是机制对照，不能被称为
识别 RET 因果效果的实验。90 日、28 日与 5 次出场均为预设常数，不搜索。

## 拟议学习与停止规则

只设 **3 个 fit 配置**：I、R、RC 各一个，不另加架构
搜索或事后混合。全组采用同一既有 326 字段、同一个 antisymmetric spline
learner：`C=.01`、`half_life=3 years`、已有 knots/shape/interaction shrinkage
规则固定。各组新增 `_diff` 入同样的线性及 odd-spline 分支；对应 `_sum`
只与 `log_rank_diff` 相乘形成 odd 项，不直接进入非对称截距。R/RC 维数一致，
I 为更小的嵌套对照；三组正则化规则一致。所有 RMS、样条输入缩放只能拟合
当期训练数据。交换 A/B 应满足 `p(B,A)=1-p(A,B)`。

预期每组 forward 2020、2021、2022 与一次最终 refit，共 12 次 fit。各训练窗
继续按目标日期 purged 21 日，训练标签最迟 2022；最终 33,833 行，2023 全
2,588 行评估、零历史也保留。2020/2021 先排除比 R4 pooled log loss 高出超过 0.005 的候选，再以总正确数
优先、相同则较低 pooled log loss、再以名称顺序选出 early 候选；冻结后仅在 2022 与旧 R4 同场比较。
升级门槛为 accuracy 增加 ≥0.005 且 log loss 增加 ≤0.005。三个模型与 R4
同列报告；2020–2022 pooled 选择同样保留 +0.005 loss guard，和 early gate 是
不同结果，不用 2023 改推荐。
任何允许继续后的首次新 2023 评分前，都必须再次冻结具体 registry 与源码。

当前来源没有完成 30/30 核验；协调者已经明确授权在此 partial evidence 限制
下开展有限实验，保持“不完整来源”标记，不以授权重写核验结论。不能只选择已核实的 10
条来构造特征，这会把官方资料可用性变成样本筛选。若存在身份/RET 方向矛盾、
时间越界或无法保持同一完整目标集，停止本分支；不因为接近 70% 修改状态规则。

## 复跑与人工核验边界

```bash
.venv/bin/python -m scripts.audit_retirement_source_round10
.venv/bin/python -m pytest tests/test_retirement_source_round10.py -q
```

第一条重建 census、冻结抽样一致性与所有 pre-2023 覆盖，并校验已判读 PDF 的
缓存 SHA256。官方证据 URL 和缓存位置见
[source manifest](../../results/round10/retirement/official_sources.json)。不同机器
可用 `--fetch-official --cache-dir /path/to/cache` 获取缺少的已成功证据；脚本
不自动重试先前失败 URL，首次 429 后停止该主机剩余请求，也不接受变更的原
成功文件哈希。官方网络可用性不保证；缺文件或内容变更会失败，不能静默复用
“verified”。人工阅读签表的 adjudication 不能由重复运行自动替代。

验证记录当前有 10 条完整官方 PDF 判读，其余 20 条保留 unresolved。没有模型、
预测或旧报告的数值产物因来源审计而发生变化。

内部授权后的实现检查（首次模型拟合前）：

* 全部 36,421 个目标生成 12 列 union，约 1 秒；三个候选输入维数经断言为
  332/336/336。新增字段的窗口、交换对称、同日聚合和未来 poisoning 合成测试通过。
* 读取既有 2006–2023 pinned raw 后，最晚目标 cutoff 前有 1,452 条 RET：
  2006–2022 的 1,387 条与来源 census 完全一致，另有 65 条过去的 2023 RET。
  已知 `EXCLUDED_PRIOR_IDS` 中没有 RET 条目进入这个池，实际额外排除数为 0。
* 2023 元数据覆盖为任一球员年度窗口有 RET 的 1,425/2,588 场；该数仅是
  输入覆盖，没有读取新预测结果来选择窗口或配置。普通 inactivity 状态覆盖
  2,580 场，三组仍保留全部 2,588 场。
* 训练入口检查授权范围仍为 10 verified/20 unresolved、0 已观察矛盾，并核对
  frozen sample、人工 reviews、official-source manifest 与审计代码哈希。改变
  adjudication 或出现矛盾会停止，不可继续使用过期 feasibility JSON。

训练入口与复用方法（仅执行上述已授权固定菜单）：

```bash
.venv/bin/python -m pytest tests/test_retirement_source_round10.py tests/test_retirement_features.py tests/test_train_retirement_round10.py -q
.venv/bin/python -m scripts.train_retirement_round10
.venv/bin/python -m scripts.train_retirement_round10 --predict-features results/round10/retirement/development_features.csv
```

训练器先写 search_registry、再执行三组 OOF，写 frozen_selection 后才拟合最终
模型并评分 2023。保存每组的三个 fold models、最终 bundle、fit_records 的训练
ID 哈希与日期、所有 OOF、完整 2023 概率、机制对照和中文结果报告。
