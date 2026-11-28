# Round 11：2022 年完赛来源前 10 项可行性核查

2026-11-28。**仅核查来源，不训练、不缩短 T−21，也不新增预测可用记录。**

[机器可读记录](../../results/round11/completion/probe_2022_first10.json)保留
[R10 工作清单](../../results/round10/completion/source_frame.csv)中全部 65 个 2022 年 source edition。
按 `source_nominal_date, source_tourney_id` 排序，固定核查前 10 个；其余 55 个明确为
`not_reviewed`。选样没有使用目标胜负或模型误差。本阶段只解析赛事 ID、名称、日期、场地、
级别和计数元数据，没有读取目标标签列、预测文件或 2024–2025 原始数据。
官方完赛报道必然描述历史比赛结果；这些内容仅用于判定来源的完成语义。

| 顺序 | Source edition | 名义开始日 | 官方全文支持的保守 completed-by 日 | 当前证据 |
| --- | --- | --- | --- | --- |
| 1 | 2022-8998 Adelaide 1 | 01-03 | 01-20 | 稍后的官方回顾，只有宽松上界 |
| 2 | 2022-9665 Melbourne | 01-03 | 01-09 | 同期官方完赛报道 |
| 3 | 2022-9663 Sydney | 01-10 | 01-15 | 同期官方完赛报道 |
| 4 | 2022-9667 Adelaide 2 | 01-10 | 01-15 | 同期官方完赛报道 |
| 5 | 2022-580 Australian Open | 01-17 | **01-31** | 次日官方分析；不以赛历 01-30 代替完成上界 |
| 6 | 2022-375 Montpellier | 01-31 | 未核实 | 预告 PDF 未完成页面核验；年末回顾仅索引 |
| 7 | 2022-891 Pune | 01-31 | 未核实 | 预告和较晚总结仅索引 |
| 8 | 2022-9158 Cordoba | 01-31 | 未核实 | 官方年报结果页已核对，但无可靠公布日／完成时间语义 |
| 9 | 2022-407 Rotterdam | 02-07 | 02-13 | 同期赛事官方完赛报道 |
| 10 | 2022-424 Dallas | 02-07 | 02-13 | 同期美国网协完赛报道 |

因此是 **6 个同期报道、1 个稍后回顾、3 个完成日上界未核实**，不是 7 个已经获准使用的历史输入。
全部 `admitted_for_prediction=false`，历史网页快照数为 0。

## 完成语义和页面日期分别记录

[Melbourne 官方报道](https://ausopen.com/articles/news/melbourne-summer-set-nadal-wins-first-title-australia-2009)
显示 1 月 9 日，并描述当晚决赛已经结束。
[Sydney 官方报道](https://ausopen.com/articles/news/badosa-karatsev-snare-sydney-silverware)
显示 1 月 15 日，并明确男子决赛在星期六完成。
[Adelaide 2 官方报道](https://www.tennis.com.au/fan-zone/news/2022/01/15/hometown-hero-kokkinakis-crowned-champion-in-adelaide)
描述 1 月 15 日当晚已完成的决赛。这些与仅公布未来赛程的公告不同。

[AO 官方决赛分析](https://ausopen.com/articles/features/analysis-how-nadal-won-final)
页面日期为 1 月 31 日，正文回顾已经结束的星期日夜间决赛。本记录采用 31 日的保守完成上界，
不从赛历的 30 日推导结束时刻，也不补造一个未经来源确认的精确时间。
[Rotterdam 赛事方](https://www.abnamro-open.nl/en/news/headlines/49th-edition/dream-comes-true-for-felix)
与 [USTA Dallas 报道](https://www.usta.com/en/home/pro/pro-media---news/reilly-opelka-wins-dallas-for-third-career-atp-title.html)
均显示 2 月 13 日，并描述已结束的决赛及赛后情节。

Adelaide 1 的 [AO 官方回顾](https://ausopen.com/articles/news/match-day-gael-force-blows-away-bublik)
显示 1 月 20 日，仅能证明该标题下描述的 Adelaide 1 决赛在此之前已结束。
不能把它改写成“1 月 9 日已发布”。这个宽松上界可能对紧邻赛事没有用途，仍原样保留。

上述 `completed_by_day` 均是当前官方文本支持的日级陈述；`displayed_publication_day`
是页面显示的日期。它们不是经过历史快照验证的首次发布时刻。全部未标明的时区保持 null，
也没有将场馆时区直接赋给网站发布日期。

## 最早正赛开始证据独立处理

[Sydney 1 月 8 日官方公告](https://ausopen.com/articles/news/murray-kyrgios-and-19-top-30-women-sydney-tennis-classic)
明确区分 1 月 9 日资格赛和 1 月 10 日开始的 ATP 正赛；
[1 月 10 日战报](https://ausopen.com/articles/news/kvitova-survives-kontaveit-scorches-sydney)
又描述了当日的男子正赛。这是公告与发生记录互相支持的例子，但仍不是涵盖所有比赛的精确时间账本。
[AO 1 月 13 日签表公告](https://ausopen.com/articles/news/draws-revealed-ao-2022)
明确宣布首轮在 1 月 17 日开始。当前仅这两项拥有已检查全文的明确正赛首日公告。

Melbourne 的[官方赛前综合公告](https://ausopen.com/articles/news/barty-and-de-minaur-light-sydney-tennis-classic)
写 1 月 3–9 日，而 ATP 索引赛事范围从 1 月 4 日开始；前者包括多项男女赛事，
不能据此自动决定 ATP 正赛最早日。Rotterdam 的
[官方赛前公告](https://www.abnamro-open.nl/en/news/headlines/49th-edition/strong-line-up-for-the-abn-amro-world-tennis-tournament)
写 2 月 5–13 日，未把资格赛与正赛拆开；不能拿它认证 2 月 7 日的最早正赛开始日。

Adelaide FAQ 和 Montpellier ATP Day 1 PDF 的初次 web 提取返回了文本，后续页面检查返回 403。
没有获取原始页面图像，因此没有把这些线索升级为经过核验的正赛首日证据；被拒后没有继续尝试。

## 可扩展来源、缓存和明确限制

[ATP 2022 results archive](https://www.atptour.com/en/scores/results-archive?year=2022)
可从索引发现多个 edition，但直接访问返回 403；它只用于发现，不用于批量认证。
Tennis TV 的 edition 页面提供回放目录，但当前提取未建立回放首次发布与实际完赛的时间关系。
此前已被拒的 ATP 2023 年鉴及 protennislive URL 没有重试或绕过。

[西班牙网协 2022 年报](https://www.rfet.es/descargar/file-rfet-memorias/17_Adjunto/rfet-memoria-2022.pdf)
已普通下载并实际渲染核对 PDF 第 18 页（印刷页码 17）。该页的赛季决赛表将 Cordoba 配在
2 月 6 日，将 AO 配在 1 月 30 日。这里确认的是原始页面内容，并非相信搜索生成的表格。
年报仍没有提供这些结果的首次公布时刻，表内的比赛日期标签也没有明确承诺是实际完成日上界。
它适合检查赛事身份和历史结果范围，不能单独承担紧邻赛事的可用性认证。
因此 Cordoba 的 `reported_final_label_day=2022-02-06` 与 `completed_by_day=null` 同时保留。

9 份 HTML 和这份年报的原始响应已缓存于本机 `/tmp/atp-round11-source-probe/2022/`；
JSON 保存实际内容 SHA-256、字节数和 UTC 检索时间。这些缓存不是 2022 年历史快照，也未作为
第三方全文提交到仓库。缓存会随本机清理而消失，不能声称别人仅凭仓库即可重现原内容。
Adelaide 2 与 Rotterdam 的文章曾由 web 工具成功读取全文，之后普通缓存请求返回 403；
相关缓存哈希仍为 null，同时保留失败记录，没有拿摘要哈希冒充来源内容哈希。
RFET 仅遇到 web 工具的文件大小限制，普通公开下载成功；不存在访问拒绝后的绕过。

下一步由主协调者统一日级证据、未知发布时区和历史网页可信度的规则，再决定是否核查下一批。
本 probe 不扩展到其余 55 个 edition，不计算全工作清单的新准入，不改目标集合，不训练模型。

## 经协调者授权的有限 target 开始日补查

完成上述固定 10-source probe 后，协调者授权仅对已支持 source 在原工作清单中可关联的
target 补查开始日。预筛选使用 `target_nominal_date - max(completed_by_day, displayed_publication_day) >= 3`
个日历日；名义 target 日期只用于发现候选，不自行充当已核开始日。
7 个已支持 source 的 33 对原有关系留下 7 对、4 个 target edition：

| Target | 候选场数 | Source | 按 target 名义日期的日差 | 正文开始日证据 |
| --- | --- | --- | --- | --- |
| 2022-580 AO | 124 | Melbourne | 8 | 1 月 17 日首轮公告，原 probe 已有 |
| 2022-495 Dubai | 31 | Rotterdam、Dallas | 8 | 2 月 21 日 opening day 官方预告 |
| 2022-807 Acapulco | 28 | Rotterdam、Dallas | 8 | 2 月 19 日资格赛、21 日正赛官方公告 |
| 2022-8996 Santiago | 27 | Rotterdam、Dallas | 8 | 本轮有限搜索未核到，保留 fallback |

上述 210 场是候选 edition 中的目标数量，并非已确认其球员均有新增历史；尚未计算直接刷新场数。
Adelaide 1 的宽松上界没有贡献，所以只用 6 个同期报道也得到相同候选集。
若错误地只按 source 名义开始日与 target 相隔至少 3 日，会得到 13 个 target；两种计数不能混淆。

[Dubai 2 月 20 日官方预告](https://dubaidutyfreetennischampionships.com/magnificent-monday-kicks-off-atp-week-at-2022-dubai-duty-free-tennis-championships/)
将星期一列为男子赛事 opening day，支持 2 月 21 日的公告开始日。全文列出首场时间，
但本记录不推定一个来源未标明的 UTC 时区。
[Acapulco 官方公告](https://abiertomexicanodetenis.com/2022/02/18/definidos-wild-cards-para-el-abierto-mexicano-telcel-presentado-por-hsbc-2022/)
明确资格赛为 2 月 19–20 日、正赛自 21 日开始；若统一协议需要更保守的全赛事开始下界，
可以提交 19 日供审阅。公告正文 dateline 为 15 日，页面显示 18 日；原 HTML 的发布时间元数据
还同时出现 `2022-02-18T13:08:09+00:00` 和 `2022-02-18T13:08:09-06:00`。
这些原始值均保留，不能任选其一声称取得无歧义的历史首次发布 UTC。

Santiago 的这轮官方域查询未找到可核验的 2022 正文；没有用后续年份页面替代，也未继续扩大来源搜索。
因此现阶段是 **3 个有正文开始公告的 target、5 对来源关系、183 场候选目标**可提交 Tier B 假设审阅，
Santiago 27 场保留原模型 fallback。这里仍没有作出预测准入决定。

## 待正式协议冻结的 Elo 实现建议

复用 [advanced_features.py](../../scripts/advanced_features.py) 的 `HistoricalState` 和
`elo_probability`，保持初值 1500、overall K32、400 分尺度及按原始赛事日期冻结后整批累加的更新。
每个有审核通过来源的 target edition 只构造一次历史子集：原有 `date <= T−21` 的 Tour inventory 行，
加已认证 source edition 在同一 inventory 中的全部合格历史行；按 match ID 去重。
从这个子集建立同一个 `HistoricalState`，读取最后准入日期的 K32 rating，不调用固定 lag 的 `features()`。
这样不虚构同日比赛顺序，也不把未确认完成的相邻赛事一起放入。

无额外来源时直接读取原始全状态在 T−21 的 rating，并要求与原 `elo_k32_diff` 和公式概率完全一致。
所有目标仍保留，采用固定 0.5 阈值。正式实现还应验证 source/target 身份、实际年份与日期、封存年份拒绝、
source_match_ids 全量集合、无重复增量、未来结果投毒、A/B 互补和空 ledger 的逐行精确 fallback。
新来源只更新胜负 Elo，不由完赛报道推断技术统计当时也已经公布。
这只是实现方案；训练菜单、Tier B 假设及正式协议哈希由协调者冻结后再执行。
