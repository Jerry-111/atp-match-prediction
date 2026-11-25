# 第二轮模型实验与复跑

用户在第一轮结果之后要求继续优化，三个并行 agent 分别实现动态实力、对手调整的技术统计、平滑非线性学习器，主 agent 负责统一实验与交接。独立实现之后进行了交叉审阅。完整的候选理由、假设、反证条件和预定替换门槛见[本轮协议](research/round2_protocol.md)。

本轮已知第一轮的2023结果，所以所有新增2023结果均为**后续探索性验证**，不是全新未见数据上的确认性结果。2024–2025保持封存。第一轮的模型、逐场概率和原先冻结的选择完整保留在`results/modeling/`。

## 新增信息与模型

- [动态实力](research/round2_ratings.md)：原始Glicko均值/不确定性更新、休赛时的不确定性增长、整体与场地评分及保守融合，共17项特征。实现对照了Glickman的数值例子；时间和场地适配属于本项目假设。
- [对手调整的发接发](research/round2_points.md)：以发球方和接发方的动态能力共同解释得分概率，固定365/730日衰减、日期批量更新、稀疏场地收缩和保发率摘要，共34项特征。它是近似递归模型，不是精确贝叶斯后验。
- [平滑非线性学习](research/round2_learning.md)：线性核心加惩罚样条和有限场地/历史可靠性交互，结构上保证交换球员时概率互补。
- 全部198项输入与紧凑输入分别测试正则化逻辑回归、CatBoost和直方图梯度提升；另比较3个原始Glicko概率。共37项新配置，复用第一轮31项候选的时间外预测。

所有历史只查询到T−21，只有训练行参与缩放、拟合和权重选择。2020–2021选择配置/集成权重，2022比较新流程与第一轮相同早期选择流程；至少降低0.001 log loss才替换原推荐。之后使用2020–2022重拟合已选流程，再查看本轮2023结果。最终推荐是否替换、准确配置及权重均保存到`results/round2/frozen_selection.json`。

## 复跑

需要第一轮锁定的Python环境。新克隆先生成第一轮被Git忽略的大型OOF和模型文件；已有这些文件则直接执行第二轮。

```sh
.venv/bin/python -m scripts.train_models --with-serve
.venv/bin/python -m scripts.train_round2
.venv/bin/python -m scripts.report_round2
.venv/bin/python -m pytest -q
```

第二轮会检查第一轮缓存的比赛ID、标签、折定义和源文件哈希。它重建历史特征，然后将新搜索注册表写盘，再开始拟合。每个候选的年度评分也保留，包括没有提升的方法。

生成物位于`results/round2/`：`temporal_cv.csv`、`validation_comparison.csv`、`validation_predictions.csv`、`search_registry.json`、`frozen_selection.json`、`summary.json`及中文`report_summary.md`。模型文件与大型开发特征/OOF表本地生成并忽略，不上传原始年度CSV。

如已按相同历史规则准备好特征矩阵，可以只做冻结推理，不重新训练：

```sh
.venv/bin/python -m scripts.train_round2 --predict-features path/to/features.csv
```

它从默认第二轮输出目录加载可信的本地`frozen_models.joblib`，写入`inference_predictions.csv`，并保留可选的`match_id`。模型文件使用pickle协议，勿加载不可信来源。

## 如何解释结果

应区分流程选择用的2022评分、本轮新配置的训练期OOF评分、集成权重在OOF上的元训练损失，以及已重复使用的2023探索性结果。2023最小值不自动成为新的推荐。

新家族之间的比较不等同于严格单特征因果贡献。配对日期块区间只衡量固定模型在验证比赛上的部分不确定性，不覆盖全部多轮搜索或跨日期球员相关性。模型最终是否更强，仍需冻结后由评估队友在2024–2025相同比赛上独立检验。
