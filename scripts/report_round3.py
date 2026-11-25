"""Render saved Round 3 history/format comparisons without fitting models."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/round3"


def load_results():
    data = {key: pd.read_csv(OUT / file) for key, file in [
        ("scores", "validation_comparison.csv"), ("cv", "temporal_cv.csv"),
        ("predictions", "validation_predictions.csv")]}
    for key, file in [("frozen", "frozen_selection.json"), ("registry", "search_registry.json"), ("summary", "summary.json")]:
        data[key] = json.loads((OUT / file).read_text())
        if data[key]["final_test_evaluated"]:
            raise ValueError("This exploratory report expects an unopened final test")
    rows = data["predictions"]
    if len(rows) != 2588 or rows.match_id.duplicated().any() or not rows.season.eq(2023).all():
        raise ValueError("Round 3 must preserve all 2,588 unique 2023 targets")
    if not data["scores"].matches.eq(len(rows)).all():
        raise ValueError("All comparison denominators must match")
    data["specs"] = {s["name"]: s for s in data["registry"]["candidates"]}
    return data


def label(name, data):
    aliases = {"round1_ensemble": "Round 1 frozen ensemble", "round1_best_single": "Round 1 best single",
               "rank_reference": "Rank-only reference", "round2_selected_single": "Round 2 selected single"}
    if name in aliases:
        return aliases[name]
    spec = data["specs"][name]
    learner = "linear" if spec["kind"] == "logistic" else "spline"
    return f"{spec['train_start']}+ · {learner} · " + ("with format" if spec["include_format"] else "without format")


def comparison_table(data):
    table = data["scores"].copy()
    table.insert(1, "description", [label(name, data) for name in table.model])
    table["pooled_2020_22_cv_log_loss"] = table.model.map(data["cv"].set_index("name").mean_log_loss)
    table["role"] = "Exploratory comparison"
    table.loc[table.model.eq("round1_ensemble"), "role"] = "Previous frozen recommendation"
    table.loc[table.model.eq(data["frozen"]["recommendation"]), "role"] = "Recommendation after 2022 gate"
    return table[["model", "description", "role", "matches", "mean_log_loss", "accuracy", "brier", "pooled_2020_22_cv_log_loss"]]


def family_table(data):
    scores = data["scores"].set_index("model")
    cv = data["cv"].set_index("name")
    rows = []
    for family, name in data["frozen"]["selected_by_family"].items():
        spec, score = data["specs"][name], scores.loc[name]
        rows.append({"training_start": spec["train_start"], "learner": "linear" if spec["kind"] == "logistic" else "spline",
                     "format_context": bool(spec["include_format"]), "selected_model": name,
                     "pooled_cv_log_loss": cv.loc[name, "mean_log_loss"],
                     "2023_log_loss": score.mean_log_loss, "2023_accuracy": score.accuracy})
    return pd.DataFrame(rows).sort_values(["learner", "format_context", "training_start"])


def contrast_tables(data):
    """Retuned family contrasts, never described as exact matched parameters."""
    table = family_table(data)
    history, context = [], []
    for (learner, format_flag), group in table.groupby(["learner", "format_context"]):
        values = group.set_index("training_start")
        history.append({"learner": learner, "format_context": format_flag,
                        "2009_model": values.loc[2009, "selected_model"], "2016_model": values.loc[2016, "selected_model"],
                        "cv_reduction_with_older_data": values.loc[2016, "pooled_cv_log_loss"]-values.loc[2009, "pooled_cv_log_loss"],
                        "2023_reduction_with_older_data": values.loc[2016, "2023_log_loss"]-values.loc[2009, "2023_log_loss"]})
    for (start, learner), group in table.groupby(["training_start", "learner"]):
        values = group.set_index("format_context")
        context.append({"training_start": start, "learner": learner,
                        "with_format_model": values.loc[True, "selected_model"], "without_format_model": values.loc[False, "selected_model"],
                        "cv_reduction_with_format": values.loc[False, "pooled_cv_log_loss"]-values.loc[True, "pooled_cv_log_loss"],
                        "2023_reduction_with_format": values.loc[False, "2023_log_loss"]-values.loc[True, "2023_log_loss"]})
    return pd.DataFrame(history), pd.DataFrame(context)


def exact_cv_controls(data):
    """Pair registered settings that differ only in training start or format."""
    specs = data["specs"]
    cv = data["cv"].set_index("name")
    pairs = []
    for dimension in ("train_start", "include_format"):
        groups = {}
        for name, spec in specs.items():
            key = {"kind": spec["kind"], "params": spec["params"], "half_life": spec.get("half_life"),
                   "train_start": spec["train_start"], "include_format": spec["include_format"]}
            del key[dimension]
            groups.setdefault(json.dumps(key, sort_keys=True), {})[spec[dimension]] = name
        before, after = (2016, 2009) if dimension == "train_start" else (False, True)
        for group in groups.values():
            if before not in group or after not in group:
                continue
            reference, candidate = group[before], group[after]
            row = {"change": "add_2009_2015_training" if dimension == "train_start" else "add_format_context",
                   "reference": reference, "candidate": candidate,
                   "pooled_cv_loss_reduction": cv.loc[reference, "mean_log_loss"]-cv.loc[candidate, "mean_log_loss"]}
            for year in (2020, 2021, 2022):
                row[f"reduction_{year}"] = cv.loc[reference, str(year)]-cv.loc[candidate, str(year)]
            pairs.append(row)
    return pd.DataFrame(pairs)


def sample_table(data):
    rows = []
    for start, group in data["cv"].groupby("train_start"):
        row = {"training_start": int(start), "eligible_rows_through_2022": data["summary"]["training_row_counts"][str(int(start))]}
        for year in (2020, 2021, 2022):
            counts = group[f"train_n_{year}"].unique()
            if len(counts) != 1:
                raise ValueError("Matched models have different fitting row counts")
            row[f"fit_rows_before_{year}"] = int(counts[0])
        rows.append(row)
    return pd.DataFrame(rows)


def draw_chart(data):
    scores = data["scores"].sort_values("mean_log_loss")
    recommended = data["frozen"]["recommendation"]
    y = np.arange(len(scores))
    colors = ["#007F7C" if name == recommended else "#304A62" for name in scores.model]
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False, "axes.spines.left": False})
    fig, (ax, accuracy) = plt.subplots(1, 2, figsize=(12.4, max(6, len(scores)*.42)), sharey=True,
                                      gridspec_kw={"width_ratios": [3,1]})
    ax.scatter(scores.mean_log_loss, y, c=colors, s=44)
    ax.set_yticks(y, [label(name,data)+( " ★" if name == recommended else "") for name in scores.model])
    ax.invert_yaxis()
    ax.set_xlim(scores.mean_log_loss.min()-.006, scores.mean_log_loss.max()+.012)
    old = scores.loc[scores.model.eq("round1_ensemble"), "mean_log_loss"].iloc[0]
    ax.axvline(old, color="#96651D", linewidth=1, linestyle="--", alpha=.65)
    for yy, value in zip(y, scores.mean_log_loss):
        ax.text(value+.00065, yy, f"{value:.5f}", va="center", fontsize=9)
    accuracy.scatter(scores.accuracy*100,y,c=colors,s=35)
    accuracy.set_xlim(scores.accuracy.min()*100-1, scores.accuracy.max()*100+3)
    for yy,value in zip(y,scores.accuracy):
        accuracy.text(value*100+.12, yy, f"{value:.1%}", va="center", fontsize=9)
    accuracy.tick_params(axis="y",labelleft=False,left=False)
    ax.set_xlabel("Mean log loss · lower is better")
    accuracy.set_xlabel("Accuracy (%)")
    for axis in (ax,accuracy):
        axis.grid(axis="x",alpha=.17)
        axis.set_axisbelow(True)
    fig.suptitle("Round 3 · training history and match-format controls",x=.02,ha="left",fontsize=16,fontweight="bold")
    fig.text(.02,.018,"Same 2,588 matches in 2023. ★ Recommendation follows the training-period gate; dashed line is the old ensemble.\n"
             "2023 has been used in previous rounds. These comparisons are exploratory; 2024–2025 remains unopened.",fontsize=9,color="#465769")
    fig.tight_layout(rect=(0,.085,1,.94))
    fig.savefig(OUT/"round3_comparison.png",dpi=180,bbox_inches="tight",facecolor="white")
    plt.close(fig)


def narrative(data, notebook=False):
    frozen, summary = data["frozen"],data["summary"]
    gate = frozen["procedure_gate"]
    chosen = frozen["recommendation"]
    scores = data["scores"].set_index("model")
    best = data["scores"].sort_values(["mean_log_loss","model"]).iloc[0]
    candidate = frozen["best_single"]
    highest_accuracy = data["scores"].sort_values(["accuracy","mean_log_loss"],ascending=[False,True]).iloc[0].model
    candidate_score,old = scores.loc[candidate],scores.loc["round1_ensemble"]
    interval = summary["comparisons_vs_round1"][candidate]
    lo,hi = interval["ci95"]
    selected_gain = old.mean_log_loss-scores.loc[chosen,"mean_log_loss"]
    conclusion = "通过门槛，采用扩展实验的模型选择流程" if gate["procedure"] == "adopt_history_model" else "未通过升级门槛，保留第一轮冻结集成"
    counts = summary["training_row_counts"]
    table = ["| 比较 | 2023 log loss ↓ | 准确率 |","| --- | ---: | ---: |"]
    aliases = {"rank_reference":"只用排名","round1_best_single":"第一轮最强单模型","round1_ensemble":"第一轮冻结集成",
               "round2_selected_single":"第二轮训练期选中单模型",highest_accuracy:"本轮表内准确率最高（描述性）",candidate:"第三轮训练期选中单模型"}
    for name in dict.fromkeys(["rank_reference","round1_best_single","round1_ensemble","round2_selected_single",candidate,highest_accuracy,chosen]):
        row=scores.loc[name]
        table.append(f"| {aliases.get(name,label(name,data))} (`{name}`) | {row.mean_log_loss:.5f} | {row.accuracy:.2%} |")
    era,context = contrast_tables(data)
    era_range = (era["2023_reduction_with_older_data"].min(),era["2023_reduction_with_older_data"].max())
    context_range = (context["2023_reduction_with_format"].min(),context["2023_reduction_with_format"].max())
    prefix = "" if notebook else "../../"
    image_prefix = "results/round3/" if notebook else ""
    uncertainty_sentence = ("区间跨过 0，尚不能据此确认候选优于旧集成。" if lo <= 0 <= hi
                            else "该条件区间未跨过 0，仍需考虑重复选择与最终测试的不确定性。")
    return f"""# 第三轮：扩大训练历史，并检验赛制信息

**结论：{conclusion}。当前推荐是 `{chosen}`。** 本轮预先登记 **{summary['new_candidates']}** 个配置；逐年分数、未胜出的配置和原有基准全部保留。

## 本轮改变了什么

将训练目标从 2016 年向前扩展到 **2009 年**，增加 2009–2015 的合格正赛，保留从 2006 年开始的历史初始化。两个训练范围分别包含 **{counts['2016']:,}** 与 **{counts['2009']:,}** 场截至 2022 年的合格目标，增加 **{counts['2009']-counts['2016']:,}** 场。每个前向验证折实际使用更早的训练行，其数量在 notebook 单独列出。

同时增加赛前已知的三盘/五盘赛制上下文及三个反对称实力交互，设置“不含赛制”的配对控制。设计包含 **2 种训练起点 × 2 种模型（线性/样条）× 2 种赛制条件 × 4 种正则/近期权重设置**。它们共享相同验证对象和 T−21 信息截止日期。

原有目标、标签和既有特征经过一致性检查；旧特征最大绝对差为 **{summary['previous_feature_max_absolute_error']:.2g}**。加入更多训练比赛不等于改变评分分母：2023 始终为同样 **2,588 场**。

## 升级门槛

先用 2020–2021 预测选择全局配置 `{gate['early_selected']}`，再在 2022 与先前选定的旧集成流程比较：新候选 **{gate['candidate_2022_loss']:.5f}**，旧流程 **{gate['reference_2022_loss']:.5f}**，改善 **{gate['absolute_improvement']:+.5f}**，所需改善至少 **{gate['required_improvement']:.3f}** log loss。决定选择流程后，使用 2020–2022 全部开发折重新选择最终配置，再查看本轮 2023 分数。

**2022 门槛年份和 2023 验证年份在前轮均已使用。** 这套门槛保持选择纪律，但不能恢复独立确认性证据；本轮也是探索实验。2024–2025 最终测试仍未评估。

## 统一验证比较

{chr(10).join(table)}

按门槛确定的当前推荐相对旧集成，log-loss 改善 **{selected_gain:+.5f}**。第三轮训练期选中候选相对旧集成的改善是 **{old.mean_log_loss-candidate_score.mean_log_loss:+.5f}**，日期分块配对区间为 **[{lo:.5f}, {hi:.5f}]**。负改善表示变差，区间未覆盖全部模型搜索或跨日期重复球员依赖。**{uncertainty_sentence}**

全部已保存比较中，2023 最低分是 `{best.model}`（**{best.mean_log_loss:.5f}**）。它是描述性排序，与门槛推荐不同，也不是 SOTA 证明。准确率最高的行同样只是描述性结果；本实验仍以 log loss 作为主要选择指标。

## 更多历史和赛制分别有什么作用

对每个模型/赛制单元分别在训练折选择设置，再比较 2009 与 2016 起点，四项 2023 log-loss 改善范围为 **[{era_range[0]:+.5f}, {era_range[1]:+.5f}]**。同理，在训练起点/模型单元中比较加入赛制，四项改善范围为 **[{context_range[0]:+.5f}, {context_range[1]:+.5f}]**。

Notebook 同时给出八个家族的全部分数、四个历史范围对照、四个赛制对照，以及 **相同正则和近期权重设置下的精确配对 CV**。家族对照会重新选正则/近期权重，不能当成固定参数下单一输入的因果效应；精确配对 CV 控制了这些设置，但仍只是预测性能比较。保留反向、无提升和年际不一致的结果。

## 复跑与交接

```bash
.venv/bin/python -m scripts.train_round3
.venv/bin/python -m scripts.report_round3
```

[执行 notebook]({prefix}round3.ipynb)、[HTML 报告]({prefix}round3.html) 和 [比较图]({image_prefix}round3_comparison.png) 单独保存，不覆盖前两轮。`search_registry.json` 记录配置和数据哈希；`temporal_cv.csv` 保存全部训练期结果；`frozen_selection.json` 保存门槛和模型选择；`validation_predictions.csv` 保存逐场概率；本地模型文件为 `frozen_models.joblib`。

下一步按冻结方案进行独立复现与最终测试，或明确登记新的探索实验。不能把重复查看 2023 得到的改进写成未见数据的测试成绩。
"""


def notebook_cells(data):
    import nbformat as nbf
    md,code=nbf.v4.new_markdown_cell,nbf.v4.new_code_cell
    return [md(narrative(data,True)),
        md("## 相同目标、不同训练规模\n\n以下 CV 训练数已经执行时间截止和训练起点过滤；合格总数不是每个折的实际训练数。"),
        code("from IPython.display import display, Image\nimport pandas as pd\nfrom scripts.report_round3 import (load_results, sample_table, comparison_table, family_table, contrast_tables, exact_cv_controls, OUT)\n"
             "data=load_results()\npd.set_option('display.max_columns',None)\npd.set_option('display.precision',5)\ndisplay(sample_table(data))"),
        md("## 所有保存的 2023 概率模型\n\n完整保留旧基准及所有家族候选。排序仅描述已查看过的验证年份。"),
        code("display(comparison_table(data))\ndisplay(Image(filename=str(OUT/'round3_comparison.png')))"),
        md("## 八个家族与分别重调参数的对照\n\n正改善表示对应改动后的 log loss 更低。这些家族各自重新选定正则和近期权重，因此不是完全相同参数的比较。"),
        code("display(family_table(data))\nera,context=contrast_tables(data)\ndisplay(era)\ndisplay(context)"),
        md("## 相同参数下的控制比较\n\n以下每对配置除了训练起点或赛制特征开关外，模型种类、正则和近期权重相同。展示训练期逐年结果，避免仅凭 2023 判断机制。"),
        code("display(exact_cv_controls(data))\ndisplay(data['cv'].sort_values('mean_log_loss'))"),
        md("## 日期分块的条件不确定性\n\n参考是第一轮冻结集成。这些区间没有修正多轮自适应设计与模型搜索。"),
        code("display(pd.DataFrame([{'model':n,'loss_reduction':v['mean_log_loss_reduction'],'lower95':v['ci95'][0],'upper95':v['ci95'][1]} for n,v in data['summary']['comparisons_vs_round1'].items()]).sort_values('loss_reduction',ascending=False))\n"
             "assert data['summary']['final_test_evaluated'] is False\nprint('2024–2025 final test remains unopened.')")]


def main():
    import nbformat as nbf
    from nbclient import NotebookClient
    from nbconvert import HTMLExporter
    from jupyter_client import KernelManager
    data=load_results()
    draw_chart(data)
    notebook=nbf.v4.new_notebook(cells=notebook_cells(data),metadata={"kernelspec":{"display_name":"Python 3","language":"python","name":"python3"}})
    manager=KernelManager(kernel_name="python3")
    manager.kernel_spec.argv=[sys.executable,"-m","ipykernel_launcher","-f","{connection_file}"]
    NotebookClient(notebook,km=manager,timeout=180,resources={"metadata":{"path":str(ROOT)}}).execute()
    nbf.write(notebook,ROOT/"round3.ipynb")
    exporter=HTMLExporter(template_name="lab")
    exporter.embed_images=True
    html,_=exporter.from_notebook_node(notebook)
    (ROOT/"round3.html").write_text(html)
    (OUT/"report_summary.md").write_text(narrative(data))
    print("Wrote separate Round 3 executed notebook, HTML, Chinese summary and comparison figure.")


if __name__=="__main__":
    main()
