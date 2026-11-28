"""Report the accuracy-first fifth round without fitting or changing selections."""
from __future__ import annotations

import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/"results/round5"
REFERENCE = "previous_probability_recommendation"
INFORMATION = {"core": "既有历史与年龄", "multilevel": "加入跨级别历史",
               "style": "加入技术组成", "both": "同时加入两组"}


def load_results():
    data = {key: pd.read_csv(OUT/name) for key, name in (
        ("scores", "validation_comparison.csv"), ("predictions", "validation_predictions.csv"),
        ("cv", "temporal_cv.csv"))}
    for key, name in (("frozen", "frozen_selection.json"), ("summary", "summary.json"),
                      ("registry", "search_registry.json")):
        data[key] = json.loads((OUT/name).read_text())
        if data[key]["final_test_evaluated"]:
            raise ValueError("This exploratory report requires an unopened final holdout")
    rows, scores = data["predictions"], data["scores"]
    if len(rows) != 2588 or rows.match_id.duplicated().any() or not rows.season.eq(2023).all():
        raise ValueError("Expected all 2,588 unchanged validation matches")
    original = pd.read_csv(ROOT/"data/processed/model_features.csv")
    identity = ["match_id", "season", "a_won"]
    if not rows[identity].equals(original.loc[original.season.eq(2023), identity].reset_index(drop=True)):
        raise ValueError("Validation target identity or orientation changed")
    if scores.model.duplicated().any() or not scores.matches.eq(len(rows)).all():
        raise ValueError("Comparison table model identities or denominators are invalid")
    if not set(scores.model).issubset(rows):
        raise ValueError("Saved model probability columns are missing")
    chosen = data["frozen"]["recommendation"]
    if scores.loc[scores.recommended, "model"].tolist() != [chosen]:
        raise ValueError("Reported recommendation disagrees with frozen selection")
    data["specs"] = {s["name"]: s for s in data["registry"]["candidates"]}
    if set(data["cv"].name) != set(data["specs"]):
        raise ValueError("Cross-validation table does not cover the frozen candidate registry")
    return data


def label(name, data):
    aliases = {REFERENCE: "Round 4 frozen probability model",
        "previous_accuracy_comparator": "Round 3 expanded linear comparator",
        "previous_exploratory_best_loss": "Round 4 exploratory loss comparator",
        "rank_reference": "Rank-only reference", "round1_best_single": "Round 1 linear comparator",
        "accuracy_grid_blend": "Accuracy-selected convex blend"}
    if name in aliases:
        return aliases[name]
    spec = data["specs"][name]
    family, kind = spec["family"], spec["kind"]
    if kind.startswith("r5_"):
        head = "logistic" if kind == "r5_logistic" else "squared hinge"
        design = spec["design"]
        if design == "rff":
            design = f"RFF {spec['params']['components']} / g={spec['params']['gamma_multiplier']:g}"
        return f"{design} · {head} · C={spec['params']['C']:g}"
    if family == "r5_existing_spline":
        return f"Existing spline control · C={spec['params']['C']:g}"
    if family == "new_catboost":
        return f"Add both groups · CatBoost d={spec['params']['depth']} / n={spec['params']['iterations']}"
    if family == "new_histboost":
        return f"Add both groups · histogram boosting / leaves={spec['params']['max_leaf_nodes']}"
    group = name.split("_", 1)[0]
    if group in INFORMATION:
        prefix = {"core": "Core + age", "multilevel": "Add cross-level history",
                  "style": "Add technical composition", "both": "Add both information groups"}[group]
        learner = "linear" if kind == "logistic" else "spline"
        half = spec.get("half_life", spec["params"].get("half_life"))
        return f"{prefix} · {learner} · C={spec['params']['C']:g} / h={half}"
    return {"new_catboost": "All new groups · CatBoost", "new_histboost": "All new groups · histogram boosting",
            "all_inputs": "All old and new information"}.get(family, family)


def comparison_table(data):
    table = data["scores"].copy()
    table.insert(1, "description", [label(n, data) for n in table.model])
    table["correct_short_of_70"] = np.maximum(0, data["summary"]["correct_required_for_70"]-table.correct)
    table["CV_accuracy"] = table.model.map(data["cv"].set_index("name").accuracy)
    table["CV_log_loss"] = table.model.map(data["cv"].set_index("name").mean_log_loss)
    pooled = data["frozen"]["pooled_selection"]
    if pooled["choice"] == "accuracy_grid_blend":
        blend = table.model.eq("accuracy_grid_blend")
        table.loc[blend, "CV_accuracy"] = pooled["selected_accuracy"]
        table.loc[blend, "CV_log_loss"] = pooled["selected_log_loss"]
    return table.sort_values(["accuracy", "mean_log_loss", "model"], ascending=[False, True, True])


def matched_information_controls(data):
    """45 exact-setting additions/removals, retaining every annual change."""
    cv = data["cv"].set_index("name")
    settings = [("linear", f"linear_C{c:g}_half{half}")
                for c in (.001, .01, .1) for half in (None, 3.)]
    settings += [("spline", f"spline_C{c:g}_half3") for c in (.003, .01, .03)]
    contrasts = [("add", "core", group) for group in ("multilevel", "style", "both")]
    contrasts += [("remove_multilevel", "both", "style"), ("remove_style", "both", "multilevel")]
    result = []
    for learner, setting in settings:
        for comparison, before, after in contrasts:
            reference, candidate = f"{before}_{setting}", f"{after}_{setting}"
            row = {"comparison": comparison, "learner": learner, "information": after,
                   "reference": reference, "candidate": candidate}
            for year in (None, 2020, 2021, 2022):
                ac = "accuracy" if year is None else f"{year}_accuracy"
                ll = "mean_log_loss" if year is None else f"{year}_log_loss"
                prefix = "pooled" if year is None else str(year)
                row[prefix+"_accuracy_gain"] = cv.loc[candidate, ac]-cv.loc[reference, ac]
                row[prefix+"_loss_reduction"] = cv.loc[reference, ll]-cv.loc[candidate, ll]
            result.append(row)
    return pd.DataFrame(result)


def matched_objective_controls(data):
    """12 identical-design/C comparisons: squared hinge minus logistic."""
    cv = data["cv"].set_index("name")
    result = []
    for spec in data["specs"].values():
        if spec["kind"] != "r5_logistic":
            continue
        reference = spec["name"]
        candidate = reference.replace("_logistic_", "_squared_hinge_")
        row = {"design": spec["design"], "components": spec["params"]["components"],
               "gamma_multiplier": spec["params"]["gamma_multiplier"], "C": spec["params"]["C"],
               "reference": reference, "candidate": candidate}
        for year in (None, 2020, 2021, 2022):
            ac = "accuracy" if year is None else f"{year}_accuracy"
            ll = "mean_log_loss" if year is None else f"{year}_log_loss"
            prefix = "pooled" if year is None else str(year)
            row[prefix+"_accuracy_gain"] = cv.loc[candidate, ac]-cv.loc[reference, ac]
            row[prefix+"_loss_reduction"] = cv.loc[reference, ll]-cv.loc[candidate, ll]
        result.append(row)
    return pd.DataFrame(result)


def family_table(data):
    cv, scores = data["cv"].set_index("name"), data["scores"].set_index("model")
    return pd.DataFrame([{
        "family": family, "model": name, "CV_accuracy": cv.loc[name, "accuracy"],
        "CV_log_loss": cv.loc[name, "mean_log_loss"], "2023_accuracy": scores.loc[name, "accuracy"],
        "2023_log_loss": scores.loc[name, "mean_log_loss"],
    } for family, name in data["frozen"]["pooled_selection"]["selected_by_family"].items()]).sort_values(
        ["CV_accuracy", "CV_log_loss"], ascending=[False, True])


def plot_comparison(data):
    table = comparison_table(data)
    chosen = data["frozen"]["recommendation"]
    colors = ["#087F72" if name == chosen else "#314C68" for name in table.model]
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9,
                         "axes.spines.top": False, "axes.spines.right": False, "axes.spines.left": False})
    fig, (acc, loss) = plt.subplots(1, 2, figsize=(15, max(7, len(table)*.37)), sharey=True,
                                  gridspec_kw={"width_ratios": [2.1, 1.3]})
    y = np.arange(len(table))
    acc.scatter(table.accuracy*100, y, c=colors, s=36)
    acc.set_yticks(y, [label(name, data)+(" *" if name == chosen else "") for name in table.model])
    acc.invert_yaxis()
    ref = float(table.loc[table.model.eq(REFERENCE), "accuracy"].iloc[0])*100
    acc.axvline(ref, color="#A16E22", linestyle="--", linewidth=1)
    acc.axvline(70, color="#777777", linestyle=":", linewidth=1)
    acc.set_xlim(min(62., table.accuracy.min()*100-1), max(71., table.accuracy.max()*100+1.8))
    for pos, row in zip(y, table.itertuples()):
        acc.text(row.accuracy*100+.08, pos, f"{row.accuracy:.2%} ({row.correct})", va="center", fontsize=8)
    loss.scatter(table.mean_log_loss, y, c=colors, s=36)
    span = max(.01, table.mean_log_loss.max()-table.mean_log_loss.min())
    loss.set_xlim(table.mean_log_loss.min()-span*.1, table.mean_log_loss.max()+span*.5)
    for pos, value in zip(y, table.mean_log_loss):
        loss.text(value+span*.018, pos, f"{value:.5f}", va="center", fontsize=8)
    loss.tick_params(axis="y", labelleft=False, left=False)
    acc.set_xlabel("Accuracy (%) and correct matches / 2,588 · higher is better")
    loss.set_xlabel("Mean log loss · lower is better")
    for axis in (acc, loss):
        axis.grid(axis="x", alpha=.17)
        axis.set_axisbelow(True)
    fig.suptitle("Round 5 · accuracy-first selection, same complete cohort", x=.02, ha="left", fontsize=16, fontweight="bold")
    fig.text(.02, .016, "* Frozen recommendation after the temporal gate. Dashed: Round 4; dotted: 70% target.\n"
             "2023 is reused exploratory validation. No final-test result or market-informed model is shown here.", fontsize=9)
    fig.tight_layout(rect=(0, .065, 1, .95))
    fig.savefig(OUT/"round5_comparison.png", dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def narrative(data, notebook=False):
    frozen, summary = data["frozen"], data["summary"]
    gate, pooled = frozen["procedure_gate"], frozen["pooled_selection"]
    chosen, candidate = frozen["recommendation"], pooled["choice"]
    scores = data["scores"].set_index("model")
    ref, selected = scores.loc[REFERENCE], scores.loc[chosen]
    best = data["scores"].sort_values(["accuracy", "mean_log_loss", "model"], ascending=[False, True, True]).iloc[0]
    best_loss = data["scores"].sort_values(["mean_log_loss", "model"]).iloc[0]
    target = summary["correct_required_for_70"]
    short = max(0, target-int(selected.correct))
    status = "通过升级门槛，采用准确率优先候选" if gate["procedure"] == "adopt_accuracy_model" else "未通过升级门槛，保留第四轮概率推荐"
    target_text = (f"当前推荐 **{int(selected.correct):,}/2,588（{selected.accuracy:.2%}）**，距离70%至少还需净增加 **{short}场**正确预测。"
                   if short else f"当前推荐在重复使用的2023样本上达到 **{selected.accuracy:.2%}**；这不等于最终测试已达到70%。")
    names = list(dict.fromkeys(["rank_reference", REFERENCE, "previous_accuracy_comparator",
        "previous_exploratory_best_loss", pooled["best_single"], candidate, chosen, best.model, best_loss.model]))
    rows = ["| 模型 / 角色 | 正确场次 | 准确率 ↑ | log loss ↓ | 距70%差额 |",
            "| --- | ---: | ---: | ---: | ---: |"]
    for name in names:
        row = scores.loc[name]
        role = "本轮正式推荐" if name == chosen else "探索性对照"
        if name == candidate and name != chosen:
            role = "训练期候选，未获升级"
        elif name == best_loss.model and name != chosen:
            role = "2023最低观察loss，探索性"
        rows.append(f"| `{name}`（{role}） | {int(row.correct):,} | {row.accuracy:.2%} | {row.mean_log_loss:.5f} | {max(0,target-int(row.correct))} |")
    comp = summary["comparisons_vs_round4"][candidate]
    ai, li = comp["accuracy"], comp["log_loss"]
    low, high = ai["ci95"]
    uncertainty = "准确率区间跨过0，不能确认优于第四轮推荐。" if low <= 0 <= high else "这个条件区间未跨过0，但未覆盖多轮搜索的不确定性。"
    controls = matched_information_controls(data)
    control_lines = []
    for (group, learner), table in controls[controls.comparison.eq("add")].groupby(["information", "learner"], sort=False):
        values = table.pooled_accuracy_gain.to_numpy()*100
        learner_label = "线性" if learner == "linear" else "样条"
        control_lines.append(f"- **{INFORMATION[group]} / {learner_label}**：{len(table)}组同设置配对，pooled CV准确率变化 **[{values.min():+.3f}, {values.max():+.3f}]个百分点**；其中{int((values>0).sum())}组提高。")
    objective = matched_objective_controls(data)
    hv = objective.pooled_accuracy_gain.to_numpy()*100
    blend_text = "、".join(f"`{name}` × {weight:.1f}" for name, weight in zip(pooled["components"], pooled["weights"]))
    early_text = "、".join(f"`{name}` × {weight:.1f}" for name, weight in zip(gate["early_selection"]["components"], gate["early_selection"]["weights"]))
    prefix = "" if notebook else "../../"
    image = "results/round5/round5_comparison.png" if notebook else "round5_comparison.png"
    return f"""# 第五轮：向全量70%准确率推进

**结论：{status}，冻结推荐为 `{chosen}`。** {target_text} 本轮最好的已观察2023准确率是 `{best.model}` 的 **{best.accuracy:.2%}**（{int(best.correct):,}场正确），距目标还差{max(0,target-int(best.correct))}场；这个排序不替代冻结选择。

主实验比较 **{summary['new_candidates']}个新配置和{summary['reused_candidates']}个旧配置**，使用{summary['features']}项候选输入、33,833场训练目标和原样保留的2,588场2023目标。不同模型使用对应输入子集。目标70%需要至少 **{target:,}场**正确；没有挑高置信比赛、减少分母或放宽T−21历史规则。市场信息属于另一个信息条件，不混入本表。

## 为什么改变方法，以及如何选择

用户授权本轮改为**准确率优先**。跨级别历史补充Tour以外已发生的对局，技术组成细化已有发球统计；学习器比较有限年龄交互、随机傅里叶残差及logistic/squared-hinge目标。原概率推荐保持独立，不因新目标被改名或覆盖。公式、来源、反证和局限见[本轮协议]({prefix}docs/research/round5_protocol.md)及[学习器说明]({prefix}docs/research/round5_learning.md)。换损失不保证更准，校准也不能改变固定0.5阈值下的胜负决策。

2020–2021先选accuracy最高的候选；log loss必须不超过第四轮参照+0.005，同accuracy再比较log loss及固定名称。每种方法取一个合格配置，从至多四个候选枚举步长0.1的非负、和为1权重，包含单模型顶点。早期锁定的是：{early_text}。

在2022门槛年，参考accuracy为 **{gate['reference_2022_accuracy']:.2%}**，候选为 **{gate['candidate_2022_accuracy']:.2%}**，变化 **{100*gate['accuracy_improvement']:+.3f}个百分点**；需至少提高 **{100*gate['required_accuracy_improvement']:.1f}个百分点**。候选log loss为 **{gate['candidate_2022_loss']:.5f}**，参考为 **{gate['reference_2022_loss']:.5f}**，允许恶化不超过 **{gate['allowed_loss_increase']:.3f}**。两项条件必须同时满足；这不是显著性检验。

全部2020–2022折的候选重选结果为：{blend_text}。若升级失败，它只作为探索性对照；正式推荐仍为旧模型。全部选择和权重在本轮2023评分前冻结。**2022与2023此前已被查看，本轮仍是后续探索，2024–2025最终测试保持关闭。** 三个旧候选的纳入也继承此前探索历史，不能称为独立预注册。

## 同一批比赛的结果

{chr(10).join(rows)}

训练期选出的本轮候选 `{candidate}` 相对第四轮模型，纠正 **{ai['corrected']}场**旧错、同时新增 **{ai['newly_wrong']}场**错误，净变化 **{ai['net_correct']:+d}场**，accuracy变化 **{100*ai['accuracy_improvement']:+.3f}个百分点**。赛事开始日期分块的配对区间为 **[{100*low:+.3f}, {100*high:+.3f}]个百分点**。**{uncertainty}** 区间还没有涵盖跨日期重复球员依赖和全部模型搜索。

候选log-loss改善（旧减新）为 **{li['mean_log_loss_reduction']:+.5f}**，对应条件区间 **[{li['ci95'][0]:+.5f}, {li['ci95'][1]:+.5f}]**。负数表示概率预测变差；准确率提升不能替代概率质量报告。完整表另外保留Brier分数。

本轮保存比较中，最低2023 log loss来自 `{best_loss.model}`：**{best_loss.mean_log_loss:.5f}**，准确率 **{best_loss.accuracy:.2%}**（{int(best_loss.correct):,}场正确）。它是值得保留的概率预测对照，但这个已观察排序不改变本轮accuracy门槛或正式推荐。

![第五轮全量准确率与概率损失]({image})

## 哪些信息和目标有帮助

以下是严格固定学习器、正则和近期权重时，加入信息后的训练期变化；范围展示多个设置，不是置信区间：

{chr(10).join(control_lines)}

相同表示和C下，squared hinge相对logistic的12项配对，pooled CV准确率变化范围为 **[{hv.min():+.3f}, {hv.max():+.3f}]个百分点**，{int((hv>0).sum())}项提高。各项log-loss变化和逐年方向同时保留，不能用单个最优设置证明某损失天然更优。

Notebook保存全部 **{len(controls)}项信息添加/删除对照**和 **{len(objective)}项学习目标对照**。删除比较以“删除后减删除前”的accuracy变化及“删除前减删除后”的loss变化计，正数表示删除后更好。相关输入的增量是条件预测比较，不能解释成因果贡献。家族赢家还经过重新选择参数，不等同于严格同参数消融。所有候选CV都保留；未满足概率护栏的家族不会被强行列为正式候选。

## 复跑与交接

```bash
.venv/bin/python -m scripts.train_round5
.venv/bin/python -m scripts.report_round5
```

[执行notebook]({prefix}round5.ipynb)、[HTML]({prefix}round5.html)、[本轮协议]({prefix}docs/research/round5_protocol.md)和独立的模型文件均保留此前实验。`search_registry.json`记录全部候选，`temporal_cv.csv`记录训练期比较，`frozen_selection.json`记录门槛与权重，`validation_predictions.csv`保存逐场概率，`frozen_models.joblib`用于冻结推理。

最终测试应比较本轮冻结推荐、第四轮概率推荐和既有强对照，使用相同信息时点和完整人群；在那之前，不将重复使用2023上的最高准确率写成已确认的70%泛化能力或SOTA。
"""


def notebook_cells(data):
    import nbformat as nbf
    md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell
    return [md(narrative(data, True)),
        md("## 完整模型表\n\n正式推荐由冻结门槛决定，2023排序只是探索性观察。"),
        code("from IPython.display import display\nimport pandas as pd\n"
             "from scripts.report_round5 import load_results,comparison_table,family_table,matched_information_controls,matched_objective_controls\n"
             "data=load_results()\npd.set_option('display.max_columns',None)\npd.set_option('display.max_rows',None)\npd.set_option('display.precision',6)\n"
             "display(comparison_table(data))"),
        md("## 全部同设置信息与学习目标对照\n\n所有变化统一以正数表示候选更好；保留逐年方向，不将模型相关性解释为因果。"),
        code("display(matched_information_controls(data))\ndisplay(matched_objective_controls(data))"),
        md("## 训练期家族选择与全部候选\n\n家族先过log-loss护栏，再按accuracy选择。未满足护栏的候选仍保存在完整CV表。"),
        code("display(family_table(data))\ndisplay(data['cv'].sort_values(['accuracy','mean_log_loss'],ascending=[False,True]))"),
        md("## 冻结决定与配对不确定性\n\n纠正旧错和新增错误均需计入净增益；日期块区间没有修复多轮模型搜索偏差。"),
        code("display(data['frozen'])\n"
             "display(pd.DataFrame([{'model':n,**v['accuracy']} for n,v in data['summary']['comparisons_vs_round4'].items()]))\n"
             "assert data['summary']['final_test_evaluated'] is False\nprint('2024–2025 final holdout remains unopened.')")]


def main():
    import nbformat as nbf
    from nbclient import NotebookClient
    from nbconvert import HTMLExporter
    from jupyter_client import KernelManager
    data = load_results()
    plot_comparison(data)
    notebook = nbf.v4.new_notebook(cells=notebook_cells(data), metadata={"kernelspec": {
        "display_name": "Python 3", "language": "python", "name": "python3"}})
    manager = KernelManager(kernel_name="python3")
    manager.kernel_spec.argv = [sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"]
    NotebookClient(notebook, km=manager, timeout=180, resources={"metadata": {"path": str(ROOT)}}).execute()
    nbf.write(notebook, ROOT/"round5.ipynb")
    exporter = HTMLExporter(template_name="lab")
    exporter.embed_images = True
    html, _ = exporter.from_notebook_node(notebook)
    (ROOT/"round5.html").write_text(html)
    (OUT/"report_summary.md").write_text(narrative(data))
    comparison_table(data).to_csv(OUT/"report_comparison.csv", index=False)
    matched_information_controls(data).to_csv(OUT/"matched_information_controls.csv", index=False)
    matched_objective_controls(data).to_csv(OUT/"matched_objective_controls.csv", index=False)
    print("Wrote independent Round 5 notebook, HTML, report, controls and comparison figure.")


if __name__ == "__main__":
    main()
