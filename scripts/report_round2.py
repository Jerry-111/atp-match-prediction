"""Report saved Round 2 exploratory results without fitting or changing Round 1."""
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
OUT = ROOT / "results" / "round2"


def load_results():
    data = {name: pd.read_csv(OUT / filename) for name, filename in [
        ("scores", "validation_comparison.csv"), ("cv", "temporal_cv.csv"),
        ("predictions", "validation_predictions.csv")]}
    for key, filename in [("frozen", "frozen_selection.json"), ("summary", "summary.json"), ("registry", "search_registry.json")]:
        data[key] = json.loads((OUT / filename).read_text())
    data["old_frozen"] = json.loads((ROOT / "results/modeling/frozen_selection.json").read_text())
    if any(data[k]["final_test_evaluated"] for k in ("frozen", "summary", "registry")):
        raise ValueError("Round 2 development report requires an unopened final test")
    predictions = data["predictions"]
    if len(predictions) != 2588 or predictions.match_id.duplicated().any() or not predictions.season.eq(2023).all():
        raise ValueError("Expected identical 2,588 unique 2023 validation matches")
    if not data["scores"].matches.eq(2588).all():
        raise ValueError("Inconsistent model comparison denominator")
    return data


def label(name):
    if name == "rank_reference":
        return "Rank-only reference"
    if name == "round1_ensemble":
        return "Round 1 frozen ensemble"
    if name == "round2_ensemble":
        return "Round 2 candidate ensemble"
    if name.startswith("r1__"):
        from scripts.model_report import model_label
        return "R1 · " + model_label(name[4:])
    if name.startswith("dynamic_linear"):
        return "Linear + dynamic ratings"
    if name.startswith("adjusted_linear"):
        return "Linear + adjusted point skill"
    if name.startswith("fusion_recent"):
        return "Fusion linear · recency weighted"
    if name.startswith("fusion_linear"):
        return "Fusion linear · all new signals"
    if name.startswith("compact_linear"):
        return "Compact fusion linear"
    if name.startswith("odd_spline"):
        return "Symmetric spline + context"
    if name.startswith("raw_"):
        return "Uncalibrated dynamic rating"
    if name.startswith("cat2"):
        return "Round 2 CatBoost · " + ("compact" if "compact" in name else "full")
    if name.startswith("hist2"):
        return "Round 2 histogram boosting"
    return name.replace("_", " ")


def comparison_table(data):
    table = data["scores"].copy()
    table.insert(1, "description", table.model.map(label))
    table["pooled_2020_22_cv_log_loss"] = table.model.map(data["cv"].set_index("name").mean_log_loss)
    table["loss_reduction_vs_old_ensemble"] = table.model.map(
        {name: interval["mean_log_loss_reduction"] for name, interval in data["summary"]["comparisons_vs_round1"].items()})
    table["role"] = "Exploratory comparison"
    table.loc[table.model.eq("round1_ensemble"), "role"] = "Previous frozen recommendation"
    table.loc[table.model.eq(data["frozen"]["recommendation"]), "role"] = "Recommendation after 2022 upgrade gate"
    return table[["model", "description", "role", "matches", "mean_log_loss", "accuracy", "brier",
                  "pooled_2020_22_cv_log_loss", "loss_reduction_vs_old_ensemble"]]


def gate_table(data):
    gate = data["frozen"]["procedure_gate"]
    return pd.DataFrame([
        {"procedure": "Earlier-selected Round 1 blend reference", "2022_log_loss": gate["reference_2022_loss"]},
        {"procedure": "Earlier-selected candidate single model", "2022_log_loss": gate["new_single_2022_loss"]},
        {"procedure": "Earlier-selected candidate blend", "2022_log_loss": gate["new_blend_2022_loss"]},
    ])


def family_table(data):
    selected = data["frozen"]["selected_by_family"]
    scores = data["scores"].set_index("model")
    old_linear = "r1__" + data["old_frozen"]["best_single"]
    reference_loss = scores.loc[old_linear, "mean_log_loss"]
    rows = []
    for family, name in selected.items():
        if family.startswith("r1__"):
            continue
        row = scores.loc[name]
        rows.append({"family": family, "model": name, "description": label(name),
                     "2023_log_loss": row.mean_log_loss, "2023_accuracy": row.accuracy,
                     "loss_reduction_vs_old_best_single": reference_loss-row.mean_log_loss})
    return pd.DataFrame(rows).sort_values("2023_log_loss")


def plot_comparison(data):
    scores = data["scores"].sort_values("mean_log_loss")
    recommendation = data["frozen"]["recommendation"]
    colors = ["#007F7C" if name == recommendation else "#8C6428" if name == "round1_ensemble" else "#304A62" for name in scores.model]
    names = [label(name) + (" ★" if name == recommendation else "") for name in scores.model]
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.spines.top": False,
                         "axes.spines.right": False, "axes.spines.left": False})
    fig, (ax, accuracy) = plt.subplots(1, 2, figsize=(12.5, max(6, .40*len(scores))), sharey=True,
                                      gridspec_kw={"width_ratios": [3, 1.1]})
    y = np.arange(len(scores))
    ax.scatter(scores.mean_log_loss, y, c=colors, s=42, zorder=3)
    ax.set_yticks(y, names)
    ax.invert_yaxis()
    left = scores.mean_log_loss.min()-.008
    right = scores.mean_log_loss.max()+.013
    ax.set_xlim(left, right)
    old_loss = float(scores.loc[scores.model.eq("round1_ensemble"), "mean_log_loss"].iloc[0])
    ax.axvline(old_loss, color="#8C6428", alpha=.55, linestyle="--", linewidth=1)
    for yy, value in zip(y, scores.mean_log_loss):
        ax.text(value+.001, yy, f"{value:.5f}", va="center", fontsize=9)
    ax.set_xlabel("Mean log loss · lower is better")
    accuracy.scatter(scores.accuracy*100, y, c=colors, s=34, zorder=3)
    accuracy.set_xlim(scores.accuracy.min()*100-1.2, scores.accuracy.max()*100+3)
    accuracy.set_xlabel("Accuracy (%)")
    accuracy.tick_params(axis="y", left=False, labelleft=False)
    for yy, value in zip(y, scores.accuracy):
        accuracy.text(value*100+.10, yy, f"{value:.1%}", va="center", fontsize=9)
    for axis in (ax, accuracy):
        axis.grid(axis="x", alpha=.18)
        axis.set_axisbelow(True)
    fig.suptitle("Round 2 · same 2,588 matches, exploratory 2023 check", x=.02, ha="left", fontsize=16, fontweight="bold")
    fig.text(.02, .015, "★ Recommendation follows the training-period upgrade gate. Dashed line: previous frozen ensemble.\n"
             "2023 was already inspected in Round 1; these are exploratory scores. Final 2024–2025 test stays unopened.",
             fontsize=9, color="#465769")
    fig.tight_layout(rect=(0, .075, 1, .95))
    fig.savefig(OUT / "round2_comparison.png", dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_calibration(data):
    from scripts.model_report import calibration_rows
    chosen = data["frozen"]["recommendation"]
    names = list(dict.fromkeys(["round1_ensemble", "round2_ensemble", chosen]))
    fig, axes = plt.subplots(1, len(names), figsize=(5*len(names), 5.4), squeeze=False)
    for ax, name in zip(axes[0], names):
        table = calibration_rows(data["predictions"], name)
        ax.plot([0,1], [0,1], "--", color="#88939E", lw=1)
        ax.plot(table.predicted, table.observed, "-o", color="#007F7C")
        for row in table.itertuples():
            ax.annotate(str(row.count), (row.predicted,row.observed), xytext=(4,7),textcoords="offset points",fontsize=8)
        ax.set(xlim=(0,1), ylim=(0,1), xlabel="Mean P(A wins)", ylabel="Observed A win rate", title=label(name))
        ax.set_aspect("equal")
        ax.grid(alpha=.15)
    fig.suptitle("Round 2 calibration · exploratory 2023", x=.02, ha="left", fontsize=15, fontweight="bold")
    fig.text(.02,.02,"Fixed 0.1-wide bins; numbers are match counts. No calibrator is fitted to these plots.",fontsize=9,color="#465769")
    fig.tight_layout(rect=(0,.10,1,.94))
    fig.savefig(OUT / "round2_calibration.png",dpi=180,bbox_inches="tight",facecolor="white")
    plt.close(fig)


def narrative(data, notebook=False):
    frozen, summary = data["frozen"], data["summary"]
    gate = frozen["procedure_gate"]
    chosen = frozen["recommendation"]
    scores = data["scores"].set_index("model")
    old = scores.loc["round1_ensemble"]
    current = scores.loc[chosen]
    candidate = scores.loc["round2_ensemble"]
    lowest = data["scores"].sort_values(["mean_log_loss", "model"]).iloc[0]
    old_single = "r1__"+data["old_frozen"]["best_single"]
    best_new = min((n for n in frozen["selected_by_family"].values() if not n.startswith("r1__")),
                   key=lambda n: scores.loc[n, "mean_log_loss"])
    gain = old.mean_log_loss-current.mean_log_loss
    direction = "降低" if gain > 0 else "升高" if gain < 0 else "保持不变，变化为"
    upgrade = "保留上一轮冻结集成" if gate["procedure"] == "retain_round1" else "采用新一轮集成流程" if gate["procedure"] == "new_blend" else "采用新一轮单模型流程"
    table = ["| 模型 | 2023 log loss ↓ | 准确率 |", "| --- | ---: | ---: |"]
    names = list(dict.fromkeys(["rank_reference", old_single, "round1_ensemble", "round2_ensemble", best_new, chosen]))
    chinese_labels = {"rank_reference": "只用排名", old_single: "第一轮最强单模型", "round1_ensemble": "第一轮冻结集成",
                      "round2_ensemble": "第二轮候选集成", best_new: "本轮验证分数最低的新家族候选"}
    table += [f"| {chinese_labels.get(n, label(n))} (`{n}`) | {scores.loc[n,'mean_log_loss']:.5f} | {scores.loc[n,'accuracy']:.2%} |" for n in names]
    interval = summary["comparisons_vs_round1"]["round2_ensemble"]
    low, high = interval["ci95"]
    family = family_table(data)
    unsuccessful = [f"`{row.model}`（{row['2023_log_loss']:.5f}）" for _, row in family.iterrows()
                    if row["2023_log_loss"] >= scores.loc[old_single, "mean_log_loss"]]
    unsuccessful_text = "、".join(unsuccessful) if unsuccessful else "本轮所有已选择的新模型家族"
    unsuccessful_sentence = ("相对上一轮最强单模型，以下候选在这次 2023 检查中没有更低的 log loss："+unsuccessful_text+"。"
                             if unsuccessful else "本轮所有已选择的新模型家族的 2023 log loss 都低于上一轮最强单模型；这仍然是重复使用验证集的探索性结果。")
    root_prefix = "" if notebook else "../../"
    image_prefix = "results/round2/" if notebook else ""
    return f"""# 第二轮优化：候选扩展与升级判定

**结论：{upgrade}，当前推荐仍按事先定义的训练期规则确定，为 `{chosen}`。** 本轮新增 **{summary['new_candidates']}** 个候选配置，复用 **{summary['reused_candidates']}** 个上一轮候选的逐年预测；保留全部尝试记录。

## 先看升级门槛，再看 2023

家族配置、组合权重和单模型候选先用 **2020–2021** 的向前验证预测选择，随后在 **2022** 检查是否比对应旧流程至少降低 **{gate['threshold']:.3f}** log loss。2022 年旧流程为 **{gate['reference_2022_loss']:.5f}**，新单模型为 **{gate['new_single_2022_loss']:.5f}**，新组合为 **{gate['new_blend_2022_loss']:.5f}**。门槛决定的是拟合/集成流程，之后才用 2020–2022 的全部开发折重新拟合选择和权重。这个门槛是预先设定的实用改进要求，不等于统计显著性检验。

**2023 在上一轮已经看过，第二轮分数必须标为探索性。** 本轮没有用新的 2023 分数反复调参，但设计受上一轮结果启发，因此不能把这次 2023 当成全新的测试集。2024–2025 年最终测试仍未打开。

## 相同 2,588 场比赛的结果

{chr(10).join(table)}

当前推荐相对旧集成的 log loss **{direction} {abs(gain):.5f}**，准确率变化 **{100*(current.accuracy-old.accuracy):+.2f} 个百分点**。
新一轮候选集成相对旧集成的 log-loss 改善为 **{old.mean_log_loss-candidate.mean_log_loss:+.5f}**；按赛事开始日期分块的配对重抽样区间为 **[{low:.5f}, {high:.5f}]**。这是给定已训练/已选择模型的探索性区间，未覆盖模型搜索和跨日期重复球员依赖。

本轮保存的全部比较中，2023 最低分是 `{lowest.model}`，**{lowest.mean_log_loss:.5f}**；它是描述性排序，不替代上面的升级规则，也不构成 SOTA 证明。

## 尝试了什么，什么没有奏效

- **动态实力与不确定性：** Glicko 风格的整体/场地评分，在长时间休赛后扩大不确定性；没有新增赛后信息。
- **对手校正的发接发能力：** 将历史技术表现与当时对手的能力联系起来，尝试区分自身状态与对手强弱。
- **平滑非线性与可靠性交互：** 对排名和技术能力差使用受正则约束的样条，并让历史样本量等上下文影响预测；仍强制交换球员后概率互补。
- **融合与重加权：** 比较线性、近期加权、CatBoost、直方图提升及非负权重组合。

{unsuccessful_sentence} 这些是不同家族在独立调参后的候选比较，**不是逐组删除且其余设置完全相同的因果或消融结论**。不能因为一种融合模型变好，就断言每一种新增信息都有帮助。完整逐年 CV、家族结果和失败候选都保留在 notebook/CSV。

## 复跑与文件

```bash
.venv/bin/python -m scripts.train_round2
.venv/bin/python -m scripts.report_round2
```

第一轮的 notebook、模型和结果目录均保留。第二轮 [执行 notebook]({root_prefix}round2.ipynb)、[可直接打开的 HTML]({root_prefix}round2.html)、[比较图]({image_prefix}round2_comparison.png) 和 [校准图]({image_prefix}round2_calibration.png) 单独保存。

`search_registry.json` 保存 {summary['new_candidates']} 个新配置及数据哈希；`temporal_cv.csv` 保存全部新旧候选；`frozen_selection.json` 保存 2022 升级门槛决定；`validation_predictions.csv` 保存逐场 2023 概率；本地 `frozen_models.joblib` 保存冻结模型与组合。评估队友应先独立核验，再按被冻结的规则显式解锁最终测试；最终成绩不能从这里推断。
"""


def notebook_cells(data):
    import nbformat as nbf
    md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell
    return [
        md(narrative(data, notebook=True)),
        md("## 全部验证比较\n\n2023 分数是重复使用该验证年份后的探索性结果。表中的 CV 列只对应候选自身的时间向前预测；集成权重的训练损失不冒充独立 CV。"),
        code("from IPython.display import display, Image\nimport pandas as pd\n"
             "from scripts.report_round2 import load_results, comparison_table, gate_table, family_table, ROOT, OUT\n"
             "data = load_results()\npd.set_option('display.max_columns', None)\npd.set_option('display.precision', 5)\n"
             "display(comparison_table(data))\ndisplay(Image(filename=str(OUT / 'round2_comparison.png')))"),
        md("## 训练期升级门槛与逐年证据\n\n2022 门槛比较使用 2020–2021 选定的配置和权重；下面家族表则展示最终用全部三个开发折选择的配置。二者不是相同的模型拟合实例。"),
        code("display(gate_table(data))\n"
             "chosen = list(data['frozen']['selected_by_family'].values())\n"
             "display(data['cv'][data['cv'].name.isin(chosen)].sort_values('mean_log_loss'))\n"
             "display(pd.DataFrame({'component': data['frozen']['ensemble_components'], 'weight': data['frozen']['ensemble_weights']}))"),
        md("## 新方法家族：增益与失败都保留\n\n这张表以第一轮最强单模型为参照。不同家族还可能改变正则、特征组合和表示方式，不能把差异直接解释为单一特征的作用。"),
        code("display(family_table(data))\n"
             "print('All registered new configurations:', data['summary']['new_candidates'])\n"
             "print('Reused first-round configurations:', data['summary']['reused_candidates'])\n"
             "display(data['cv'][~data['cv'].reused].sort_values('mean_log_loss'))"),
        md("## 概率校准\n\n每个标签是该固定宽度概率区间的比赛数。图只用于诊断，不据此重新拟合校准参数。"),
        code("display(Image(filename=str(OUT / 'round2_calibration.png')))"),
        md("## 不确定性与交接\n\n每行区间比较同一批比赛的 log loss，参考是上一轮冻结集成。所有区间都是条件于已选择模型的探索性结果；不能修正全部反复搜索偏差。"),
        code("intervals = data['summary']['comparisons_vs_round1']\n"
             "display(pd.DataFrame([{'model': name, 'loss_reduction': r['mean_log_loss_reduction'], 'lower95': r['ci95'][0], 'upper95': r['ci95'][1]} for name,r in intervals.items()]).sort_values('loss_reduction',ascending=False))\n"
             "assert data['summary']['final_test_evaluated'] is False\n"
             "print('Final 2024–2025 test remains unopened.')"),
    ]


def main():
    import nbformat as nbf
    from nbclient import NotebookClient
    from nbconvert import HTMLExporter
    from jupyter_client import KernelManager

    data = load_results()
    plot_comparison(data)
    plot_calibration(data)
    notebook = nbf.v4.new_notebook(cells=notebook_cells(data), metadata={
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}})
    manager = KernelManager(kernel_name="python3")
    manager.kernel_spec.argv = [sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"]
    NotebookClient(notebook, km=manager, timeout=180, resources={"metadata": {"path": str(ROOT)}}).execute()
    nbf.write(notebook, ROOT / "round2.ipynb")
    exporter = HTMLExporter(template_name="lab")
    exporter.embed_images = True
    html, _ = exporter.from_notebook_node(notebook)
    (ROOT / "round2.html").write_text(html)
    (OUT / "report_summary.md").write_text(narrative(data))
    print("Wrote separate Round 2 executed notebook, HTML, Chinese summary and two figures.")


if __name__ == "__main__":
    main()
