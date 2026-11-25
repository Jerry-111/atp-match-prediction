"""Render the frozen modeling results, executed notebooks and standalone HTML.

This reports saved predictions only; it never fits or selects a model or reads
final-test outcomes. Run after ``python -m scripts.train_models --with-serve``.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "modeling"
SECTION_MARKER = "<!-- frozen-modeling-results -->"
COLORS = {"navy": "#203C58", "teal": "#007F7C", "gold": "#A36A15", "gray": "#7B8794"}


def load_results(output=OUT):
    output = Path(output)
    data = {
        "comparison": pd.read_csv(output / "validation_comparison.csv"),
        "predictions": pd.read_csv(output / "validation_predictions.csv"),
        "cv": pd.read_csv(output / "temporal_cv.csv"),
        "frozen": json.loads((output / "frozen_selection.json").read_text()),
        "summary": json.loads((output / "summary.json").read_text()),
    }
    if data["summary"]["final_test_evaluated"] or data["frozen"]["final_test_evaluated"]:
        raise ValueError("This development report expects an unopened final holdout")
    rows = data["predictions"]
    if not rows.season.eq(2023).all() or rows.match_id.duplicated().any():
        raise ValueError("Validation predictions must be unique 2023 matches")
    if len(rows) != 2588 or not data["comparison"].matches.eq(len(rows)).all():
        raise ValueError("Comparison must use the complete shared 2,588-match validation set")
    return data


def model_label(name):
    if name.startswith("lr_"):
        bits = name.split("_")[1]
        groups = ["Rank"] + [label for flag, label in zip(bits, ["form", "surface", "H2H"]) if flag == "1"]
        return " + ".join(groups)
    if name.startswith("raw_elo_"):
        return "Overall Elo " + name.split("_")[2].upper()
    if name == "oof_convex_ensemble":
        return "Temporal OOF ensemble"
    return name.replace("_", " ")


def comparison_table(data):
    table = data["comparison"].copy()
    table.insert(1, "description", table.model.map(model_label))
    table["selection_role"] = "Diagnostic comparison"
    table.loc[table.model.eq(data["summary"]["prescribed_subset_selected_on_2023"]), "selection_role"] = "8-subset winner selected on 2023"
    table.loc[table.model.eq(data["frozen"]["advanced_champion"]), "selection_role"] = "Advanced recommendation frozen before 2023"
    cv = data["cv"].set_index("name").mean_log_loss
    table["training_period_cv_log_loss"] = table.model.map(cv)
    # Ensemble weights were fit on OOF rows, so their fit loss is deliberately
    # not placed beside true out-of-fold candidate prediction losses.
    return table[["model", "description", "selection_role", "matches", "mean_log_loss",
                  "accuracy", "brier", "training_period_cv_log_loss"]]


def subset_contributions(data):
    selected = data["frozen"]["selected_by_family"]
    scores = data["comparison"].set_index("model")
    full = selected["lr_111"]
    rows = []
    for family, removed in [("lr_011", "Recent overall form"), ("lr_101", "Same-surface results"), ("lr_110", "Head-to-head history")]:
        reduced = selected[family]
        rows.append({"removed_from_four_feature_model": removed,
                     "full_model_log_loss": scores.loc[full, "mean_log_loss"],
                     "reduced_model_log_loss": scores.loc[reduced, "mean_log_loss"],
                     "loss_increase_when_removed": scores.loc[reduced, "mean_log_loss"] - scores.loc[full, "mean_log_loss"],
                     "reduced_model": reduced})
    return pd.DataFrame(rows)


def advanced_diagnostics(data):
    selected = data["frozen"]["selected_by_family"]
    families = ["all_history_reference", "without_elo", "without_serve", "without_rank"]
    names = [selected[f] for f in families if f in selected]
    table = comparison_table(data).set_index("model").loc[names].reset_index()
    reference = selected.get("all_history_reference")
    if reference:
        reference_loss = data["comparison"].set_index("model").loc[reference, "mean_log_loss"]
        table["loss_increase_vs_full_linear_reference"] = table.mean_log_loss - reference_loss
    return table


def probability_metrics(rows, name):
    y = rows.a_won.to_numpy(dtype=float)
    p = np.clip(rows[name].to_numpy(dtype=float), 1e-8, 1 - 1e-8)
    return {"matches": len(rows), "log_loss": float(np.mean(-y*np.log(p)-(1-y)*np.log1p(-p))),
            "accuracy": float(np.mean((p >= .5) == y)), "brier": float(np.mean((p-y)**2))}


def subgroup_table(data):
    rows = data["predictions"]
    names = list(dict.fromkeys([data["frozen"]["selected_by_family"]["lr_000"], data["frozen"]["advanced_champion"]]))
    groups = [("All validation", np.ones(len(rows), dtype=bool))]
    groups += [(surface, rows.surface.eq(surface)) for surface in ("Hard", "Clay", "Grass")]
    groups += [("Rank gap ≤ 10", (rows.rank_a-rows.rank_b).abs().le(10))]
    return pd.DataFrame([dict(group=label, model=name, **probability_metrics(rows[mask], name))
                         for label, mask in groups for name in names])


def worked_match(data):
    rows = data["predictions"]
    example = rows[rows.match_id.eq("2023-422:55")]
    if len(example) != 1:
        raise ValueError("Missing Cincinnati worked match 2023-422:55")
    names = list(dict.fromkeys([data["frozen"]["selected_by_family"][f] for f in ("lr_000", "lr_111", "raw_elo")]
                              + [data["frozen"]["advanced_champion"]]))
    metadata = example[["match_id", "tourney_date", "player_a_name", "player_b_name", "rank_a", "rank_b",
                        "log_rank_diff", "overall_rate_diff", "surface_rate_diff", "h2h_rate_diff", "a_won"]]
    probabilities = pd.DataFrame([{"model": name, "P(Alcaraz wins)": float(example[name].iloc[0])} for name in names])
    return metadata, probabilities


def draw_comparison(data, path):
    table = data["comparison"].sort_values("mean_log_loss", ascending=True)
    champion = data["frozen"]["advanced_champion"]
    labels = [model_label(name) + ("  ★" if name == champion else "") for name in table.model]
    colors = [COLORS["teal"] if name == champion else COLORS["navy"] for name in table.model]
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.spines.top": False,
                         "axes.spines.right": False, "axes.spines.left": False})
    fig, (ax, acc) = plt.subplots(1, 2, figsize=(12, max(5.5, len(table)*.39)),
                                 gridspec_kw={"width_ratios": [3, 1]}, sharey=True)
    y = np.arange(len(table))
    ax.barh(y, table.mean_log_loss, color=colors, height=.66)
    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    ax.set_xlabel("Mean log loss · lower is better")
    ax.set_xlim(0, table.mean_log_loss.max() * 1.13)
    for yy, loss in zip(y, table.mean_log_loss):
        ax.text(loss + .006, yy, f"{loss:.4f}", va="center", fontsize=9)
    acc.scatter(table.accuracy * 100, y, color=colors, s=36)
    for yy, accuracy in zip(y, table.accuracy):
        acc.text(accuracy*100+.15, yy, f"{accuracy:.1%}", va="center", fontsize=9)
    acc.set_xlim(min(table.accuracy.min()*100-2, 60), table.accuracy.max()*100+4)
    acc.set_xlabel("Accuracy (%)")
    acc.tick_params(axis="y", left=False, labelleft=False)
    for axis in (ax, acc):
        axis.grid(axis="x", alpha=.18)
        axis.set_axisbelow(True)
    fig.suptitle("Same 2,588 matches · 2023 validation", x=.02, ha="left", fontsize=16, fontweight="bold")
    fig.text(.02, .015, "★ Advanced recommendation frozen using 2020–2022 folds. Ranking-rule accuracy shown in notebook.\n"
             "2024–2025 final test remains unopened. Validation ordering does not establish state of the art.", fontsize=9, color="#485666")
    fig.tight_layout(rect=(0, .065, 1, .95))
    fig.savefig(path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def calibration_rows(rows, model, bins=10):
    p = rows[model].to_numpy()
    # Fixed bins, no data-driven calibration or model adjustment.
    bucket = np.minimum((p*bins).astype(int), bins-1)
    table = pd.DataFrame({"bin": bucket, "p": p, "y": rows.a_won.to_numpy()}).groupby("bin").agg(
        predicted=("p", "mean"), observed=("y", "mean"), count=("y", "size"))
    return table


def draw_calibration(data, path):
    selected = data["frozen"]["selected_by_family"]
    models = list(dict.fromkeys([selected["lr_000"], selected["raw_elo"], data["frozen"]["advanced_champion"]]))
    fig, axes = plt.subplots(1, len(models), figsize=(13, 5.2), squeeze=False)
    for ax, name in zip(axes[0], models):
        table = calibration_rows(data["predictions"], name)
        ax.plot([0, 1], [0, 1], "--", color="#9AA3AB", linewidth=1)
        ax.plot(table.predicted, table.observed, "-o", color=COLORS["teal"], linewidth=1.8)
        for row in table.itertuples():
            ax.annotate(str(row.count), (row.predicted, row.observed), xytext=(4, 7), textcoords="offset points", fontsize=8)
        ax.set(xlim=(0, 1), ylim=(0, 1), xlabel="Mean predicted P(A wins)", ylabel="Observed A win rate",
               title=model_label(name))
        ax.set_aspect("equal", adjustable="box")
        ax.grid(alpha=.15)
    fig.suptitle("Probability calibration · 2023 validation", x=.02, ha="left", fontsize=16, fontweight="bold")
    fig.text(.02, .015, "Fixed 0.1-wide bins; labels give match counts. Small bins are noisy. These plots do not refit calibration.",
             fontsize=9, color="#485666")
    fig.tight_layout(rect=(0, .13, 1, .94))
    fig.savefig(path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def narrative(data):
    scores = data["comparison"].set_index("model")
    frozen, summary = data["frozen"], data["summary"]
    champion = frozen["advanced_champion"]
    rank = frozen["selected_by_family"]["lr_000"]
    c, r = scores.loc[champion], scores.loc[rank]
    interval = summary["paired_vs_rank_logistic"][champion]
    low, high = interval["ci95"]
    best2023 = data["comparison"].sort_values("mean_log_loss").iloc[0].model
    return f"""The advanced recommendation is **`{champion}`**, frozen using training-period forward folds before scoring 2023.
On the same **{int(c.matches):,} validation matches**, its log loss is **{c.mean_log_loss:.5f}**, compared with **{r.mean_log_loss:.5f}** for rank-only logistic regression.
That is an absolute log-loss reduction of **{r.mean_log_loss-c.mean_log_loss:.5f}** ({(r.mean_log_loss-c.mean_log_loss)/r.mean_log_loss:.2%} relative).
Accuracy changes from **{r.accuracy:.2%}** to **{c.accuracy:.2%}** ({100*(c.accuracy-r.accuracy):+.2f} percentage points).
The deterministic ranking rule achieves **{summary['ranking_rule']['accuracy']:.2%}** and has no probability log-loss score.

The paired tournament-start-date bootstrap interval for the log-loss reduction is **[{low:.5f}, {high:.5f}]**.
This is a conditional, exploratory interval: it does not account for the model search, dependence across dates or future-season drift.

The lowest saved 2023 log loss belongs to **`{best2023}`**. This is a descriptive validation result, distinct from the pre-2023 advanced recommendation.
The course's eight-subset protocol selects **`{summary['prescribed_subset_selected_on_2023']}`** on 2023.
**No 2024–2025 test result has been computed; state-of-the-art superiority is not established.**"""


def chinese_summary(data):
    frozen, summary = data["frozen"], data["summary"]
    selected = frozen["selected_by_family"]
    champion = frozen["advanced_champion"]
    scores = data["comparison"].set_index("model")
    rank, chosen = scores.loc[selected["lr_000"]], scores.loc[champion]
    improvement = rank.mean_log_loss - chosen.mean_log_loss
    low, high = summary["paired_vs_rank_logistic"][champion]["ci95"]
    names = list(dict.fromkeys([selected["lr_000"], selected["lr_111"], selected["raw_elo"], champion]))
    rows = ["| 模型 | 2023 log loss ↓ | 准确率 |", "| --- | ---: | ---: |"]
    rows += [f"| `{name}` | {scores.loc[name, 'mean_log_loss']:.5f} | {scores.loc[name, 'accuracy']:.2%} |" for name in names]
    contributions = subset_contributions(data)
    group_labels = {"Recent overall form": "近期整体状态", "Same-surface results": "同场地战绩", "Head-to-head history": "历史交手"}
    feature_lines = []
    for row in contributions.itertuples():
        delta = row.loss_increase_when_removed
        implication = "点估计略优，尚不能据此证明稳定增益" if delta > 0 else "该信息在这组对照中没有显示增益"
        feature_lines.append(f"- 去掉**{group_labels[row.removed_from_four_feature_model]}**后，log loss 变化 {delta:+.5f}；{implication}。")
    best2023 = data["comparison"].sort_values("mean_log_loss").iloc[0].model
    return f"""# 建模与验证交接：用结果选择模型

推荐交给最终测试队友的是 **`{champion}`**。这个选择在查看 2023 年模型评分前，已经根据 2020–2022 年按时间向前验证的结果冻结。它是本次实验的强候选，**目前不能称为已证明的 SOTA**。

## 同一批比赛上的提升

下面全部使用相同的 **2,588 场 2023 年比赛**。log loss 衡量概率质量，越低越好；准确率衡量胜者猜对比例。

{chr(10).join(rows)}

相对只用排名的逻辑回归，推荐模型的 log loss 减少 **{improvement:.5f}（相对变化 {improvement/rank.mean_log_loss:.2%}）**，准确率变化 **{100*(chosen.accuracy-rank.accuracy):+.2f} 个百分点**。确定性的“总选排名更高者”准确率为 **{summary['ranking_rule']['accuracy']:.2%}**，不报告概率 log loss。

对同一赛事开始日期分块、配对重抽样得到的 log-loss 改善区间是 **[{low:.5f}, {high:.5f}]**。这只反映固定模型下的探索性不确定性，未计入模型搜索、不同日期之间的依赖或未来赛季变化。

## 哪些信息有帮助

以下对照是在原四组特征逻辑回归中，每次去掉一组；正变化表示删除后变差。每个组合的正则强度都只在训练年份内选择。这些是给定其余特征后的预测增量，不是因果效应。

{chr(10).join(feature_lines)}

扩展实验还比较了多时间尺度的整体/场地 Elo、近期状态、历史经验、休赛间隔及历史发接发统计，并测试线性模型、CatBoost、直方图梯度提升和非负加权集成。扩展删组对照以 `all_history_reference` 为参照，分别重新选择正则强度；`without_elo` 只删显式 Elo 评分，仍保留用 Elo 预期胜率算出的状态残差，不能解释为删除全部 Elo 信息。具体数值见 notebook 的扩展诊断表。原八组的最佳组合相对排名基准的改善区间跨过零，因此不应把这些微小差异写成已经确定的稳定提升。

**三个选择需要区分：** 原计划八种组合按 2023 年 log loss 选中 `{summary['prescribed_subset_selected_on_2023']}`；所有已保存模型中 2023 年分数最低的是 `{best2023}`；推荐进入最终测试的扩展方案仍是事先冻结的 `{champion}`。不会因为看到验证排名而悄悄改写冻结决定。

## 复跑与下一位队友

在仓库根目录按锁定的依赖环境运行：

```bash
python -m scripts.train_models --with-serve
python -m scripts.model_report
```

[完整 notebook](../../modeling.ipynb)、[课程 notebook](../../analysis.ipynb)、[可直接打开的 HTML](../../workbook.html)包含所有模型对比、概率校准、分场地/接近排名比赛结果，以及 Cincinnati 决赛 `2023-422:55` 的实例。

`search_registry.json` 保存候选配置与数据哈希；`frozen_selection.json` 保存选择；`frozen_models.joblib` 保存已拟合模型；`validation_predictions.csv` 保存逐场概率。结果图为 [模型比较](model_comparison.png) 和 [校准图](calibration.png)。

下一位先独立复跑并抽查 A/B 排序、T−21 截止日期，再显式解锁 **2024–2025 年最终测试**，用冻结流程报告分年和合并结果。当前开发特征入口故意拒绝测试年份，需要评估队友按交接规则扩展。最终测试尚未计算；本次也没有与同样本、同时间切分的赔率市场或所有最新论文模型比较，不能据此宣布世界最好。
"""


def report_cells(data):
    import nbformat as nbf
    md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell
    return [
        md(SECTION_MARKER + "\n# Modeling and validation: frozen extended experiment\n\n"
           "These saved results extend the prescribed eight ranking/history models with past-only strength, form and technical statistics. "
           "All numeric comparisons below use identical 2023 matches. The foundation notebook's original code and saved outputs are preserved. "
           "The final test seasons remain unopened."),
        md("## Decision and measured improvement\n\n" + narrative(data)),
        code("from pathlib import Path\nimport pandas as pd\nfrom IPython.display import display, Image\n"
             "from scripts.model_report import (load_results, comparison_table, subset_contributions, advanced_diagnostics, subgroup_table, worked_match, ROOT)\n"
             "report = load_results(ROOT / 'results' / 'modeling')\n"
             "pd.set_option('display.max_columns', None)\npd.set_option('display.precision', 5)\n"
             "display(comparison_table(report))\n"
             "display(Image(filename=str(ROOT / 'results/modeling/model_comparison.png')))"),
        md("## Temporal validation and the frozen choice\n\n"
           "Candidates are tuned on expanding forward folds for 2020, 2021 and 2022. Each fit uses only earlier rows available by the first fold date minus 21 days. "
           "The advanced family winners and ensemble choice are written before 2023 scoring. For the blend gate, family configurations and weights are chosen using only 2020–2021 predictions, then compared on 2022. "
           "After selecting the blend procedure, family configurations and weights are refitted using all three development folds. "
           "Ensemble weights fitted to pooled out-of-fold predictions have a meta-training loss, which must not be called an unbiased CV estimate. "
           "All models are symmetric under exchanging A and B. Historical result features use the same T−21 cutoff."),
        code("selected_names = list(report['frozen']['selected_by_family'].values())\n"
             "display(report['cv'][report['cv'].name.isin(selected_names)].sort_values('mean_log_loss'))\n"
             "display(pd.DataFrame({'component': report['frozen']['ensemble_components'], 'weight': report['frozen']['ensemble_weights']}))\n"
             "print('Advanced recommendation:', report['frozen']['advanced_champion'])\n"
             "print('2022 blend gate:', report['frozen']['ensemble_gate_2022'])"),
        md("## What does history add?\n\n"
           "The first table removes one history group from the original four-feature logistic model. Positive loss increases mean removing that information worsened validation predictions, conditional on the other features. "
           "These are predictive comparisons, not causal effects; correlated features can substitute for each other. Each subset's regularization was chosen within training years.\n\n"
           "The extended linear diagnostics remove explicit Elo rating features, technical statistics or ranking information from the all-history linear reference. "
           "`without_elo` retains form residuals calculated against Elo expected wins, so it does not remove all Elo-derived information. "
           "Regularization is retuned independently under the same three-value grid. "
           "Do not attribute a nonlinear champion's gain to a single feature from these diagnostics."),
        code("display(subset_contributions(report))\ndisplay(advanced_diagnostics(report))"),
        md("## Probability reliability and difficult matches\n\n"
           "Calibration uses fixed-width probability bins and displays counts. Subgroups are descriptive checks on the same validation season, not additional tuning folds. "
           "Grass and close-ranking subsets have fewer matches and more uncertainty."),
        code("display(Image(filename=str(ROOT / 'results/modeling/calibration.png')))\ndisplay(subgroup_table(report))"),
        md("## Worked match: Cincinnati final, 2023\n\n"
           "Match `2023-422:55` has A = Carlos Alcaraz and B = Novak Djokovic. Its recorded tournament start date is 14 August 2023, "
           "so historical results stop at 24 July 2023. The actual final took place later in that tournament. "
           "The outcome is revealed here only to explain the saved validation prediction; it is never an input feature."),
        code("example_features, example_probabilities = worked_match(report)\ndisplay(example_features)\ndisplay(example_probabilities)"),
        md("## Reproduction and final-test handoff\n\n"
           "From the repository root, use the locked modeling environment and run:\n\n"
           "```bash\npython -m scripts.train_models --with-serve\npython -m scripts.model_report\n```\n\n"
           "The first command reruns training-period selection and 2023 scoring. The second renders saved results without fitting. "
           "`results/modeling/search_registry.json` records the candidate grid and source hashes; `frozen_selection.json` records the pre-2023 recommendation; "
           "`frozen_models.joblib` contains fitted estimators. `validation_predictions.csv` preserves every match-level probability.\n\n"
           "The next teammate should independently reproduce the results, verify the Cincinnati cutoff and feature orientation, then explicitly unlock 2024–2025 for one frozen evaluation. "
           "Preserve the same population and T−21 rule, report 2024, 2025 and pooled metrics, and disclose any change before claiming final-test evidence. "
           "The current feature entry point deliberately rejects held-out targets.\n\n"
           "Limitations: completed ranked Tour/major main draws only; no betting-market benchmark; tournament-week dates require conservative lag; repeated model selection limits validation inference. "
           "A stronger score here is not a cross-study SOTA comparison. See `docs/research/sota_evidence.md` for source-backed context."),
    ]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    import nbformat as nbf
    from nbclient import NotebookClient
    from nbconvert import HTMLExporter
    from jupyter_client import KernelManager

    data = load_results()
    draw_comparison(data, OUT / "model_comparison.png")
    draw_calibration(data, OUT / "calibration.png")
    notebook = nbf.v4.new_notebook(cells=report_cells(data), metadata={
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}})
    manager = KernelManager(kernel_name="python3")
    manager.kernel_spec.argv = [sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"]
    NotebookClient(notebook, km=manager, timeout=180, resources={"metadata": {"path": str(ROOT)}}).execute()
    nbf.write(notebook, ROOT / "modeling.ipynb")
    original = nbf.read(ROOT / "analysis.ipynb", as_version=4)
    # Idempotent regeneration replaces only this appended section, preserving
    # every existing foundation cell and its saved output.
    boundary = next((i for i, cell in enumerate(original.cells) if SECTION_MARKER in cell.source), len(original.cells))
    original.cells = original.cells[:boundary] + notebook.cells
    # Refresh obsolete status prose without executing or changing foundation
    # code cells, which load the wider source inventory.
    for cell in original.cells[:boundary]:
        if cell.cell_type == "markdown" and cell.source.startswith("## 3. Modeling\n"):
            cell.source = ("## 3. Modeling\n\n**Implemented.** Eight ranking-containing logistic subsets, an overall Elo baseline and "
                           "the extended historical-strength models have now been fitted and scored. The complete executed comparison, "
                           "feature ablations and frozen choice appear in the appended **Modeling and validation: frozen extended experiment** section "
                           "and in [modeling.ipynb](modeling.ipynb).\n\n"
                           "The original [four-feature table](data/processed/model_features.csv) supplies the prescribed comparisons. "
                           "All probability models use the [shared scoring function](scripts/metrics.py) and the same 2,588 validation matches.")
        elif cell.cell_type == "markdown" and cell.source.startswith("## 4. Evaluation\n"):
            cell.source = ("## 4. Evaluation\n\nThe foundation ranking-rule result is retained below. The appended modeling section reports "
                           "all probability-model scores on the same eligible 2023 rows. The course's eight-feature-subset winner is selected "
                           "by 2023 log loss; the advanced recommendation was frozen using training-period forward folds before 2023 scoring. "
                           "Accuracy is a secondary metric. **Final evaluation on 2024–2025 remains pending.**")
    nbf.write(original, ROOT / "analysis.ipynb")
    exporter = HTMLExporter(template_name="lab")
    exporter.embed_images = True
    html, _ = exporter.from_notebook_node(original)
    (ROOT / "workbook.html").write_text(html)
    (OUT / "report_summary.md").write_text(chinese_summary(data).rstrip() + "\n")
    print("Wrote executed modeling.ipynb, appended analysis.ipynb, workbook.html and two result figures.")


if __name__ == "__main__":
    main()
