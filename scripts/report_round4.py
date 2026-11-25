"""Render the paper-grounded fourth experiment; never fit models or overwrite earlier reports."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/"results/round4"
GROUPS={"margin":"比分加权实力", "graph":"时间加权对手网络", "point":"逐分到整场概率", "profile":"年龄与排名积分"}
DIAGNOSTICS={"age":"仅年龄表示", "points":"仅排名积分"}


def load_results():
    data={key:pd.read_csv(OUT/file) for key,file in [("scores","validation_comparison.csv"),
                                                   ("predictions","validation_predictions.csv"),("cv","temporal_cv.csv")]}
    for key,file in [("summary","summary.json"),("frozen","frozen_selection.json"),("registry","search_registry.json")]:
        data[key]=json.loads((OUT/file).read_text())
        if data[key]["final_test_evaluated"]:
            raise ValueError("Round 4 exploratory report requires an unopened final holdout")
    rows=data["predictions"]
    if len(rows)!=2588 or rows.match_id.duplicated().any() or not rows.season.eq(2023).all():
        raise ValueError("Expected the unchanged 2,588 validation matches")
    if not data["scores"].matches.eq(len(rows)).all():
        raise ValueError("Model scoring denominator mismatch")
    data["specs"]={s["name"]:s for s in data["registry"]["candidates"]}
    data["round3_cv"]=pd.read_csv(ROOT/"results/round3/temporal_cv.csv")
    data["round3_registry"]=json.loads((ROOT/"results/round3/search_registry.json").read_text())
    return data


def label(name,data):
    aliases={"round3_recommended":"Round 3 prior recommendation", "round1_best_single":"Round 1 best linear",
             "round1_ensemble":"Round 1 frozen ensemble", "rank_reference":"Rank-only reference", "round3_linear":"Round 3 expanded linear"}
    if name in aliases:return aliases[name]
    spec=data["specs"][name]
    family=spec["family"]
    if family=="raw_margin_model":
        column=spec["columns"][0]
        if "welo_games" in column:return "Uncalibrated game-weighted Elo"
        if "welo_sets" in column:return "Uncalibrated set-weighted Elo"
        return "Uncalibrated ordinary Elo"
    groups={"margin":"weighted Elo", "graph":"opponent network", "point":"point-to-match model", "profile":"age and rank points",
            "age":"age only", "points":"rank points only"}
    if family.startswith("add_"):text="Add "+groups[family[4:]]
    elif family.startswith("without_"):text="All except "+groups[family[8:]]
    else:text={"all_papers":"All paper families", "paper_compact":"Compact paper features",
               "paper_catboost":"All papers · CatBoost", "paper_histboost":"All papers · histogram boosting",
               "raw_point_model":"Uncalibrated point model"}.get(family,family)
    if spec["kind"] in ("logistic","odd_spline"):
        text+=" · "+("linear" if spec["kind"]=="logistic" else "spline")
    return text


def comparison_table(data):
    table=data["scores"].copy()
    table.insert(1,"description",[label(n,data) for n in table.model])
    table["pooled_2020_22_cv_log_loss"]=table.model.map(data["cv"].set_index("name").mean_log_loss)
    table["role"]="Exploratory comparison"
    table.loc[table.model.eq("round3_recommended"),"role"]="Prior Round 3 recommendation"
    table.loc[table.model.eq(data["frozen"]["recommendation"]),"role"]="Recommendation after Round 4 gate"
    return table[["model","description","role","matches","mean_log_loss","accuracy","brier","pooled_2020_22_cv_log_loss"]]


def matched_cv_controls(data):
    """Addition/removal contrasts for same C, spline and three-year weighting."""
    cv=data["cv"].set_index("name")
    previous=data["round3_cv"].set_index("name")
    base={float(s["params"]["C"]):s["name"] for s in data["round3_registry"]["candidates"]
          if s["kind"]=="odd_spline" and s["train_start"]==2009 and s["include_format"]
          and s["params"].get("half_life")==3.}
    rows=[]
    for group in GROUPS:
        for c in (.003,.01):
            name=f"add_{group}_spline_C{c:g}"
            reference=base[c]
            row={"comparison":"add_group_to_round3", "group":GROUPS[group], "C":c,
                 "reference":reference, "candidate":name}
            for metric in ("mean_log_loss","2020","2021","2022"):
                row["reduction_"+metric]=previous.loc[reference,metric]-cv.loc[name,metric]
            rows.append(row)
            reference=f"all_papers_spline_C{c:g}"
            removed=f"without_{group}_spline_C{c:g}"
            row={"comparison":"remove_group_from_all", "group":GROUPS[group], "C":c,
                 "reference":reference, "candidate":removed}
            for metric in ("mean_log_loss","2020","2021","2022"):
                row["reduction_"+metric]=cv.loc[reference,metric]-cv.loc[removed,metric]
            rows.append(row)
    for group,description in DIAGNOSTICS.items():
        for c in (.003,.01):
            name=f"add_{group}_spline_C{c:g}"
            if name not in cv.index:continue
            reference=base[c]
            row={"comparison":"add_diagnostic_to_round3", "group":description, "C":c,
                 "reference":reference,"candidate":name}
            for metric in ("mean_log_loss","2020","2021","2022"):
                row["reduction_"+metric]=previous.loc[reference,metric]-cv.loc[name,metric]
            rows.append(row)
    return pd.DataFrame(rows)


def family_table(data):
    scores=data["scores"].set_index("model")
    cv=data["cv"].set_index("name")
    return pd.DataFrame([{"family":family,"model":name,"description":label(name,data),
                          "CV_log_loss":cv.loc[name,"mean_log_loss"],
                          "2023_log_loss":scores.loc[name,"mean_log_loss"],
                          "2023_accuracy":scores.loc[name,"accuracy"]}
                         for family,name in data["frozen"]["selected_by_family"].items()]).sort_values("CV_log_loss")


def paper_map():
    return """| 论文路线 | 本轮实现 | 复现与改编边界 |
| --- | --- | --- |
| Angelini、Candila、De Angelis： [Weighted Elo](https://doi.org/10.1016/j.ejor.2021.04.011) | 历史比分的局/盘占比加权 Elo，保留相同动态 K 的普通 Elo 对照 | 按核对过的原方程实现权重；T−21、同日批量、场地评分银行和派生特征是项目适配，不复现论文原样本成绩或投注结果。 |
| Arcagni、Candila、Grassi： [eigenvector centrality](https://pmc.ncbi.nlm.nih.gov/articles/PMC8900648/) | 败者→胜者的时间加权网络与入向主特征向量 | 使用论文的双曲衰减和归一化中心性核心；伪链接正则、未知球员先验、仅历史节点及场地网络属于明确改编。 |
| Spanias、Knottenbelt：[low-level point model](https://doi.org/10.1093/imaman/dps010)，并核对作者学位论文和 O'Malley 概率示例 | 既有赛前发球能力 → 局、盘、抢七与整场概率 | 递推保持发球顺序；采用统一抢七规则和项目能力估计，未完整复现论文能力估计器或所有历史赛事决胜盘规则。 |
| Buhamra、Groll、Gerharz：[modern forecasting approaches](https://doi.org/10.1177/22150218251338954)，及相关增强变量研究 | 赛前年龄、距 30 岁/[28,32]区间距离、ATP 排名积分和缺失标记 | 借鉴已定义变量变换，未移植赔率或论文完整机器学习流程；30 岁是固定参考假设，不是从本数据证明的最优年龄。 |
"""


def plot_comparison(data):
    scores=data["scores"].sort_values("mean_log_loss")
    chosen=data["frozen"]["recommendation"]
    colors=["#007F7C" if n==chosen else "#304A62" for n in scores.model]
    plt.rcParams.update({"font.family":"DejaVu Sans","font.size":10,"axes.spines.top":False,
                         "axes.spines.right":False,"axes.spines.left":False})
    fig,(ax,acc)=plt.subplots(1,2,figsize=(13,max(6.5,len(scores)*.38)),sharey=True,
                              gridspec_kw={"width_ratios":[3,1]})
    y=np.arange(len(scores))
    ax.scatter(scores.mean_log_loss,y,c=colors,s=40)
    ax.set_yticks(y,[label(n,data)+(" ★" if n==chosen else "") for n in scores.model])
    ax.invert_yaxis()
    width=max(.02,scores.mean_log_loss.max()-scores.mean_log_loss.min())
    ax.set_xlim(scores.mean_log_loss.min()-width*.12,scores.mean_log_loss.max()+width*.32)
    old=float(scores.loc[scores.model.eq("round3_recommended"),"mean_log_loss"].iloc[0])
    ax.axvline(old,color="#A36A15",ls="--",alpha=.65,lw=1)
    for yy,p in zip(y,scores.mean_log_loss):ax.text(p+width*.018,yy,f"{p:.5f}",va="center",fontsize=9)
    acc.scatter(scores.accuracy*100,y,c=colors,s=34)
    acc.set_xlim(scores.accuracy.min()*100-1.2,scores.accuracy.max()*100+3)
    for yy,p in zip(y,scores.accuracy):acc.text(p*100+.12,yy,f"{p:.1%}",va="center",fontsize=9)
    acc.tick_params(axis="y",labelleft=False,left=False)
    ax.set_xlabel("Mean log loss · lower is better");acc.set_xlabel("Accuracy (%)")
    for axis in (ax,acc):axis.grid(axis="x",alpha=.17);axis.set_axisbelow(True)
    fig.suptitle("Round 4 · paper-grounded methods, same 2,588 matches",x=.02,ha="left",fontsize=16,fontweight="bold")
    fig.text(.02,.018,"★ Recommendation follows the 2022 upgrade gate against the Round 3 procedure; dashed line is the prior Round 3 model.\n"
             "2023 is reused exploratory validation. Adapted paper methods do not establish replicated SOTA or final-test performance.",
             fontsize=9,color="#465769")
    fig.tight_layout(rect=(0,.07,1,.95))
    fig.savefig(OUT/"round4_comparison.png",dpi=180,bbox_inches="tight",facecolor="white")
    plt.close(fig)


def narrative(data,notebook=False):
    frozen,summary=data["frozen"],data["summary"]
    gate=frozen["procedure_gate"]
    chosen,candidate=frozen["recommendation"],frozen["best_single"]
    scores=data["scores"].set_index("model")
    old=scores.loc["round3_recommended"]
    gain=old.mean_log_loss-scores.loc[candidate,"mean_log_loss"]
    low,high=summary["comparisons_vs_round3"][candidate]["ci95"]
    best=data["scores"].sort_values(["mean_log_loss","model"]).iloc[0]
    status="通过升级门槛，采用本轮论文候选流程" if gate["procedure"]=="adopt_paper_model" else "没有达到升级门槛，保留第三轮推荐"
    aliases={"rank_reference":"只用排名","round1_best_single":"第一轮最强线性模型","round1_ensemble":"第一轮冻结集成",
             "round3_recommended":"第三轮此前推荐","round3_linear":"第三轮扩展线性对照",candidate:"第四轮训练期选中候选"}
    table=["| 比较 | 2023 log loss ↓ | 准确率 |","| --- | ---: | ---: |"]
    for name in dict.fromkeys(["rank_reference","round1_best_single","round1_ensemble","round3_recommended","round3_linear",candidate,chosen]):
        row=scores.loc[name]
        table.append(f"| {aliases.get(name,label(name,data))} (`{name}`) | {row.mean_log_loss:.5f} | {row.accuracy:.2%} |")
    interval_text="区间跨过 0，不能据此确认优于第三轮此前推荐。" if low<=0<=high else "这个条件区间未跨过 0，但仍没有覆盖多轮模型搜索不确定性。"
    controls=matched_cv_controls(data)
    addition=controls[controls.comparison.isin(["add_group_to_round3","add_diagnostic_to_round3"])]
    control_summary=[]
    for group,rows in addition.groupby("group",sort=False):
        lower,upper=rows.reduction_mean_log_loss.min(),rows.reduction_mean_log_loss.max()
        implication="两种C下均改善" if lower>0 else "两种C下均未改善" if upper<=0 else "方向随C改变"
        control_summary.append(f"- **{group}**：匹配样条设置的 pooled CV log-loss 改善范围 **[{lower:+.5f}, {upper:+.5f}]**，{implication}。")
    diagnostic_note=("前两次运行均在本轮 2023 评分之前中断：首次发现 2019 年华盛顿站积分快照问题，随后全量审计又发现同站年龄混合日期问题。"
                     "最终修复将该赛事年龄和积分同时视为缺失，再从头运行。仅年龄和仅积分两个诊断分支是在首次已见部分训练期结果后追加，"
                     "属于明确记录的自适应补充，并非从最初就固定的独立试验。"
                     if controls.comparison.eq("add_diagnostic_to_round3").any() else "")
    profile_meta=data["registry"]["paper_metadata"].get("profile",{})
    if profile_meta.get("points_snapshot_masked_appearances"):
        diagnostic_note+=(f" 对日期/属性快照已审计但缺少可信赛前年龄和积分的赛事，保留目标比赛并将相关属性设缺失；"
                          f"积分遮蔽 {profile_meta['points_snapshot_masked_appearances']:,} 次球员出场，"
                          f"年龄遮蔽 {profile_meta.get('age_snapshot_masked_appearances',0):,} 次，使用固定填充值和缺失标记。")
    source_note=""
    if controls.comparison.eq("add_diagnostic_to_round3").any():
        age=controls.loc[controls.group.eq(DIAGNOSTICS["age"]),"reduction_mean_log_loss"].to_numpy()
        points=controls.loc[controls.group.eq(DIAGNOSTICS["points"]),"reduction_mean_log_loss"].to_numpy()
        if np.all(age>points):
            source_note="在两种匹配 C 下，单独加入年龄的改善都大于单独加入积分；这些控制比较支持年龄表示是更强的来源，但不能把联合 profile 的全部增益因果归于年龄。"
        elif np.all(points>age):
            source_note="在两种匹配 C 下，单独加入积分的改善都大于单独加入年龄；这些控制比较支持积分是更强的来源，但不能把联合 profile 的全部增益因果归于积分。"
        else:
            source_note="年龄和积分的相对表现随 C 改变；当前匹配比较不能给出一致的主导来源。"
    prefix="" if notebook else "../../"
    image_prefix="results/round4/" if notebook else ""
    return f"""# 第四轮：把论文方法变成同口径实验

**结论：{status}。当前推荐为 `{chosen}`。** 本轮登记 **{summary['new_candidates']} 个配置**，候选特征池共 **{summary['features']} 项**，各模型使用对应子集；训练比赛为 **{summary['training_rows']:,} 场**。所有最终比较保持同样的 2,588 场 2023 比赛。

## 四条论文路线，分别复用了什么

{paper_map()}

各分支的公式、原始全文访问记录、数值例子与局限见 `docs/research/round4_protocol.md` 及对应方法文档。这里检验的是**来源明确的项目适配**，不是直接重现作者发表的整套实验。论文使用不同赛事、预测时点或信息条件时，其准确率不能直接与本表比较。

{diagnostic_note}

## 升级门槛比较的是当前第三轮流程

2020–2021 的向前折先选定 `{gate['early_selected']}`；2022 年新候选 log loss 为 **{gate['candidate_2022_loss']:.5f}**，第三轮相同早期选择流程的参照为 **{gate['reference_2022_loss']:.5f}**，改善 **{gate['absolute_improvement']:+.5f}**，要求至少 **{gate['required_improvement']:.3f}**。没有退回较弱的第一轮流程降低门槛。作出流程决定后，才用全部 2020–2022 开发折重选配置，并计算本轮 2023 分数。

**2022 和 2023 已被前轮使用，因此这仍然是后续探索。** 本轮评分前登记的规则有助于避免随结果改口，但不等于独立预注册或恢复未见验证数据。2024–2025 最终测试没有打开。

## 同一批 2023 比赛的比较

{chr(10).join(table)}

第四轮训练期选中候选相对第三轮推荐的 log-loss 改善为 **{gain:+.5f}**，准确率变化 **{100*(scores.loc[candidate,'accuracy']-old.accuracy):+.2f} 个百分点**。按赛事开始日期分块的配对改善区间为 **[{low:.5f}, {high:.5f}]**。**{interval_text}** 负改善表示变差；区间还没有覆盖跨日期重复球员依赖。

全部保存比较中，2023 最低 log loss 是 `{best.model}`（**{best.mean_log_loss:.5f}**）；这个描述性排序不替代上面的门槛决定，不能据此宣布 SOTA。

## 添加与删除信息的控制比较

添加组对照固定 2009 训练起点、样条表示、3 年近期权重及相同 C，与第三轮对应配置比较；pooled CV 数值越正，加入该组后越好：

{chr(10).join(control_summary)}

{source_note}

删除组对照固定相同 C 的 `all_papers` 样条模型，移除一个论文分支。Notebook 保留全部 **{len(controls)} 项精确设置配对**及 2020、2021、2022 各年变化。删除对照也采用“参考损失−删除后损失”的符号：正数表示删掉该组反而更好，负数表示删除后变差。仅年龄/仅积分诊断用于区分 profile 分支的来源，但两类信息可能互相补充，不能假设联合增益必然等于单独增益相加。分支相关时，单独加入和条件删除回答不同问题；这些是预测信息比较，不是因果贡献。

家族最终 2023 表独立展示训练期选中的线性、样条或树配置，不能把跨算法和重新选择正则后的差异当成同参数消融。全部失败候选与原始公式概率基准均保留。

## 复跑与交接

```bash
.venv/bin/python -m scripts.train_round4
.venv/bin/python -m scripts.report_round4
```

[执行 notebook]({prefix}round4.ipynb)、[HTML]({prefix}round4.html) 和 [比较图]({image_prefix}round4_comparison.png) 单独保存，不覆盖以前实验。`search_registry.json` 记录候选和各论文分支元数据，`temporal_cv.csv` 保存全部训练期结果，`frozen_selection.json` 保存升级门槛和最终配置，`validation_predictions.csv` 保存逐场概率，本地 `frozen_models.joblib` 保存模型与旧对照。

最终测试应独立核验后按冻结方案执行；也应保留第三轮推荐、旧强线性模型和本轮候选，避免把探索性小提升写成已确认的泛化结论。
"""


def notebook_cells(data):
    import nbformat as nbf
    md,code=nbf.v4.new_markdown_cell,nbf.v4.new_code_cell
    return [md(narrative(data,True)),
        md("## 完整 2023 对比\n\n星号表示门槛决定的推荐，不是把验证排序第一名自动换成推荐。"),
        code("from IPython.display import display, Image\nimport pandas as pd\nfrom scripts.report_round4 import load_results,comparison_table,family_table,matched_cv_controls,OUT\n"
             "data=load_results()\npd.set_option('display.max_columns',None)\npd.set_option('display.precision',5)\ndisplay(comparison_table(data))\ndisplay(Image(filename=str(OUT/'round4_comparison.png')))"),
        md("## 同设置添加/删除组 CV\n\n正数统一表示候选比参考损失低；因此删除组行的正数表示删组后改善。仅比较相同样条、C、训练期与近期权重，完整保留逐年方向。"),
        code("display(matched_cv_controls(data))"),
        md("## 所有家族与候选\n\n家族最终候选重新选择了模型种类或正则，因此此表属于模型家族比较，不是上述精确设置配对。"),
        code("display(family_table(data))\ndisplay(data['cv'].sort_values('mean_log_loss'))\n"
             "display(pd.DataFrame([{'branch':g,'features':len(cols)} for g,cols in data['registry']['feature_groups'].items()]))"),
        md("## 门槛和条件不确定性\n\n基准是第三轮当前推荐。区间不能修正重复查看验证年份和多轮设计选择带来的全部不确定性。"),
        code("display(pd.DataFrame([data['frozen']['procedure_gate']]))\n"
             "display(pd.DataFrame([{'model':n,'loss_reduction':v['mean_log_loss_reduction'],'lower95':v['ci95'][0],'upper95':v['ci95'][1]} for n,v in data['summary']['comparisons_vs_round3'].items()]).sort_values('loss_reduction',ascending=False))\n"
             "assert data['summary']['final_test_evaluated'] is False\nprint('2024–2025 final holdout remains unopened.')")]


def main():
    import nbformat as nbf
    from nbclient import NotebookClient
    from nbconvert import HTMLExporter
    from jupyter_client import KernelManager
    data=load_results()
    plot_comparison(data)
    notebook=nbf.v4.new_notebook(cells=notebook_cells(data),metadata={"kernelspec":{"display_name":"Python 3","language":"python","name":"python3"}})
    manager=KernelManager(kernel_name="python3")
    manager.kernel_spec.argv=[sys.executable,"-m","ipykernel_launcher","-f","{connection_file}"]
    NotebookClient(notebook,km=manager,timeout=180,resources={"metadata":{"path":str(ROOT)}}).execute()
    nbf.write(notebook,ROOT/"round4.ipynb")
    exporter=HTMLExporter(template_name="lab");exporter.embed_images=True
    html,_=exporter.from_notebook_node(notebook)
    (ROOT/"round4.html").write_text(html)
    (OUT/"report_summary.md").write_text(narrative(data))
    print("Wrote separate Round 4 executed notebook, HTML, Chinese summary and comparison figure.")


if __name__=="__main__":main()
