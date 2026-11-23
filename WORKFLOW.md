# How the four of us collaborate

Anyone can take the next piece of work. The GitHub issue identifies what is being picked up, the PR shows work in progress, and [HANDOFF.md](HANDOFF.md) describes what a new teammate can reproduce on `main`.

## Pick up or hand over work

1. **Pick a task.** Start with the linked issue in the handoff. Assign yourself or leave a short claim on the issue when you start. Link an existing branch or PR when resuming mid-task.
2. **Work in parallel.** Use a short-lived branch for each independent change. If two people are active, agree in the issue or PR who is editing [analysis.ipynb](analysis.ipynb). Another teammate can work on a Python function, check a paper or draft narration from a finished figure.
3. **Share something reproducible.** Open a PR with the change, the check you ran and what the next person should do. A draft PR is enough for an unfinished handoff. Give the next person its URL or branch and latest commit, and say which cells still need running.
4. **Bring the team up to date.** On completing a milestone, include the updated handoff in the same PR as its evidence. Link the work issue with `Closes #N` when the PR completes it. A teammate who joins later can pull `main` and follow the new next task.

## When a research idea changes

| What someone said | What to record |
| --- | --- |
| “Maybe we should also try this.” | Add the question to **Ideas to revisit** in [HANDOFF.md](HANDOFF.md), with the reason it might matter and the smallest experiment that would answer it. |
| “Let's change the plan to this.” | Update the adopted choice in [PROJECT-PLAN.zh.md](PROJECT-PLAN.zh.md). Record the old choice, the new one and the reason in the handoff's decision history. Link the implementation issue or PR and say which results, notebook cells or slides need refreshing. |
| “We ran it and here's what happened.” | Save the reproducible output in the notebook, then update the verified milestone with the dataset split, result and evidence link. If the measurement changes a conclusion, refresh downstream slides or mark them pending. |

For example, if grass-court history looks sparse and someone suggests a two-year window, capture it as an idea. If the team adopts a window comparison, write down what stays fixed, check whether early seasons have enough warm-up history, and compare coverage and performance on 2023. The handoff can say “plan adopted, results pending” while implementation is underway. Once scores exist, link the executed comparison. If final test scores have already been viewed, label the follow-up exploratory.

If another teammate is already implementing the earlier plan, update their open issue's acceptance criteria and put the decision link in their active PR as well. On resuming that branch, read the linked change and sync the merged plan from `main` before rerunning the affected cells. This gives both the active teammate and someone joining later a route to the same decision.

## Copy into a new agent chat

> Sync this repo and read AGENTS.md and HANDOFF.md. Check the linked issue and any open PR before editing. Continue the next task I assigned, following the agreed experiment in PROJECT-PLAN.zh.md. Leave a branch or PR link, what you verified, and what the next person should pick up.

> 同步仓库，先读 AGENTS.md 和 HANDOFF.md，再看对应的 issue 和进行中的 PR。按我这次说的修改方案；把已经确定的决定写进项目计划，把还在讨论的想法放进候选清单。交接时给出改动链接、怎么验证、下一步谁可以接。
