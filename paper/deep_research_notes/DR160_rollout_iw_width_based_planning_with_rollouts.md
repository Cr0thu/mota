# DR160 Rollout IW: Width-based Planning with Rollouts

- Year: 2018
- Area: width-planning
- URL: https://arxiv.org/abs/1806.03355
- Experiment role: planner_core

## 问题
这篇工作被纳入精读，是因为它处理了长视野、稀疏反馈、组合搜索、奖励学习或示范数据使用中的一个关键环节。

## 方法
用 novelty 维度鼓励新资源组合、新可杀怪集合和新楼层连通，避免只追逐短期 HP 或楼层分数。

## 对魔塔的启示
rollout width can cheaply expand many route variants

## 工程取舍
如果进入当前前十层实验，优先转化为可测试的搜索、reward、ranker 或 ablation 模块；如果属于世界模型/LLM/大规模泛化方向，则先作为后续 50 层扩展参考。
