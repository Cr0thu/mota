# DR187 Resource Constrained Shortest Paths

- Year: 1980
- Area: numeric-planning
- URL: https://doi.org/10.1002/net.3230100109
- Experiment role: planner_core

## 问题
这篇工作被纳入精读，是因为它处理了长视野、稀疏反馈、组合搜索、奖励学习或示范数据使用中的一个关键环节。

## 方法
提供资源约束最短路、label-setting 和 dominance pruning 的理论依据。

## 对魔塔的启示
label-setting and dominance are mathematically aligned with HP/key constrained routing

## 工程取舍
如果进入当前前十层实验，优先转化为可测试的搜索、reward、ranker 或 ablation 模块；如果属于世界模型/LLM/大规模泛化方向，则先作为后续 50 层扩展参考。
