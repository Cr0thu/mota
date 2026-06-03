# DR178 Policy Invariance under Reward Transformations

- Year: 1999
- Area: reward-credit
- URL: https://people.eecs.berkeley.edu/~pabbeel/cs287-fa09/readings/NgHaradaRussell-shaping-ICML1999.pdf
- Experiment role: reward_design

## 问题
这篇工作被纳入精读，是因为它处理了长视野、稀疏反馈、组合搜索、奖励学习或示范数据使用中的一个关键环节。

## 方法
用于设计 PBRS、回报重分配和阶段 reward machine，缓解 boss 成败对早期拿宝石/盾的信用分配问题。

## 对魔塔的启示
PBRS is the safest way to add dense Mota rewards

## 工程取舍
如果进入当前前十层实验，优先转化为可测试的搜索、reward、ranker 或 ablation 模块；如果属于世界模型/LLM/大规模泛化方向，则先作为后续 50 层扩展参考。
