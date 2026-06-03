# DR174 AWAC: Accelerating Online RL with Offline Datasets

- Year: 2020
- Area: demo-offline-rl
- URL: https://arxiv.org/abs/2006.09359
- Experiment role: hp403_ablation

## 问题
这篇工作被纳入精读，是因为它处理了长视野、稀疏反馈、组合搜索、奖励学习或示范数据使用中的一个关键环节。

## 方法
用于回答 hp403 如何使用：它可以 warm-start，但必须和 pure no-demo 实验分开报告。

## 对魔塔的启示
hp403 can be an offline seed before online improvement

## 工程取舍
如果进入当前前十层实验，优先转化为可测试的搜索、reward、ranker 或 ablation 模块；如果属于世界模型/LLM/大规模泛化方向，则先作为后续 50 层扩展参考。
