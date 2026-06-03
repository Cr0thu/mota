# DR184 Graph Attention Networks

- Year: 2017
- Area: graph-co
- URL: https://arxiv.org/abs/1710.10903
- Experiment role: graph_q_model

## 问题
这篇工作被纳入精读，是因为它处理了长视野、稀疏反馈、组合搜索、奖励学习或示范数据使用中的一个关键环节。

## 方法
支持全节点 action scoring：每个资源/怪/门/NPC 都作为 token，输出 masked Q 或 ranking score。

## 对魔塔的启示
GAT is a lightweight encoder for all-node Q values

## 工程取舍
如果进入当前前十层实验，优先转化为可测试的搜索、reward、ranker 或 ablation 模块；如果属于世界模型/LLM/大规模泛化方向，则先作为后续 50 层扩展参考。
