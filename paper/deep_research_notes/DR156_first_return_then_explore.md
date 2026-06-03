# DR156 First Return, Then Explore

- Year: 2020
- Area: hard-exploration
- URL: https://www.nature.com/articles/s41586-020-03157-9
- Experiment role: planner_core

## 问题
这篇工作被纳入精读，是因为它处理了长视野、稀疏反馈、组合搜索、奖励学习或示范数据使用中的一个关键环节。

## 方法
用 archive/cell 保存阶段性状态并允许回到状态继续探索，解决稀疏奖励下随机 RL 找不到钥匙/盾/红钥匙的问题。

## 对魔塔的启示
robustified Go-Explore suggests separating deterministic archive discovery from policy training

## 工程取舍
如果进入当前前十层实验，优先转化为可测试的搜索、reward、ranker 或 ablation 模块；如果属于世界模型/LLM/大规模泛化方向，则先作为后续 50 层扩展参考。
