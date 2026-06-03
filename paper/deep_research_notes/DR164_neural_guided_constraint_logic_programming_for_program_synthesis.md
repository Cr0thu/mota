# DR164 Neural Guided Constraint Logic Programming for Program Synthesis

- Year: 2018
- Area: policy-guided-search
- URL: https://arxiv.org/abs/1809.02840
- Experiment role: learned_search

## 问题
这篇工作被纳入精读，是因为它处理了长视野、稀疏反馈、组合搜索、奖励学习或示范数据使用中的一个关键环节。

## 方法
把策略网络当作搜索 prior/value，而不是直接端到端控制；这样可以保留确定性模拟器和 strict replay。

## 对魔塔的启示
neural guidance over symbolic search mirrors node ranking in Mota

## 工程取舍
如果进入当前前十层实验，优先转化为可测试的搜索、reward、ranker 或 ablation 模块；如果属于世界模型/LLM/大规模泛化方向，则先作为后续 50 层扩展参考。
