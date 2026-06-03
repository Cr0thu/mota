# DR098 Reward Design via Online Gradient Ascent

- Year: 2010
- Area: reward-design
- URL: https://papers.nips.cc/paper_files/paper/2010/file/168908dd3227b8358eababa07fcaf091-Paper.pdf
- Experiment role: reward_design

## 问题
这篇工作被纳入精读，是因为它处理了长视野、稀疏反馈、组合搜索、奖励学习或示范数据使用中的一个关键环节。

## 方法
PGRD 直接优化 reward 参数，适合自动调 heuristic 权重

## 对魔塔的启示
PGRD 直接优化 reward 参数，适合自动调 heuristic 权重

## 工程取舍
如果进入当前前十层实验，优先转化为可测试的搜索、reward、ranker 或 ablation 模块；如果属于世界模型/LLM/大规模泛化方向，则先作为后续 50 层扩展参考。
