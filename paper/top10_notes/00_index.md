# 十篇必读论文阅读笔记索引

生成日期：2026-05-13  
PDF 目录：`paper/top10_pdfs/`  
MinerU 抽取目录：`paper/top10_extracted/`  
逐篇笔记目录：`paper/top10_notes/`

## 阅读顺序

1. `01_policy_invariance_reward_shaping.md` - PBRS，决定魔塔 reward 怎么加才不改变目标。
2. `02_reward_design_pgrd.md` - reward 权重自动调参，把人工调 reward 变成优化问题。
3. `03_rudder_return_decomposition.md` - 长延迟奖励重分配，适合分析早期钥匙/宝石选择。
4. `04_searchformer.md` - 学搜索过程，而不是只学最终路线。
5. `05_halfweg.md` - 递归 landmark/subgoal，直接对应魔塔 staged solver。
6. `06_path_channels.md` - Sokoban RNN 内部规划机制，指导后续网络结构。
7. `07_muzero.md` - 隐式世界模型 + MCTS 的主路线。
8. `08_gumbel_muzero.md` - 少模拟预算下的 MuZero/AlphaZero 改进。
9. `09_thinker.md` - 学会何时、如何调用世界模型规划。
10. `10_unizero.md` - Transformer latent world model，适合长历史、多任务和跨楼层记忆。

## 对当前魔塔项目的优先级

当前前十层卡点不是“缺少一个更大的深度 RL 算法”，而是：

1. 缺少稳定可解释的 `Phi(s)` 状态势函数。
2. 搜索器只保存最终路线，缺少可训练的 search trace。
3. 全局 heuristic 不够，应该改成 staged solver。
4. 攻防临界值、钥匙期权、Boss 生存边际需要从搜索日志中回测，不应长期靠手调。

因此近期最应该从第 1、3、4、5 篇落地工程；第 7-10 篇作为后续升级路线。
