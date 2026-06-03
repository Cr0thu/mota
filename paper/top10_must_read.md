# 《魔塔》RL+规划项目十篇最应该读的论文

生成日期：2026-05-12  
筛选口径：优先服务当前前十层目标，即“确定性模拟器 + 搜索专家 + reward/heuristic 因子 + 行为克隆/RL 微调”，而不是泛泛覆盖所有强化学习方向。

## 先读顺序

1. 先读 reward 与 credit assignment：1-3。
2. 再读搜索轨迹蒸馏与层次子目标：4-6。
3. 最后读世界模型和 MCTS 系列：7-10。

如果时间只够读三篇，读第 1、4、7 篇。

## 1. Policy Invariance under Reward Transformations

- 作者：Ng, Harada, Russell
- 年份：1999
- 链接：https://people.eecs.berkeley.edu/~pabbeel/cs287-fa09/readings/NgHaradaRussell-shaping-ICML1999.pdf
- 核心问题：如何加中间奖励而不改变原任务最优策略。
- 核心方法：提出 potential-based reward shaping，额外奖励写成 `gamma * Phi(s') - Phi(s)`。
- 对魔塔最重要的启示：我们现在所有“钥匙价值、攻击阈值、防御阈值、Boss 生存边际”都应该尽量先写成势函数 `Phi(s)`，再用势能差给 reward。这样比直接写 `拿钥匙 +1、上楼 +5` 更不容易诱导错误策略。
- 工程落点：把当前 `dynamic_pbrs_potential(state)` 当作主接口，后续所有因子先进入 `Phi(s)`，再做 ablation。

## 2. Reward Design via Online Gradient Ascent

- 作者：Sorg, Lewis, Singh
- 年份：2010
- 链接：https://papers.nips.cc/paper_files/paper/2010/file/168908dd3227b8358eababa07fcaf091-Paper.pdf
- 核心问题：reward 里的权重能不能自动学，而不是人工反复调。
- 核心方法：PGRD 直接对 reward 参数做梯度更新，目标是最大化真实任务表现，而不是最大化手写奖励本身。
- 对魔塔最重要的启示：攻击临界、钥匙压力、Boss 伤害下降、HP margin 这些都可以看成 reward 因子；权重不应长期手调，而应通过搜索成功率、最终 HP、死局率来调。
- 工程落点：做一个 `reward_weight_sweep` 或更进一步的 PGRD/贝叶斯优化脚本，目标函数用 `solved + best_floor + boss_damage_margin - expansions_cost`。

## 3. RUDDER: Return Decomposition for Delayed Rewards

- 作者：Arjona-Medina et al.
- 年份：2019
- 链接：https://arxiv.org/abs/1806.07857
- 核心问题：长视野任务里，最终奖励太晚，早期关键动作拿不到信用。
- 核心方法：用序列模型把最终回报重新分配到真正导致回报变化的关键时间点。
- 对魔塔最重要的启示：比如 5F 取剑、9F 取盾、某个早期黄钥匙选择，可能几百步后才体现价值；RUDDER 的思路能把最终“打不过骷髅队长”的失败回推到早期资源错误。
- 工程落点：保存每条搜索/游玩轨迹的状态因子和最终 outcome，训练一个 return predictor，再看哪些动作导致预测回报跃迁。

## 4. Beyond A*: Better Planning with Transformers via Search Dynamics Bootstrapping

- 常用名：Searchformer
- 年份：2024
- 链接：https://arxiv.org/abs/2402.14083
- 核心问题：模型不只学最终答案，而是学搜索过程本身。
- 核心方法：把 A* 搜索中的 open/close/expand trace 编码成序列，让 Transformer 学会搜索动态，再用 expert iteration 提升。
- 对魔塔最重要的启示：不要只保存最终路线 `route.jsonl`；更应该保存每次扩展的候选动作、打分、mask 原因、dominance 剪枝原因。这些就是训练“魔塔 Searchformer”的数据。
- 工程落点：改 solver 日志格式，新增 `artifacts/search_traces/*.jsonl`，每个 node 记录 `state_features/action/features/score/pruned_reason/final_outcome`。

## 5. Solving Sokoban using Hierarchical Reinforcement Learning with Landmarks

- 常用名：HalfWeg
- 年份：2025
- 链接：https://arxiv.org/abs/2504.04366
- 核心问题：Sokoban 这类长程确定性解谜，直接从初始状态规划到终局太难。
- 核心方法：自动生成中间 landmark/subgoal，把长路径递归拆成多个短路径。
- 对魔塔最重要的启示：前十层不该只靠一个全局 heuristic。更自然的子目标是“小偷剧情后状态 -> 铁剑 -> 8F/9F关键资源 -> 铁盾 -> Boss-ready 状态 -> 10F 骷髅队长”。
- 工程落点：下一步最应该做 staged solver，而不是继续盲目加 expansions。每个阶段有不同因子权重和不同成功谓词。

## 6. Path Channels and Plan Extension Kernels

- 作者/主题：Sokoban RNN 规划机制可解释性
- 年份：2026
- 链接：https://openreview.net/forum?id=aAshH4kQ1v
- 核心问题：无模型网络到底有没有学会“规划”。
- 核心方法：分析 Sokoban DRC/RNN 内部隐藏通道，发现路径通道、计划扩展核、负激活回溯等结构。
- 对魔塔最重要的启示：魔塔中的门、怪、钥匙、Boss 都是“可通行性会随资源变化而改变”的障碍。网络结构如果要学规划，必须允许多次 internal ticks，而不是一步卷积就输出动作。
- 工程落点：后续 policy 网络不要只用普通 MLP；至少保留循环/Transformer 迭代计算接口，并记录 attention/hidden channel 与门怪阻塞的关系。

## 7. Mastering Atari, Go, Chess and Shogi by Planning with a Learned Model

- 常用名：MuZero
- 年份：2019
- 链接：https://arxiv.org/abs/1911.08265
- 核心问题：没有显式规则模型时，如何仍然做搜索规划。
- 核心方法：representation、dynamics、prediction 三个网络在 latent space 中支持 MCTS。
- 对魔塔最重要的启示：魔塔当前有显式模拟器，所以第一阶段不需要 MuZero；但如果以后做多版本魔塔、未知事件、自动泛化，MuZero 是主路线。
- 工程落点：当前接口要保留 `model.predict(state, action) -> next_latent/reward/value/policy` 的可替换位置，方便以后接 LightZero/MuZero。

## 8. Policy Improvement by Planning with Gumbel

- 常用名：Gumbel MuZero
- 年份：2021
- 链接：https://openreview.net/forum?id=bERaNdoegnO
- 核心问题：标准 MuZero 搜索预算大，动作选择效率不够。
- 核心方法：用 Gumbel top-k 和改进的策略改善过程，让少量模拟也能产生更好的策略提升。
- 对魔塔最重要的启示：魔塔宏动作空间虽然被 mask 后不算巨大，但关键节点很敏感。Gumbel MuZero 的动作候选筛选思想适合“只对可达且有战略意义的目标展开”。
- 工程落点：如果接 LightZero，优先试 Gumbel MuZero，而不是原始 MuZero。

## 9. Thinker: Learning to Plan and Act

- 年份：2023
- 链接：https://arxiv.org/abs/2307.14993
- 核心问题：agent 什么时候应该多想，什么时候应该直接行动。
- 核心方法：把世界模型变成 agent 可交互的内部环境，让 agent 自己决定规划深度。
- 对魔塔最重要的启示：走廊移动不该花大搜索预算；黄钥匙不足、遇到攻防阈值、临近 Boss 时应该自动增加 beam/MCTS 深度。
- 工程落点：先实现轻量版 Thinker：根据 `key_pressure/boss_margin/new_floor/critical_enemy_delta` 动态调整 `beam_width/max_expansions`。

## 10. UniZero: Generalized and Efficient Planning with Scalable Latent World Models

- 年份：2024
- 链接：https://arxiv.org/abs/2406.10667
- 核心问题：长历史任务中，latent state 和历史信息纠缠会降低规划质量。
- 核心方法：用 Transformer latent world model 解耦潜状态与潜历史，并同时优化模型与决策。
- 对魔塔最重要的启示：魔塔跨楼层记忆非常强，比如低层未开的门、高层拿到钥匙后返回、前期资源选择影响后期 Boss。UniZero 的历史解耦思想比普通 MuZero 更贴近 50 层全流程。
- 工程落点：这不是前十层第一实现，但应该作为“多楼层泛化世界模型”的主研究线。

## 暂不放进十篇但建议扫读

- Rainbow / Beyond the Rainbow：用于 DQN baseline 和 PER/Noisy/多步组合，链接：https://arxiv.org/abs/1710.02298 与 https://arxiv.org/abs/2411.03820
- EfficientZero：如果样本很贵再重点读，链接：https://arxiv.org/abs/2111.00210
- NetHack Learning Environment：环境封装和符号观测参考，链接：https://arxiv.org/abs/2006.13760
- Hindsight Experience Replay：失败轨迹重标目标，适合“没打过 Boss 但到达 MT9”的路线，链接：https://arxiv.org/abs/1707.01495

## 读完后的直接产出

1. 从第 1-3 篇产出：`Phi(s)` 因子列表、权重调参脚本、return redistribution 日志格式。
2. 从第 4-6 篇产出：搜索 trace 记录器、staged solver、landmark/subgoal 定义。
3. 从第 7-10 篇产出：后续 LightZero/MuZero/Thinker 接入接口设计，不阻塞前十层验收。
