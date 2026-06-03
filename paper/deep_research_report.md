# 《魔塔》前十层 Deep Research Report

生成方式：合并既有主文献、reward/因子文献，并补充硬探索、width-based planning、policy-guided search、offline/demo RL、资源约束规划等方向。

## Manifest 概览

- 总条目：200
- 精读条目：60
- Manifest：`paper/deep_research_manifest_200.csv`
- 精读笔记目录：`paper/deep_research_notes/`

## 角色分布

- `background`: 124
- `candidate_method`: 18
- `future_world_model`: 11
- `graph_q_model`: 4
- `hp403_ablation`: 6
- `learned_search`: 10
- `planner_core`: 12
- `reward_design`: 15

## No Demonstration RL vs hp403 Warm-start

### 没有训练数据时怎么做 RL

魔塔前十层不是一个适合从随机 PPO/DQN 直接起步的任务。奖励极稀疏，动作后果会跨越几十到几百个宏动作，错误开门或过早打怪会让后续状态不可逆。因此无专家数据主线应当先由搜索产生训练数据，而不是让神经网络裸探索。

可执行路线是：

1. 用确定性模拟器做 Go-Explore/BFWS/staged search，保存成功和失败状态，而不是只保存最终路线。
2. 把失败路线按阶段重标：没打过队长但拿到剑、盾、红钥匙，都可以成为阶段成功样本。
3. 用搜索 trace 训练 Graph Q / policy-value：输入全节点资源图，输出每个节点 Q、stage value、deadend risk。
4. 把 learned prior 接回搜索，形成 expert-iteration：搜索生成更好数据，模型再提高下一轮搜索效率。
5. 所有路线必须 strict replay；relaxed negative-HP 只用于发现结构。

### 使用 hp403 会不会更好

会更好，但它回答的是另一个实验问题。`hp403` 提供完整通关阶段顺序，可以显著缓解早期探索真空，适合做行为克隆初始化、offline Q 初始化、self-imitation 或 DQfD 风格 warm-start。它还能作为 reward sanity check：如果某个 reward 配置给 `hp403` 很低分，说明 reward 很可能方向错误。

风险也很明确：只有一条路线，模型可能过拟合单一路线；它可能遮蔽更优路线；如果把它放进训练 replay buffer，就不能再把结果称为纯无专家数据 RL。因此实验必须分为三条线：

- `pure_search_rl`：不用 `hp403`，只用搜索自举数据。
- `hp403_warmstart`：允许用 `hp403` 做 BC/offline Q/self-imitation 初始化。
- `hp403_benchmark`：只做可视化和最终对照。

### 推荐结论

科研主线应该优先做 `pure_search_rl`，保证方法论干净；工程推进可以并行做 `hp403_warmstart`，用它验证模型结构、reward 因子和可视化播放链路。最后报告中分开比较成功率、final HP、route length、expansions 和是否找到不同于 `hp403` 的路线。

## 工程路线

- 建模：独立 `MotaResourceGraph`，不替换可视化工具。
- 规划：Go-Explore archive + BFWS novelty + dominance pruning。
- Reward：PBRS，Phi 包含英雄资源、可达资源、不可达关键资源、怪物伤害、ATK/DEF 临界、10F 资源和 boss margin。
- 学习：GraphMaskedQNet / policy-value 只做搜索 prior 和状态排序，不直接承担裸探索。
- 验收：100 局 fixed seed，`10f战胜骷髅队长=true` 成功率 >=95%。

## 60 篇精读摘要

### DR156 First Return, Then Explore
- 问题：长视野魔塔求解中与 `planner_core` 对应的瓶颈。
- 方法：用 archive/cell 保存阶段性状态并允许回到状态继续探索，解决稀疏奖励下随机 RL 找不到钥匙/盾/红钥匙的问题。
- 对魔塔启示：robustified Go-Explore suggests separating deterministic archive discovery from policy training
- 工程取舍：当前只接入能通过 strict replay 和 ablation 验证的部分。

### DR157 Agent57: Outperforming the Atari Human Benchmark
- 问题：长视野魔塔求解中与 `planner_core` 对应的瓶颈。
- 方法：用 archive/cell 保存阶段性状态并允许回到状态继续探索，解决稀疏奖励下随机 RL 找不到钥匙/盾/红钥匙的问题。
- 对魔塔启示：meta-controller over exploration policies motivates phase-specific exploration schedules
- 工程取舍：当前只接入能通过 strict replay 和 ablation 验证的部分。

### DR158 Width and Serialization of Classical Planning Problems
- 问题：长视野魔塔求解中与 `planner_core` 对应的瓶颈。
- 方法：用 novelty 维度鼓励新资源组合、新可杀怪集合和新楼层连通，避免只追逐短期 HP 或楼层分数。
- 对魔塔启示：novelty width gives a principled alternative to greedy route scoring
- 工程取舍：当前只接入能通过 strict replay 和 ablation 验证的部分。

### DR159 Best-First Width Search: Exploration and Exploitation in Classical Planning
- 问题：长视野魔塔求解中与 `planner_core` 对应的瓶颈。
- 方法：用 novelty 维度鼓励新资源组合、新可杀怪集合和新楼层连通，避免只追逐短期 HP 或楼层分数。
- 对魔塔启示：BFWS is a strong fit for resource-key novelty in Mota
- 工程取舍：当前只接入能通过 strict replay 和 ablation 验证的部分。

### DR160 Rollout IW: Width-based Planning with Rollouts
- 问题：长视野魔塔求解中与 `planner_core` 对应的瓶颈。
- 方法：用 novelty 维度鼓励新资源组合、新可杀怪集合和新楼层连通，避免只追逐短期 HP 或楼层分数。
- 对魔塔启示：rollout width can cheaply expand many route variants
- 工程取舍：当前只接入能通过 strict replay 和 ablation 验证的部分。

### DR161 Planning with Pixels in Atari with IW
- 问题：长视野魔塔求解中与 `planner_core` 对应的瓶颈。
- 方法：用 novelty 维度鼓励新资源组合、新可杀怪集合和新楼层连通，避免只追逐短期 HP 或楼层分数。
- 对魔塔启示：even simple novelty features can solve hard sparse-reward games
- 工程取舍：当前只接入能通过 strict replay 和 ablation 验证的部分。

### DR187 Resource Constrained Shortest Paths
- 问题：长视野魔塔求解中与 `planner_core` 对应的瓶颈。
- 方法：提供资源约束最短路、label-setting 和 dominance pruning 的理论依据。
- 对魔塔启示：label-setting and dominance are mathematically aligned with HP/key constrained routing
- 工程取舍：当前只接入能通过 strict replay 和 ablation 验证的部分。

### DR188 A Survey of Resource Constrained Shortest Path Problems
- 问题：长视野魔塔求解中与 `planner_core` 对应的瓶颈。
- 方法：提供资源约束最短路、label-setting 和 dominance pruning 的理论依据。
- 对魔塔启示：formalizes label dominance used by Mota route search
- 工程取舍：当前只接入能通过 strict replay 和 ablation 验证的部分。

### DR189 Planning with Numeric State Variables
- 问题：长视野魔塔求解中与 `planner_core` 对应的瓶颈。
- 方法：提供资源约束最短路、label-setting 和 dominance pruning 的理论依据。
- 对魔塔启示：PDDL/numeric planning is a fallback formalization for attack/defense/key effects
- 工程取舍：当前只接入能通过 strict replay 和 ablation 验证的部分。

### DR190 The Metric-FF Planning System
- 问题：长视野魔塔求解中与 `planner_core` 对应的瓶颈。
- 方法：提供资源约束最短路、label-setting 和 dominance pruning 的理论依据。
- 对魔塔启示：numeric relaxed planning offers heuristics for HP/key/resource constraints
- 工程取舍：当前只接入能通过 strict replay 和 ablation 验证的部分。

### DR191 Fast Downward
- 问题：长视野魔塔求解中与 `planner_core` 对应的瓶颈。
- 方法：提供资源约束最短路、label-setting 和 dominance pruning 的理论依据。
- 对魔塔启示：planning-system architecture informs clean separation of model, heuristic, and search
- 工程取舍：当前只接入能通过 strict replay 和 ablation 验证的部分。

### DR192 Landmarks, Critical Paths and Abstractions
- 问题：长视野魔塔求解中与 `planner_core` 对应的瓶颈。
- 方法：提供资源约束最短路、label-setting 和 dominance pruning 的理论依据。
- 对魔塔启示：landmarks formalize sword/shield/red-key/boss-ready as mandatory subgoals
- 工程取舍：当前只接入能通过 strict replay 和 ablation 验证的部分。

### DR162 Policy-guided Heuristic Search with Guarantees
- 问题：长视野魔塔求解中与 `learned_search` 对应的瓶颈。
- 方法：把策略网络当作搜索 prior/value，而不是直接端到端控制；这样可以保留确定性模拟器和 strict replay。
- 对魔塔启示：policy prior can guide Mota search without replacing admissible-style pruning
- 工程取舍：当前只接入能通过 strict replay 和 ablation 验证的部分。

### DR163 Learning to Search Better than Your Teacher
- 问题：长视野魔塔求解中与 `learned_search` 对应的瓶颈。
- 方法：把策略网络当作搜索 prior/value，而不是直接端到端控制；这样可以保留确定性模拟器和 strict replay。
- 对魔塔启示：expert iteration over search traces is the right no-demo training loop
- 工程取舍：当前只接入能通过 strict replay 和 ablation 验证的部分。

### DR164 Neural Guided Constraint Logic Programming for Program Synthesis
- 问题：长视野魔塔求解中与 `learned_search` 对应的瓶颈。
- 方法：把策略网络当作搜索 prior/value，而不是直接端到端控制；这样可以保留确定性模拟器和 strict replay。
- 对魔塔启示：neural guidance over symbolic search mirrors node ranking in Mota
- 工程取舍：当前只接入能通过 strict replay 和 ablation 验证的部分。

### DR165 Learning Heuristics for Domain-Independent Planning
- 问题：长视野魔塔求解中与 `learned_search` 对应的瓶颈。
- 方法：把策略网络当作搜索 prior/value，而不是直接端到端控制；这样可以保留确定性模拟器和 strict replay。
- 对魔塔启示：learned value functions should rank frontier states, not directly replace the simulator
- 工程取舍：当前只接入能通过 strict replay 和 ablation 验证的部分。

### DR166 Neural A*: Learning Heuristic Functions for Path Planning
- 问题：长视野魔塔求解中与 `learned_search` 对应的瓶颈。
- 方法：把策略网络当作搜索 prior/value，而不是直接端到端控制；这样可以保留确定性模拟器和 strict replay。
- 对魔塔启示：differentiable search ideas inform trainable heuristics over route graphs
- 工程取舍：当前只接入能通过 strict replay 和 ablation 验证的部分。

### DR167 The Boxoban Level Collection
- 问题：长视野魔塔求解中与 `learned_search` 对应的瓶颈。
- 方法：Sokoban 的不可逆操作和长程死锁对应魔塔的钥匙、HP、攻防临界；搜索 trace 比最终路线更有训练价值。
- 对魔塔启示：dataset design and puzzle split methodology for deterministic planning games
- 工程取舍：当前只接入能通过 strict replay 和 ablation 验证的部分。

### DR168 Learning to Plan in High Dimensions via Neural Exploration-Exploitation Trees
- 问题：长视野魔塔求解中与 `learned_search` 对应的瓶颈。
- 方法：Sokoban 的不可逆操作和长程死锁对应魔塔的钥匙、HP、攻防临界；搜索 trace 比最终路线更有训练价值。
- 对魔塔启示：tree search plus learned policies informs macro-action expansion
- 工程取舍：当前只接入能通过 strict replay 和 ablation 验证的部分。

### DR169 Sokoban and the Growth of the Search Space
- 问题：长视野魔塔求解中与 `learned_search` 对应的瓶颈。
- 方法：Sokoban 的不可逆操作和长程死锁对应魔塔的钥匙、HP、攻防临界；搜索 trace 比最终路线更有训练价值。
- 对魔塔启示：deadlock and irreversibility analysis maps to Mota key/HP traps
- 工程取舍：当前只接入能通过 strict replay 和 ablation 验证的部分。

### DR170 Solving Sokoban with Forward-Backward Reinforcement Learning
- 问题：长视野魔塔求解中与 `learned_search` 对应的瓶颈。
- 方法：Sokoban 的不可逆操作和长程死锁对应魔塔的钥匙、HP、攻防临界；搜索 trace 比最终路线更有训练价值。
- 对魔塔启示：backward hints can turn terminal states into training signals
- 工程取舍：当前只接入能通过 strict replay 和 ablation 验证的部分。

### DR171 Thinking Like Transformers: Searchformer
- 问题：长视野魔塔求解中与 `learned_search` 对应的瓶颈。
- 方法：Sokoban 的不可逆操作和长程死锁对应魔塔的钥匙、HP、攻防临界；搜索 trace 比最终路线更有训练价值。
- 对魔塔启示：train on search dynamics, not just final routes
- 工程取舍：当前只接入能通过 strict replay 和 ablation 验证的部分。

### DR094 Policy Invariance under Reward Transformations: Theory and Application to Reward Shaping
- 问题：长视野魔塔求解中与 `reward_design` 对应的瓶颈。
- 方法：势函数 shaping 的理论底座，避免改变最优解
- 对魔塔启示：势函数 shaping 的理论底座，避免改变最优解
- 工程取舍：当前只接入能通过 strict replay 和 ablation 验证的部分。

### DR095 Potential-Based Shaping and Q-Value Initialization are Equivalent
- 问题：长视野魔塔求解中与 `reward_design` 对应的瓶颈。
- 方法：shaping 可转化为初始化价值，适合搜索启发式
- 对魔塔启示：shaping 可转化为初始化价值，适合搜索启发式
- 工程取舍：当前只接入能通过 strict replay 和 ablation 验证的部分。

### DR096 Dynamic Potential-Based Reward Shaping
- 问题：长视野魔塔求解中与 `reward_design` 对应的瓶颈。
- 方法：动态势函数对应不同楼层阶段
- 对魔塔启示：动态势函数对应不同楼层阶段
- 工程取舍：当前只接入能通过 strict replay 和 ablation 验证的部分。

### DR097 Potential-Based Reward Shaping for Reinforcement Learning in Multi-Agent Systems
- 问题：长视野魔塔求解中与 `reward_design` 对应的瓶颈。
- 方法：局部奖励分解思想可用于子目标
- 对魔塔启示：局部奖励分解思想可用于子目标
- 工程取舍：当前只接入能通过 strict replay 和 ablation 验证的部分。

### DR098 Reward Design via Online Gradient Ascent
- 问题：长视野魔塔求解中与 `reward_design` 对应的瓶颈。
- 方法：PGRD 直接优化 reward 参数，适合自动调 heuristic 权重
- 对魔塔启示：PGRD 直接优化 reward 参数，适合自动调 heuristic 权重
- 工程取舍：当前只接入能通过 strict replay 和 ablation 验证的部分。

### DR099 Learning Intrinsic Rewards for Policy Gradient Methods
- 问题：长视野魔塔求解中与 `reward_design` 对应的瓶颈。
- 方法：学习内在奖励，适合稀疏通关任务
- 对魔塔启示：学习内在奖励，适合稀疏通关任务
- 工程取舍：当前只接入能通过 strict replay 和 ablation 验证的部分。

### DR100 Learning to Incentivize Other Learning Agents
- 问题：长视野魔塔求解中与 `reward_design` 对应的瓶颈。
- 方法：奖励学习器和策略学习器解耦
- 对魔塔启示：奖励学习器和策略学习器解耦
- 工程取舍：当前只接入能通过 strict replay 和 ablation 验证的部分。

### DR132 Using Reward Machines for High-Level Task Specification and Decomposition in Reinforcement Learning
- 问题：长视野魔塔求解中与 `reward_design` 对应的瓶颈。
- 方法：把魔塔里程碑写成自动机 reward
- 对魔塔启示：把魔塔里程碑写成自动机 reward
- 工程取舍：当前只接入能通过 strict replay 和 ablation 验证的部分。

### DR133 Reward Machines: Exploiting Reward Function Structure in Reinforcement Learning
- 问题：长视野魔塔求解中与 `reward_design` 对应的瓶颈。
- 方法：系统化 reward machines，适合长视野任务分解
- 对魔塔启示：系统化 reward machines，适合长视野任务分解
- 工程取舍：当前只接入能通过 strict replay 和 ablation 验证的部分。

### DR136 Lifelong Reinforcement Learning with Temporal Logic Formulas and Reward Machines
- 问题：长视野魔塔求解中与 `reward_design` 对应的瓶颈。
- 方法：多任务 reward machine 迁移
- 对魔塔启示：多任务 reward machine 迁移
- 工程取舍：当前只接入能通过 strict replay 和 ablation 验证的部分。

### DR138 Text2Reward: Reward Shaping with Language Models for Reinforcement Learning
- 问题：长视野魔塔求解中与 `reward_design` 对应的瓶颈。
- 方法：LLM 根据环境代码生成 dense reward
- 对魔塔启示：LLM 根据环境代码生成 dense reward
- 工程取舍：当前只接入能通过 strict replay 和 ablation 验证的部分。

### DR139 Eureka: Human-Level Reward Design via Coding Large Language Models
- 问题：长视野魔塔求解中与 `reward_design` 对应的瓶颈。
- 方法：LLM 进化 reward 代码，可用于魔塔 reward 候选
- 对魔塔启示：LLM 进化 reward 代码，可用于魔塔 reward 候选
- 工程取舍：当前只接入能通过 strict replay 和 ablation 验证的部分。

### DR178 Policy Invariance under Reward Transformations
- 问题：长视野魔塔求解中与 `reward_design` 对应的瓶颈。
- 方法：用于设计 PBRS、回报重分配和阶段 reward machine，缓解 boss 成败对早期拿宝石/盾的信用分配问题。
- 对魔塔启示：PBRS is the safest way to add dense Mota rewards
- 工程取舍：当前只接入能通过 strict replay 和 ablation 验证的部分。

### DR179 Using Reward Machines for High-Level Task Specification and Decomposition
- 问题：长视野魔塔求解中与 `reward_design` 对应的瓶颈。
- 方法：用于设计 PBRS、回报重分配和阶段 reward machine，缓解 boss 成败对早期拿宝石/盾的信用分配问题。
- 对魔塔启示：stage automata can formalize sword/shield/red-key/boss progress
- 工程取舍：当前只接入能通过 strict replay 和 ablation 验证的部分。

### DR180 Learning What to Do by Simulating the Past
- 问题：长视野魔塔求解中与 `reward_design` 对应的瓶颈。
- 方法：用于设计 PBRS、回报重分配和阶段 reward machine，缓解 boss 成败对早期拿宝石/盾的信用分配问题。
- 对魔塔启示：counterfactual replay helps identify early resource mistakes
- 工程取舍：当前只接入能通过 strict replay 和 ablation 验证的部分。

### DR181 Neural Combinatorial Optimization with Reinforcement Learning
- 问题：长视野魔塔求解中与 `graph_q_model` 对应的瓶颈。
- 方法：支持全节点 action scoring：每个资源/怪/门/NPC 都作为 token，输出 masked Q 或 ranking score。
- 对魔塔启示：node selection over resources resembles pointer-style combinatorial policies
- 工程取舍：当前只接入能通过 strict replay 和 ablation 验证的部分。

### DR182 Attention, Learn to Solve Routing Problems!
- 问题：长视野魔塔求解中与 `graph_q_model` 对应的瓶颈。
- 方法：支持全节点 action scoring：每个资源/怪/门/NPC 都作为 token，输出 masked Q 或 ranking score。
- 对魔塔启示：attention over all graph nodes is a direct model template
- 工程取舍：当前只接入能通过 strict replay 和 ablation 验证的部分。

### DR183 Learning Combinatorial Optimization Algorithms over Graphs
- 问题：长视野魔塔求解中与 `graph_q_model` 对应的瓶颈。
- 方法：支持全节点 action scoring：每个资源/怪/门/NPC 都作为 token，输出 masked Q 或 ranking score。
- 对魔塔启示：GNN message passing supports resource graph scoring
- 工程取舍：当前只接入能通过 strict replay 和 ablation 验证的部分。

### DR184 Graph Attention Networks
- 问题：长视野魔塔求解中与 `graph_q_model` 对应的瓶颈。
- 方法：支持全节点 action scoring：每个资源/怪/门/NPC 都作为 token，输出 masked Q 或 ranking score。
- 对魔塔启示：GAT is a lightweight encoder for all-node Q values
- 工程取舍：当前只接入能通过 strict replay 和 ablation 验证的部分。

### DR172 Deep Q-learning from Demonstrations
- 问题：长视野魔塔求解中与 `hp403_ablation` 对应的瓶颈。
- 方法：用于回答 hp403 如何使用：它可以 warm-start，但必须和 pure no-demo 实验分开报告。
- 对魔塔启示：hp403 can warm-start Q learning but must be separated from pure RL
- 工程取舍：当前只接入能通过 strict replay 和 ablation 验证的部分。

### DR173 Self-Imitation Learning
- 问题：长视野魔塔求解中与 `hp403_ablation` 对应的瓶颈。
- 方法：用于回答 hp403 如何使用：它可以 warm-start，但必须和 pure no-demo 实验分开报告。
- 对魔塔启示：successful route fragments can be replayed without external experts
- 工程取舍：当前只接入能通过 strict replay 和 ablation 验证的部分。

### DR174 AWAC: Accelerating Online RL with Offline Datasets
- 问题：长视野魔塔求解中与 `hp403_ablation` 对应的瓶颈。
- 方法：用于回答 hp403 如何使用：它可以 warm-start，但必须和 pure no-demo 实验分开报告。
- 对魔塔启示：hp403 can be an offline seed before online improvement
- 工程取舍：当前只接入能通过 strict replay 和 ablation 验证的部分。

### DR175 Offline Reinforcement Learning with Implicit Q-Learning
- 问题：长视野魔塔求解中与 `hp403_ablation` 对应的瓶颈。
- 方法：用于回答 hp403 如何使用：它可以 warm-start，但必须和 pure no-demo 实验分开报告。
- 对魔塔启示：conservative value learning from route buffers avoids extrapolation issues
- 工程取舍：当前只接入能通过 strict replay 和 ablation 验证的部分。

### DR176 Conservative Q-Learning for Offline Reinforcement Learning
- 问题：长视野魔塔求解中与 `hp403_ablation` 对应的瓶颈。
- 方法：用于回答 hp403 如何使用：它可以 warm-start，但必须和 pure no-demo 实验分开报告。
- 对魔塔启示：single-route hp403 training needs pessimism to avoid overvaluing unseen actions
- 工程取舍：当前只接入能通过 strict replay 和 ablation 验证的部分。

### DR177 Decision Transformer
- 问题：长视野魔塔求解中与 `hp403_ablation` 对应的瓶颈。
- 方法：用于回答 hp403 如何使用：它可以 warm-start，但必须和 pure no-demo 实验分开报告。
- 对魔塔启示：route sequence modeling is useful after a diverse archive exists
- 工程取舍：当前只接入能通过 strict replay 和 ablation 验证的部分。

### DR001 Mastering Chess and Shogi by Self-Play with a General Reinforcement Learning Algorithm
- 问题：长视野魔塔求解中与 `future_world_model` 对应的瓶颈。
- 方法：local_pdf_exists
- 对魔塔启示：local_pdf_exists
- 工程取舍：当前只接入能通过 strict replay 和 ablation 验证的部分。

### DR002 Mastering the Game of Go without Human Knowledge
- 问题：长视野魔塔求解中与 `future_world_model` 对应的瓶颈。
- 方法：url_recorded
- 对魔塔启示：url_recorded
- 工程取舍：当前只接入能通过 strict replay 和 ablation 验证的部分。

### DR003 Mastering Atari with Deep Reinforcement Learning
- 问题：长视野魔塔求解中与 `background` 对应的瓶颈。
- 方法：url_recorded
- 对魔塔启示：url_recorded
- 工程取舍：当前只接入能通过 strict replay 和 ablation 验证的部分。

### DR004 Human-level Control through Deep Reinforcement Learning
- 问题：长视野魔塔求解中与 `background` 对应的瓶颈。
- 方法：url_recorded
- 对魔塔启示：url_recorded
- 工程取舍：当前只接入能通过 strict replay 和 ablation 验证的部分。

### DR005 Deep Reinforcement Learning with Double Q-learning
- 问题：长视野魔塔求解中与 `background` 对应的瓶颈。
- 方法：url_recorded
- 对魔塔启示：url_recorded
- 工程取舍：当前只接入能通过 strict replay 和 ablation 验证的部分。

### DR006 Dueling Network Architectures for Deep Reinforcement Learning
- 问题：长视野魔塔求解中与 `background` 对应的瓶颈。
- 方法：url_recorded
- 对魔塔启示：url_recorded
- 工程取舍：当前只接入能通过 strict replay 和 ablation 验证的部分。

### DR007 Prioritized Experience Replay
- 问题：长视野魔塔求解中与 `background` 对应的瓶颈。
- 方法：url_recorded
- 对魔塔启示：url_recorded
- 工程取舍：当前只接入能通过 strict replay 和 ablation 验证的部分。

### DR008 Noisy Networks for Exploration
- 问题：长视野魔塔求解中与 `background` 对应的瓶颈。
- 方法：url_recorded
- 对魔塔启示：url_recorded
- 工程取舍：当前只接入能通过 strict replay 和 ablation 验证的部分。

### DR009 A Distributional Perspective on Reinforcement Learning
- 问题：长视野魔塔求解中与 `background` 对应的瓶颈。
- 方法：url_recorded
- 对魔塔启示：url_recorded
- 工程取舍：当前只接入能通过 strict replay 和 ablation 验证的部分。

### DR010 Rainbow Combining Improvements in Deep Reinforcement Learning
- 问题：长视野魔塔求解中与 `background` 对应的瓶颈。
- 方法：url_recorded
- 对魔塔启示：url_recorded
- 工程取舍：当前只接入能通过 strict replay 和 ablation 验证的部分。

### DR011 Beyond the Rainbow High Performance Deep Reinforcement Learning on a Desktop PC
- 问题：长视野魔塔求解中与 `background` 对应的瓶颈。
- 方法：url_recorded
- 对魔塔启示：url_recorded
- 工程取舍：当前只接入能通过 strict replay 和 ablation 验证的部分。

### DR012 Mastering Atari Go Chess and Shogi by Planning with a Learned Model
- 问题：长视野魔塔求解中与 `future_world_model` 对应的瓶颈。
- 方法：url_recorded
- 对魔塔启示：url_recorded
- 工程取舍：当前只接入能通过 strict replay 和 ablation 验证的部分。

### DR013 Learning and Planning in Complex Action Spaces
- 问题：长视野魔塔求解中与 `future_world_model` 对应的瓶颈。
- 方法：url_recorded
- 对魔塔启示：url_recorded
- 工程取舍：当前只接入能通过 strict replay 和 ablation 验证的部分。

