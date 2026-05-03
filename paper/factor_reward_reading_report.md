# 魔塔 RL 与量化因子挖掘联动阅读报告

生成日期：2026-05-01  
范围：围绕“把魔塔 reward/heuristic 设计转化为因子挖掘问题”，完成 110 篇文献的一轮阅读整理，并选择 20 篇做精读。  
配套清单：`paper/factor_reward_paper_manifest_100.csv`
本地 PDF：`paper/pdfs/factor_reward/` 已下载 19 篇精读论文 PDF；FR003 为期刊/DOI 记录，未强制下载。

## 0. 阅读口径

这份报告不是继续堆强化学习算法，而是把问题换成一个更工程化的视角：

- 量化里，因子挖掘寻找 `feature(state_t) -> future_return` 的稳定预测关系。
- 魔塔里，因子挖掘寻找 `feature(game_state_t) -> future_solve / future_hp / future_dead_end` 的稳定预测关系。
- 量化回测检验因子是否稳定；魔塔回测检验因子能否提升搜索通关率、减少 expansions、减少钥匙死局。
- 量化里的 IC、RankIC、分组回测、因子中性化，对应魔塔里的状态价值相关性、动作排序命中率、按楼层/步数分组、进度中性化。

阅读分三层：

- 110 篇清单：每篇至少筛读问题设定、核心方法、和对魔塔的可迁移点。
- 20 篇精读：展开“问题-方法-魔塔启示-工程落地-风险”。
- 最终工程结论：优先实现“魔塔因子挖掘流水线”，再把稳定因子转成 heuristic、reward shaping 和离线训练特征。

## 1. 110 篇文献地图

文献按用途分成六组：

1. 量化因子与 factor zoo：FR001-FR015  
   重点是因子定义、组合、过拟合、多重检验、分组回测。魔塔里对应“哪些状态特征是真的有效，哪些只是因为已经走到高层所以看起来有效”。

2. 自动 alpha / 公式因子挖掘：FR016-FR030  
   重点是 LLM、遗传规划、RL、MCTS 如何搜索可解释公式。魔塔里对应自动搜索状态打分函数，例如 `hp_margin + key_option_value - door_pressure`。

3. 符号回归与公式发现：FR031-FR037  
   重点是从数据里发现简洁表达式。魔塔里可用于把搜索日志蒸馏成可读的价值函数。

4. reward 设计、reward shaping、credit assignment：FR038-FR069  
   重点是怎样加中间奖励而不改变最优策略，以及怎样把延迟奖励回分配给早期关键动作。

5. IRL、模仿学习、偏好学习、reward machine：FR070-FR089  
   重点是从专家轨迹、人类偏好、时序逻辑里学习 reward。魔塔里适合从“成功/失败路线对比”里学习更稳定的状态价值。

6. LLM reward、Sokoban、NetHack、离线 RL、层次规划：FR090-FR110  
   重点是 LLM 自动写 reward、长视野解谜、搜索轨迹蒸馏、递归子目标。魔塔里直接对应前十层路线搜索和后续扩展到 50 层。

## 2. 20 篇精读分析

### FR001 101 Formulaic Alphas

**问题。** 这篇论文把大量交易信号写成可执行公式，核心价值不在于每个 alpha 都必然赚钱，而在于建立了“公式化、可批量评估、可组合、可审计”的因子范式。

**方法。** 每个 alpha 都由市场原始变量和算子组合而成，例如 rank、delay、correlation、ts_rank 等。研究者可以批量回测、比较、组合，也可以把它们作为后续自动因子挖掘的种子库。

**对魔塔启示。** 魔塔的状态因子也应该公式化，而不是写散落在 heuristic 里的魔法数字。候选因子可以包括 `yellow_key_pressure`、`reachable_attack_gain`、`damage_to_next_milestone`、`boss_survival_margin`、`door_unlock_value`。它们必须能在每个状态上自动计算、记录、排序、回测。

**工程取舍。** 先做 50-100 个简单公式因子，不急着上神经网络。每次搜索扩展状态时记录因子值和最终 outcome，再做相关性检验。这样能解释为什么 200000 expansions 卡在 MT7：不是算力不足，而是当前排序函数没有正确计价黄钥匙期权。

### FR003 Taming the Factor Zoo

**问题。** 量化领域有大量“看起来有效”的因子，但很多只是数据挖掘和多重检验的产物。直接把所有候选因子加入模型，会导致过拟合和失真。

**方法。** 论文通过严谨的检验框架评估新因子是否能在已有因子解释之外提供增量信息。它强调控制虚假发现率、考虑因子相关性、避免重复表达同一个经济逻辑。

**对魔塔启示。** 魔塔也会出现伪因子。例如“楼层越高越好”与成功高度相关，但它不能指导早期决策；“HP 越高越好”也可能误导 agent 不敢打必要怪。真正有用的是在相同楼层、相同步数区间、相似装备状态下还能区分好坏路线的因子。

**工程取舍。** 因子评估必须做进度中性化：按 `floor_id`、`story_stage`、`steps_bucket` 分组计算 IC。只有分组后仍然稳定的因子，才加入 reward 或 heuristic。否则会产生“奖励上楼”这类短视策略。

### FR016 Alpha-GPT

**问题。** 传统 alpha mining 很难把研究员的自然语言想法转成可运行因子。Alpha-GPT 尝试用 LLM 做交互式因子生成，让人类提供思路，模型生成公式，再由回测反馈筛选。

**方法。** LLM 负责把自然语言假设转成公式化 alpha，系统用历史数据回测，研究员根据结果继续修改。它不是让 LLM 直接判断好坏，而是把 LLM 放在“候选生成器”的位置，评估仍然交给数据。

**对魔塔启示。** 这正适合 reward 设计。我们可以让 LLM 根据游戏机制生成候选因子或 reward 代码，例如“黄钥匙在 7F 前价值更高”“攻击力跨阈值时价值非线性增加”。但不能让 LLM 直接决定最终权重，必须用模拟器回测。

**工程取舍。** 后续可以加一个 `scripts/propose_mota_factors.py`：读取搜索失败日志，生成新因子表达式，再自动跑 10000/100000 expansions 做 ablation。LLM 只提出假设，solver 负责验真。

### FR017 AlphaForge

**问题。** 公式 alpha 不仅要生成，还要动态组合。单个因子在不同市场阶段表现会变，固定权重组合容易失效。

**方法。** AlphaForge 用生成-预测网络挖掘公式因子，再用组合模型根据近期表现动态选择和加权因子。它强调多样性、可解释性和动态适应。

**对魔塔启示。** 魔塔也不是一个全局固定权重问题。1F-3F 关键是剧情和基础资源，4F-7F 关键是黄钥匙和剑，8F-10F 关键是蓝钥匙、盾和 boss 生存边际。一个统一 heuristic 权重很容易失真。

**工程取舍。** 我们应该做阶段化因子组合：`stage=post_thief`、`stage=pre_sword`、`stage=pre_mt8_key`、`stage=pre_shield`、`stage=pre_boss`。每个阶段有不同因子权重。比全局调一个 heuristic 分数更稳。

### FR018 Synergistic Formulaic Alpha Generation based on RL

**问题。** 自动因子生成可以看成一个序列决策问题：逐步选择变量、算子、参数，构造一个完整公式，并用回测结果作为 reward。

**方法。** 论文用强化学习在公式空间中生成协同因子，重点不是单个公式，而是与已有因子组合后的增量价值。

**对魔塔启示。** 魔塔的 reward/heuristic 也可以被看成公式生成问题。动作是选择一个基础特征、算子、阈值、权重，结果是一个状态评分公式。评价标准不是公式本身，而是它是否提升 solver 成功率、缩短路线、降低扩展数。

**工程取舍。** 第一阶段不用训练复杂 RL 生成器，可以先用随机搜索/遗传搜索公式。目标函数设为 `solve_success * 1000 + best_floor * 20 + final_hp_margin - expansions_penalty`。这比手动调一堆常数更系统。

### FR021 Navigating the Alpha Jungle

**问题。** 公式因子搜索空间巨大，遗传规划容易低效，LLM 生成又容易不受约束。该工作把 LLM 和 MCTS 结合，用 MCTS 控制表达式搜索，用 LLM 提供先验和变异建议。

**方法。** MCTS 在公式树空间中展开，LLM 提供候选节点、解释和重写。回测结果反哺搜索树，使得搜索既有探索，也有基于历史收益的利用。

**对魔塔启示。** 这与我们当前的问题非常像：魔塔 solver 卡在局部最优，需要搜索更好的状态价值公式。可以把 reward 表达式当作一棵语法树，用 MCTS 搜索 `factor + factor * weight + nonlinear_threshold` 结构。

**工程取舍。** 后续建议做 `factor_mcts`，不是先做 PPO。每个 reward 候选跑一个小预算搜索，记录 `solved / best_floor / key_deadend / route_len`，把结果作为表达式搜索 reward。这样能把“人调 reward”变成可复现实验。

### FR038 Policy Invariance under Reward Transformations

**问题。** reward shaping 最大风险是改变原问题。加了中间奖励后，agent 可能学会刷奖励而不是通关。

**方法。** Ng 等提出 potential-based reward shaping：额外奖励形如 `F(s,a,s') = gamma * Phi(s') - Phi(s)`，在折扣 MDP 下不改变最优策略。也就是说，势函数可以加速学习，但不会改变最优解集合。

**对魔塔启示。** 我们现在的“里程碑奖励”如果直接写成 `拿钥匙 +1、上楼 +5`，理论上可能诱导 agent 走短视路线。更稳的是学习或手写一个势函数 `Phi(s)`，奖励使用差分形式。比如 `Phi` 估计“当前状态离打败骷髅队长还有多近”。

**工程取舍。** 先把因子组合成 `Phi(s)`，再在 Gym 环境里返回 `gamma*Phi(next)-Phi(curr)`。搜索 heuristic 也可以直接用 `Phi(s)` 排序。这样 reward 和 heuristic 共用一个价值函数，实验更统一。

### FR042 Reward Design via Online Gradient Ascent

**问题。** PGRD 把 reward 参数本身作为需要优化的对象，而不是人工固定。目标不是最大化代理 reward，而是最大化真实 objective。

**方法。** 论文定义一个可参数化 reward，策略在该 reward 下学习；外层用真实性能对 reward 参数做梯度更新。它把 reward 设计变成双层优化。

**对魔塔启示。** 我们可以定义 reward/heuristic 参数，例如钥匙压力权重、HP 权重、攻击阈值权重、里程碑权重。内层是 solver/RL 用这些权重跑路线，外层评价是否击败 10F boss、扩展数多少、是否陷入钥匙死局。

**工程取舍。** 由于当前环境是确定性的，不一定需要真正求梯度。可以先做黑盒优化：CMA-ES、贝叶斯优化、随机搜索。PGRD 的核心启示是：reward 权重必须被真实通关目标检验，而不是凭直觉固定。

### FR047 RUDDER

**问题。** 长视野任务里，最终奖励离关键动作太远，TD 学习很难把信用分配回早期动作。魔塔里“7F 黄钥匙用完”可能要到 10F 才暴露后果。

**方法。** RUDDER 用序列模型预测整条轨迹回报，再通过贡献分析把最终回报重分配到真正导致回报变化的时间点。理想情况下，重分配后未来期望奖励接近 0，学习难度大幅下降。

**对魔塔启示。** 魔塔很适合 RUDDER 思路，因为关键事件稀疏且可解释：拿剑、保留黄钥匙、开错门、拿盾、触发机关。我们可以用成功/失败路线训练一个 return predictor，再用特征贡献找出早期关键动作。

**工程取舍。** 不需要马上完整实现 RUDDER。先做简化版：对每条路线记录里程碑和最终结果，训练树模型或线性模型预测成功，再用 SHAP/置换贡献找“哪些动作导致失败”。这会直接指导 reward 修正。

### FR048 Randomized Return Decomposition

**问题。** RUDDER 依赖完整轨迹回报分解，训练和解释成本较高。RRD 试图用随机片段学习长期 reward redistribution。

**方法。** 它从轨迹中采样片段，用片段级别的目标学习代理 reward，使得稀疏终局信号被拆成更密集的局部信号。

**对魔塔启示。** 我们当前 solver 会产生大量失败前缀。即使没通关，这些前缀也包含信息：哪些片段推进到了更高楼层，哪些片段导致钥匙耗尽。RRD 的片段学习比只看整条路线更适合早期数据不足阶段。

**工程取舍。** 建议在 `artifacts/search_logs/` 里保存每次扩展产生的状态片段，然后训练 `segment_value_model`。输出不是动作策略，而是局部片段价值，先用于排序候选动作。

### FR063 Thinker

**问题。** 标准 MCTS 每步都花固定预算搜索，但长视野游戏并不是每个状态都需要深搜。走廊移动不需要想，资源分支和 boss 前才需要想。

**方法。** Thinker 把世界模型变成 agent 可调用的内部环境，使 agent 学会何时思考、思考多久、何时行动。它在 Sokoban 等任务上体现了动态计算分配的优势。

**对魔塔启示。** 魔塔的计算资源应该集中在“分岔点”：开哪扇门、是否打怪、是否先拿装备、钥匙如何保留。普通移动已经被宏动作压缩，不需要 RL 学箭头级动作。

**工程取舍。** 当前项目可以先实现轻量版 Thinker：宏动作环境中，每个候选目标先用快速 evaluator 过滤；在黄钥匙低、进入新楼层、遇到装备/商店/boss 前自动加大 beam width。也就是让 planner 根据状态风险动态调整预算。

### FR072 Maximum Entropy Inverse Reinforcement Learning

**问题。** 专家路线通常不是唯一最优。只模仿一条路线容易过拟合，而且无法解释专家为什么这样走。MaxEnt IRL 用最大熵原则从专家行为中推断 reward。

**方法。** 在匹配专家特征期望的同时，最大化轨迹分布熵，避免把概率全部压到一条路线。这样可以学习到“多种合理路线背后的共同 reward”。

**对魔塔启示。** 如果后续我们生成多条能过前十层的路线，可以用 MaxEnt IRL 学 reward 权重。例如成功路线共同特征可能不是“固定经过某坐标”，而是“在 8F 前保留足够黄钥匙，并拿到攻防阈值”。

**工程取舍。** 在只有一条路线前，MaxEnt IRL 不急着做。先通过搜索和扰动生成多条成功/近成功路线，再用特征期望学习 reward。输出的 reward 应该是可解释线性/树模型，而不是黑盒网络。

### FR075 AIRL

**问题。** GAIL 可以模仿专家，但学到的 discriminator reward 往往不是真正可迁移的 reward。AIRL 目标是学到更接近环境任务本质的 reward，从而跨动态泛化。

**方法。** AIRL 把 reward 结构拆成状态/动作相关项和 shaping 项，通过对抗训练恢复更稳健的 reward 表达。

**对魔塔启示。** 魔塔后续会从前十层扩展到 50 层，不能只学“这张图上专家在哪些坐标走过”。需要学更通用的 reward：节约钥匙、跨阈值、提高可达资源、避免不可逆死局。

**工程取舍。** AIRL 可以作为后期路线。当前更实际的是保留 AIRL 的结构思想：reward 分成 `task_reward` 和 `potential_shaping`。前者绑定通关事件，后者来自因子势函数。

### FR078 Deep RL from Human Preferences

**问题。** 有些任务很难手写 reward，但人类可以比较两段行为哪段更好。该工作用片段偏好训练 reward model，再用 RL 优化。

**方法。** 系统采样短轨迹片段，请人类做二选一偏好标注；reward model 学会预测偏好，再作为 RL 的奖励信号。

**对魔塔启示。** 魔塔里用户很容易判断两条路线哪条更合理：一条保留钥匙拿装备，一条血很多但卡死。我们可以让人只比较片段，而不是手写 reward 权重。

**工程取舍。** 后续可以做一个简单工具：从搜索日志里抽两段路线，显示状态差异和事件序列，用户选择更优。训练一个 pairwise ranking model，输出 `Phi(s)`。这会比手调 reward 更快收敛。

### FR084 Reward Machines

**问题。** 很多任务的奖励依赖历史事件，不是单个状态能完全表达。Reward Machine 用有限状态机表示任务进度和奖励结构。

**方法。** 用户定义高层事件，自动机根据事件转移并发放奖励。RL agent 在原环境状态和 reward-machine 状态的乘积空间中学习。

**对魔塔启示。** 魔塔前十层天然是 reward machine：小偷剧情结束 -> 拿 5F 剑 -> 到 8F -> 拿蓝钥匙/盾 -> 触发 10F 机关 -> 击败骷髅队长。这个顺序不是简单 reward 加法，而是事件依赖结构。

**工程取舍。** 建议立即实现一个 `MotaProgressMachine`，输出 progress state 和 potential。它既能作为 reward shaping，也能作为搜索日志标签。这样后面做因子中性化时，可以按 progress machine 阶段分组。

### FR090 Text2Reward

**问题。** 手写 dense reward 成本高，且容易漏掉任务约束。Text2Reward 让语言模型根据任务描述和环境代码生成 reward shaping。

**方法。** LLM 读取环境接口和任务文本，生成可执行 reward 代码，并在训练中评估和迭代。它强调 reward 要直接连接可观测状态和任务目标。

**对魔塔启示。** 魔塔环境是确定性的，状态变量清晰，非常适合 LLM 生成 reward 候选。比如 LLM 可以读到怪物表、门坐标、楼层事件，提出钥匙压力、战斗阈值、装备必要性等因子。

**工程取舍。** 不能直接相信 LLM reward。正确做法是：LLM 生成候选 reward 函数 -> 单元测试验证无非法字段 -> 搜索回测 -> 因子检验 -> 只保留提升真实通关指标的版本。

### FR091 Eureka

**问题。** Eureka 进一步把 LLM reward 设计做成进化搜索：模型生成 reward 代码，训练策略，根据表现反馈给模型，再生成下一代 reward。

**方法。** 它利用 LLM 的代码生成和自我改进能力，在多个候选 reward 之间做进化式优化。评价标准来自环境真实表现，而不是文本打分。

**对魔塔启示。** 这正是魔塔 reward 设计的长期方向。我们可以把 reward 函数作为 Python 代码片段进化，每一代在 solver/RL 上跑固定预算，选择成功率和扩展效率更好的版本。

**工程取舍。** 第一阶段不要让 LLM 改 simulator，只允许改 `factor_expr.py` 或 `reward_candidates/*.py`。每个候选必须通过 replay 和测试，防止 reward 代码利用环境 bug。

### FR098 Planning in a Recurrent Neural Network that Plays Sokoban

**问题。** 模型无关 RL 是否真的会规划？Sokoban 研究通过分析 DRC 网络发现，循环计算步数会显著影响解题能力，网络内部出现类似规划的行为。

**方法。** 研究者固定环境动作前的内部循环 ticks，观察成功率、隐状态和规划迹象。结果显示 DRC 不是只做反射式策略，而是在内部展开多步推理。

**对魔塔启示。** 魔塔如果做神经策略，不能只用普通 MLP/CNN 一步输出动作。它需要内部计算时间，尤其是在门、怪、钥匙和装备的长依赖场景中。

**工程取舍。** 近期不建议直接训练 DRC。但可以借鉴“内部 ticks”的思想：在宏动作 policy 前加入 evaluator 多次迭代，或者让 transformer/GRU 读取候选动作列表和局部搜索结果，再做排序。

### FR100 Searchformer

**问题。** 直接让 Transformer 输出解题路径很难；但让它学习 A* 的搜索动态，再逐步蒸馏出更短搜索过程，会更稳定。

**方法。** Searchformer 把 A* 搜索轨迹编码成 token 序列，包括节点加入、弹出、扩展等过程。模型先模仿搜索动态，再通过 expert iteration 优化搜索效率。

**对魔塔启示。** 我们现在已经有 solver 搜索日志。不要只保存最终路线，应该保存开放表扩展、候选动作评分、被剪枝原因、dominance 结果。这些就是训练“魔塔 Searchformer”的原料。

**工程取舍。** 下一步应该把 solver 改成可记录 search trace：`state_id, parent_id, action, factors, score, expanded, pruned_reason, outcome_estimate`。先用这个训练一个动作排序模型，比从零 PPO 更靠谱。

### FR109 Solving Sokoban using Hierarchical Reinforcement Learning with Landmarks

**问题。** 长视野解谜很难直接从初始状态规划到最终状态。HalfWeg 思路把问题递归拆成中间 landmark/subgoal。

**方法。** 高层策略生成中间状态，低层策略尝试从当前状态到中间状态、再到终点。通过多层级递归，长任务被拆成更短、更可学习的片段。

**对魔塔启示。** 魔塔的 landmark 非常明确：小偷剧情结束、拿 5F 剑、通过 7F、拿 8F 蓝钥匙、拿 9F 盾、10F boss。与其让 solver 在全局空间盲搜，不如强制评估这些 landmark 的可达性和资源余量。

**工程取舍。** 建议做 `landmark_solver`：每个阶段只优化到下一个 landmark，并把到达时的资源余量作为阶段得分。阶段之间再做 beam 合并。这样比一个全局优先队列更不容易在 MT7 局部最优里打转。

## 3. 对当前魔塔项目的直接结论

当前项目最该补的不是 PPO，而是三件事：

1. **状态因子日志。**  
   每次 solver 扩展状态时记录一行：状态摘要、候选动作、因子值、score、是否扩展、是否剪枝、最终 best outcome。没有这个日志，就无法判断 reward/heuristic 是不是有效。

2. **因子评估。**  
   对每个候选因子计算：
   - 全局 IC：因子和未来 `best_floor / solved / final_hp` 的相关性；
   - 分组 IC：按楼层、progress machine、步数分桶后再算；
   - 单调分组：高因子组是否更容易到 MT8/MT9/MT10；
   - 稳定性：不同搜索预算和随机 tie-break seed 下是否一致。

3. **势函数化 reward。**  
   先把有效因子组合为 `Phi(s)`，再用 `gamma * Phi(s_next) - Phi(s)` 做 shaping。这样比直接给“上楼 +5”更少改变真实目标。

## 4. 推荐的魔塔因子库 v0

建议先实现以下基础因子，每个因子都要进入日志：

- `floor_progress`: 楼层进度，但只作为分组变量，不直接作为强 reward。
- `progress_machine_state`: 里程碑自动机状态。
- `hp_margin_to_known_boss`: 当前 HP 对关键怪/boss 的生存边际。
- `yellow_key_pressure`: 剩余黄钥匙与未来必要黄门之间的压力。
- `blue_key_option_value`: 蓝钥匙打开高价值区域的期权价值。
- `reachable_attack_gain`: 不额外开门情况下可拿攻击增益。
- `reachable_defense_gain`: 不额外开门情况下可拿防御增益。
- `door_unlock_value`: 开某扇门后新增可达资源价值减去钥匙机会成本。
- `combat_efficiency`: 击杀怪物的金币/经验/路径收益除以 HP 损耗。
- `threshold_distance_atk`: 距离下一个关键攻击阈值差多少。
- `threshold_distance_def`: 距离下一个关键防御阈值差多少。
- `deadend_key_risk`: 钥匙为 0 且关键资源未拿时的风险。
- `landmark_reachability`: 下一个 landmark 是否可达。
- `resource_liquidity`: 可立即转化为战力的资源量，而不是账面资源量。
- `irreversible_cost`: 打怪、开门、吃血瓶等不可逆动作的机会成本。

## 5. 推荐实验路线

### 阶段 A：先做搜索日志和因子回测

目标不是马上通关，而是把失败解释清楚。

1. 改 solver，保存 `artifacts/search_logs/search_trace_*.parquet/jsonl`。
2. 每个状态计算 v0 因子。
3. 用当前 10000、200000 expansions 的失败轨迹做第一版因子 IC。
4. 找出导致 MT7 卡死的最强负因子，预计会是黄钥匙压力和门后资源价值估计错误。

### 阶段 B：把稳定因子变成 heuristic

只保留分组后仍有效的因子，组合成：

```text
score(s) = base_progress
         + w1 * Phi_landmark(s)
         + w2 * key_option_value(s)
         + w3 * combat_threshold_value(s)
         - w4 * deadend_risk(s)
         - w5 * irreversible_cost(s)
```

权重不手拍，先用随机搜索或贝叶斯优化，目标函数是：

```text
objective = solved * 10000
          + best_landmark * 1000
          + best_floor * 100
          + final_hp_margin
          - log(expansions)
          - key_deadend_penalty
```

### 阶段 C：再做 reward shaping 与 BC/RL

当 solver 能稳定产出成功路线后：

1. 用成功路线做 BC，不再用当前失败路线做主要监督。
2. Gym reward 使用 `Phi` 差分，而不是手写稠密奖励。
3. PPO/DQN 只作为微调，不作为第一阶段主力。
4. 如果有多条成功路线，再尝试 MaxEnt IRL/AIRL 学 reward。

### 阶段 D：LLM/进化式 reward 搜索

参考 Alpha-GPT、AlphaForge、Eureka：

1. LLM 生成候选因子表达式。
2. 单元测试检查表达式只读合法状态字段。
3. solver 小预算回测。
4. 保留表现提升且可解释的因子。
5. 每轮把失败日志摘要反馈给 LLM，生成下一代因子。

## 6. 近期最小可实施任务

为了让这份文献研究真正推动当前项目，建议下一步直接做：

1. 新增 `src/mota_solver/factors.py`：实现 v0 因子。
2. 修改 `src/mota_solver/search.py`：每次扩展记录 `state_id/action/factors/score/outcome/pruned_reason`。
3. 新增 `scripts/analyze_factors.py`：输出 IC、分组 IC、top/bottom 分组胜率。
4. 新增 `src/mota_solver/progress_machine.py`：小偷剧情后到 10F boss 的 reward machine。
5. 用 10000 expansions 先生成日志，再跑一次 200000，只比较“同预算下 best landmark 是否提升”。

## 7. 关键判断

魔塔前十层当前不是“缺一个更大的 RL 算法”，而是“缺一个能解释状态价值的因子系统”。量化因子挖掘给我们的最强方法论是：

- 先把状态价值拆成可解释候选因子；
- 再用大量搜索日志检验因子；
- 然后把稳定因子组合为势函数；
- 最后再让 RL 学这个被验证过的结构。

这比从零训练 PPO 更符合魔塔这种确定性、长视野、强死局、强资源约束的环境。
