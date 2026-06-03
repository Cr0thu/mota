# 01 - Policy Invariance under Reward Transformations

## 基本信息

- 论文：Policy Invariance under Reward Transformations: Theory and Application to Reward Shaping
- 作者：Andrew Y. Ng, Daishi Harada, Stuart Russell
- 年份：1999
- 链接：https://people.eecs.berkeley.edu/~pabbeel/cs287-fa09/readings/NgHaradaRussell-shaping-ICML1999.pdf
- 本地 PDF：`paper/top10_pdfs/01_pbrs_ng_harada_russell_1999.pdf`
- 抽取文本：`paper/top10_extracted/01_pbrs_ng_harada_russell_1999.md`

## 一句话结论

如果我们要给魔塔加中间奖励，最稳妥的形式是把状态评估写成势函数 `Phi(s)`，再用 `F(s,a,s') = gamma * Phi(s') - Phi(s)` 做奖励塑形；除此之外的 shaping 在一般 MDP 中可能改变最优策略。

## 论文解决的问题

强化学习里经常会给 agent 加辅助奖励，例如靠近目标给奖励、触碰某个对象给奖励、完成子目标给奖励。但这种做法有一个根本风险：agent 可能学会刷辅助奖励，而不是完成原任务。论文开头举了两个典型 bug：自行车任务中 agent 学会在起点附近绕圈来反复获得“靠近目标”的奖励；足球机器人任务中 agent 学会在球旁边抖动来反复获得“碰球”奖励。

论文的问题是：对一个 MDP 的 reward 做什么变换，才能保证最优策略不变？

## 核心方法

论文证明了两个策略保持的 reward 变换：

1. 正线性变换：`R' = alpha * R + beta`，其中 `alpha > 0`。
2. 势能差塑形：额外 reward 写成 `F(s,a,s') = gamma * Phi(s') - Phi(s)`。

关键定理是必要且充分条件：如果没有额外假设，要保证对所有可能的转移和 reward 都不改变最优策略，那么 shaping reward 必须是势能差形式。

直观理解是：势能差在任意闭环路径上的累计值会抵消，不会制造“绕圈刷分”的正收益循环。论文还指出，如果 `Phi(s)` 接近真实最优价值函数 `V*(s)`，学习会变得更容易；但即便 `Phi` 只是粗糙的距离或子目标估计，也能显著加快学习。

## 实验结论

论文用网格世界展示两类势函数：

1. 距离型势函数：用曼哈顿距离估计到目标的剩余步数。
2. 子目标型势函数：按已完成的子目标数量给不同势能。

两者都能明显减少到达目标所需训练时间。更重要的是，实验不是为了证明 PBRS 一定最强，而是说明一个粗糙但方向正确的 `Phi(s)` 也能有效。

## 对魔塔的启发

魔塔里最危险的 reward 设计是直接写：

- 拿黄钥匙 `+1`
- 上楼 `+5`
- 杀怪 `+经验/金币`
- 开门 `+1`

这些都可能诱导短视行为。例如 agent 可能为了上楼错过关键资源，或为了拿眼前奖励消耗不可替代的钥匙。按这篇论文，应该把“状态好坏”写成 `Phi(s)`，然后用势能差产生即时 reward。

魔塔中的 `Phi(s)` 可以包括：

- 当前 HP、ATK、DEF、钥匙、金币的资产价值。
- 对骷髅队长和关键怪物的伤害下降。
- 未开启关键区域的可达性。
- 当前是否已经获得铁剑、铁盾、关键蓝钥匙。
- Boss 生存边际：`HP - damage_to_skeleton_captain`。

然后 reward 使用 `gamma * Phi(next) - Phi(curr)`，而不是对每个事件手写常数奖励。

## 工程落地

当前项目应该把 `src/mota_env/rewards.py` 里的动态 reward 继续收敛为一个统一接口：

```text
Phi(s) = asset_factor + threshold_factor + key_option_factor + unlock_factor + boss_margin_factor
reward = env_terminal_reward + gamma * Phi(s_next) - Phi(s)
```

每新增一个 reward 想法，先问：它能不能写进 `Phi(s)`？如果不能，就要特别警惕是否会改变目标。

## 局限

PBRS 保证的是最优策略不变，不保证训练一定快，也不保证函数近似、动作 mask、有限搜索预算下完全安全。魔塔目前还有搜索预算和剪枝误差，所以 PBRS 是必要基础，但不能单独解决通关。

## 给本项目的下一步

1. 把当前 reward 因子拆成可记录的 `Phi` components。
2. 每次 solver 扩展状态时记录 `Phi` 分解。
3. 做 ablation：只开 asset、只开 threshold、只开 key option、组合后比较搜索最远进度和 Boss damage。
