# 09 - Thinker

## 基本信息

- 论文：Thinker: Learning to Plan and Act
- 作者：Stephen Chung, Ivan Anokhin, David Krueger
- 年份：2023
- 链接：https://arxiv.org/abs/2307.14993
- 本地 PDF：`paper/top10_pdfs/09_thinker_2307.14993.pdf`
- 抽取文本：`paper/top10_extracted/09_thinker_2307.14993.md`

## 一句话结论

Thinker 把世界模型包装进环境，让 agent 像执行真实动作一样执行“想象动作”，从而学会何时规划、规划什么、何时执行。

## 论文解决的问题

大多数 model-based RL 使用手写规划器，例如 MCTS、穷举 rollout、I2A 的固定 rollout。Thinker 问的是：能不能让 agent 自己学会如何使用世界模型，而不是由研究者规定搜索算法？

这正好对应魔塔中的预算分配问题。走廊移动不需要深思；黄钥匙紧张、临近 Boss、攻防临界时需要大量搜索。

## 核心方法

Thinker 把原 MDP 转换成 augmented MDP。一个真实环境 step 被拆成一个 stage：

- 前 `K-1` 步是 imaginary steps，动作发给世界模型。
- 第 `K` 步是真实 step，动作发给真实环境。

在 imaginary step 中，agent 选择：

- 一个想象动作。
- 是否 reset 到 root。

模型沿动作展开，agent 看到的是 augmented state，其中包含：

- root node 信息。
- current node 信息。
- 预测 reward、value、policy。
- rollout return 的均值和最大值。
- visit count。
- 当前 stage 内步数。

真实 reward 只在真实 step 给出，imaginary step reward 为 0。折扣设为 `gamma^(1/K)`，保证 augmented MDP 的 return 与真实 MDP 对齐。

## 世界模型

Thinker 使用 dual network：

1. state-reward network：预测未来状态、reward、终止概率。
2. value-policy network：预测未来 value 和 policy。

论文认为直接像素 L2 可能浪费容量，所以提出 feature loss，让状态预测更关注对决策有用的特征。

## 实验结论

Sokoban 上，Thinker 在 5e7 frames 达到约 95% 解题率，明显优于同样 actor-critic 在 raw MDP 上的表现，也超过多个 baseline。测试时改变 `K` 显示，更多 planning steps 能显著提高性能。Atari 200M frames 上，Thinker 的 agent policy 也明显优于 raw MDP baseline，但还没有达到最强 SOTA。

重要 ablation：

- `K=1` 没有模型交互，性能差。
- `K=10` 已接近最优，不需要 MCTS 那种数千 simulations。
- hints 或 RNN memory 至少要有一个，否则 agent 学不会总结 rollouts。
- base policy/value 是学习规划的重要启发。

## 对魔塔的启发

当前魔塔不需要马上实现完整 Thinker，但它给出一个很实用的工程方向：动态规划预算。

我们可以做轻量版 Thinker：

- 普通状态：只扩展少量宏动作。
- key pressure 高：增加 expansions。
- Boss damage 接近 HP：增加 rollout depth。
- 新楼层首次进入：增加搜索。
- 攻击/防御临界点附近：增加候选动作采样。

也就是说，让 solver 根据状态风险分配搜索预算，而不是固定 `--max-expansions`。

## 工程落地

建议实现一个 budget controller：

```text
budget_multiplier =
  1
  + key_pressure_bonus
  + boss_margin_risk_bonus
  + threshold_nearby_bonus
  + new_floor_bonus
  + deadend_risk_bonus
```

用于控制：

- 每个节点展开的候选动作数。
- 局部 rollout depth。
- stage 内保留的 top-K 状态。
- 是否调用昂贵的 boss damage evaluator。

## 局限

Thinker 训练成本很高，并且 primitive action 规划在动作效果很小的游戏中收益有限。魔塔如果直接用方向键作为 primitive action，会浪费大量想象步。因此应该先在宏动作环境中做 Thinker-like planner。

## 给本项目的下一步

1. 不做完整 Thinker，先做动态 beam/search budget。
2. 把 `key_pressure`、`boss_margin`、`threshold_delta` 变成预算因子。
3. 对比固定 200000 expansions 与动态预算的最远进度和内存消耗。
