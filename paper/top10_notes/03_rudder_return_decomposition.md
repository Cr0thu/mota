# 03 - RUDDER: Return Decomposition for Delayed Rewards

## 基本信息

- 论文：RUDDER: Return Decomposition for Delayed Rewards
- 作者：Arjona-Medina et al.
- 年份：2019
- 链接：https://arxiv.org/abs/1806.07857
- 本地 PDF：`paper/top10_pdfs/03_rudder_1806.07857.pdf`
- 抽取文本：`paper/top10_extracted/03_rudder_1806.07857.md`

## 一句话结论

长延迟任务中，关键不是把最终奖励慢慢 TD 回传，而是学习哪些早期状态转移真正改变了最终回报，并把奖励重分配到这些转移上。

## 论文解决的问题

RUDDER 针对 delayed reward。普通 TD 学习要把最终奖励一步步往前传，延迟越长，需要的更新越多；Monte Carlo 又会因为未来随机性产生高方差。论文的目标是构造一个 return-equivalent 的新过程，使未来期望奖励接近 0，这样 Q 值估计退化为估计即时 reward 均值。

对魔塔来说，最典型的 delayed reward 是：

- 早期少拿一把黄钥匙，几百步后在 9F/10F 卡死。
- 早期少拿一颗攻击宝石，后面 Boss damage 多几十到几百。
- 早期杀错怪，HP 没归零，但最终不够打骷髅队长。

这些都不是当前动作立刻能看出来的。

## 核心方法

RUDDER 有两个核心概念：

1. Reward redistribution：把同一条 episode 的总回报重新分配到序列中的关键时间点，保持 return-equivalent，因此最优策略不变。
2. Return decomposition：训练一个序列模型预测整条轨迹的最终 return，然后用相邻时间点预测值的差分作为贡献。

论文给出一个重要结论：理想情况下，重分配后的即时 reward 等于原延迟任务中相邻 Q 值的差。这样未来期望奖励为 0，Q 学习变成更简单的即时均值估计。

实现上，RUDDER 用 LSTM 读完整轨迹，在每个时间点预测最终 return。相邻预测差：

```text
contribution_t = predicted_return_t - predicted_return_{t-1}
```

就是分配给当前转移的 reward。

## 实验结论

论文设计了三个 delayed reward 人工任务：

1. Grid World：测试 TD 对长延迟奖励传播的困难。
2. The Choice：测试 MC 在未来无关随机奖励下的高方差。
3. Trace-Back：测试 potential shaping 也可能无法高效传播关键早期动作。

RUDDER 在这些任务上明显快于 TD、MC、MCTS 和几类 shaping。Atari 上，RUDDER 叠加 PPO 后在 delayed reward 明显的游戏中改进更大，例如 Bowling、Solaris、Venture、Seaquest。

## 对魔塔的启发

魔塔项目现在只保存最终路线 JSONL 不够。我们应该保存失败路线和搜索过程，再训练一个“最终结果预测器”：

输入：每一步状态因子和动作。

输出：

- 是否最终击败骷髅队长。
- 最终 HP。
- 骷髅队长 damage margin。
- 是否钥匙死局。
- 最远楼层/里程碑。

然后分析预测值在哪些动作后发生大幅变化。那些动作就是应该被 reward/heuristic 强化或惩罚的关键点。

例如：

- 开某扇黄门后成功概率大幅下降，说明这扇门消耗了关键黄钥匙。
- 拿铁剑后成功概率大幅上升，说明铁剑是强 landmark。
- 某次战斗后预测 HP margin 下降过多，说明该怪应被规避或延后。

## 工程落地

建议新增 `artifacts/trajectory_logs/`，每一步记录：

```json
{
  "step": 123,
  "state_hash": "...",
  "floor": "MT7",
  "pos": [x, y],
  "hp": 726,
  "atk": 20,
  "def": 10,
  "keys": {"yellow": 2, "blue": 1, "red": 0},
  "action": {"kind": "battle", "target": "skeletonSoldier"},
  "phi_components": {...},
  "reachable_items": {...},
  "boss_damage": 630,
  "event": "...",
  "outcome": null
}
```

轨迹结束后补全 outcome。之后训练一个轻量模型，不必一开始用 LSTM，可以先用梯度提升树或小 Transformer。

## 局限

RUDDER 需要足够多样的完整 episode。当前我们还没有成功路线，只有 partial route 和失败路线。因此第一阶段可以把 outcome 设成连续信号，例如最远阶段、Boss damage、HP margin，而不是只用成功/失败二值。

## 给本项目的下一步

1. solver 不只保存 best route，还保存若干失败但有代表性的 routes。
2. 为每条 route 计算最终 outcome。
3. 训练 return predictor。
4. 用相邻预测差分挖掘“奖励因子”和“危险动作”。
