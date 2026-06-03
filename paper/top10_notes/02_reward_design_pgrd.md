# 02 - Reward Design via Online Gradient Ascent

## 基本信息

- 论文：Reward Design via Online Gradient Ascent
- 作者：Jonathan Sorg, Satinder Singh, Richard L. Lewis
- 年份：2010
- 链接：https://papers.nips.cc/paper_files/paper/2010/file/168908dd3227b8358eababa07fcaf091-Paper.pdf
- 本地 PDF：`paper/top10_pdfs/02_pgrd_sorg_lewis_singh_2010.pdf`
- 抽取文本：`paper/top10_extracted/02_pgrd_sorg_lewis_singh_2010.md`

## 一句话结论

reward 不一定要人工固定；可以把 reward 参数当成策略参数的一部分，用真实目标表现来在线优化 reward 权重。

## 论文解决的问题

经典 RL 通常默认“设计者想最大化的 reward”和“agent 学习时使用的 reward”是同一个。但在计算受限、规划深度有限、模型错误或部分可观测的 agent 中，直接使用真实目标 reward 可能效果很差。

论文把这个问题称为 optimal reward problem：设计一个 agent 内部使用的 reward，使它最终最大化设计者关心的 objective reward。

这和魔塔很像。我们真正关心的是击败 10F 骷髅队长；但训练时如果只给最终胜利奖励，agent 几乎学不到东西。我们加的钥匙、宝石、Boss damage、里程碑 reward，本质上都是 agent 内部 reward。

## 核心方法

PGRD 的关键视角是：如果 agent 的行为由 reward 参数 `theta` 决定，那么 reward 参数也可以被看成策略参数。论文考虑一类有限深度 model-based planning agent：

1. agent 有一个内部 reward `R(i,o,a;theta)`。
2. agent 用这个 reward 做深度 `d` 的局部规划，得到动作分数。
3. 动作通过 softmax 采样。
4. 真实环境给出设计者关心的 objective reward。
5. 用 policy gradient 思路更新 `theta`，让 objective reward 变大。

和普通 policy gradient 的区别是，梯度要穿过“reward 参数 -> 局部规划 Q 值 -> softmax policy”这条链。论文给出了深度 `d` 规划下 `Q` 对 reward 参数的递归梯度。

## 实验结论

论文做了几个受限 agent 实验：

1. Foraging fully observable：有限规划深度越浅，越需要 reward 调整。PGRD 能在在线经验中提升 objective return。
2. Poor model：即使模型错误，表达力足够的 reward 参数也能缓解错误模型带来的损失。
3. Partially observable foraging：在部分可观测且模型假设错误时，固定 reward 的规划可能反而更差，PGRD 调 reward 后显著改善。
4. Acrobot：在连续控制任务中，带规划的 PGRD 优于不带规划的策略梯度版本。

一个重要发现是：更深的规划不总是更好。如果模型错，深规划会放大错误；这对魔塔也有启发，模拟器正确时可以深搜，学习模型不准时不能盲目深搜。

## 对魔塔的启发

当前我们手调 reward/heuristic 时遇到的问题是：

- 加强红宝石/蓝宝石奖励会让 agent 消耗太多钥匙。
- 重视 Boss damage 会导致早期路线变差。
- 重视上楼会让 agent 忽略低层资源。

PGRD 给出的思路是：不要长期手调权重，而是把这些项参数化，然后用真实目标回测更新。

可参数化的魔塔 reward 因子：

- `hp_weight`
- `atk_weight`
- `def_weight`
- `yellow_key_weight_by_stage`
- `blue_key_weight_by_stage`
- `boss_damage_drop_weight`
- `boss_survival_margin_weight`
- `unlock_reachable_item_weight`
- `milestone_weight`
- `deadend_penalty`

真实 objective 不应该是这些 reward 的加和，而应该是：

```text
success * 100000
+ reached_stage_score
+ final_hp_margin
- route_length_penalty
- expansion_cost
```

## 工程落地

短期不需要完整实现 PGRD，可以先做低成本版本：

1. 把当前 heuristic/reward 权重放进 JSON config。
2. 每组权重跑固定预算搜索，例如 10000、50000、200000 expansions。
3. 记录成功率、最远楼层、Boss damage、最终 HP、钥匙死局、路线长度。
4. 用随机搜索、CMA-ES、贝叶斯优化或网格搜索调权重。

这等价于把 reward 设计变成可复现实验，而不是凭直觉改常数。

## 局限

PGRD 假设可以估计 reward 参数对策略的梯度；魔塔当前主流程是确定性搜索，不是可微 planning agent。所以第一阶段更适合做黑盒参数搜索，而不是直接套 PGRD。

## 给本项目的下一步

1. 新增 `configs/reward_weights/*.json`。
2. 新增 `scripts/sweep_reward_weights.py`。
3. 每次搜索输出统一 metrics。
4. 先用 50-200 组权重跑小预算，筛出前 5 组再跑大预算。
