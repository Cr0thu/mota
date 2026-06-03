# 07 - MuZero

## 基本信息

- 论文：Mastering Atari, Go, Chess and Shogi by Planning with a Learned Model
- 作者：Julian Schrittwieser et al.
- 年份：2019
- 链接：https://arxiv.org/abs/1911.08265
- 本地 PDF：`paper/top10_pdfs/07_muzero_1911.08265.pdf`
- 抽取文本：`paper/top10_extracted/07_muzero_1911.08265.md`

## 一句话结论

MuZero 不学习重建环境，而是学习一个只服务于规划的隐式模型：给定 latent state 和 action，预测下一 latent state、reward、policy、value，并在 latent space 中做 MCTS。

## 论文解决的问题

AlphaZero 依赖完美规则模拟器。现实任务往往没有规则，或者规则太复杂。传统 model-based RL 试图预测下一帧或真实状态，但这会浪费大量模型容量在与决策无关的细节上。MuZero 的问题是：能否不学习真实规则，也能获得可搜索的模型？

## 核心架构

MuZero 有三个函数：

1. 表示函数 `h`：把历史观测编码为 root latent state `s0`。
2. 动态函数 `g`：输入 `s^{k-1}` 和假想动作 `a^k`，输出 reward `r^k` 和下一 latent state `s^k`。
3. 预测函数 `f`：输入 latent state，输出 policy `p^k` 和 value `v^k`。

训练目标不是重建观测，而是对齐三类决策相关目标：

- reward target：真实即时 reward。
- value target：n-step return 或最终胜负。
- policy target：MCTS 后的 improved policy。

MCTS 在 latent state 上运行，节点边保存 visit count、Q、prior、reward 和 transition。

## 搜索细节

MuZero 的搜索类似 AlphaZero：

- selection 用 PUCT。
- expansion 调用 dynamics 和 prediction。
- backup 把预测 reward 与 leaf value 折扣回传。
- root visit count 归一化成 improved policy。

在 Atari 中，MuZero 使用更少 simulations；在棋类中使用 800 simulations。论文还用 min-max 归一化处理非固定范围 Q 值。

## 实验结论

MuZero 在围棋、国际象棋、将棋上匹配或超过 AlphaZero，但不使用规则；在 Atari 57 个游戏上超过当时 model-free 和 model-based 方法。MuZero Reanalyze 通过用新网络重新搜索旧轨迹提升样本效率。

一个关键 ablation 是：用同样网络换成 model-free Q-learning 后，Ms. Pacman 表现远低于 MuZero。这说明 MCTS policy improvement 给了更强的学习信号。

## 对魔塔的启发

当前魔塔前十层有显式模拟器，所以第一阶段不需要 MuZero。原因很简单：我们已经有精确转移和伤害公式，没必要先让模型学习规则。

但 MuZero 对后续有价值：

1. 如果要泛化到不同魔塔版本、未知事件脚本、自动解析地图，MuZero-style latent model 是主线。
2. 如果要把搜索器变成学习系统，MCTS improved policy 可以作为监督信号。
3. reward/value/policy 三头结构适合魔塔宏动作环境。

魔塔中的 dynamics model 必须学会：

- 钥匙扣除和门消失。
- 战斗伤害公式。
- 物品拾取和属性变化。
- 楼层跳转和事件 flag。
- 死亡和不可达状态。

这些在前十层显式模拟器里已经有，所以先用搜索专家更现实。

## 工程落地

当前代码应保留 MuZero 接口，而不是马上实现完整 MuZero：

```text
encode(observation) -> latent
dynamics(latent, macro_action) -> next_latent, reward
predict(latent) -> policy_logits, value
search(latent, action_mask) -> improved_policy
```

这样后续可以接 LightZero 或 UniZero，而不重写环境。

## 局限

MuZero 训练成本高，且标准 MuZero 对长历史和部分可观测任务不够好。魔塔 50 层跨楼层记忆、钥匙回收路线和远期门控更接近 POMDP/long-history 问题，标准 MuZero 可能不如 UniZero 或显式 staged search。

## 给本项目的下一步

1. 前十层先用确定性 simulator + staged search。
2. 用 solver 的 MCTS/beam 输出做 policy/value 监督数据。
3. 只有当专家路线稳定后，再考虑接 LightZero/MuZero。
