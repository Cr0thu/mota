# 05 - HalfWeg

## 基本信息

- 论文：Solving Sokoban using Hierarchical Reinforcement Learning with Landmarks
- 作者：Sergey Pastukhov
- 年份：2025
- 链接：https://arxiv.org/abs/2504.04366
- 本地 PDF：`paper/top10_pdfs/05_halfweg_2504.04366.pdf`
- 抽取文本：`paper/top10_extracted/05_halfweg_2504.04366.md`

## 一句话结论

长程解谜不要直接从起点规划到终点；应该学习或定义中间 landmark，把一个长任务递归拆成多个短任务。

## 论文解决的问题

Sokoban 的困难在于长程、不可逆和稀疏奖励。HalfWeg 尝试不依赖人工领域知识，学习一个多层级策略体系，让高层输出中间状态，低层负责实现中间状态。

魔塔前十层的问题结构非常类似。直接从小偷剧情后状态搜到 10F 骷髅队长，会在钥匙、装备、血量、攻防阈值之间产生巨大组合空间。实际应该先过若干中继状态。

## 核心方法

HalfWeg 把任务定义为 planning problem instance：

```text
(u, v, b)
```

- `u`：起始状态。
- `v`：目标状态。
- `b`：方向标志。`b=0` 表示尽量接近目标，`b=1` 表示远离目标，用来促进探索。

它使用两类模型：

1. `MA(u,v,b)`：输出长度为 `d` 的动作序列。
2. `MS(u,v,b,r)`：输出一个 landmark state `w`，把 `u -> v` 拆成 `u -> w` 和 `w -> v`。

策略层级是递归的：

- `PL0` 直接调用动作模型，输出短动作序列。
- `PLi` 先用 `MS` 生成中间状态，再调用 `PL(i-1)` 两次，拼接两段路径。

这样 `PLi` 输出长度约为 `2^i * d` 的计划。论文中使用 6 层策略。

## 训练方式

HalfWeg 的训练循环包括：

1. 自玩和随机探索，收集状态池。
2. 从状态池采样 `(u,v,b)` planning instances。
3. 用不同层级策略和搜索生成候选 plan。
4. 选择效果最好的 plan 作为监督数据。
5. 用这些数据训练 `MA` 和 `MS`。

它不是简单 model-free RL，而是“搜索产生训练数据，模型模仿搜索，模型再辅助搜索”的循环。

## 实验结论

在 Boxoban 上，单次前向调用的高层策略能解一部分测试题；当增加目标枚举和搜索调用次数后，解题率显著提升。论文报告高层策略 `PL5` 在 36 个目标和 1000 搜索下可以解超过 90% 的 Boxoban 测试题。模型规模并不大，核心优势来自递归 landmark 结构。

## 对魔塔的启发

魔塔前十层非常适合 staged solver。与其调一个全局 heuristic，不如把目标拆成：

1. 小偷剧情后起点状态。
2. 获得 5F 铁剑。
3. 拿到 8F/9F 关键钥匙和可达资源。
4. 获得 9F 铁盾。
5. 达到 Boss-ready 状态：`HP > skeletonCaptainDamage + buffer`。
6. 到达 10F 并击败骷髅队长。

每个阶段的价值函数不同：

- pre-sword：攻击力、黄钥匙、到 5F 的路线更重要。
- pre-shield：蓝钥匙、盾路线、低损战斗更重要。
- pre-boss：攻防阈值、HP margin、红/蓝宝石价值更重要。

这解释了为什么当前全局 reward 调参容易失灵：早期给宝石过高奖励会导致钥匙耗尽，后期不给宝石奖励又会导致攻防太低。

## 工程落地

第一阶段不必端到端学习 `MS`。可以先手写 landmarks，做 staged search：

```text
stage_1_success = has_item("sword1")
stage_2_success = reached_MT8_or_key_area
stage_3_success = has_item("shield1")
stage_4_success = boss_damage < hp - buffer
stage_5_success = flag_10f_skeleton_captain
```

每个 stage 有独立 heuristic weights，并把上一阶段找到的若干 Pareto 状态作为下一阶段起点。这会比单次 200000 expansions 更可控。

## 局限

HalfWeg 对 Sokoban 状态间距离定义比较自然；魔塔状态既有地图消耗，又有资源数值，目标状态不能只用网格相似度表示。魔塔 landmark 必须包含资源阈值，例如 `ATK>=25`、`DEF>=23`、`yellow_key>=x`，不能只包含坐标。

## 给本项目的下一步

1. 实现 `src/mota_solver/staged_search.py`。
2. 每阶段保留 top-K 非支配状态，而不是只保留一条 best route。
3. 阶段间记录资源差异，回测哪些 landmark 真正提高最终成功率。
