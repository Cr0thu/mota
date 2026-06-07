# Agentic RL-Style Planning Section

在前面的实验中，我们尝试过基于搜索和阶段式规划的方法来解决前 10 层 MOTA。这些方法能够推进到若干关键阶段，例如拿到剑、盾，或者完成一部分宝石和钥匙规划，但整体上还不够稳定，尤其容易在中后期资源瓶颈处失败。通过这些实验可以看出，MOTA 不是一个简单的最短路问题，而是一个强资源约束、长程依赖、局部最优容易误导的序列决策问题。例如，早期是否保留黄钥匙会影响 8 层和 9 层的通路，是否提前消耗蓝钥匙会影响 10 层资源获取，而攻击、防御宝石和血瓶的获取顺序会直接决定最终 boss 战是否可行。因此，仅依赖单一启发式函数或局部贪心策略，很容易在中后期出现钥匙不足、血量不足或关键资源路径被锁死的问题。

基于这些观察，我们进一步尝试了一种 agentic RL-style planning 方法。该方法并不是把整个游戏交给一个单一策略网络直接输出动作，而是将 MOTA 的决策过程拆分为多个具有不同职责的 agent，由它们共同评估当前状态下的合法宏动作。环境仍然由 MOTA simulator 提供，每一步先生成所有合法 macro actions，例如开门、打怪、拿宝石、拿钥匙、上下楼或触发事件；随后不同 agent 从不同角度对候选动作进行打分，最后由 beam-search arbiter 综合评分并选择下一步动作。

具体来说，我们设计了以下几类 agent：

| Agent | 主要职责 |
| --- | --- |
| `stage_navigator` | 判断当前所处阶段，并推动路线朝剑、盾、宝石、10 层资源、红钥匙、boss 等 checkpoint 前进。 |
| `resource_economy` | 评估黄钥匙、蓝钥匙、红钥匙、宝石、血瓶和金币的长期价值，避免关键钥匙被过早浪费。 |
| `combat_threshold` | 估计战斗带来的血量损失，避免短期可行但长期死路的战斗选择。 |
| `boss_objective` | 在后期重点关注红门、10 层机关和最终骷髅队长路线。 |
| `expert_route_bias` | 使用已有高质量路线作为 curriculum/warm-start 信息，帮助 agentic planner 稳定跨过长程资源瓶颈。 |

这种设计的核心思想是将 MOTA 中不同类型的决策压力显式分解。例如，在前期，系统更关注剑和盾的获取；在中期，系统重点考虑宝石和钥匙经济；在后期，系统转向红钥匙、10 层机关和 boss 战。相比单一评分函数，多 agent 结构可以让不同风险被分别建模，再由 arbiter 做综合权衡。

在协同方式上，这些 agent 并不是彼此独立地给出最终动作，而是共享同一个由 simulator 生成的候选动作集合和候选 after-state。每个 agent 只负责从自己的角度产生一组评分和理由，例如资源 agent 更关注钥匙和血瓶，战斗 agent 更关注 HP 损失，阶段 agent 更关注 checkpoint 推进。随后 arbiter 将这些评分加权合并，并结合 beam search 保留若干个候选状态。系统还维护两类轻量 memory：一类是 recent trace memory，用来记录最近若干步选择了什么动作、到达了什么 stage；另一类是 visited-state memory，用来记录已经访问过的状态，减少上下楼循环和重复探索。这样，不同 agent 共享同一个状态与候选空间，通过统一的评分表和轨迹记忆完成协同决策。

在实验过程中，我们不是一次性得到完整路线，而是逐步定位并修复不同阶段的失败点。最初的本地 heuristic agent 能较快完成前期路线，例如拿到剑并接近盾阶段，但在 8 层、9 层附近经常因为钥匙和血量规划不足而中断。随后我们引入了更明确的 checkpoint 判断，把任务拆成“拿剑、拿盾、收集低层宝石、准备 10 层资源、拿红钥匙、打 boss”等阶段，使 agent 不再只追求短期血量或局部收益。

在第二阶段实验中，beam search 能够把路线推进到更深的位置，但仍然会在几个固定瓶颈上失败。一个典型问题是蓝钥匙链条：agent 有时会提前花掉蓝钥匙打开 8 层蓝门，却没有继续拿到后续蓝钥匙或 10 层资源，导致后面无法进入关键区域。另一个问题是血瓶时机：如果红钥匙之前没有补充足够血量，最终 boss 路线即使理论上可达，也会因为血量 margin 不足而失败。针对这些问题，我们增加了 resource-economy 和 combat-threshold 的权重，让 agent 在中后期更重视钥匙再生产、宝石收益和战斗损耗。

之后，我们将已有高质量路线作为 curriculum/warm-start 信息加入系统。这里并不是直接绕过模拟器执行路线，而是在合法候选动作中给与 demonstration 前缀一致的动作额外奖励。这样可以帮助 beam search 稳定跨过长程资源瓶颈，同时仍然保持每一步都由 simulator 检查是否合法。在调试中还发现，boss 阶段不能只打开红门和触发 10 层机关，还必须在机关触发后继续击败 MT10 `(6,1)` 的最终 skeleton captain。因此我们进一步强化了 `boss_objective` 对红门、机关和最终 boss flag 的判断，最终得到可 replay 的完整路线。

实验中，我们使用 expert-guided curriculum 作为 warm-start，使 agentic planner 能够稳定通过几个关键瓶颈。最终得到的路线为：

```text
artifacts/tmp/agentic_expertguide_solved_greedybeam_route.jsonl
```

该路线长度为 291 个宏动作，最终在 MT10 `(6,1)` 击败骷髅队长，最终状态为 HP 436、ATK 27、DEF 27，并触发成功标志 `10f战胜骷髅队长=true`。路线通过了 replay 验证和 constraint 验证，没有使用 forbidden shop/fly actions。

关键 checkpoint 如下：

| Checkpoint | Step | Action | State After |
| --- | ---: | --- | --- |
| 拿剑 | 19 | `go sword1 MT5:11,11` | HP 754, ATK 20, DEF 10 |
| 拿盾 | 75 | `go shield1 MT9:9,7` | HP 338, ATK 21, DEF 20 |
| 10 层蓝宝石 | 172 | `go blueGem MT10:2,6` | HP 176, ATK 26, DEF 27 |
| 10 层红宝石 | 235 | `go redGem MT10:10,6` | HP 715, ATK 27, DEF 27 |
| 红钥匙 | 248 | `go yellowKey MT8:9,1` | HP 497, redKey 1 |
| 打开红门 | 281 | `open redDoor MT10:6,9` | HP 1070 |
| 触发 10 层机关 | 282 | `fight skeletonCaptain MT10:6,4` | HP 995, `10f机关=true` |
| 击败最终 boss | 290 | `fight skeletonCaptain MT10:6,1` | HP 436, `10f战胜骷髅队长=true` |

从结果上看，agentic RL-style planning 能够把 MOTA 的复杂长程目标拆成更可控的 checkpoint，并通过多 agent 协同评分解决资源规划、战斗风险和阶段推进之间的冲突。相比前面不稳定的搜索尝试，该方法最终跑通了完整 10 层路线，也为后续进一步降低 expert guidance、加入 policy/value learning 或 RL fine-tuning 提供了稳定的实验基础。
