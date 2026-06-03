# AlphaZero-style 训练报告 2026-05-29

## 结论

本轮确认了一条 **strict replay 成功** 的 AlphaZero-style warm-start 路线，并修复了“击败队长后无法继续拿 10F 顶部资源”的终止条件问题。

- 队长路线文件：`artifacts/runs/4090_hp403_alpha_success_stable_20260529_v2/eval_mcts64_deep/boss_az_route.jsonl`
- 全宝石路线文件：`artifacts/runs/4090_hp403_alpha_success_stable_20260529_v2/eval_mcts64_deep/boss_az_route_with_all_gems.jsonl`
- `boss_all_gems` MCTS 文件：`artifacts/runs/4090_hp403_alpha_success_stable_20260529_v2/eval_mcts64_boss_all_gems/boss_all_gems_az_route.jsonl`
- 队长 replay 结果：317 个 macro transition，最终 HP=403、ATK=27、DEF=27
- 全宝石 replay 结果：319 个 macro transition，最终 HP=1003、ATK=30、DEF=30，剩余攻防宝石为 0
- `boss_all_gems` MCTS replay 结果：319 个 macro transition，最终 HP=603、ATK=30、DEF=30，剩余攻防宝石为 0
- 成功 flag：`10f战胜骷髅队长=true`
- 约束检查：无 4F 商店、无飞行器；MT6/MT7 商人按规则使用

这条路线使用 `hp403` 作为 warm-start/benchmark，不属于 no-demonstration pure RL 结果。no-hp403 线目前最远稳定到 `mt10_yellow_ready`：可到 10F 并拿左侧蓝宝石，最好状态约 HP=93、ATK=25、DEF=27，但后续缺黄钥匙和可回补血量，尚未 strict 通关。

## 怎么用的 AlphaZero 那一套

我们没有照搬围棋里的二人零和 AlphaZero，而是改成适合《魔塔》的单智能体确定性规划版本：

1. 状态不是棋盘，而是资源交互图：英雄、怪物、门、钥匙、宝石、血瓶、楼梯、事件都编码成 graph tokens。
2. 动作不是上下左右，而是 macro action：选择一个当前可执行的交互节点，例如打某个怪、开某个门、拿某个资源。
3. 网络是 policy-value model：输入 graph state，输出每个节点的 policy prior 和当前状态 value。
4. MCTS 使用 PUCT：用网络 prior 选择扩展节点，用 value/reward potential 评估叶子节点。
5. 自博弈改成自举路线生成：MCTS 每一步选 macro action，形成路线，再用 MCTS 访问分布作为 policy target、阶段成功/最终结果作为 value target 训练网络。
6. 加了阶段目标：拿剑、拿盾、拿宝石、拿红钥匙、触发 10F 机关、打骷髅队长，避免超长 horizon 直接崩掉。

关键代码入口：

- MCTS：`src/mota_solver/az_mcts.py`
- AlphaZero-style 训练：`scripts/train_alpha_mota_stage.py`
- 图状态模型：`src/mota_rl/graph_policy_value_model.py`

## 本轮新改动

- 新增 `SimulatorConfig.stop_on_boss`，默认仍为 `True`，保持旧实验“击败队长即终止”的语义；当目标为 `boss_all_gems` 或显式传入 `--continue-after-boss` 时，允许队长后继续行动。
- 新增阶段目标 `boss_all_gems`：必须满足 `10f战胜骷髅队长=true` 且 `remaining_attack_defense_gems=0`。
- 修正路线 replay：`scripts/replay_route.py --continue-after-boss` 可以严格验证队长后继续拿资源的路线。
- 给 `mt10_yellow_ready` 增加缺蓝钥匙时回 6F 买蓝钥匙的约束，修复了 8F/9F 楼梯循环。
- 给 `mt10_resources` 增加从 10F 返回下层回补的动作过滤，避免 10F 无资源时直接停止。
- 尝试在 `pre_shield_gems` / `shield` 阶段延后 7F 底部蝙蝠和黄钥匙，目标是把后期钥匙/血量留给 10F 回补。

## 当前实验状态

成功线：

- `hp403_warmstart` AlphaZero-style 队长路线 strict 成功，最终 HP=403、ATK=27、DEF=27。
- `hp403_warmstart` 队长后资源 sweep strict 成功，最终 HP=1003、ATK=30、DEF=30，10F 顶部 6 个宝石全部收完。
- `hp403_warmstart` 远程 `boss_all_gems` MCTS strict 成功，最终 HP=603、ATK=30、DEF=30，证明不是本地手工追加造成的假成功。

no-hp403 线：

- `4090_no_hp403_mt10_yellow_direct_from_mt8d_20260529_v3` 成功到 10F 左侧蓝宝石，最终 HP=93、ATK=25、DEF=27。
- `mt10_resources_from_direct_v3` 失败，原因是进入 10F 后黄钥匙为 0，低层可用回补已经被早期路线消耗。
- `delayed_refill_preshield` 尝试把 7F 底部资源后移，但会导致早期 `pre_shield_gems` HP/ATK/DEF 同时达标困难。

## 路线关键节点

成功路线中的关键状态：

- 第 19 步：拿剑，HP=754、ATK=20、DEF=10
- 第 77 步：拿盾，HP=418、ATK=21、DEF=20
- 第 218 步：10F 左侧蓝宝石后，HP=338、ATK=26、DEF=27
- 第 256 步：10F 右侧红宝石后，HP=836、ATK=27、DEF=27
- 第 284 步：8F 蓝血瓶和红钥匙后，HP=903、红钥匙=1
- 第 316 步：击败 10F 骷髅队长，HP=403
- 第 318 步：收完 10F 顶部 6 个攻防宝石，HP=1003、ATK=30、DEF=30

## 下一步建议

不要继续盲目增加 PPO/DQN 时长。当前瓶颈是阶段资源保留，不是 GPU 时长。

建议分两条线：

1. `hp403_warmstart`：把成功路线作为示范，训练 policy/value/reward，先做稳定可复现的课程项目结果。
2. `pure_search_rl`：继续做科研线，重点改 Go-Explore archive 和阶段资源保留，不把 hp403 放进训练标签。

对 no-hp403，下一步应单独优化 `pre_shield_gems -> shield -> mt8_gems` 的资源保留，不要从 10F 末端硬补。核心指标是：到 10F 左侧蓝宝石后仍保留至少 1-2 把黄钥匙，且 7F/1F/8F 至少有一个大血瓶尚未消耗。
