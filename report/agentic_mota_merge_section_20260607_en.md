# Agentic RL-Style Planning Section

In the previous experiments, we explored search-based and staged-planning methods for solving the first 10 floors of MOTA. These methods were able to reach several important intermediate stages, such as obtaining the sword and shield or completing part of the gem and key planning. However, the overall performance was still unstable, especially around mid- and late-game resource bottlenecks. These experiments show that MOTA is not a simple shortest-path problem, but a long-horizon sequential decision-making problem with strong resource constraints and misleading local optima. For example, preserving yellow keys in the early game affects access routes on floors 8 and 9, spending a blue key too early can block later access to floor-10 resources, and the order of attack gems, defense gems, and potions directly determines whether the final boss fight is feasible. Therefore, a single heuristic function or local greedy policy can easily lead to insufficient keys, insufficient HP, or locked critical resource paths in the later stages.

Based on these observations, we further explored an agentic RL-style planning method. Instead of giving the whole game to a single policy network that directly outputs actions, we decomposed the MOTA decision process into multiple agents with different responsibilities. These agents jointly evaluate the legal macro actions available in the current state. The environment is still provided by the MOTA simulator. At each step, the simulator first generates all legal macro actions, such as opening doors, fighting enemies, collecting gems, collecting keys, moving between floors, or triggering events. Then different agents score each candidate action from different perspectives, and a beam-search arbiter combines these scores to select the next action.

Specifically, we designed the following agents:

| Agent | Main Responsibility |
| --- | --- |
| `stage_navigator` | Identifies the current stage and pushes the route toward checkpoints such as sword, shield, gems, floor-10 resources, red key, and boss. |
| `resource_economy` | Evaluates the long-term value of yellow keys, blue keys, red keys, gems, potions, and gold, and avoids wasting critical keys too early. |
| `combat_threshold` | Estimates HP loss from combat and avoids fights that are locally feasible but lead to long-term dead ends. |
| `boss_objective` | Focuses on the red door, floor-10 mechanism, and final skeleton captain route in the late game. |
| `expert_route_bias` | Uses a high-quality route as curriculum and warm-start information to help the agentic planner pass long-horizon resource bottlenecks. |

The core idea is to explicitly decompose the different decision pressures in MOTA. In the early game, the system focuses more on obtaining the sword and shield. In the middle game, it pays more attention to gems and key economy. In the late game, it shifts toward the red key, floor-10 mechanism, and boss fight. Compared with a single scoring function, the multi-agent structure allows different risks to be modeled separately and then combined by the arbiter.

The agents collaborate through a shared candidate-action interface rather than making isolated final decisions. At each step, all agents observe the same simulator state, the same legal macro-action list, and the same candidate after-states. Each agent contributes scores and reasons from its own perspective: the resource agent focuses on keys and potions, the combat agent focuses on HP loss, and the stage agent focuses on checkpoint progress. The arbiter then merges these scores and uses beam search to preserve promising future states. The system also maintains two lightweight forms of memory. The first is recent trace memory, which records the latest selected actions, stages, and state summaries. The second is visited-state memory, which stores state keys to reduce repeated loops such as unnecessary floor transitions. In this way, the agents share information through the common candidate table, the trajectory trace, and the beam-search state archive.

During the experiments, the complete route was not obtained in one step. Instead, we gradually located and fixed failures at different stages. The initial local heuristic agents could quickly complete the early route, such as obtaining the sword and approaching the shield stage, but they often failed around floors 8 and 9 due to insufficient key and HP planning. We then introduced more explicit checkpoint detection and decomposed the task into stages such as obtaining the sword, obtaining the shield, collecting lower-floor gems, preparing floor-10 resources, collecting the red key, and fighting the boss. This prevented the agents from only pursuing short-term HP or local reward.

In the second stage of experiments, beam search was able to push the route deeper, but several fixed bottlenecks still appeared. One typical issue was the blue-key chain: the agent sometimes spent a blue key to open the floor-8 blue door too early, but failed to continue toward the later blue key or floor-10 resources, making the later route infeasible. Another issue was potion timing: if the agent did not collect enough HP before the red key, the final boss route could be theoretically reachable but still fail due to insufficient HP margin. To address these problems, we strengthened the resource-economy and combat-threshold components so that the agents placed more emphasis on key regeneration, gem value, and combat loss in the middle and late stages.

After that, we introduced a high-quality route as curriculum and warm-start information. This does not bypass the simulator or directly execute the demonstration route. Instead, among legal candidate actions, actions that match the demonstration prefix receive an additional score bonus. This helps beam search pass long-horizon resource bottlenecks while still ensuring that every action is checked by the simulator. During debugging, we also found that the boss stage cannot stop after opening the red door and triggering the floor-10 mechanism. The route must continue after the mechanism is triggered and defeat the final skeleton captain at MT10 `(6,1)`. Therefore, we further strengthened the `boss_objective` component for the red door, mechanism, and final boss flag, eventually obtaining a complete replayable route.

In the final experiment, we used expert-guided curriculum as a warm start so that the agentic planner could reliably pass the key bottlenecks. The final route is:

```text
artifacts/tmp/agentic_expertguide_solved_greedybeam_route.jsonl
```

This route contains 291 macro actions. It defeats the skeleton captain at MT10 `(6,1)`, ending with HP 436, ATK 27, and DEF 27, and triggers the success flag `10f战胜骷髅队长=true`. The route passes both replay validation and constraint validation, and it does not use forbidden shop or fly actions.

The key checkpoints are shown below:

| Checkpoint | Step | Action | State After |
| --- | ---: | --- | --- |
| Sword | 19 | `go sword1 MT5:11,11` | HP 754, ATK 20, DEF 10 |
| Shield | 75 | `go shield1 MT9:9,7` | HP 338, ATK 21, DEF 20 |
| Floor-10 blue gem | 172 | `go blueGem MT10:2,6` | HP 176, ATK 26, DEF 27 |
| Floor-10 red gem | 235 | `go redGem MT10:10,6` | HP 715, ATK 27, DEF 27 |
| Red key | 248 | `go yellowKey MT8:9,1` | HP 497, redKey 1 |
| Red door opened | 281 | `open redDoor MT10:6,9` | HP 1070 |
| Floor-10 mechanism | 282 | `fight skeletonCaptain MT10:6,4` | HP 995, `10f机关=true` |
| Final boss defeated | 290 | `fight skeletonCaptain MT10:6,1` | HP 436, `10f战胜骷髅队长=true` |

Overall, agentic RL-style planning can decompose the complex long-horizon objective in MOTA into more controllable checkpoints, and the multi-agent scoring mechanism helps resolve conflicts among resource planning, combat risk, and stage progression. Compared with the earlier unstable search attempts, this method successfully completes the full 10-floor route and provides a stable experimental foundation for reducing expert guidance or adding policy/value learning in later experiments.
