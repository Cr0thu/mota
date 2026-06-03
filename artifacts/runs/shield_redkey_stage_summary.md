# Shield -> Red Key Staged Search Summary

Date: 2026-05-13

## Code Changes

- External stage order is now `shield -> red_key -> trap -> boss`.
- `shield` is a coarse milestone. Before sword is collected, the solver scores states with the hidden `sword` subgoal.
- `red_key` is a coarse milestone. Before the hero can survive the two MT8 yellow guards plus an 80 HP path buffer, the solver scores states with the hidden `guard_ready` subgoal.
- `guard_ready` now explicitly rewards key low-floor resources, MT8 bottom resources, yellow-guard attack breakpoints, and low-damage clearing fights.
- Red-key navigation now points low-floor states upward toward MT8, and avoids generic stair oscillation on MT8.

## Current Best Runs

| Run | Result | Best Red-Key Stage State |
| --- | --- | --- |
| `route_first10_staged_shield_redkey_10k_v3.jsonl` | shield solved, red key failed | HP 296, ATK 23, DEF 21 |
| `route_first10_staged_shield_redkey_20k_v3.jsonl` | shield solved, red key failed | HP 462, ATK 24, DEF 23 |
| `route_first10_staged_shield_redkey_20k_v4.jsonl` | shield solved, red key failed | HP 615, ATK 24, DEF 23 |
| `route_first10_staged_shield_redkey_100k_v3.jsonl` | shield solved, red key failed | HP 544, ATK 26, DEF 26 |

For `100k_v3`, yellow-guard damage is 264 per guard. With the current 80 HP path buffer, readiness is:

```text
544 - 2 * 264 - 80 = -64
```

So the run is close but still short of the red-key chamber requirement.

## Interpretation

The first milestone is now stable: the staged solver reaches the 9F shield quickly. The remaining bottleneck is not the shield route, but the red-key preparation route: the agent must collect enough low-floor HP/stat resources, then return to MT8 without wasting HP on unnecessary fights.

This is still CPU search work, not GPU training work. A 4090 becomes useful for training the attention value/ranking model or running many reward-weight trials in parallel, but it is not required for the next single-search iteration.

## Next Implementation Step

Split the hidden `guard_ready` subgoal into two scored subgoals:

1. `guard_stats_ready`: reach roughly ATK 26+ and DEF 25+.
2. `guard_route_ready`: after stats are ready, collect enough HP/key resources and navigate to MT8 with positive yellow-guard margin.

This should reduce the current conflict where the search sometimes prefers HP-heavy routes and sometimes ATK-heavy routes, but does not consistently combine both.
