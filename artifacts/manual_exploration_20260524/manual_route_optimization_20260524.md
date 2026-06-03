# Manual Route Optimization - 2026-05-24

## Result

Best optimized route:

`artifacts/manual_exploration_20260524/manual_success_no_shop_true_10f_trap_optimized_hp403.jsonl`

Compared with the previous successful route:

- Final HP: `6 -> 403`
- Previous best manual routes: `233 -> 280 -> 299 -> 315 -> 375 -> 403`
- Route length: `270 -> 317` macro actions
- Final stats unchanged: `ATK=27`, `DEF=27`
- Final flag unchanged: `10f战胜骷髅队长=true`
- No 4F shop, no fly
- Legal one-time merchants still used: MT7 yellow-key merchant and MT6 blue-key merchant

Validation:

```bash
PYTHONPATH=src python scripts/replay_route.py \
  --route artifacts/manual_exploration_20260524/manual_success_no_shop_true_10f_trap_optimized_hp403.jsonl

PYTHONPATH=src python scripts/validate_route_constraints.py \
  artifacts/manual_exploration_20260524/manual_success_no_shop_true_10f_trap_optimized_hp403.jsonl
```

## Accepted Optimizations

1. Delay 4F lower-left key branch until after sword.
   - Old: fight `bat MT4:4,9` at `ATK=10 DEF=10`, damage `112`.
   - New: take `sword1` first, then fight it at `ATK=20 DEF=10`, damage `56`.
   - Also reduces the nearby green slime damage from `24` to `8`.

2. Delay the 5F top-right key sub-branch until after shield.
   - Keep only the 5F bat gate needed to reach the upper-left route.
   - Delay `redSlime MT5:11,2`, `yellowDoor MT5:10,1`, `greenSlime MT5:9,2`, and `yellowKey MT5:8,4`.
   - These become zero-damage or near-zero-damage after shield.

3. Delay 7F blue-priest potion branch until after shield.
   - Keep `redSlime MT7:7,9` before the MT7 merchant because its money is needed.
   - Delay `bluePriest MT7:7,10` and `bluePotion MT7:7,11`.

4. Take 9F blue-gem branch immediately after shield.
   - This gives `DEF+1` before the delayed 7F blue priest and later low-floor fights.

5. Reorder low-floor gems.
   - Best tested order: 4F right red gem, 3F left blue gem, 3F right branch, 3F left red gem, 4F left branch.
   - This makes several blue-priest/skeleton fights happen after one more attack or defense point.

6. Reorder 1F.
   - Best tested order: 1F right-side red/blue gems first, then top-right potion, then left-side skeleton-soldier branch.
   - This avoids fighting the 1F skeleton soldier before the 1F gems.

7. Reorder 5F/6F.
   - Best tested order: 5F blue gem branch, 6F lower blue-gem branch, then 6F upper merchant branch.

8. Reorder upper floors.
   - Best tested order: 7F key branch, 8F left potion branch, 9F lower branch, 10F resource branch, then 9F top key branch.

## Round 2 Search

After the `hp233` route, a second optimization pass treated the non-stair interactions as an objective sequence and replayed candidates with automatic floor navigation. The pass randomly moved small objective blocks, then kept only legal routes that still triggered `flag:10f战胜骷髅队长=true`.

- Candidates tried: `5962`
- Solved candidates: `279`
- Best final HP found: `280`
- Best route saved as: `manual_success_no_shop_true_10f_trap_optimized_hp280.jsonl`
- A follow-up prune pass found no removable action block that preserved `HP >= 280`, so the route remains `280` macro actions.

The main gain is from delaying several damaging fights until after more attack/defense resources are collected. For example, the MT7 blue-priest and bat fights happen later at lower damage. This route is longer than `hp233`, but it has a better final HP margin.

## Round 3 Search

The third pass was moved into a reusable script:

```bash
PYTHONPATH=src python scripts/optimize_manual_route.py \
  --base-route artifacts/manual_exploration_20260524/manual_success_no_shop_true_10f_trap_optimized_hp280.jsonl \
  --prefix manual_success_no_shop_true_10f_trap_round3_seed14 \
  --iterations 3000 \
  --seed 2026052414 \
  --max-segment 20 \
  --bias \
  --save-each-improvement
```

The script extracts non-stair interaction objectives, automatically inserts legal stair navigation during replay, and repeatedly mutates the objective order. Mutations include small segment moves, segment swaps, delaying fights, and pulling resources such as gems, keys, sword, and shield earlier when legal.

Round 3 results:

- Smoke run: `280 -> 283`
- First parallel generation: best `294`
- Second parallel generation: best `297`
- Third short generation: best `299`
- Final prune: removed one redundant explicit `go event MT10:6,5` objective; the path still triggers the MT10 trap naturally.
- Stable route: `manual_success_no_shop_true_10f_trap_optimized_hp299.jsonl`

Validation for the final route:

- `solved=true`
- Final HP: `299`
- Route length: `301` macro actions
- Final stats: `ATK=27`, `DEF=27`
- No 4F shop and no fly
- `pytest -q tests/test_visualizer_route_player.py tests/test_simulator.py tests/test_visualizer_environment.py`: `16 passed`

## Round 4 Search

Round 4 continued from `hp299` with the same objective-order optimizer and added one explicit extra-target insertion scan.

Search results:

- From `hp299`: four parallel runs of `2500` iterations; best route improved to `hp302`.
- From `hp302`: three parallel runs of `1800` iterations; best route improved to `hp315`.
- From `hp315`: three shorter runs of `1200` iterations; no HP improvement, but the best same-HP route shortened to `313` macro actions.
- Extra-target insertion scan: tried `3213` insertions of route targets not present in the current objective list; `2341` remained solvable, but the best was only `hp302`. This suggests the `hp315` gain is mainly order-dependent, not caused by an omitted extra resource branch.
- Final prune: no single objective could be removed while preserving `HP >= 315`.

Stable route:

`manual_success_no_shop_true_10f_trap_optimized_hp315.jsonl`

Validation for the final route:

- `solved=true`
- Final HP: `315`
- Route length: `313` macro actions
- Final stats: `ATK=27`, `DEF=27`
- No 4F shop and no fly
- `pytest -q tests/test_visualizer_route_player.py tests/test_simulator.py tests/test_visualizer_environment.py`: `16 passed`

## Round 5 Shield And Bat Threshold Check

Round 5 focused on the issue that the route still looked too slow to prioritize the shield and did not obviously respect bat thresholds.

Findings:

- Bat stats are `HP=35`, `ATK=38`, `DEF=3`.
- With hero `ATK=20`, bats require 3 hero attacks. At `DEF=10`, each early bat costs `56` HP.
- At hero `ATK=21`, bats drop to 2 hero attacks. With `DEF=20`, each bat costs only `18` HP.
- Therefore one pre-bat red gem would be extremely valuable, but a single-objective move scan found that no existing red/blue gem objective can be moved before the first four bats while preserving route legality.
- The current unavoidable-looking early bat block remains four bats before shield, total `224` HP damage.
- Shield can be moved earlier within the existing objective order: from objective index `65` to `63`.

The final `hp315` route now uses the shield-priority ordering:

- Shield row in route: `72`
- Shield state: `HP=374`, `ATK=20`, `DEF 10 -> 20`
- After shield, the 9F red gem is collected before the next bat, so the next bat is fought at `ATK=21`, `DEF=20`, damage `18`.
- Final HP remains `315`, route length remains `313`, primitive steps improve from `3251` to `3245`.

I also ran three standalone staged searches stopping at shield. They did not beat the current route:

- Seed `2026052471`: shield HP `184`
- Seed `2026052472`: shield HP `199`
- Seed `2026052473`: shield HP `178`

These staged results confirm that the current staged reward is overvaluing pre-shield gems and money relative to shield-arrival HP. For future RL/search work, the shield stage reward should penalize pre-shield bat/blue-priest/skeleton damage more strongly and should treat `ATK=21` before bats as a high-value threshold.

## Round 6 Early 4F Red Gem And Key Circulation

Round 6 implemented the key insight that the 4F red gem should be pulled before the early 5F bats, while doors should not be opened unless the route actually needs the target behind them.

Targeted change:

- After sword, the route still must fight the first 4F bat to reach the 4F key pocket.
- Then it collects the 4F yellow keys, opens `MT4:8,8`, fights `bluePriest MT4:8,9`, and takes `redGem MT4:7,10`.
- This reaches `ATK=21` before the next three early bats.
- Those three bats drop from `56` damage each to `28` damage each.
- The early 4F blue priest is more expensive than fighting it later, but the bat threshold gain dominates.

A direct early-4F-red-gem route reached `HP=355`. It initially broke later at `MT9:4,5` because the earlier 4F door spent one yellow key that the old route implicitly relied on. The fix was to make 9F central resources explicit before that door:

- `go redGem MT9:6,5`, or equivalently the central yellow-key path that passes through it, restores the key/attack flow before opening the left 9F door.

After local search from the repaired route:

- Best route: `manual_success_no_shop_true_10f_trap_optimized_hp375.jsonl`
- Final HP: `375`
- Route length: `314` macro actions
- Final stats: `ATK=27`, `DEF=27`
- No 4F shop and no fly
- Single-objective prune removed `0` actions; all explicit targets are currently needed.
- `pytest -q tests/test_visualizer_route_player.py tests/test_simulator.py tests/test_visualizer_environment.py`: `16 passed`

## Round 7 Delaying The 4F Left Bat

The first `hp375` route still had a bad ordering: it fought `bat MT4:4,9` before taking the 4F red gem. That was unnecessary.

Clarification:

- `open yellowDoor MT4:4,8` is still before sword because it is part of the path from 4F right side to `upFloor MT4:1,11`, which is needed for the 5F sword route.
- But `fight bat MT4:4,9` is not needed before the 4F red gem.

The repaired route now uses a different yellow-key circulation:

- After sword, fight `bat MT5:6,4`.
- Pick `yellowKey MT5:6,2`.
- Use that key to open the 4F red-gem route.
- Fight `bluePriest MT4:8,9`.
- Take `redGem MT4:7,10`, reaching `ATK=21`.
- Only then fight `bat MT4:4,9`.

This changes the 4F left bat from:

- Before: `ATK=20`, damage `56`
- After: `ATK=21`, damage `28`

The final HP remains `375`; route length becomes `315` macro actions. I chose this version as the current stable `hp375` route because it obeys the intended resource logic: do not fight the 4F left bat before the 4F red gem. A short three-seed local search from this repaired structure did not find a higher-HP route.

Validation:

- `solved=true`
- Final HP: `375`
- Final stats: `ATK=27`, `DEF=27`
- No 4F shop and no fly
- `pytest -q tests/test_visualizer_route_player.py tests/test_simulator.py tests/test_visualizer_environment.py`: `16 passed`

## Round 8 Delaying The 4F Left Door

Round 7 still contained a key-circulation mistake: it opened `yellowDoor MT4:4,8` before the sword even though the route did not fight the 4F left bat until later. That prematurely tied up one yellow key.

Corrected ordering:

- Do not open `yellowDoor MT4:4,8` on the first 4F pass.
- Keep that yellow key while going to 5F and taking `sword1`.
- After sword, go directly back to 4F and open the right-side `yellowDoor MT4:8,8`.
- Fight `bluePriest MT4:8,9`.
- Take `redGem MT4:7,10`, reaching `ATK=21`.
- Then fight `bat MT5:6,4`, collect `yellowKey MT5:6,2`, and only then open the delayed 4F left door.

This matters because `bat MT5:6,4` crosses the bat turn threshold:

- Before: fought at `ATK=20`, damage `56`.
- After: fought at `ATK=21`, damage `28`.

Final route:

- Best route: `manual_success_no_shop_true_10f_trap_optimized_hp403.jsonl`
- Final HP: `403`
- Route length: `317` macro actions
- Final stats: `ATK=27`, `DEF=27`
- No 4F shop and no fly
- A 1000-candidate local reorder pass from `hp403` found no higher-HP route before being stopped.

Validation:

- `solved=true`
- `flag:10f战胜骷髅队长=true`
- `PYTHONPATH=src python scripts/replay_route.py --route artifacts/manual_exploration_20260524/manual_success_no_shop_true_10f_trap_optimized_hp403.jsonl`
- `PYTHONPATH=src python scripts/validate_route_constraints.py artifacts/manual_exploration_20260524/manual_success_no_shop_true_10f_trap_optimized_hp403.jsonl`
- `PYTHONPATH=src pytest -q tests/test_visualizer_route_player.py tests/test_simulator.py tests/test_visualizer_environment.py`: `16 passed`

## Round 9 Delaying The 7F Skeleton Within The Floor

The `hp403` route still fought `skeleton MT7:9,5` immediately after entering 7F. That is not structurally required.

Corrected 7F order:

- Open `yellowDoor MT7:11,7`.
- Fight `bluePriest MT7:4,6`.
- Open `blueDoor MT7:5,5`.
- Fight `redSlime MT7:5,3`.
- Then fight `skeleton MT7:9,5`.
- Take the right-top key/potion chain via `yellowKey MT7:9,1`.
- Open `yellowDoor MT7:7,7`.
- Buy 5 yellow keys from the 7F merchant.

This keeps final HP unchanged at `403`, because no attack/defense threshold changes before the skeleton. It is still useful because it removes the false dependency that the 7F skeleton must be the first action on the floor.

I also tested delaying this skeleton past the 7F merchant or past shield. That failed or became worse under the current resource line:

- Without the skeleton/key branch, the route reaches the 7F merchant with only `47` money.
- The merchant requires `50` money.
- Replacing the skeleton money with extra early 6F fights is possible, but costs more HP than it saves from delaying the skeleton.

Validation after updating the main `hp403` route:

- `solved=true`
- Final HP: `403`
- Final stats: `ATK=27`, `DEF=27`
- `flag:10f战胜骷髅队长=true`
- `PYTHONPATH=src python scripts/replay_route.py --route artifacts/manual_exploration_20260524/manual_success_no_shop_true_10f_trap_optimized_hp403.jsonl`
- `PYTHONPATH=src python scripts/validate_route_constraints.py artifacts/manual_exploration_20260524/manual_success_no_shop_true_10f_trap_optimized_hp403.jsonl`
- `PYTHONPATH=src pytest -q tests/test_visualizer_route_player.py tests/test_simulator.py tests/test_visualizer_environment.py`: `16 passed`

## Cleanup

Intermediate candidate routes from these optimization passes were moved to:

`artifacts/manual_exploration_20260524/obsolete_expert_routes/optimization_probes_20260524/`

`artifacts/manual_exploration_20260524/obsolete_expert_routes/optimization_probes_20260524_round2/`

`artifacts/manual_exploration_20260524/obsolete_expert_routes/optimization_probes_20260524_round3/`

`artifacts/manual_exploration_20260524/obsolete_expert_routes/optimization_probes_20260524_round4/`

`artifacts/manual_exploration_20260524/obsolete_expert_routes/optimization_probes_20260524_round5/`

`artifacts/manual_exploration_20260524/obsolete_expert_routes/optimization_probes_20260524_round6/`

`artifacts/manual_exploration_20260524/obsolete_expert_routes/optimization_probes_20260524_round7/`

`artifacts/manual_exploration_20260524/obsolete_expert_routes/optimization_probes_20260524_round8/`

`artifacts/manual_exploration_20260524/obsolete_expert_routes/optimization_probes_20260524_round9/`

The visualizer ignores those directories. The main route dropdown should now show only the original success route plus the stable optimized routes `hp233`, `hp280`, `hp299`, `hp315`, `hp375`, and `hp403`, without temporary probes.
