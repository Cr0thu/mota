# 4090 AlphaZero / Search Run Report - 2026-05-29

## Strict success status

- `hp403_warmstart/benchmark` strict replay: success.
- Route: `artifacts/runs/4090_hp403_alpha_success_stable_20260529_v2/eval_mcts64_boss_all_gems/boss_all_gems_az_route.jsonl`
- Current-code replay summary: `artifacts/runs/4090_hp403_warmstart_current_20260529/strict_replay_summary.json`
- Final state: MT10, HP 603, ATK 30, DEF 30, yellowKey 3, boss flag true.

## Pure no-hp403 line status

Pure no-hp403 has not solved the full task yet.

Best pure suffix found this run:

- Prefix: `artifacts/runs/4090_no_hp403_low_refill_alpha_v6_20260529/repair_keyed/skip_mt3_potion_room.jsonl`
- After 10F resources: HP 494, ATK 27, DEF 27, yellowKey 1, red-key margin +45.
- Focused red-key route: `artifacts/runs/4090_no_hp403_low_refill_alpha_v12_20260529/redkey_focused_refill/red_key_az_route.jsonl`
- After red key chamber resources: HP 295, ATK 27, DEF 27, yellowKey 2, redKey 1, boss margin -339.

Important diagnosis:

- The late red-key suffix is now structurally correct after adding focused `red_key` filtering and requiring chamber resources.
- The remaining failure is not 10F resource collection; it is early/mid-game HP loss before shield.
- Pure route gets shield much too late/low-HP. It reached shield previously around HP 229, while hp403 benchmark reaches shield at HP 418.
- Pure `mt4_redgem` route from the pure sword prefix reaches HP 590/ATK21/DEF10, but yellowKey becomes 0, which blocks clean shield routing.
- MT2 blue potions remain on the map but are unreachable from the simplified post-thief topology, so they cannot be counted as recoverable HP in the current environment.

## Code changes made this run

- `scripts/repair_route.py`
  - Added `--min-yellow-keys` soft quality target.
  - Increased route-quality weight for red-key margin and yellow-key preservation.

- `scripts/run_az_mcts_stage.py`
  - Added `--hp-aware-success-value`, `--hp-success-base`, and `--hp-success-scale` passthrough to MCTS.

- `src/mota_solver/az_mcts.py`
  - Added `red_key`, `boss_ready`, `trap`, and `boss` to filtered resource stages.
  - Added focused red-key action tokens.
  - Added MT8 red-key chamber resources to late refill tokens.
  - Added MT2 blue potions to low-floor refill targets.
  - Changed late boss-refill descent condition to use `boss_route_margin < 0`, not only HP < 300.

- `src/mota_env/rewards.py`
  - Changed `red_key_taken` stage completion to require red key plus MT8 red-key chamber potions and yellow keys.

## What worked

- Route repair improved the pure 10F-resource endpoint from HP452/yellowKey0 to HP533/yellowKey0.
- Segment deletion found a better pure candidate: HP494/yellowKey1 after 10F resources.
- Focused red-key MCTS solved the red-key chamber from that candidate in 14 macros.

## What did not work

- Direct shield MCTS from reset failed and loops around MT1/MT2.
- Shield MCTS from pure sword prefix failed because it wastes yellow keys and does not reproduce the necessary 4F/5F/6F key cycle.
- hp403 reward leaf alone was too slow for shield search at 1024 simulations and was killed after 4.5 minutes without output.

## Next recommended experiment

Use `hp403_warmstart` explicitly instead of pretending it is pure RL:

1. Extract hp403 route prefixes for `sword`, `mt4_redgem`, `shield`, `mid_gems`, `mt8_gems`, `mt10_resources`, `red_key`, `boss`.
2. Train the graph policy/value model with DQfD-style supervised policy loss on those prefixes plus MCTS-generated alternatives.
3. Evaluate whether the learned policy can reproduce shield and then improve route HP by search.
4. Keep pure line separate: use Go-Explore archive with explicit cell keys for yellow-key buffer and shield HP, because pure MCTS/planner currently does not preserve key cycles.
