# Manual Route Success Report - 2026-05-24

## Result

Superseded by the optimized route in:

`artifacts/manual_exploration_20260524/manual_route_optimization_20260524.md`

Successful route:

`artifacts/manual_exploration_20260524/manual_success_no_shop_true_10f_trap.jsonl`

Replay result:

- `solved=true`
- final flag: `10f战胜骷髅队长=true`
- final HP: `6`
- final stats: `ATK=27`, `DEF=27`
- route length: `270` macro actions
- no 4F shop
- no fly
- 10F red door and trap are used

Validation commands:

```bash
PYTHONPATH=src python scripts/replay_route.py \
  --route artifacts/manual_exploration_20260524/manual_success_no_shop_true_10f_trap.jsonl

PYTHONPATH=src python scripts/validate_route_constraints.py \
  artifacts/manual_exploration_20260524/manual_success_no_shop_true_10f_trap.jsonl
```

## Key Fixes

The failed route was not fundamentally impossible. The main issue was early HP waste before and around sword acquisition.

1. 5F bat was moved after sword.
   - Old: fight `bat MT5:6,4` before sword, damage `112`.
   - New: take `sword1` first, then fight the same bat, damage `56`.

2. 5F top-right key branch was moved after sword.
   - Old route fought `redSlime MT5:11,2` and `greenSlime MT5:9,2` before sword.
   - New route opens the direct sword path first, takes sword, then comes back for the key branch.
   - This keeps the needed yellow keys while reducing early damage.

3. Shield route keeps the delayed 7F red gem idea, but only as route optimization.
   - It does not move 1F gems before shield.
   - It avoids spending HP on the 7F red-gem side branch before shield, then collects it after shield with lower damage.

4. After shield, 8F resources are collected before the 1F/5F cleanup.
   - This makes later 1F and 5F fights cheaper by improving attack/defense earlier.

## Important Checkpoints

- Sword: `HP=618`, `ATK=20`, `DEF=10`
- Shield: `HP=358`, `ATK=21`, `DEF=21`
- Before 10F red door: `HP=640`, `ATK=27`, `DEF=27`, `redKey=1`
- After 10F trap guards and captain dialogue: `HP=310`
- After skeleton captain: `HP=6`, `10f战胜骷髅队长=true`

## Notes

The route uses the legal MT7 and MT6 one-time merchants:

- `buy 5 yellowKey MT7:6,1`
- `buy blueKey MT6:8,4`

These are NPC merchant events, not the disabled 4F shop. The route constraint check forbids 4F shop and fly and passes.
