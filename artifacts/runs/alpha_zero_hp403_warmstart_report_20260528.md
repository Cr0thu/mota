# AlphaZero-style hp403 Warm-start Report

Date: 2026-05-28

## What changed

- Added route-supervised policy/value warm-start for the AlphaZero-style graph policy:
  - `scripts/train_alpha_from_route.py`
- Updated MCTS action selection:
  - root action tie-breaks can include value/prior;
  - `run_az_mcts_stage.py` now avoids revisiting already-seen states during rollout.
- Added dense single-player value-target options in `train_alpha_mota_stage.py`:
  - `--value-target-mode final|root|mixed`
  - `--final-action-value-weight`
  - `--final-action-prior-weight`

## Key result

Using `hp403` only as a warm-start route, the AlphaZero-style pipeline now reaches the 10F skeleton captain and passes strict replay.

Best verified route:

- Remote: `artifacts/runs/4090_alpha_hp403_warmstart_v2_boss_20260528/eval_mcts16_weak_reward/boss_az_route.jsonl`
- Local copy: `artifacts/runs/4090_alpha_hp403_warmstart_v2_boss_20260528/eval_mcts16_weak_reward/boss_az_route.jsonl`

Strict replay result:

```json
{
  "steps": 317,
  "solved": true,
  "final": {
    "floor": "MT10",
    "x": 6,
    "y": 1,
    "hp": 403,
    "atk": 27,
    "def": 27,
    "money": 262,
    "keys": {"yellowKey": 0, "blueKey": 0, "redKey": 0},
    "flags": {"10f机关": true, "10f战胜骷髅队长": true}
  }
}
```

Constraint check:

- no 4F shop
- no fly
- no route constraint violations

## Experiments

### Sword

- Warm-start samples: 20
- Policy-only MCTS-lite reached sword:
  - route length: 43 macro steps
  - final HP: 6
- Earlier full reward-leaf MCTS failed because reward leaf overrode the learned policy.

### Shield

- Warm-start samples: 78
- Overfit policy check:
  - top1: 97.4%
  - top3: 100%
  - expert action mean probability: 0.972
- 16-sim MCTS without reward reached shield.
- 16-sim MCTS with weak reward leaf also reached shield.
- Final shield state:
  - HP=418, ATK=21, DEF=20

### Red key

- Warm-start samples: 283
- Policy-only reached red key.
- 16-sim weak-reward MCTS reached red key.
- Final red-key state:
  - HP=653, ATK=27, DEF=27, redKey=1

### Boss

- Warm-start samples: 317
- Policy-only reached boss.
- 16-sim weak-reward MCTS reached boss.
- Final boss state:
  - HP=403, ATK=27, DEF=27
  - `flag:10f战胜骷髅队长=true`

## Interpretation

The useful AlphaZero version for this project is not raw MCTS from scratch. It is:

1. Train a graph policy/value network from a seed route or planner archive.
2. Use policy-guided MCTS with a small number of simulations.
3. Keep reward leaf weak; it should be a tie-breaker, not the main driver.
4. Use Go-Explore / staged planner to generate route variants, then repeat expert iteration.

Strong reward leaf without a trained policy caused early detours and stair loops. The current winning configuration keeps the learned policy dominant and uses MCTS as a controlled policy-improvement layer.
