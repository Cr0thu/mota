# Agentic RL-Style Planning for 10F MOTA

Date: 2026-06-07

## 1. Overview

This project builds an agentic RL-style planning system for the first 10 floors of MOTA. The task is a long-horizon sequential decision problem with sparse terminal reward, strict resource constraints, and many irreversible decisions. A successful policy must coordinate HP, keys, attack/defense gems, route access, red-key timing, and the final 10F boss sequence.

Our system solves the 10F benchmark with a legal replayable route:

- Route file: `artifacts/tmp/agentic_expertguide_solved_greedybeam_route.jsonl`
- Run summary: `artifacts/runs/agentic_expertguide_solved_greedybeam/summary.json`
- Trace file: `artifacts/runs/agentic_expertguide_solved_greedybeam/trace.jsonl`
- Route length: 291 macro actions
- Final state: MT10 `(6,1)`, HP 436, ATK 27, DEF 27
- Success flag: `10f战胜骷髅队长 = true`

The route passes replay validation and constraint validation. It does not use forbidden shop/fly actions.

## 2. Problem Characteristics

MOTA is difficult for simple greedy search because local rewards are misleading. A move that improves immediate HP or key count can block a later door, while a move that temporarily loses HP may be required for future attack/defense gains.

The first-10-floor task has several important constraints:

- The sword and shield must be acquired in a feasible order.
- Yellow and blue keys must be preserved for specific gates.
- Attack/defense gems strongly affect later combat margins.
- MT10 requires prepared HP and key buffers.
- Red key timing matters because taking it too early or too late changes the boss route.
- The final boss sequence requires triggering the 10F mechanism and then defeating the skeleton captain.

Because of these dependencies, we formulate the problem as checkpoint-driven agentic planning over legal macro actions.

## 3. System Design

The environment is represented by the existing MOTA simulator. At every step, the simulator generates legal macro actions such as moving to an item, opening a door, fighting an enemy, entering a floor, or triggering an event. The planner never directly edits the game state; all transitions are applied through the simulator.

The decision pipeline is:

1. Read current simulator state.
2. Generate legal macro actions.
3. Convert actions into candidates with predicted after-states.
4. Score each candidate using multiple specialized agents.
5. Combine scores in a beam-search arbiter.
6. Execute the selected macro action.
7. Save the transition into a replayable JSONL route.

The successful command was:

```powershell
conda run -n dl python scripts\run_agentic_rl.py `
  --backend heuristic `
  --beam-width 2 `
  --candidate-top-k 1 `
  --episodes 1 `
  --max-steps 340 `
  --expert-weight 2.5 `
  --out-dir artifacts\runs\agentic_expertguide_solved_greedybeam `
  --route-out artifacts\tmp\agentic_expertguide_solved_greedybeam_route.jsonl
```

## 4. Agent Roles

The planner uses several local agents. Each agent evaluates the same legal candidates from a different perspective.

| Agent | Function |
| --- | --- |
| `stage_navigator` | Tracks the current milestone and prefers actions that advance the route stage. |
| `resource_economy` | Evaluates keys, gems, potions, money, and door costs. |
| `combat_threshold` | Penalizes unsafe fights and evaluates HP loss against future progress. |
| `boss_objective` | Prioritizes red door, 10F mechanism, boss access, and final skeleton captain victory. |
| `expert_route_bias` | Provides curriculum-style warm-start guidance from a successful demonstration trajectory. |

The agent scores are combined into a candidate score. The beam arbiter keeps promising states and avoids immediately committing to a single fragile path.

## 5. Stage And Checkpoint Design

The route is organized around explicit strategic checkpoints:

1. Sword acquisition.
2. Shield acquisition.
3. Lower-floor attack/defense gem collection.
4. MT10 access preparation.
5. MT10 resource collection.
6. Pre-red-key HP/resource buffer.
7. Red key acquisition.
8. MT10 red door opening.
9. 10F mechanism activation.
10. Final skeleton captain defeat.

This checkpoint structure improves long-horizon credit assignment. Instead of treating all actions equally, the planner understands that different resources matter at different stages. For example, a blue key is highly valuable before MT10 access, while HP and red-key timing dominate the final boss sequence.

## 6. Curriculum Guidance

The system uses a successful route as curriculum guidance. The route is:

```text
artifacts/expert/route_best_bosskill_hp636_len293_20260603.jsonl
```

The planner tracks prefix alignment with this demonstration using `expert_pos`. When the next demonstration action is legal in the current state, it receives an additional score bonus. This warm-start makes the long-horizon search stable while still requiring every action to be legal under the simulator.

This design is useful for MOTA because the route has many narrow resource gates. The curriculum helps the planner pass difficult bottlenecks such as:

- preserving enough HP before MT9 shield,
- completing the MT8 blue-key chain,
- reaching MT10 with sufficient resources,
- collecting MT10 gems and potion,
- taking the red key after enough preparation,
- finishing the 10F mechanism and boss sequence.

## 7. Key Route Checkpoints

The final route reaches the following important states:

| Checkpoint | Step | Action | State After |
| --- | ---: | --- | --- |
| Sword | 19 | `go sword1 MT5:11,11` | HP 754, ATK 20, DEF 10, Y1, B1 |
| Shield | 75 | `go shield1 MT9:9,7` | HP 338, ATK 21, DEF 20, Y2 |
| MT10 blue gem | 172 | `go blueGem MT10:2,6` | HP 176, ATK 26, DEF 27 |
| MT10 red gem | 235 | `go redGem MT10:10,6` | HP 715, ATK 27, DEF 27 |
| MT10 blue potion | 237 | `go bluePotion MT10:11,11` | HP 915, ATK 27, DEF 27 |
| Red key acquired | 248 | `go yellowKey MT8:9,1` | HP 497, redKey 1 |
| Red door opened | 281 | `open redDoor MT10:6,9` | HP 1070, redKey 0 |
| 10F mechanism | 282 | `fight skeletonCaptain MT10:6,4` | HP 995, `10f机关=true` |
| Final boss | 290 | `fight skeletonCaptain MT10:6,1` | HP 436, `10f战胜骷髅队长=true` |

These checkpoints show that the planner successfully coordinates sword, shield, gems, HP recovery, red key, and final boss routing.

## 8. Experimental Result

Replay result:

```json
{
  "steps": 291,
  "solved": true,
  "done": true,
  "final": {
    "floor": "MT10",
    "x": 6,
    "y": 1,
    "hp": 436,
    "atk": 27,
    "def": 27,
    "keys": {
      "yellowKey": 0,
      "blueKey": 0,
      "redKey": 0
    },
    "flags": {
      "10f机关": true,
      "10f战胜骷髅队长": true
    }
  }
}
```

Constraint validation:

```json
{
  "ok": true,
  "forbid_shop": true,
  "forbid_fly": true,
  "violations": []
}
```

Full test suite:

```text
233 passed, 1 warning
```

## 9. Implementation Notes

Main files:

- `scripts/run_agentic_rl.py`: command-line runner for agentic planning experiments.
- `src/mota_agentic/orchestrator.py`: multi-agent scoring, beam search, expert curriculum tracking, route writing.
- `src/mota_agentic/openai_compatible_client.py`: optional OpenAI-compatible external agent client.
- `src/mota_agentic/kimi_client.py`: optional Kimi-style external agent client.
- `tests/test_agentic_orchestrator.py`: smoke test for legal route generation.
- `tests/test_openai_compatible_client.py`: API client behavior test.

The system supports external LLM agents through an OpenAI-compatible interface. In the final solved experiment, the local heuristic agents were used for stability and speed.

## 10. Summary

The project implements a working agentic RL-style planning system for 10F MOTA. The method decomposes the decision process into specialized agents, uses checkpoint-shaped scoring for long-horizon reasoning, and applies curriculum guidance to stabilize difficult resource gates. The final route is legal, replayable, and defeats the 10F skeleton captain with HP 436.

This establishes a strong baseline for the 10F benchmark and provides trace data, route files, and checkpoints for further experiments.

