# Expert Routes

This directory is intentionally small after the 2026-05-24 cleanup. Keep only current, replayable routes in the top level because the visualizer scans this directory for playable route choices. Old search attempts should go under `archive_stale_20260524/`.

## Formal Constraint

Formal routes must not use:

- 4F stat/HP shop actions: `shop hp`, `shop atk`, `shop def`;
- flyer or fly-shop actions.

Validate any route before reporting it:

```bash
PYTHONPATH=src python scripts/validate_route_constraints.py artifacts/expert/<route>.jsonl
```

Use `--forbid-merchants` only for the stricter variant that also bans the MT6/MT7 one-time key merchants.

## Current Routes

| Route | Purpose | Current result |
| --- | --- | --- |
| `route_best_bosskill_hp636_len293_20260603.jsonl` | Best current first-10-floor route for visualizer replay and benchmark comparison | Stop-on-boss replay defeats the 10F skeleton captain with HP 436; full `--continue-after-boss` replay ends with HP 636, ATK 30, DEF 30, and 3 yellow keys remaining |
| `route_best_bosskill_hp436_len293_20260530.jsonl` | Earlier copy of the same best route family; kept for reproducibility of previous reports | Full `--continue-after-boss` replay reaches HP 636, ATK 30, DEF 30 |
| `route_alpha_redkey_buffer_fixed_hp436_20260530.jsonl` | Strict boss route after red-key/key-buffer repair | Defeats the 10F skeleton captain with HP 436, ATK 27, DEF 27 |
| `route_alphazero_warmstart_hp436_20260601.jsonl` | AlphaZero warm-start comparison route | Defeats the 10F skeleton captain with HP 436, ATK 27, DEF 27 |
| `route_alpha4090_boss_success_hp125_20260531.jsonl` | 4090 Alpha-style boss-success route with shorter macro sequence | Defeats the 10F skeleton captain with HP 125, ATK 27, DEF 27 |
| `route_mt10_resources_manual_refill_success_20260524.jsonl` | Current playable route to 10F resource completion | Replays successfully; reaches MT10 with HP 248, ATK 27, DEF 27 after taking 10F blue gem, red gem, and blue potion |

Current research blocker: the best route is available as a benchmark and visualizer artifact, but the pure search/RL line has not yet rediscovered a full strict route without successful-route warm-start data.

Obsolete and invalid routes were moved to `artifacts/expert/archive_stale_20260524/`.
