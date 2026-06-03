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
| `route_mt10_resources_manual_refill_success_20260524.jsonl` | Current playable route to 10F resource completion | Replays successfully; reaches MT10 with HP 248, ATK 27, DEF 27 after taking 10F blue gem, red gem, and blue potion |

Current blocker: this route proves 10F resource collection is feasible, but it does not yet defeat the skeleton captain.

Obsolete and invalid routes were moved to `artifacts/expert/archive_stale_20260524/`.
