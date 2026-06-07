# Mota AlphaZero

This repository contains a deterministic first-10-floor Magic Tower simulator,
route replay/validation tools, a local visualizer, and the current
AlphaZero-style search/training code.

The active research setting starts after the compulsory MT3 demon / MT2 thief
story, keeps floors MT1-MT10, removes the 4F shop and flyer, and targets
defeating the 10F skeleton captain. The MT6/MT7 one-time key merchants remain
modeled as legal NPC actions when money is sufficient.

## Current Direction

The current training line is **not DQN**. Use:

- `MotaSimulator` for deterministic state transitions.
- `GraphStateBuilder` for the global interaction graph.
- `GraphPolicyValueNet` for policy/value prediction over graph nodes.
- `AlphaMCTS` for AlphaZero-style PUCT planning.
- Go-Explore / resource planner scripts only as search/data generators, not as
  DQN trainers.

Do not use the removed Graph-DQN scripts for new experiments. Historical hp403
routes are benchmarks or warm-start ablations only; keep them separate from
pure no-hp403 runs.

## Local Setup

```bash
cd /Users/cr0/Documents/项目/mota
PYTHONPATH=src python -m pytest
```

The committed data extract is:

```text
artifacts/data/mota_first10.json
```

The original `game/` directory is ignored by git and is only needed to
regenerate this JSON or compare behavior visually against the game assets.

## Useful Commands

Replay a route:

```bash
PYTHONPATH=src python scripts/replay_route.py \
  --route artifacts/expert/route_alpha4090_boss_success_hp125_20260531.jsonl
```

Validate no-shop/no-fly constraints:

```bash
PYTHONPATH=src python scripts/validate_route_constraints.py \
  artifacts/expert/route_alpha4090_boss_success_hp125_20260531.jsonl
```

Run a local AlphaZero/MCTS stage smoke test:

```bash
PYTHONPATH=src python scripts/train_alpha_mota_stage.py \
  --protocol no_agent_manual \
  --target-stage sword \
  --episodes 2 \
  --simulations 16 \
  --max-macros 80 \
  --out-dir artifacts/runs/local_alpha_smoke
```

Run the agentic RL-style planner:

```bash
PYTHONPATH=src python scripts/run_agentic_rl.py \
  --backend heuristic \
  --beam-width 2 \
  --candidate-top-k 1 \
  --episodes 1 \
  --max-steps 340 \
  --expert-weight 2.5 \
  --out-dir artifacts/runs/agentic_expertguide_solved_greedybeam \
  --route-out artifacts/tmp/agentic_expertguide_solved_greedybeam_route.jsonl
```

Generate the agentic report figures:

```bash
PYTHONPATH=src python tools/generate_agentic_report_figures.py
```

Run the visualizer:

```bash
open tools/visualizer/run_visualizer_iterm.command
```

Regenerate first-10-floor data from the local HTML5 project:

```bash
node scripts/extract_mota_data.js
node scripts/extract_mota_data.js /path/to/51_2/project artifacts/data/mota_first10.json
```

## 4090 Workflow

Experiments should run on the 4090 pod under:

```text
/root/mota/mota
```

Use the `humanoid` conda environment on the pod. The current remote line is
AlphaZero-style policy/value + MCTS; keep each run in its own
`artifacts/runs/<run_id>/` directory and save `protocol.json` with
`dqn_used=false` for no-DQN runs.

## Repository Hygiene

Keep in git:

- simulator, graph state, reward, search, AlphaZero/MCTS source code;
- tests;
- `artifacts/data/mota_first10.json`;
- a small set of important routes in `artifacts/expert/`;
- paper manifests/reports and proposal source.

Do not commit:

- original `game/` binaries/source dumps;
- local training payloads, checkpoints, and run directories;
- downloaded/generated PDFs outside intentional paper/proposal deliverables;
- Python caches and local smoke-test output.

## Agentic RL-Style Planning Baseline

The repository also contains an agentic planning harness under
`src/mota_agentic/`. It decomposes MOTA decision-making into specialized
scoring agents:

- `stage_navigator` for checkpoint progress;
- `resource_economy` for keys, gems, potions, and money;
- `combat_threshold` for HP-loss and dead-end risk;
- `boss_objective` for red-door, mechanism, and final boss routing;
- `expert_route_bias` for curriculum-style warm-start guidance.

At each step, the simulator generates legal macro actions. All agents evaluate
the same candidate action table and candidate after-states. A beam-search
arbiter merges their scores, keeps promising states, and writes a replayable
route JSONL. The planner also keeps recent trace memory and visited-state
memory to summarize recent decisions and reduce repeated loops.

The report snippets and figures for this line are in:

```text
report/agentic_mota_merge_section_20260607.md
report/agentic_mota_merge_section_20260607_en.md
report/figures/
```
