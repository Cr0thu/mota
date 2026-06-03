# Cleanup Report 2026-05-31

## Scope

The repository was cleaned around the current AlphaZero-style research line:

- deterministic simulator;
- graph state builder;
- AlphaZero-style policy/value network and MCTS;
- Go-Explore/resource-planner helpers;
- local visualizer and route replay;
- paper/proposal material.

## Deleted

- Root scratch files accidentally created by pasted one-off Python snippets.
- Python caches, `.pytest_cache`, `.DS_Store`, `output/`, `tmp/`.
- Old local run payload directories under `artifacts/runs/`.
- Old archived/obsolete route payloads under `artifacts/archive/`,
  `artifacts/tmp/`, `artifacts/expert/archive_stale_20260524/`, and
  `artifacts/manual_exploration_20260524/obsolete_expert_routes/`.
- DQN / Graph-Q training entry points:
  - `scripts/train_graph_dqn_pure_rl.py`
  - `scripts/train_graph_masked_q.py`
  - DQN-focused 4090 launch scripts
  - `src/mota_rl/graph_q_model.py`
  - `tests/test_graph_q_model.py`

## Kept

- `artifacts/data/mota_first10.json`
- important success / benchmark routes under `artifacts/expert/`
- manual route notes under `artifacts/manual_exploration_20260524/`
- current simulator, reward, graph state, AlphaZero/MCTS, Go-Explore, and
  resource planner source code
- visualizer code and assets
- paper manifests, reading notes, downloaded paper PDFs, and proposal files

## Verification

After cleanup:

```text
PYTHONPATH=src pytest -q
80 passed, 1 skipped
```

Approximate local artifact size after cleanup:

```text
artifacts: 2.4M
scripts:   552K
src:       620K
tests:     104K
tools:     2.1M
```
