# No-Agent-Manual Protocol

The `route_alpha4090_boss_success_hp125_20260531.jsonl` result is a manual-guided
debugging baseline. It must not be reported as pure RL or no-demonstration RL.

Valid no-agent-manual runs must use:

- `scripts/train_alpha_mota_stage.py --protocol no_agent_manual`
- no `--start-route`
- no `--init-checkpoint`
- `--heuristic-prior-mix 0`
- `--disable-stage-action-filter` enabled automatically
- `--stage-value-mode binary` enabled automatically
- no `--reward-weights-file`

This means MCTS expands all legal macro actions from the simulator. Coordinate,
token, or route-specific filters authored after observing failures are disabled.
The value target is the declared stage outcome only: success is `+1`, failure is
`-1`. Hand-shaped failure values based on HP, floor, keys, boss margin, or
similar route observations are not used in this protocol.

Allowed ingredients:

- fixed simulator rules
- graph state
- generic macro actions from reachable interactions
- generic terminal/stage predicates
- policy/value learning from self-generated MCTS trajectories
- automatic hyperparameter search when the search space is declared before the
  run and the selection metric is fixed in the run manifest

Disallowed ingredients:

- adding coordinate or label tokens after inspecting a failed trajectory
- starting from `hp403`, `hp436`, or any manual route
- using the current HP=125 route as imitation data
- modifying reward weights by hand based on a specific failed route
- using hand-shaped failure values or a learned/manual reward file inside the
  no-agent-manual AlphaZero run
- selecting a new hyperparameter value because a specific route looked bad;
  follow-up runs must either reuse the existing search space or declare a new
  search-space version before launching

Current status:

- Manual-guided baseline: success, HP=125.
- Fixed-parameter no-agent smoke: runnable, not a main result.
- Auto-sweep runner: `scripts/remote_4090_no_agent_manual_sweep_job.sh`.
- No-agent-manual main result: not yet established.
