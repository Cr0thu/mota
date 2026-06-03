# 4090 Experiment Reorg Status

Date: 2026-05-27

## Current Discipline

- Local machine is now used for code edits, lightweight tests, visualization, and result review only.
- Training, reward search, large search, and DQN-style experiments are run on the AILab 4090 pod under `/root/mota/mota`.
- Pure and hp403 experiments are kept separate:
  - `pure_search_rl`: no hp403 data.
  - `hp403_warmstart`: hp403 is only a warm-start / DQfD-style comparison.

## Implemented Remote Entrypoints

- `scripts/remote_4090_hp403_warmstart_job.sh`
  - Runs the explicit hp403 warm-start branch.
  - Does not mix results into pure experiments.
- `scripts/remote_4090_pure_dqn_v2_job.sh`
  - Runs direct Graph DQN from pure staged routes.
  - Later runs showed this branch is still too weak without a stronger staged prior.
- `scripts/remote_4090_action_ranker_v1_job.sh`
  - Trains stage-specific action rankers from pure staged/search routes only.
  - Stopped after from-scratch boss beam became CPU-bound.
- `scripts/remote_4090_ranker_chain_v2_job.sh`
  - Adds staged chain decode: shield -> 10F resources -> red key -> boss.
  - Stopped because first-stage shield beam still over-valued early 3F resource/fight actions.
- `scripts/remote_4090_mt10_repair_v3_job.sh`
  - Starts from full strict staged prefixes and repairs the 10F-resource segment.
  - Stopped because the full prefixes entered 10F too late with only 0-2 yellow keys.
- `scripts/remote_4090_mt10_prefix_v4_job.sh`
  - Starts from earlier high-key prefix cut points via `--start-route-max-steps`.
  - v6 added per-depth best-route checkpoint output from `beam_decode`, so timeout no longer loses the best partial route.
- `scripts/remote_4090_gem_chain_v7_job.sh`
  - New chain experiment: `low_gems -> mid_gems -> mt8_gems -> mt10_resources -> red_key -> boss`.
  - Uses checkpoint routes as phase inputs even when a phase times out before writing a summary.
  - Purpose: fix the v6 failure mode where high-key states climbed to 10F before enough attack/defense resources were collected.
- `scripts/remote_4090_guard_chain_v8_job.sh`
  - New guard-readiness chain from v7 mid-gem routes:
    `mt8_hp_ready -> mt8_gems -> guard_ready -> red_key -> mt10_resources -> boss_ready -> boss`.
  - Purpose: fix the v7 failure mode where low/mid gems were recovered but HP stayed too low for the 8F red-key route.
- `scripts/remote_4090_relaxed_redkey_probe_v1_job.sh`
  - Small relaxed probe from the two most informative v7 mid-gem routes.
  - Allows negative HP only to discover red-key route structure and HP debt; these routes are never counted as strict successes.
- `scripts/remote_4090_mt10_first_v12_job.sh`
  - New MT10-first chain: `mt10_resources -> guard_ready -> red_key -> boss_ready -> boss`.
  - Supports `INPUT_ROUTES_FILE`, so the same script can start from mid-gem routes or HP-ready routes.
  - Purpose: enforce the current research decision that 10F blue/red gem and potion should be collected before the 8F red-key guard route.

## Remote Runs Started

- `4090_pure_search_rl_20260527_104212`
  - Pure staged search produced strict prefixes but stalled in reward tuning.
  - Best strict prefixes reached MT10 with HP 149-162, ATK 26, DEF 27, but lacked enough yellow keys for all 10F resources.
- `4090_hp403_warmstart_20260527_105344`
  - hp403 benchmark strict replay is valid, but warm-start DQN did not yet generalize.
- `4090_action_ranker_v1_20260527_152511`
  - Stage action-ranker training improved route top-1 ranking, but greedy eval remained unsuccessful.
- `4090_ranker_chain_v2_20260527_163600`
  - Stopped; shield beam repeated early bad choices.
- `4090_mt10_repair_v3_20260527_165622`
  - Stopped; continuation from staged route endpoints was too late.
- `4090_mt10_prefix_v4_20260527_173048`
  - Superseded by v5/v6 after reconnecting the AILab tunnel.
- `4090_mt10_prefix_v6_20260527_204235`
  - Running on the 4090 pod in tmux session `mota_mt10_prefix_v6_4090`.
  - Best checkpoints preserve 5-7 yellow keys and reach MT9/MT10, but still have low stats around ATK/DEF 24-25 and at most `mt10_resource_progress=1`.
  - Current conclusion: key preservation improved, but the search enters 10F too early and must explicitly recover MT8/MT9/lower gems first.
- `4090_gem_chain_v7_20260527_210353`
  - Running on the 4090 pod in tmux session `mota_gem_chain_v7_4090`.
  - First phase is backtracking from shield/high-key prefixes to complete low-level gems before continuing to mid gems, 8F resources, 10F resources, red key, and boss.
  - Result so far: no red-key success. Best useful failure has low/mid gems completed and ATK/DEF around 26/26, but HP is too low and `red_key_route_margin` remains about -500.
- `4090_guard_chain_v8_20260527_233620`
  - Running on the 4090 pod in tmux session `mota_guard_chain_v8_4090`.
  - Early HP-ready checkpoint improved one branch to HP 233, ATK/DEF 25/25, yellow keys 6, blue keys 1, but it still needs further guard-margin improvement before red key.
  - Stopped after discovering that `guard_ready` adaptive margin weights were not being applied by `beam_decode`.
- `4090_guard_chain_v9_20260528_012259`
  - Running on the 4090 pod in tmux session `mota_guard_chain_v9_4090`.
  - Uses the fixed `guard_ready` adaptive reward and explicit 10F-resource bias.
  - Stopped because the first HP-ready phase still delayed the actual guard-ready test.
- `4090_guard_direct_v10_20260528_013012`
  - Running on the 4090 pod in tmux session `mota_guard_direct_v10_4090`.
  - Directly starts `guard_ready` from v7 mid-gem routes using the fixed margin reward, then chains red key, 10F resources, boss-ready, and boss if successful.
  - Stopped after diagnosing that 9F side fights/doors were still ranked ahead of the MT9 -> MT10 critical path.
- `4090_guard_direct_v11_20260528_014032`
  - Stopped after no MT10 resource progress appeared; replaced by the harder MT10-first chain.
  - Adds focused `guard_ready` bias for `bat MT9:7,10 -> yellowDoor MT9:6,11 -> blueDoor MT9:3,11 -> upFloor MT9:1,11`.
- `4090_relaxed_redkey_probe_20260527_233944`
  - Completed as diagnostic only.
  - Purpose is diagnostic: determine whether red-key structure is reachable with HP debt and measure the gap.
- `4090_mt10_first_v12_20260528_0150`
  - Stopped after confirming that low-HP mid-gem prefixes still do not reliably enter 10F resources.
  - Superseded by HP-ready starts.
- `4090_mt10_first_v13_hpready_20260528_0200`
  - Stopped after a beam-search pruning bug was fixed.
  - Diagnostic value: HP-ready prefix gives a promising state at MT8 with HP 283, ATK/DEF 25/25, yellow keys 5, blue keys 1, and `mt10_access_ready=True`.
- `4090_mt10_first_v14_seenbest_20260528_0204`
  - Stopped after the fixed revisiting logic still allowed MT8 fights to outrank the direct MT8 -> MT9 stair path.
  - Code fix: `seen_best` now allows a later higher-score route to replace a lower-score route to the same simulator state.
- `4090_mt10_first_v15_no_mt8_fights_20260528_0210`
  - Running on the 4090 pod in tmux session `mota_mt10_first_v15_no_mt8_fights_4090`.
  - Adds hard action bias in `mt10_resources`: once `mt10_access_ready=True`, suppress MT8 fights/doors and prefer `upFloor MT8:6,1`.
  - Current early checkpoint still has the promising HP-ready MT8 state; waiting for deeper expansion toward 9F/10F resources.
- `4090_mt10_first_v21_from_mt9_pathfix_20260528_0232`
  - Strictly solved `mt10_first_resource` from the MT9 access checkpoint after adding the missing MT9 path step `redSlime MT9:7,6`.
  - Final checkpoint: MT10 blue gem taken, HP 128, ATK/DEF 25/26, yellow keys 4, `mt10_resource_progress=1`.
- `4090_mt10_resources_v22_from_blue_20260528_0234`
  - Strictly solved `mt10_resources` from v21.
  - Final checkpoint: all three 10F resources taken, HP 260, ATK/DEF 26/26, yellow keys 1, `red_key_route_margin=-348`.
  - This is a real improvement over earlier MT10 failures, but still not enough for red-key guards.
- `4090_guard_from_mt10_v23_20260528_0236`
  - Started from v22 and targeted `guard_ready`.
  - Stopped after diagnosing a stage-order issue: all 10F resources first leaves too little low-floor refill flexibility.
- `4090_guard_after_first_resource_v25_20260528_0242` / `4090_guard_after_first_resource_v27_from_v20_20260528_0248`
  - Tested the benchmark-inspired order: take the first 10F resource, then descend for refill before red-key guards.
  - Result: the search now descends correctly, but these pure routes had already consumed too many low-floor potions earlier.
- `4090_guard_low_refill_v28_from_v20_20260528_0254`
  - Added narrow `guard_low_refill` target to force low-floor refill exploration after the first 10F resource.
  - Diagnostic result: reached MT1 quickly but did not choose temporary-loss blockers, so no margin improvement.
- `4090_guard_low_refill_v29_from_v20_bias_20260528_0258`
  - Adds low-floor refill bias for `greenSlime MT1:9,11`, `skeleton MT1:2,4/2,7`, 1F red potions, and 2F/4F blue potions.
  - Completed without solving `guard_ready`.
  - Diagnosis: low-floor refill is feasible only if earlier stages preserve those potions; the v20 prefix already consumed too many low-floor resources, so the route-repair target must penalize pre-10F potion consumption before the first 10F resource.
- `4090_delay_refill_v30_20260528_continue`
  - Running on the 4090 pod in tmux session `mota_v30_delay_refill`.
  - Code change: `beam_decode` now supports a weight-controlled `delayed_refill_penalty` that applies before the first 10F resource in `low_gems/mid_gems/mt8_gems/mt10_* / guard_*` stages.
  - The penalty is generic resource management, not hp403 imitation: it discourages consuming low-floor potions before `mt10_resource_progress > 0`, with stronger weights for blue potions and known refill-critical low-floor blue potions.
  - First completed low-gems candidate is only a diagnostic prefix already at MT8: HP 141, ATK/DEF 25/25, yellow keys 4, blue keys 1, `red_key_route_margin=-684`. The meaningful earlier-prefix beams are still running.
- `4090_pure_dqn_v31_parallel_20260528`
  - Failed immediately because the old DQN launcher did not activate the `humanoid` conda env and fell back to Python 3.13 without torch/numpy.
  - Fix: `scripts/remote_4090_pure_dqn_v2_job.sh` now activates `${CONDA_ENV:-humanoid}` and exports `PYTHONPATH=src`.
- `4090_pure_dqn_v32_humanoid_20260528`
  - Stopped after strict replay of current `best_train_route.jsonl` showed all best routes were relaxed negative-HP failures that did not reach 10F.
  - Pure line only: uses staged pure routes for bootstrap/guided rollout, not hp403.
  - Environment confirmed: Python 3.10.20, PyTorch 2.5.1+cu121, 4 x 4090D visible.
  - Progress before stop: 420 bootstrap transitions loaded per process; GPU pretrain reached update 80.
  - Failure mode: `allow_negative_hp=True` and `hp_debt_penalty=0.06` were too permissive, so Graph DQN selected negative-HP routes as best.
- `4090_pure_dqn_v33_strict_20260528`
  - Running on the 4090 pod in tmux session `mota_dqn_v33_strict`.
  - Same pure bootstrap setup as v32, but `ALLOW_NEGATIVE=0`, `HP_DEBT_PENALTY=0.25`, `DEATH_PENALTY=180`, `DEADEND_PENALTY=45`, `REVISIT_PENALTY=9`.
  - Purpose: test whether Graph DQN can learn legal stage progress when negative-HP exploration is disabled.

## Current Remote Access

- The AILab tunnel is working again through an explicit SSH forward to the API IP:
  `127.0.0.1:6443 -> 10.1.0.217:6443`.
- Active pod: `ruichen-intelligent-robot-hw2-6b7fd8f4d5-qhp2b`.
- GPU resource: 4 x NVIDIA GeForce RTX 4090 D.

## Observed Technical Conclusion

- Direct DQN and direct learned-prior rollout are still too local.
- The useful search signal is now clearer: staged routes fail for two reasons, not one:
  - late endpoints enter MT10 with too few yellow keys;
  - high-key prefixes enter MT10 with too little ATK/DEF/HP because lower/8F resources were skipped.
- The next useful experiment is staged resource-chain search first, then Graph Q/ranker training from repaired suffixes if the chain produces strict candidates.
- A concrete search bug was found and fixed: `beam_decode` previously discarded every repeated simulator state after the first visit, even when the later route had a higher score. It now prunes only when the previous score is at least as good.
- The current best pure-search direction is no longer "take all 10F resources first".
- The better structure is: first 10F blue gem -> descend/refill -> red-key guards -> return for remaining 10F resources/boss.
- The next useful repair is an earlier-stage route search that explicitly delays low-floor HP potions until after the first 10F resource. This should be encoded as reward/label penalties, not as hp403 imitation data.
- That repair is now implemented as `delayed_refill_penalty`; v30 is testing whether the route distribution changes enough to preserve refill resources before the first MT10 resource.
- In parallel, v32 is the current pure Graph DQN/PER run on the 4090s. It is still a secondary line because the search quality is the bottleneck, but it now runs in the correct CUDA environment.

## Local Verification

- Shell syntax check passed for the current 4090 remote scripts.
- `PYTHONPATH=src python -m pytest -q tests/test_beam_decode.py tests/test_rewards.py` passes locally and remotely.
