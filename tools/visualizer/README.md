# Magic Tower Visualizer

This directory contains the Tkinter visualizer imported from the `mota-with-window`
branch. It is intentionally isolated from the main `src/mota_env` simulator because
the teammate branch uses a separate environment implementation.

## Run Locally

```bash
open tools/visualizer/run_visualizer_iterm.command
```

The launcher opens iTerm and runs the visualizer with the first usable Python
environment it can find. On this machine the preferred environment is:

```text
/Users/cr0/anaconda3/envs/Cr0/bin/python
```

That environment has `torch`, `tkinter`, `Pillow`, and `pandas`, so both manual
visualization and PPO demo/training code can be loaded.

Manual mode supports:

- floor switching
- legal-action list
- executing selected macro actions
- one-step backtracking
- syncing to the local algorithm start state after the 2F thief sequence
- loading and replaying JSONL routes generated under `artifacts/expert/`
- connection-line toggling
- action detail explanations
- single-click execution from map target tiles
- restored early 3F demon-king plot and 2F thief sequence for manual play
- physical removal of the 4F stat/HP shop and the flyer item
- MT6/MT7 key merchants, gated by the original money checks
- a clearer speed control using delay in milliseconds (`0` is fastest)

Keyboard shortcuts:

- `Enter`: execute selected action
- `Backspace`: backtrack one step
- `R`: reset
- `L`: show/hide green connection lines
- `+` / `-`: view upper/lower floor

Important: the floor buttons only change the viewed floor. They do not move the
hero. To move the hero, execute one of the legal macro actions in the right-side
action list.

## Replay Local Algorithm Routes

Generate a route from the main project first:

```bash
PYTHONPATH=src python -m mota_solver.solve_staged \
  --data artifacts/data/mota_first10.json \
  --max-expansions-per-stage 20000 \
  --write-route \
  --route-out artifacts/expert/visualizer_route.jsonl
```

Then open the visualizer. The route selector automatically scans
`artifacts/expert/*.jsonl`; choose one route from the dropdown, click
`同步算法起点`, and use `单步路线` or `播放路线`. Playback follows the solver
route corridor instead of requiring a one-to-one action match. This matters
because the solver records stairs and long movement segments, while the
visualizer hides stairs as internal graph edges and may split one solver macro
action into several visible item/monster actions. It also skips `fakeWall` steps
because fake walls are already simplified to passable floor in the visualizer.

## Direct Run

To bypass iTerm, run:

```bash
tools/visualizer/run_visualizer.command
```

You can force a specific virtual environment with:

```bash
MOTA_VISUALIZER_PYTHON=/path/to/python tools/visualizer/run_visualizer.command
```

## PPO Training / Demo

The PPO buttons in the UI require `torch` and the local model file. The CLI
training path is:

```bash
python tools/visualizer/train.py --rounds 1000 --save model/ppo_10floor.pth
python tools/visualizer/train.py --demo --save model/ppo_10floor.pth
```

The uploaded model is kept locally at `tools/visualizer/model/ppo_10floor.pth`,
but `*.pth` is ignored by git in this repository.

## Notes

- `run_this.py` changes the process working directory to this folder so that
  image assets under `pictures/` and model paths resolve correctly.
- This visualizer currently optimizes and demonstrates the 5F sword objective,
  not the final first-10-floor skeleton-captain objective.
