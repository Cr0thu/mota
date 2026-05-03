# Mota RL

This repository builds a deterministic environment and hybrid planning/RL pipeline for the
HTML5 50-floor Magic Tower project in `game/Falsh原版魔塔合集/51_2/project`.

The active experiment starts after the MT3 thief plot, removes shop/fly mechanics, keeps only
floors 1-10, and targets defeating the skeleton captain on floor 10.

## Quick Start

```bash
node scripts/extract_mota_data.js
PYTHONPATH=src python3 -m mota_solver.solve_first10 --max-expansions 1000 --write-route --write-best
PYTHONPATH=src python3 scripts/replay_route.py
PYTHONPATH=src python3 -m pytest
```

The default scenario is already post-thief, no shop, and no flyer. To increase search budget:

```bash
PYTHONPATH=src python3 -m mota_solver.solve_first10 \
  --max-expansions 10000 \
  --heuristic landmark_key_potential \
  --write-route \
  --write-best

PYTHONPATH=src python3 scripts/replay_route.py
```

`--write-best` writes the best legal partial route when the current search budget does not yet
defeat the 10F skeleton captain. The simulator and replay pipeline are deterministic; the next
engineering step is to promote the 8F blue-key and 9F shield milestones into the expert search so
`artifacts/expert/route_first10.jsonl` reaches `flag:10f战胜骷髅队长=true`.

Current best simplified-scenario heuristic run:

```bash
PYTHONPATH=src python3 -m mota_solver.solve_first10 \
  --max-expansions 50000 \
  --heuristic landmark_key_potential \
  --write-route \
  --write-best \
  --route-out artifacts/expert/route_first10_landmark_50k_v3.jsonl
```

The run reaches MT9 with the 9F shield, but has not yet defeated the 10F skeleton captain. See
`artifacts/runs/landmark_heuristic_summary.md`.

Optional RL dependencies:

```bash
uv venv --python 3.10 .venv
source .venv/bin/activate
uv pip install -e '.[rl,test]'
PYTHONPATH=src python -m mota_rl.behavior_clone --route artifacts/expert/route_first10.jsonl
PYTHONPATH=src python -m mota_rl.train_masked_ppo --timesteps 50000
```

Remote pod workflow keeps the existing global ailab config unchanged:

```bash
zsh -lc 'source ~/.zshrc && ailab-sync-up /Users/cr0/Documents/项目/mota /root/mota/mota'
zsh -lc 'source ~/.zshrc && ailab-exec bash -lc "cd /root/mota/mota && PYTHONPATH=src python3 -m mota_solver.solve_first10 --max-expansions 10000 --write-route --write-best"'
```

## Paper Research

- `paper/paper_manifest.csv`: 56 papers/projects with open PDF or URL.
- `paper/pdfs/`: downloaded PDFs for MuZero, UniZero, Thinker, Searchformer, NLE, Rainbow, PER, BTR, EfficientZero, Gumbel MuZero.
- `paper/reading_report.md`: per-paper notes and project-specific recommendations.
- `paper/factor_reward_paper_manifest_100.csv`: 110 papers linking quant factor mining, reward design, IRL, and long-horizon puzzle solving.
- `paper/factor_reward_reading_report.md`: 20-paper deep-read report and a concrete factor/reward engineering plan.

## Repository Hygiene

The GitHub repository intentionally excludes local game binaries/source dumps, SWF files, generated
PDF downloads, caches, and large training artifacts. The extracted first-10-floor JSON under
`artifacts/data/` is kept so tests and simulator experiments can run after clone.

If you need to regenerate data from the local HTML5 game project, place the original game assets
under `game/Falsh原版魔塔合集/51_2/project` and run:

```bash
node scripts/extract_mota_data.js
```
