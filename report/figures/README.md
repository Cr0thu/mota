# Agentic MOTA Figures

This folder contains figures generated for the agentic RL-style MOTA report section. Each figure is saved as both PNG and SVG. The PNG files are convenient for quick insertion into reports, while the SVG files are better for LaTeX or high-resolution editing.

Generation command:

```powershell
conda run -n dl python tools\generate_agentic_report_figures.py
```

## 1. `agentic_architecture`

Files:

- `agentic_architecture.png`
- `agentic_architecture.svg`

This figure shows the overall agentic planning framework. The MOTA simulator first provides the current state and generates legal macro actions. These candidate actions are then evaluated by several specialized agents: `stage_navigator`, `resource_economy`, `combat_threshold`, `boss_objective`, and `expert_curriculum`. Their scores are combined by a beam-search arbiter, which selects the next action and writes the final trajectory as a replayable route JSONL file.

Recommended placement:

- Method section.
- Agent design section.

Suggested caption:

> Overview of the agentic RL-style planning framework. The simulator generates legal macro actions, multiple specialized agents score each candidate, and a beam-search arbiter selects actions to produce a replayable MOTA route.

## 2. `checkpoint_timeline`

Files:

- `checkpoint_timeline.png`
- `checkpoint_timeline.svg`

This figure visualizes the major milestones reached by the solved route along the macro-action step axis. It highlights when the agent obtains the sword, obtains the shield, collects important floor-10 gems, gets the red key, opens the red door, triggers the floor-10 mechanism, and defeats the final boss.

Recommended placement:

- Results section.
- Route analysis section.

Suggested caption:

> Checkpoint timeline of the solved route. The agent reaches the sword at step 19, shield at step 75, floor-10 resources in the middle-late stage, red key at step 248, and defeats the final boss at step 290.

## 3. `resource_progression`

Files:

- `resource_progression.png`
- `resource_progression.svg`

This figure plots HP, ATK, and DEF over the solved route. It shows that the agent does not simply maximize short-term HP. Instead, it accepts HP drops to obtain important resources, then recovers HP before the red-key and boss stages. The ATK/DEF curves also show how attack and defense gems gradually improve the combat margin.

Recommended placement:

- Results section.
- Resource planning discussion.

Suggested caption:

> HP, attack, and defense progression over the solved route. The curve shows how the agent trades HP for key resources and gradually improves combat capability before the final boss fight.

## 4. `key_inventory`

Files:

- `key_inventory.png`
- `key_inventory.svg`

This figure shows the inventory count of yellow, blue, and red keys over the solved route. It is useful for explaining why MOTA requires long-horizon planning: keys must be spent and preserved at the correct time. The red key only appears late in the route and is consumed by the final red door before the boss sequence.

Recommended placement:

- Resource economy subsection.
- Appendix if the main report is short.

Suggested caption:

> Key inventory during the solved route. Yellow and blue keys are repeatedly spent and replenished, while the red key appears only in the late stage and is consumed to open the final red door.

## 5. `experiment_progression`

Files:

- `experiment_progression.png`
- `experiment_progression.svg`

This figure summarizes the experimental progression across several agentic planning variants. Earlier attempts reached only partial checkpoints, while later variants with checkpoint shaping and curriculum guidance reached the mechanism and finally defeated the boss.

Recommended placement:

- Experiment process section.
- Appendix.

Suggested caption:

> Experimental progression of agentic planning variants. The final curriculum-guided agentic planner reaches the final boss checkpoint, while earlier variants stop at intermediate stages.

## Suggested Figure Selection

For a short report section, use:

1. `agentic_architecture`
2. `checkpoint_timeline`
3. `resource_progression`

For a longer report or appendix, additionally include:

4. `key_inventory`
5. `experiment_progression`

