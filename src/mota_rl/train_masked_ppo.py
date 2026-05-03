from __future__ import annotations

import argparse
import json
from pathlib import Path

from mota_env.gym_env import MotaMacroEnv
from mota_env.rewards import reward_scheme_names


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="artifacts/data/mota_first10.json")
    parser.add_argument("--timesteps", type=int, default=50_000)
    parser.add_argument("--out-dir", default="artifacts/runs/masked_ppo")
    parser.add_argument("--reward-scheme", choices=reward_scheme_names(), default="label_dense")
    args = parser.parse_args()

    try:
        from sb3_contrib import MaskablePPO
        from sb3_contrib.common.maskable.utils import get_action_masks
    except Exception as exc:
        raise SystemExit(
            "Maskable PPO requires optional dependencies on Python 3.10/3.11:\n"
            "  pip install gymnasium stable-baselines3 sb3-contrib torch\n"
            f"Import error: {exc}"
        ) from exc

    class MaskedEnv(MotaMacroEnv):
        def action_masks(self):
            return self.action_mask()

    env = MaskedEnv(args.data, reward_scheme=args.reward_scheme)
    model = MaskablePPO("MultiInputPolicy", env, verbose=1, seed=20260430)
    model.learn(total_timesteps=args.timesteps, use_masking=True)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model.save(out_dir / "model.zip")
    observation, info = env.reset()
    successes = 0
    episodes = 20
    for _ in range(episodes):
        observation, info = env.reset()
        done = False
        while not done:
            action_masks = get_action_masks(env)
            action, _ = model.predict(observation, action_masks=action_masks, deterministic=True)
            observation, reward, terminated, truncated, info = env.step(int(action))
            done = terminated or truncated
        successes += int(env.state is not None and env.state.flags.get("10f战胜骷髅队长", False))
    (out_dir / "eval.json").write_text(
        json.dumps({"episodes": episodes, "successes": successes}, indent=2),
        encoding="utf8",
    )
    print(json.dumps({"out_dir": str(out_dir), "successes": successes, "episodes": episodes}, indent=2))


if __name__ == "__main__":
    main()
