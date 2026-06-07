from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from mota_agentic import (
    AgenticRLConfig,
    AgenticRLOrchestrator,
    KimiAgentClient,
    OpenAICompatibleAgentClient,
    PlaceholderAgentClient,
)
from mota_env import MotaSimulator, SimulatorConfig, load_game_data


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="artifacts/data/mota_first10.json")
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=340)
    parser.add_argument("--seed", type=int, default=20260606)
    parser.add_argument("--out-dir", default="artifacts/runs/agentic_rl_smoke")
    parser.add_argument("--route-out", default="")
    parser.add_argument(
        "--backend",
        choices=("heuristic", "placeholder", "kimi", "kimi-code", "openai-compatible", "deepseek"),
        default="heuristic",
        help="heuristic uses local agents only; kimi adds external Kimi votes.",
    )
    parser.add_argument("--agent-base-url", default="")
    parser.add_argument("--agent-model", default="")
    parser.add_argument(
        "--external-roles",
        default="planner",
        help="Comma-separated external agent roles, e.g. planner or planner,critic,explorer.",
    )
    parser.add_argument("--kimi-model", default="")
    parser.add_argument("--kimi-base-url", default="")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--beam-width", type=int, default=1)
    parser.add_argument("--candidate-top-k", type=int, default=1)
    parser.add_argument(
        "--expert-route",
        default="artifacts/expert/route_best_bosskill_hp636_len293_20260603.jsonl",
    )
    parser.add_argument("--expert-weight", type=float, default=1.0)
    parser.add_argument("--continue-after-boss", action="store_true")
    args = parser.parse_args()

    sim = MotaSimulator(
        load_game_data(args.data),
        SimulatorConfig(stop_on_boss=not args.continue_after_boss),
    )
    if args.backend == "placeholder":
        client = PlaceholderAgentClient()
    elif args.backend in {"kimi", "kimi-code"}:
        client = KimiAgentClient(
            model=args.kimi_model or None,
            base_url=args.kimi_base_url or None,
            code_mode=args.backend == "kimi-code",
        )
    elif args.backend in {"openai-compatible", "deepseek"}:
        client = OpenAICompatibleAgentClient(
            base_url=args.agent_base_url or None,
            model=args.agent_model or None,
        )
    else:
        client = None
    orchestrator = AgenticRLOrchestrator(
        sim,
        AgenticRLConfig(
            episodes=args.episodes,
            max_steps=args.max_steps,
            seed=args.seed,
            temperature=args.temperature,
            external_roles=tuple(role.strip() for role in args.external_roles.split(",") if role.strip()),
            beam_width=args.beam_width,
            candidate_top_k=args.candidate_top_k,
            expert_route=args.expert_route,
            expert_weight=args.expert_weight,
        ),
        client=client,
    )
    outcome = orchestrator.run()
    outputs = orchestrator.write_outputs(
        outcome,
        out_dir=args.out_dir,
        route_out=args.route_out or None,
    )
    print(
        json.dumps(
            {
                "solved": outcome.solved,
                "done": outcome.done,
                "episodes": outcome.episodes,
                "best_episode": outcome.best_episode,
                "route_len": len(outcome.route),
                "final": outcome.final,
                "outputs": outputs,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
