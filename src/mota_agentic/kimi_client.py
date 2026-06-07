from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any

from .client import AgentClientError


def _extract_json_array(text: str) -> list[Any]:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\[[\s\S]*\]", text)
        if match is None:
            raise
        data = json.loads(match.group(0))
    if not isinstance(data, list):
        raise ValueError("Kimi response must be a JSON array")
    return data


@dataclass
class KimiAgentClient:
    """OpenAI-compatible Kimi/Moonshot backend for action ranking."""

    api_key: str | None = None
    base_url: str | None = None
    model: str | None = None
    code_mode: bool = False
    timeout: float = 45.0
    max_completion_tokens: int = 700

    def __post_init__(self) -> None:
        self.api_key = self.api_key or os.environ.get("KIMI_API_KEY") or os.environ.get("MOONSHOT_API_KEY")
        self.code_mode = self.code_mode or os.environ.get("KIMI_CODE_MODE", "").strip().lower() in {
            "1",
            "true",
            "yes",
            "on",
        }
        default_base_url = "https://api.kimi.com/coding/v1" if self.code_mode else "https://api.moonshot.cn/v1"
        default_model = "kimi-for-coding" if self.code_mode else "kimi-k2.6"
        self.base_url = (self.base_url or os.environ.get("KIMI_BASE_URL") or default_base_url).rstrip("/")
        if "api.kimi.com/coding" in self.base_url:
            self.code_mode = True
        self.model = self.model or os.environ.get("KIMI_MODEL") or default_model
        if not self.api_key:
            raise AgentClientError("KIMI_API_KEY is not set")

    def rank_actions(
        self,
        *,
        role: str,
        task: str,
        state: dict[str, Any],
        actions: list[dict[str, Any]],
        memory: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        payload = {
            "model": self.model,
            "temperature": 0.2,
            "max_completion_tokens": self.max_completion_tokens,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are one specialist in a multi-agent Magic Tower route search. "
                        "Rank only legal actions supplied by the simulator. Reply with a JSON "
                        "array and no prose. Each element must be "
                        '{"action_index": int, "score": number, "reason": string}.'
                    ),
                },
                {
                    "role": "user",
                    "content": json.dumps(
                        {
                            "role": role,
                            "task": task,
                            "state": state,
                            "recent_memory": memory[-8:],
                            "legal_actions": actions[:80],
                            "ranking_guidance": self._role_guidance(role),
                        },
                        ensure_ascii=False,
                    ),
                },
            ],
        }
        request = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=json.dumps(payload, ensure_ascii=False).encode("utf8"),
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
                "User-Agent": "mota-agentic-rl/0.1",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                body = response.read().decode("utf8")
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf8", errors="replace")
            raise AgentClientError(f"Kimi HTTP {exc.code}: {detail[:500]}") from exc
        except urllib.error.URLError as exc:
            raise AgentClientError(f"Kimi request failed: {exc}") from exc

        try:
            data = json.loads(body)
            content = data["choices"][0]["message"]["content"]
            rows = _extract_json_array(content)
        except (KeyError, IndexError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise AgentClientError(f"Could not parse Kimi ranking response: {body[:500]}") from exc

        proposals: list[dict[str, Any]] = []
        valid_indices = {int(action["action_index"]) for action in actions if "action_index" in action}
        for row in rows:
            if not isinstance(row, dict):
                continue
            try:
                action_index = int(row["action_index"])
                score = float(row.get("score", 0.0))
            except (KeyError, TypeError, ValueError):
                continue
            if action_index not in valid_indices:
                continue
            proposals.append(
                {
                    "action_index": action_index,
                    "score": max(-100.0, min(100.0, score)),
                    "reason": str(row.get("reason", ""))[:180],
                }
            )
        return proposals[:12]

    @staticmethod
    def _role_guidance(role: str) -> str:
        guidance = {
            "planner": (
                "Follow the macro goal chain: sword -> shield -> stat gems -> red key -> boss. "
                "Prefer actions that advance the current stage without breaking keys or HP."
            ),
            "critic": (
                "Down-rank actions that waste yellow keys, eat potions too early, fight high-damage "
                "monsters before stat thresholds, or make the red-key/boss route impossible."
            ),
            "explorer": (
                "Suggest non-obvious order changes that may cross ATK/DEF thresholds earlier, while "
                "still respecting legal macro actions and future key needs."
            ),
        }
        return guidance.get(role, "Rank actions by long-term route quality.")
