from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .client import AgentClientError

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_LOCAL_KEY_PATH = REPO_ROOT / "artifacts" / "tmp" / "agent_api_key.txt"


def _extract_json_array(text: str) -> list[Any]:
    text = text.strip()
    text = re.sub(r"<think>[\s\S]*?</think>", "", text, flags=re.IGNORECASE).strip()
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
    if isinstance(data, dict) and isinstance(data.get("rankings"), list):
        data = data["rankings"]
    if not isinstance(data, list):
        raise ValueError("model response must be a JSON array")
    return data


def _read_local_key() -> str | None:
    path = Path(os.environ.get("AGENT_API_KEY_FILE") or DEFAULT_LOCAL_KEY_PATH)
    if not path.exists():
        return None
    value = path.read_text(encoding="utf8").strip()
    return value or None


@dataclass
class OpenAICompatibleAgentClient:
    """Generic chat-completions client for action ranking agents."""

    api_key: str | None = None
    base_url: str | None = None
    model: str | None = None
    timeout: float | None = None
    max_completion_tokens: int = 1200

    def __post_init__(self) -> None:
        self.api_key = (
            self.api_key
            or os.environ.get("AGENT_API_KEY")
            or os.environ.get("DEEPSEEK_API_KEY")
            or os.environ.get("OPENAI_API_KEY")
            or _read_local_key()
        )
        self.base_url = (
            self.base_url
            or os.environ.get("AGENT_BASE_URL")
            or os.environ.get("DEEPSEEK_BASE_URL")
            or "https://new.53hk.cn"
        ).rstrip("/")
        self.model = self.model or os.environ.get("AGENT_MODEL") or os.environ.get("DEEPSEEK_MODEL") or "deepseek-chat"
        self.timeout = float(self.timeout or os.environ.get("AGENT_TIMEOUT") or 12.0)
        if not self.api_key:
            raise AgentClientError("AGENT_API_KEY or DEEPSEEK_API_KEY is not set")

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
            "max_tokens": self.max_completion_tokens,
            "response_format": {"type": "json_object"},
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You rank legal macro actions for a deterministic Magic Tower simulator. "
                        "Do not explain. Do not reveal reasoning. Do not use markdown. "
                        "Your entire response must be a JSON object with one key named rankings. "
                        "rankings must be an array. Each row must be "
                        '{"action_index": int, "score": number, "reason": string}. Return at most 5 rows.'
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
                            "guidance": self._role_guidance(role),
                            "output_contract": (
                                "Return only JSON like "
                                '{"rankings":[{"action_index":0,"score":10,"reason":"short reason"}]}. '
                                "No prose before or after."
                            ),
                        },
                        ensure_ascii=False,
                    ),
                },
            ],
        }
        request = urllib.request.Request(
            f"{self.base_url}/v1/chat/completions"
            if not self.base_url.endswith("/v1")
            else f"{self.base_url}/chat/completions",
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
            raise AgentClientError(f"OpenAI-compatible HTTP {exc.code}: {detail[:500]}") from exc
        except TimeoutError as exc:
            raise AgentClientError(f"OpenAI-compatible request timed out after {self.timeout}s") from exc
        except urllib.error.URLError as exc:
            raise AgentClientError(f"OpenAI-compatible request failed: {exc}") from exc

        try:
            data = json.loads(body)
            content = data["choices"][0]["message"]["content"]
            rows = _extract_json_array(content)
        except (KeyError, IndexError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise AgentClientError(f"Could not parse action ranking response: {body[:500]}") from exc

        valid_indices = {int(action["action_index"]) for action in actions if "action_index" in action}
        proposals: list[dict[str, Any]] = []
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
        if role == "planner":
            return (
                "Follow the goal chain: sword -> shield -> all stat gems -> healing potions before red key -> red key -> boss. "
                "All stat gems means every redGem and blueGem that raises ATK/DEF in the first ten floors. "
                "After all stat gems and before red key, prefer collecting reachable HP potions to create boss-route buffer. "
                "During shield stage, the target is the MT9 shield, not side gems. "
                "Do not spend the last yellow key on side resources before shield unless it directly unlocks shield progress."
            )
        if role == "critic":
            return "Reject moves that waste keys, take high damage before stat thresholds, or block red-key/boss access."
        if role == "explorer":
            return "Suggest non-obvious legal order changes that may improve ATK/DEF thresholds or preserve HP."
        return "Rank actions by expected route quality."
