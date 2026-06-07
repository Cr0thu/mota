from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol


class AgentClientError(RuntimeError):
    """Raised when a configured external agent backend cannot answer."""


class AgentClient(Protocol):
    """Minimal interface for future LLM/agent API backends.

    The orchestrator passes a compact state/action payload to each role.  A real
    backend should return either action labels or action indices with scores.
    Until an API is provided, the local heuristic agents keep the loop runnable.
    """

    def rank_actions(
        self,
        *,
        role: str,
        task: str,
        state: dict[str, Any],
        actions: list[dict[str, Any]],
        memory: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Return ranked action proposals.

        Expected row shape:
        {"action_index": int, "score": float, "reason": str}
        """


@dataclass
class PlaceholderAgentClient:
    """Empty API hook to be replaced once credentials/endpoints are known."""

    backend_name: str = "placeholder"

    def rank_actions(
        self,
        *,
        role: str,
        task: str,
        state: dict[str, Any],
        actions: list[dict[str, Any]],
        memory: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        raise AgentClientError(
            f"{self.backend_name} has no configured API for role {role!r}; "
            "use local heuristic agents or provide an AgentClient implementation."
        )
