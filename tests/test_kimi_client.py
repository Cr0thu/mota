from __future__ import annotations

from mota_agentic.kimi_client import _extract_json_array


def test_extract_json_array_accepts_fenced_response() -> None:
    rows = _extract_json_array(
        """```json
        [{"action_index": 2, "score": 9.5, "reason": "take shield"}]
        ```"""
    )
    assert rows == [{"action_index": 2, "score": 9.5, "reason": "take shield"}]
