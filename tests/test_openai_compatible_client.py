from __future__ import annotations

from mota_agentic.openai_compatible_client import _extract_json_array, _read_local_key


def test_openai_compatible_extract_json_array_from_text() -> None:
    rows = _extract_json_array(
        'Sure: [{"action_index": 1, "score": 8, "reason": "advance stage"}]'
    )
    assert rows == [{"action_index": 1, "score": 8, "reason": "advance stage"}]


def test_read_local_key_from_configured_file(tmp_path, monkeypatch) -> None:
    key_path = tmp_path / "key.txt"
    key_path.write_text("sk-test\n", encoding="utf8")
    monkeypatch.setenv("AGENT_API_KEY_FILE", str(key_path))
    assert _read_local_key() == "sk-test"
