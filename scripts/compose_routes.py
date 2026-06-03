from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf8").splitlines() if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("routes", nargs="+")
    args = parser.parse_args()

    rows: list[dict[str, Any]] = []
    for route in args.routes:
        for row in load_jsonl(Path(route)):
            copied = dict(row)
            copied["index"] = len(rows)
            rows.append(copied)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf8",
    )
    print(json.dumps({"out": str(out), "rows": len(rows)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
