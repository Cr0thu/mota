from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


SHOP_KINDS = {"shop", "fly_shop"}
FLY_KINDS = {"fly", "fly_shop"}
MERCHANT_LABELS = ("buy blueKey", "buy 5 yellowKey")


def load_route(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def validate_route(
    path: Path,
    forbid_shop: bool = True,
    forbid_fly: bool = True,
    forbid_merchants: bool = False,
) -> dict[str, Any]:
    violations: list[dict[str, Any]] = []
    for index, row in enumerate(load_route(path)):
        action = row.get("action", {})
        kind = action.get("kind", "")
        label = action.get("label", "")

        if forbid_shop and (
            kind in SHOP_KINDS
            or label.startswith("shop ")
            or label.startswith("fly shop ")
        ):
            violations.append(
                {
                    "index": index,
                    "type": "shop",
                    "label": label,
                    "kind": kind,
                }
            )

        if forbid_fly and (
            kind in FLY_KINDS
            or label.startswith("fly ")
        ):
            violations.append(
                {
                    "index": index,
                    "type": "fly",
                    "label": label,
                    "kind": kind,
                }
            )

        if forbid_merchants and any(token in label for token in MERCHANT_LABELS):
            violations.append(
                {
                    "index": index,
                    "type": "merchant",
                    "label": label,
                    "kind": kind,
                }
            )

    return {
        "route": str(path),
        "ok": not violations,
        "forbid_shop": forbid_shop,
        "forbid_fly": forbid_fly,
        "forbid_merchants": forbid_merchants,
        "violations": violations,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("routes", nargs="+")
    parser.add_argument("--allow-shop", action="store_true")
    parser.add_argument("--allow-fly", action="store_true")
    parser.add_argument(
        "--forbid-merchants",
        action="store_true",
        help="Also forbid the one-time MT6/MT7 key merchants.",
    )
    args = parser.parse_args()

    results = [
        validate_route(
            Path(route),
            forbid_shop=not args.allow_shop,
            forbid_fly=not args.allow_fly,
            forbid_merchants=args.forbid_merchants,
        )
        for route in args.routes
    ]
    payload = {
        "ok": all(result["ok"] for result in results),
        "results": results,
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    if not payload["ok"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
