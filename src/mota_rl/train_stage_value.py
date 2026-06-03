from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from mota_rl.attention_model import build_attention_model, tokenize_feature_snapshot


def load_rows(path: str | Path) -> list[dict[str, Any]]:
    rows = []
    with Path(path).open(encoding="utf8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    if not rows:
        raise SystemExit(f"No dataset rows found in {path}")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="artifacts/stage_dataset/staged_route.jsonl")
    parser.add_argument("--out-dir", default="artifacts/runs/stage_value_attention")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--max-tokens", type=int, default=160)
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda", "mps"))
    args = parser.parse_args()

    try:
        import torch
        from torch import nn
        from torch.utils.data import DataLoader, Dataset
    except Exception as exc:
        raise SystemExit(
            "Stage value training requires torch. Install optional RL deps first:\n"
            "  pip install -e '.[rl]'\n"
            f"Import error: {exc}"
        ) from exc

    rows = load_rows(args.dataset)

    class StageDataset(Dataset):
        def __len__(self) -> int:
            return len(rows)

        def __getitem__(self, index: int):
            row = rows[index]
            tokenized = tokenize_feature_snapshot(row["after_features"], max_tokens=args.max_tokens)
            target = row.get("target", {})
            return {
                "token_types": torch.tensor(tokenized.token_types, dtype=torch.long),
                "values": torch.tensor(tokenized.values, dtype=torch.float32),
                "mask": torch.tensor(tokenized.mask, dtype=torch.bool),
                "success": torch.tensor(float(target.get("solved_final", False)), dtype=torch.float32),
                "stage_value": torch.tensor(
                    float(target.get("stage_value_after", 0.0)) / 500.0,
                    dtype=torch.float32,
                ),
                "boss_margin": torch.tensor(
                    float(target.get("boss_margin_after", -2000.0)) / 2000.0,
                    dtype=torch.float32,
                ),
            }

    loader = DataLoader(StageDataset(), batch_size=args.batch_size, shuffle=True)
    if args.device == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(args.device)
    model = build_attention_model(args.d_model, args.heads, args.layers).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    bce = nn.BCEWithLogitsLoss()
    mse = nn.MSELoss()
    history: list[dict[str, float]] = []

    for epoch in range(args.epochs):
        total_loss = 0.0
        total_rows = 0
        for batch in loader:
            batch = {key: value.to(device) for key, value in batch.items()}
            out = model(batch["token_types"], batch["values"], batch["mask"])
            loss = (
                bce(out["success_logit"], batch["success"])
                + mse(out["stage_value"], batch["stage_value"])
                + mse(out["boss_margin"], batch["boss_margin"])
            )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += float(loss.detach()) * batch["token_types"].shape[0]
            total_rows += batch["token_types"].shape[0]
        row = {"epoch": epoch, "loss": total_loss / max(1, total_rows), "device": str(device)}
        history.append(row)
        print(json.dumps(row))

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": {
                "d_model": args.d_model,
                "heads": args.heads,
                "layers": args.layers,
                "max_tokens": args.max_tokens,
                "device": str(device),
            },
        },
        out_dir / "model.pt",
    )
    (out_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf8")
    print(json.dumps({"out_dir": str(out_dir), "rows": len(rows)}, indent=2))


if __name__ == "__main__":
    main()
