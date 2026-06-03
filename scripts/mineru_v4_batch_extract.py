from __future__ import annotations

import argparse
import json
import os
import time
import zipfile
from pathlib import Path

import requests


API_ROOT = "https://mineru.net/api/v4"
MINERU_ROOT = Path("/Users/cr0/MinerU")


def request_upload_urls(token: str, pdfs: list[Path], out_dir: Path) -> str:
    payload = {
        "files": [
            {"name": pdf.name, "data_id": pdf.stem, "is_ocr": False}
            for pdf in pdfs
        ],
        "model_version": "vlm",
        "enable_formula": True,
        "enable_table": True,
        "language": "en",
    }
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
    }
    response = requests.post(
        f"{API_ROOT}/file-urls/batch", headers=headers, json=payload, timeout=60
    )
    response.raise_for_status()
    result = response.json()
    (out_dir / "mineru-submit-response.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    if result.get("code") != 0:
        raise RuntimeError(f"MinerU upload URL request failed: {result}")

    batch_id = result["data"]["batch_id"]
    upload_urls = result["data"]["file_urls"]
    if len(upload_urls) != len(pdfs):
        raise RuntimeError(
            f"MinerU returned {len(upload_urls)} URLs for {len(pdfs)} PDFs"
        )

    for pdf, upload_url in zip(pdfs, upload_urls, strict=True):
        with pdf.open("rb") as handle:
            upload = requests.put(upload_url, data=handle, timeout=300)
        if upload.status_code != 200:
            raise RuntimeError(f"Upload failed for {pdf}: HTTP {upload.status_code}")
        print(f"uploaded {pdf.name}")

    (out_dir / "mineru-batch-id.txt").write_text(batch_id, encoding="utf-8")
    return batch_id


def poll_results(token: str, batch_id: str, out_dir: Path, timeout: int) -> dict:
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
    }
    deadline = time.time() + timeout
    last_result: dict | None = None
    while time.time() < deadline:
        response = requests.get(
            f"{API_ROOT}/extract-results/batch/{batch_id}",
            headers=headers,
            timeout=60,
        )
        response.raise_for_status()
        result = response.json()
        last_result = result
        (out_dir / "mineru-v4-result.json").write_text(
            json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        if result.get("code") != 0:
            raise RuntimeError(f"MinerU poll failed: {result}")

        rows = result.get("data", {}).get("extract_result", [])
        states: dict[str, int] = {}
        for row in rows:
            states[row.get("state", "unknown")] = states.get(row.get("state", "unknown"), 0) + 1
        print(f"states: {states}")

        if rows and all(row.get("state") in {"done", "failed"} for row in rows):
            return result
        time.sleep(30)

    raise TimeoutError(f"Timed out waiting for MinerU batch {batch_id}: {last_result}")


def download_outputs(result: dict, out_dir: Path, batch_id: str) -> None:
    rows = result.get("data", {}).get("extract_result", [])
    failures = []
    for row in rows:
        file_name = row.get("file_name", "unknown.pdf")
        stem = Path(file_name).stem
        if row.get("state") != "done":
            failures.append(row)
            continue
        zip_url = row["full_zip_url"]
        target_dir = MINERU_ROOT / f"{stem}-{batch_id}"
        target_dir.mkdir(parents=True, exist_ok=True)
        zip_path = target_dir / "result.zip"
        response = requests.get(zip_url, timeout=300)
        response.raise_for_status()
        zip_path.write_bytes(response.content)
        with zipfile.ZipFile(zip_path) as archive:
            archive.extractall(target_dir)
        (target_dir / "v4-result.json").write_text(
            json.dumps(row, ensure_ascii=False, indent=2), encoding="utf-8"
        )

        full_md_candidates = list(target_dir.rglob("full.md"))
        if not full_md_candidates:
            print(f"warning: no full.md found for {file_name}")
            continue
        full_md = full_md_candidates[0]
        extracted_target = out_dir / f"{stem}.md"
        extracted_target.write_text(full_md.read_text(encoding="utf-8"), encoding="utf-8")
        print(f"extracted {file_name} -> {extracted_target}")

    if failures:
        (out_dir / "mineru-failures.json").write_text(
            json.dumps(failures, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        raise RuntimeError(f"{len(failures)} MinerU tasks failed")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pdf-dir", default="paper/top10_pdfs")
    parser.add_argument("--out-dir", default="paper/top10_extracted")
    parser.add_argument("--timeout", type=int, default=7200)
    parser.add_argument("--batch-id")
    args = parser.parse_args()

    token = os.environ.get("MINERU_API_TOKEN")
    if not token:
        raise RuntimeError("MINERU_API_TOKEN is not set")

    pdf_dir = Path(args.pdf_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pdfs = sorted(pdf_dir.glob("*.pdf"))
    if not pdfs:
        raise RuntimeError(f"No PDFs found under {pdf_dir}")

    batch_id = args.batch_id
    if not batch_id:
        saved = out_dir / "mineru-batch-id.txt"
        if saved.exists():
            batch_id = saved.read_text(encoding="utf-8").strip()
            print(f"resuming batch {batch_id}")
        else:
            batch_id = request_upload_urls(token, pdfs, out_dir)
            print(f"submitted batch {batch_id}")

    result = poll_results(token, batch_id, out_dir, args.timeout)
    download_outputs(result, out_dir, batch_id)


if __name__ == "__main__":
    main()
