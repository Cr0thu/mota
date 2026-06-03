from __future__ import annotations

import csv
import re
import shutil
import subprocess
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PAPER_DIR = ROOT / "paper"
TMP_DIR = ROOT / "tmp" / "pdfs"
OUTPUT_DIR = ROOT / "output" / "pdf"
SOURCE_MD = TMP_DIR / "mota_paper_reading_compendium_reader.md"
HEADER_TEX = TMP_DIR / "mota_reader_header.tex"
STYLE_CSS = TMP_DIR / "mota_reader.css"
OUTPUT_HTML = OUTPUT_DIR / "mota_paper_reading_compendium_reader.html"
FINAL_PDF = OUTPUT_DIR / "mota_paper_reading_compendium_reader.pdf"
PAPER_COPY = PAPER_DIR / "mota_paper_reading_compendium_reader.pdf"


def sanitize(text: str) -> str:
    replacements = {
        "\u2010": "-",
        "\u2011": "-",
        "\u2012": "-",
        "\u2013": "-",
        "\u2014": "--",
        "\ufeff": "",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text


def demote_tables(markdown: str) -> str:
    lines = markdown.splitlines()
    out: list[str] = []
    table: list[str] = []

    def flush() -> None:
        nonlocal table
        if not table:
            return
        rows: list[list[str]] = []
        for raw in table:
            stripped = raw.strip()
            if not stripped or re.fullmatch(r"\|?[\s:\-|]+\|?", stripped):
                continue
            cells = [cell.strip() for cell in stripped.strip("|").split("|")]
            if cells:
                rows.append(cells)
        if rows:
            headers = rows[0]
            for cells in rows[1:] if len(rows) > 1 else rows:
                parts = []
                for idx, cell in enumerate(cells):
                    label = headers[idx] if idx < len(headers) and len(rows) > 1 else ""
                    parts.append(f"{label}: {cell}" if label else cell)
                out.append("- " + "; ".join(parts))
            out.append("")
        table = []

    for line in lines:
        if line.lstrip().startswith("|"):
            table.append(line)
            continue
        flush()
        out.append(line)
    flush()
    return "\n".join(out)


def read_markdown(path: Path) -> str:
    if not path.exists():
        return f"> Missing file: `{path.relative_to(ROOT)}`\n"
    return demote_tables(sanitize(path.read_text(encoding="utf8", errors="replace")))


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf8", newline="") as handle:
        return list(csv.DictReader(handle))


def first_value(row: dict[str, str], keys: tuple[str, ...]) -> str:
    for key in keys:
        value = row.get(key, "").strip()
        if value:
            return sanitize(value)
    return ""


def manifest_section(
    path: Path,
    *,
    title: str,
    intro: str,
    connection_keys: tuple[str, ...],
    limit: int | None = None,
) -> str:
    rows = read_csv_rows(path)
    selected = rows[:limit] if limit else rows
    chunks = [f"# {title}\n", f"{intro}\n", f"Source: `{path.relative_to(ROOT)}`. Entries: {len(rows)}.\n"]
    for index, row in enumerate(selected, start=1):
        paper_id = first_value(row, ("id", "paper_id"))
        paper_title = first_value(row, ("title",)) or "Untitled"
        year = first_value(row, ("year",))
        topic = first_value(row, ("topic", "area", "category"))
        url = first_value(row, ("url", "pdf_url", "source"))
        connection = first_value(row, connection_keys)
        heading = f"## {index:03d}. {paper_title}"
        if year:
            heading += f" ({year})"
        chunks.append(heading + "\n")
        meta = []
        if paper_id:
            meta.append(f"ID: `{paper_id}`")
        if topic:
            meta.append(f"Topic: {topic}")
        if meta:
            chunks.append("**" + " | ".join(meta) + "**\n")
        if url:
            chunks.append(f"URL: <{url}>\n")
        if connection:
            chunks.append(connection + "\n")
    if limit and len(rows) > limit:
        chunks.append(f"\n_Only the first {limit} entries are included in this reader edition._\n")
    return "\n".join(chunks)


def build_source() -> str:
    today = datetime.now().strftime("%Y-%m-%d %H:%M")
    parts = [
        "---\n"
        "title: \"《魔塔》RL 与规划论文阅读合集\"\n"
        "subtitle: \"Reader Edition：Reward 设计、图状态 Q-learning、搜索规划、世界模型与分层探索\"\n"
        "author: \"Mota RL Project\"\n"
        f"date: \"{today}\"\n"
        "toc: true\n"
        "toc-depth: 2\n"
        "numbersections: true\n"
        "---\n",
        "\\newpage\n\n# 编译说明\n\n"
        "这是一份重新排版的 reader edition。它不是把原始论文 PDF 拼在一起，而是把项目内的阅读报告、十篇精读笔记和文献清单重新编译成适合连续阅读的讲义版。\n\n"
        "新版改动：正文使用 Pandoc/XeLaTeX 编译，保留目录和章节编号；长表格改成逐篇条目；页边距、行距和标题层级按阅读材料处理，而不是按数据报表处理。\n\n",
        "\\newpage\n\n" + read_markdown(PAPER_DIR / "top10_must_read.md"),
        "\\newpage\n\n" + read_markdown(PAPER_DIR / "deep_research_report.md"),
        "\\newpage\n\n" + read_markdown(PAPER_DIR / "reading_report.md"),
        "\\newpage\n\n" + read_markdown(PAPER_DIR / "factor_reward_reading_report.md"),
        "\\newpage\n\n# 十篇精读逐篇笔记\n\n",
    ]
    for note in sorted((PAPER_DIR / "top10_notes").glob("*.md")):
        if note.name == "00_index.md":
            continue
        parts.append("\\newpage\n\n" + read_markdown(note))
    parts.extend(
        [
            "\\newpage\n\n"
            + manifest_section(
                PAPER_DIR / "deep_research_manifest_200.csv",
                title="Deep Research 200 篇文献清单",
                intro="覆盖硬探索、width-based planning、policy-guided search、Sokoban、reward、图组合优化、offline/demo RL 与世界模型。",
                connection_keys=("mota_connection", "experiment_role", "notes"),
            ),
            "\\newpage\n\n"
            + manifest_section(
                PAPER_DIR / "paper_manifest.csv",
                title="RL/规划文献清单",
                intro="覆盖 AlphaZero/MuZero、Sokoban、NLE、Thinker、Searchformer、HRL 等方向。",
                connection_keys=("mota_relevance", "mota_connection", "notes"),
            ),
            "\\newpage\n\n"
            + manifest_section(
                PAPER_DIR / "factor_reward_paper_manifest_100.csv",
                title="Reward/因子挖掘文献清单",
                intro="覆盖势函数、奖励学习、回报分解、因子挖掘与 LLM 生成 reward。",
                connection_keys=("mota_factor_connection", "mota_connection", "notes", "url"),
            ),
        ]
    )
    return "\n\n".join(parts)


def write_css() -> None:
    STYLE_CSS.write_text(
        """
@page {
  size: A4;
  margin: 18mm 20mm 19mm 20mm;
}

html {
  color: #172033;
  background: #ffffff;
}

body {
  font-family: "Songti SC", "PingFang SC", "Hiragino Sans GB", "Microsoft YaHei", serif;
  font-size: 11.3pt;
  line-height: 1.68;
  letter-spacing: 0;
  max-width: 760px;
  margin: 0 auto;
}

.title {
  margin-top: 22mm;
  font-size: 28pt;
  line-height: 1.25;
  color: #0f172a;
  text-align: center;
}

.subtitle {
  margin: 8mm auto 5mm;
  max-width: 620px;
  color: #475569;
  font-size: 13pt;
  line-height: 1.55;
  text-align: center;
}

.author, .date {
  text-align: center;
  color: #64748b;
}

#TOC {
  page-break-before: always;
  page-break-after: always;
  border-top: 1px solid #cbd5e1;
  border-bottom: 1px solid #cbd5e1;
  padding: 8mm 0;
}

#TOC > ul {
  columns: 2;
  column-gap: 12mm;
}

#TOC ul {
  padding-left: 1.2em;
}

#TOC li {
  break-inside: avoid;
  margin: 0.18rem 0;
}

h1, h2, h3, h4 {
  font-family: "PingFang SC", "Songti SC", sans-serif;
  letter-spacing: 0;
  color: #0f172a;
  break-after: avoid;
}

h1 {
  page-break-before: always;
  margin: 0 0 0.7em;
  padding-bottom: 0.2em;
  border-bottom: 2px solid #dbeafe;
  font-size: 21pt;
  line-height: 1.35;
}

h1:first-of-type {
  page-break-before: auto;
}

h2 {
  margin-top: 1.25em;
  margin-bottom: 0.35em;
  color: #1d4ed8;
  font-size: 15.4pt;
  line-height: 1.42;
}

h3 {
  margin-top: 0.95em;
  margin-bottom: 0.25em;
  color: #334155;
  font-size: 12.6pt;
}

p {
  margin: 0.38em 0 0.7em;
  text-align: justify;
  overflow-wrap: anywhere;
}

ul, ol {
  margin: 0.2em 0 0.8em;
  padding-left: 1.55em;
}

li {
  margin: 0.22em 0;
  break-inside: avoid;
}

a {
  color: #1d4ed8;
  text-decoration: none;
  overflow-wrap: anywhere;
  word-break: break-word;
}

blockquote {
  margin: 0.85em 0;
  padding: 0.55em 0.9em;
  border-left: 3px solid #93c5fd;
  background: #f8fafc;
  color: #334155;
}

code {
  font-family: Menlo, Monaco, Consolas, monospace;
  font-size: 0.88em;
  background: #f1f5f9;
  padding: 0.06em 0.25em;
  border-radius: 3px;
}

pre {
  white-space: pre-wrap;
  background: #0f172a;
  color: #e2e8f0;
  padding: 0.8em;
  border-radius: 6px;
  line-height: 1.45;
  break-inside: avoid;
}

table {
  width: 100%;
  border-collapse: collapse;
  font-size: 9.5pt;
}

th, td {
  border-bottom: 1px solid #e2e8f0;
  padding: 0.35em 0.45em;
  vertical-align: top;
}

hr {
  border: 0;
  border-top: 1px solid #cbd5e1;
  margin: 1.2em 0;
}
""".strip()
        + "\n",
        encoding="utf8",
    )


def build_pdf() -> Path:
    if shutil.which("pandoc") is None:
        raise SystemExit("pandoc is required to build the reader PDF.")
    chrome = Path("/Applications/Google Chrome.app/Contents/MacOS/Google Chrome")
    if not chrome.exists():
        raise SystemExit("Google Chrome is required for HTML-to-PDF printing.")

    TMP_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    SOURCE_MD.write_text(build_source(), encoding="utf8")
    write_css()

    html_cmd = [
        "pandoc",
        str(SOURCE_MD),
        "-o",
        str(OUTPUT_HTML),
        "--standalone",
        "--embed-resources",
        "--toc",
        "--number-sections",
        "--css",
        str(STYLE_CSS),
        "--metadata",
        "lang=zh-CN",
    ]
    subprocess.run(html_cmd, cwd=ROOT, check=True)
    pdf_cmd = [
        str(chrome),
        "--headless=new",
        "--disable-gpu",
        "--no-sandbox",
        "--no-pdf-header-footer",
        f"--print-to-pdf={FINAL_PDF}",
        OUTPUT_HTML.as_uri(),
    ]
    subprocess.run(pdf_cmd, cwd=ROOT, check=True)
    PAPER_COPY.write_bytes(FINAL_PDF.read_bytes())
    return FINAL_PDF


if __name__ == "__main__":
    print(build_pdf())
