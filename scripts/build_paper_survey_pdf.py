from __future__ import annotations

import csv
import re
from datetime import datetime
from pathlib import Path
from typing import Iterable

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (
    BaseDocTemplate,
    Frame,
    KeepTogether,
    PageBreak,
    PageTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
)
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.cidfonts import UnicodeCIDFont


ROOT = Path(__file__).resolve().parents[1]
PAPER_DIR = ROOT / "paper"
OUTPUT_DIR = ROOT / "output" / "pdf"
FINAL_PDF = OUTPUT_DIR / "mota_paper_reading_compendium.pdf"
PAPER_COPY = PAPER_DIR / "mota_paper_reading_compendium.pdf"


def register_fonts() -> tuple[str, str]:
    font_name = "STSong-Light"
    pdfmetrics.registerFont(UnicodeCIDFont(font_name))
    return font_name, font_name


def make_styles(font_name: str, bold_font: str):
    styles = getSampleStyleSheet()
    styles.add(
        ParagraphStyle(
            "CoverTitle",
            parent=styles["Title"],
            fontName=bold_font,
            fontSize=24,
            leading=34,
            alignment=TA_CENTER,
            spaceAfter=18,
            textColor=colors.HexColor("#111827"),
            wordWrap="CJK",
        )
    )
    styles.add(
        ParagraphStyle(
            "CoverSub",
            parent=styles["Normal"],
            fontName=font_name,
            fontSize=11.5,
            leading=18,
            alignment=TA_CENTER,
            textColor=colors.HexColor("#475569"),
            wordWrap="CJK",
        )
    )
    styles.add(
        ParagraphStyle(
            "SectionTitle",
            parent=styles["Heading1"],
            fontName=bold_font,
            fontSize=19,
            leading=27,
            spaceAfter=7,
            textColor=colors.white,
            wordWrap="CJK",
        )
    )
    styles.add(
        ParagraphStyle(
            "SectionSub",
            parent=styles["BodyText"],
            fontName=font_name,
            fontSize=10,
            leading=15,
            textColor=colors.HexColor("#e2e8f0"),
            wordWrap="CJK",
        )
    )
    styles.add(
        ParagraphStyle(
            "H1CJK",
            parent=styles["Heading1"],
            fontName=bold_font,
            fontSize=16,
            leading=23,
            spaceBefore=12,
            spaceAfter=8,
            textColor=colors.HexColor("#0f172a"),
            wordWrap="CJK",
        )
    )
    styles.add(
        ParagraphStyle(
            "H2CJK",
            parent=styles["Heading2"],
            fontName=bold_font,
            fontSize=13,
            leading=19,
            spaceBefore=10,
            spaceAfter=5,
            textColor=colors.HexColor("#1e293b"),
            wordWrap="CJK",
        )
    )
    styles.add(
        ParagraphStyle(
            "H3CJK",
            parent=styles["Heading3"],
            fontName=bold_font,
            fontSize=11.2,
            leading=16.5,
            spaceBefore=6,
            spaceAfter=3,
            textColor=colors.HexColor("#334155"),
            wordWrap="CJK",
        )
    )
    styles.add(
        ParagraphStyle(
            "BodyCJK",
            parent=styles["BodyText"],
            fontName=font_name,
            fontSize=10.1,
            leading=16.4,
            spaceAfter=5.2,
            alignment=TA_LEFT,
            textColor=colors.HexColor("#111827"),
            wordWrap="CJK",
        )
    )
    styles.add(
        ParagraphStyle(
            "BulletCJK",
            parent=styles["BodyText"],
            fontName=font_name,
            fontSize=9.8,
            leading=15.6,
            leftIndent=15,
            firstLineIndent=-9,
            spaceAfter=3.8,
            textColor=colors.HexColor("#1f2937"),
            wordWrap="CJK",
        )
    )
    styles.add(
        ParagraphStyle(
            "SmallCJK",
            parent=styles["BodyText"],
            fontName=font_name,
            fontSize=8.3,
            leading=12.2,
            spaceAfter=3,
            textColor=colors.HexColor("#475569"),
            wordWrap="CJK",
        )
    )
    styles.add(
        ParagraphStyle(
            "NoteTitle",
            parent=styles["Heading2"],
            fontName=bold_font,
            fontSize=12.2,
            leading=17,
            spaceBefore=5,
            spaceAfter=3,
            textColor=colors.HexColor("#0f172a"),
            wordWrap="CJK",
        )
    )
    styles.add(
        ParagraphStyle(
            "ManifestTitle",
            parent=styles["Heading3"],
            fontName=bold_font,
            fontSize=10.8,
            leading=15.2,
            spaceBefore=6,
            spaceAfter=2,
            textColor=colors.HexColor("#0f172a"),
            wordWrap="CJK",
        )
    )
    styles.add(
        ParagraphStyle(
            "ManifestMeta",
            parent=styles["BodyText"],
            fontName=font_name,
            fontSize=8.2,
            leading=11.8,
            spaceAfter=2,
            textColor=colors.HexColor("#64748b"),
            wordWrap="CJK",
        )
    )
    styles.add(
        ParagraphStyle(
            "ManifestBody",
            parent=styles["BodyText"],
            fontName=font_name,
            fontSize=9.2,
            leading=14,
            spaceAfter=4,
            textColor=colors.HexColor("#1f2937"),
            wordWrap="CJK",
        )
    )
    return styles


def soft_wrap_urls(value: str) -> str:
    def repl(match: re.Match[str]) -> str:
        url = match.group(0)
        return (
            url.replace("/", "/ ")
            .replace("?", "? ")
            .replace("&", "& ")
            .replace("=", "= ")
            .replace("-", "- ")
            .replace("_", "_ ")
        )

    return re.sub(r"https?://[^\s)]+", repl, value)


def esc(text: object) -> str:
    value = "" if text is None else str(text)
    value = re.sub(r"\[(.*?)\]\((.*?)\)", r"\1 (\2)", value)
    value = soft_wrap_urls(value)
    value = value.replace("`", "")
    value = value.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    value = re.sub(r"\*\*(.*?)\*\*", r"<b>\1</b>", value)
    return value


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf8", newline="") as handle:
        return list(csv.DictReader(handle))


def paragraph(text: object, style) -> Paragraph:
    return Paragraph(esc(text), style)


def section_page(story: list, styles, title: str, subtitle: str) -> None:
    story.append(PageBreak())
    table = Table(
        [
            [
                [
                    paragraph(title, styles["SectionTitle"]),
                    paragraph(subtitle, styles["SectionSub"]),
                ]
            ]
        ],
        colWidths=[165 * mm],
    )
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#1e293b")),
                ("LEFTPADDING", (0, 0), (-1, -1), 13 * mm),
                ("RIGHTPADDING", (0, 0), (-1, -1), 13 * mm),
                ("TOPPADDING", (0, 0), (-1, -1), 15 * mm),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 15 * mm),
                ("BOX", (0, 0), (-1, -1), 0.8, colors.HexColor("#0f172a")),
            ]
        )
    )
    story.append(Spacer(1, 62 * mm))
    story.append(table)
    story.append(PageBreak())


def add_markdown(story: list, path: Path, styles, *, title: str | None = None) -> None:
    if title:
        story.append(paragraph(title, styles["H1CJK"]))
    if not path.exists():
        story.append(paragraph(f"缺失文件：{path.relative_to(ROOT)}", styles["BodyCJK"]))
        return

    lines = path.read_text(encoding="utf8", errors="replace").splitlines()
    pending: list[str] = []
    table_lines: list[str] = []

    def flush_pending() -> None:
        if pending:
            text = " ".join(item.strip() for item in pending if item.strip())
            if text:
                story.append(paragraph(text, styles["BodyCJK"]))
            pending.clear()

    def flush_table() -> None:
        if not table_lines:
            return
        rows = []
        for raw in table_lines:
            clean = raw.strip().strip("|")
            if not clean or re.fullmatch(r"[:\-\s|]+", raw.strip()):
                continue
            cells = [cell.strip() for cell in clean.split("|")]
            if cells:
                rows.append(" / ".join(cells))
        if rows:
            story.append(Spacer(1, 1.5 * mm))
            for row in rows:
                story.append(paragraph("- " + row, styles["SmallCJK"]))
            story.append(Spacer(1, 1.5 * mm))
        table_lines.clear()

    for raw in lines:
        line = raw.rstrip()
        if line.startswith("|"):
            flush_pending()
            table_lines.append(line)
            continue
        flush_table()
        if not line.strip():
            flush_pending()
            continue
        if line.startswith("# "):
            flush_pending()
            story.append(paragraph(line[2:].strip(), styles["H1CJK"]))
        elif line.startswith("## "):
            flush_pending()
            story.append(paragraph(line[3:].strip(), styles["H2CJK"]))
        elif line.startswith("### "):
            flush_pending()
            story.append(paragraph(line[4:].strip(), styles["H3CJK"]))
        elif line.startswith("- ") or re.match(r"^\d+\.\s+", line):
            flush_pending()
            clean = re.sub(r"^\d+\.\s+", "", line)
            clean = clean[2:] if clean.startswith("- ") else clean
            story.append(paragraph("- " + clean, styles["BulletCJK"]))
        else:
            pending.append(line)
    flush_table()
    flush_pending()


def add_toc(story: list, styles) -> None:
    story.append(paragraph("内容目录", styles["H1CJK"]))
    entries = [
        ("1", "编译说明与使用方式"),
        ("2", "十篇最应该读的论文"),
        ("3", "RL 与规划阅读报告"),
        ("4", "Reward 与量化因子挖掘阅读报告"),
        ("5", "十篇精读逐篇笔记"),
        ("6", "RL/规划文献清单"),
        ("7", "Reward/因子挖掘文献清单"),
    ]
    for number, title in entries:
        story.append(paragraph(f"{number}. {title}", styles["BulletCJK"]))
    story.append(Spacer(1, 6 * mm))
    story.append(
        paragraph(
            "新版排版把长表格改成逐篇条目，优先保证可读性。原始 PDF 文件没有合并进本文档，仍保存在 paper/top10_pdfs/、paper/pdfs/ 与 paper/pdfs/factor_reward/。",
            styles["BodyCJK"],
        )
    )


def manifest_value(row: dict[str, str], keys: Iterable[str]) -> str:
    for key in keys:
        value = row.get(key, "").strip()
        if value:
            return value
    return ""


def add_manifest_entries(
    story: list,
    path: Path,
    styles,
    *,
    title: str,
    subtitle: str,
    connection_keys: tuple[str, ...],
) -> None:
    rows = read_csv(path)
    story.append(paragraph(title, styles["H1CJK"]))
    story.append(paragraph(f"{subtitle} 来源：{path.relative_to(ROOT)}；共 {len(rows)} 条。", styles["SmallCJK"]))
    story.append(Spacer(1, 3 * mm))

    for index, row in enumerate(rows, start=1):
        paper_id = manifest_value(row, ("id", "paper_id"))
        paper_title = manifest_value(row, ("title",))
        year = manifest_value(row, ("year",))
        topic = manifest_value(row, ("topic", "area", "category"))
        url = manifest_value(row, ("url", "pdf_url", "source"))
        connection = manifest_value(row, connection_keys)

        heading_bits = [f"{index:03d}"]
        if paper_id:
            heading_bits.append(paper_id)
        heading = " | ".join(heading_bits)
        if paper_title:
            heading = f"{heading}. {paper_title}"
        if year:
            heading = f"{heading} ({year})"

        block = [
            paragraph(heading, styles["ManifestTitle"]),
            paragraph(f"Topic/Area: {topic or 'N/A'}", styles["ManifestMeta"]),
        ]
        if url:
            block.append(paragraph(f"URL: {url}", styles["ManifestMeta"]))
        if connection:
            block.append(paragraph(connection, styles["ManifestBody"]))
        story.append(KeepTogether(block[:2]))
        for item in block[2:]:
            story.append(item)
        story.append(Spacer(1, 1.7 * mm))


def add_page_number(canvas, doc):
    canvas.saveState()
    canvas.setStrokeColor(colors.HexColor("#cbd5e1"))
    canvas.setLineWidth(0.4)
    canvas.line(20 * mm, 14 * mm, A4[0] - 20 * mm, 14 * mm)
    canvas.setFont("STSong-Light", 8)
    canvas.setFillColor(colors.HexColor("#64748b"))
    canvas.drawString(20 * mm, 9 * mm, "Mota RL Paper Reading Compendium")
    canvas.drawRightString(A4[0] - 20 * mm, 9 * mm, f"{doc.page}")
    canvas.restoreState()


def build_pdf() -> Path:
    font_name, bold_font = register_fonts()
    styles = make_styles(font_name, bold_font)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    doc = BaseDocTemplate(
        str(FINAL_PDF),
        pagesize=A4,
        rightMargin=20 * mm,
        leftMargin=20 * mm,
        topMargin=18 * mm,
        bottomMargin=19 * mm,
        title="Mota RL Paper Reading Compendium",
        author="Mota RL Project",
    )
    frame = Frame(doc.leftMargin, doc.bottomMargin, doc.width, doc.height, id="normal")
    doc.addPageTemplates([PageTemplate(id="main", frames=[frame], onPage=add_page_number)])

    story: list = []
    story.append(Spacer(1, 35 * mm))
    story.append(paragraph("《魔塔》RL 与规划论文阅读合集", styles["CoverTitle"]))
    story.append(
        paragraph(
            "面向前十层无专家数据 RL：Reward 设计、图状态 Q-learning、搜索规划、世界模型与分层探索",
            styles["CoverSub"],
        )
    )
    story.append(Spacer(1, 5 * mm))
    story.append(paragraph(f"生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M')}", styles["CoverSub"]))
    story.append(PageBreak())

    add_toc(story, styles)

    section_page(story, styles, "1. 编译说明", "如何使用这份阅读合集，以及它与原始论文 PDF 的关系。")
    story.append(paragraph("编译说明", styles["H1CJK"]))
    story.append(
        paragraph(
            "本 PDF 是阅读笔记、文献清单和工程启示的统一阅读版，不是原始论文 PDF 的简单拼接。"
            "它适合组内先快速建立共同语境，再回到原文精读关键章节。",
            styles["BodyCJK"],
        )
    )
    story.append(
        paragraph(
            "为了提高可读性，新版不再使用小字号横向大表格。所有文献清单均按逐篇条目展示；长 URL 会自动断行。",
            styles["BodyCJK"],
        )
    )

    section_page(story, styles, "2. 十篇最应该读的论文", "Reward、搜索、世界模型和机制可解释性四条线的核心入口。")
    add_markdown(story, PAPER_DIR / "top10_must_read.md", styles)

    section_page(story, styles, "3. RL 与规划阅读报告", "从 DQN/MuZero/Thinker/NetHack/Sokoban 到《魔塔》工程路线。")
    add_markdown(story, PAPER_DIR / "reading_report.md", styles)

    section_page(story, styles, "4. Reward 与因子挖掘", "把动态 reward 设计和量化因子挖掘联系起来。")
    add_markdown(story, PAPER_DIR / "factor_reward_reading_report.md", styles)

    section_page(story, styles, "5. 十篇精读逐篇笔记", "每篇按问题、方法、对魔塔启示和工程取舍整理。")
    for note in sorted((PAPER_DIR / "top10_notes").glob("*.md")):
        if note.name == "00_index.md":
            continue
        story.append(paragraph(note.stem.replace("_", " "), styles["NoteTitle"]))
        add_markdown(story, note, styles)
        story.append(Spacer(1, 4 * mm))

    section_page(story, styles, "6. RL/规划文献清单", "不再用拥挤表格，改为逐篇条目。")
    add_manifest_entries(
        story,
        PAPER_DIR / "paper_manifest.csv",
        styles,
        title="RL/规划文献清单",
        subtitle="覆盖 AlphaZero/MuZero、Sokoban、NLE、Thinker、Searchformer、HRL 等方向。",
        connection_keys=("mota_relevance", "mota_connection", "notes"),
    )

    section_page(story, styles, "7. Reward/因子挖掘文献清单", "Reward learning、PBRS、IRL、LLM reward 和 alpha mining 的参考。")
    add_manifest_entries(
        story,
        PAPER_DIR / "factor_reward_paper_manifest_100.csv",
        styles,
        title="Reward/因子挖掘文献清单",
        subtitle="覆盖势函数、奖励学习、回报分解、因子挖掘与 LLM 生成 reward。",
        connection_keys=("mota_factor_connection", "mota_connection", "notes", "url"),
    )

    doc.build(story)
    PAPER_COPY.write_bytes(FINAL_PDF.read_bytes())
    return FINAL_PDF


if __name__ == "__main__":
    print(build_pdf())
