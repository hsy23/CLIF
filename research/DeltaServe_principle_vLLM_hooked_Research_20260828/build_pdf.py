from __future__ import annotations

import html
import re
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)


ROOT = Path(__file__).resolve().parents[2]
SOURCE = Path(__file__).with_name("report.md")
OUTPUT = ROOT / "output" / "pdf" / "deltaserve_vllm_principle_report_20260828.pdf"
NAVY = colors.HexColor("#003D5C")
LIGHT = colors.HexColor("#F3F6F8")


def inline_markup(text: str) -> str:
    text = html.escape(text)
    text = re.sub(r"`([^`]+)`", r"<font name='Courier'>\1</font>", text)
    text = re.sub(r"\*\*([^*]+)\*\*", r"<b>\1</b>", text)
    text = re.sub(r"\[([^]]+)\]\(([^)]+)\)", r"<a href='\2' color='#003D5C'>\1</a>", text)
    return text


def footer(canvas, doc) -> None:
    canvas.saveState()
    canvas.setStrokeColor(colors.HexColor("#D1D5DB"))
    canvas.line(18 * mm, 14 * mm, A4[0] - 18 * mm, 14 * mm)
    canvas.setFont("Deng", 8)
    canvas.setFillColor(colors.HexColor("#4A5568"))
    canvas.drawString(18 * mm, 9 * mm, "DeltaServe-style vLLM 单副本原型评估")
    canvas.drawRightString(A4[0] - 18 * mm, 9 * mm, f"{doc.page}")
    canvas.restoreState()


def build() -> None:
    pdfmetrics.registerFont(TTFont("Deng", r"C:\Windows\Fonts\Deng.ttf"))
    pdfmetrics.registerFont(TTFont("DengBold", r"C:\Windows\Fonts\Dengb.ttf"))
    pdfmetrics.registerFontFamily("Deng", normal="Deng", bold="DengBold")
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    styles = getSampleStyleSheet()
    body = ParagraphStyle(
        "BodyCN",
        parent=styles["BodyText"],
        fontName="Deng",
        fontSize=9.5,
        leading=15,
        textColor=colors.HexColor("#1A1A1A"),
        alignment=TA_LEFT,
        spaceAfter=6,
        splitLongWords=False,
    )
    h1 = ParagraphStyle(
        "TitleCN",
        parent=body,
        fontSize=22,
        leading=30,
        textColor=NAVY,
        alignment=TA_CENTER,
        spaceAfter=14,
    )
    h2 = ParagraphStyle(
        "H2CN",
        parent=body,
        fontSize=15,
        leading=21,
        textColor=NAVY,
        spaceBefore=12,
        spaceAfter=8,
        keepWithNext=True,
    )
    h3 = ParagraphStyle(
        "H3CN",
        parent=body,
        fontSize=11.5,
        leading=17,
        textColor=colors.HexColor("#243746"),
        spaceBefore=9,
        spaceAfter=5,
        keepWithNext=True,
    )
    bib = ParagraphStyle(
        "BibCN",
        parent=body,
        fontSize=8.2,
        leading=12,
        leftIndent=8 * mm,
        firstLineIndent=-8 * mm,
        spaceAfter=5,
    )
    table_style = TableStyle(
        [
            ("BACKGROUND", (0, 0), (-1, 0), NAVY),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTNAME", (0, 0), (-1, -1), "Deng"),
            ("FONTSIZE", (0, 0), (-1, -1), 7.5),
            ("LEADING", (0, 0), (-1, -1), 10),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#D1D5DB")),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, LIGHT]),
            ("LEFTPADDING", (0, 0), (-1, -1), 5),
            ("RIGHTPADDING", (0, 0), (-1, -1), 5),
            ("TOPPADDING", (0, 0), (-1, -1), 4),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ]
    )

    story = []
    lines = SOURCE.read_text(encoding="utf-8").splitlines()
    story.append(Spacer(1, 30 * mm))
    story.append(Paragraph(inline_markup(lines[0].removeprefix("# ")), h1))
    story.append(Paragraph("WSL + RTX 5070 Ti 实测 / 2026-08-28", ParagraphStyle(
        "Subtitle", parent=body, alignment=TA_CENTER, textColor=colors.HexColor("#4A5568"), fontSize=10.5
    )))
    story.append(Spacer(1, 12 * mm))
    metrics = [
        ["6/6", "2", "16", "LM head"],
        ["功能不变量", "GPU 进程", "研究来源", "当前 LoRA 范围"],
    ]
    metric_table = Table(metrics, colWidths=[42 * mm] * 4)
    metric_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), LIGHT),
        ("TEXTCOLOR", (0, 0), (-1, 0), NAVY),
        ("FONTNAME", (0, 0), (-1, -1), "Deng"),
        ("FONTSIZE", (0, 0), (-1, 0), 16),
        ("FONTSIZE", (0, 1), (-1, 1), 8),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E1")),
        ("TOPPADDING", (0, 0), (-1, 0), 10),
        ("BOTTOMPADDING", (0, 0), (-1, 0), 5),
        ("TOPPADDING", (0, 1), (-1, 1), 4),
        ("BOTTOMPADDING", (0, 1), (-1, 1), 8),
    ]))
    story.append(metric_table)
    story.append(PageBreak())

    index = 1
    in_bibliography = False
    while index < len(lines):
        line = lines[index].strip()
        if not line:
            index += 1
            continue
        if line.startswith("## "):
            title = line[3:]
            in_bibliography = title == "Bibliography"
            story.append(Paragraph(inline_markup(title), h2))
            index += 1
            continue
        if line.startswith("### "):
            story.append(Paragraph(inline_markup(line[4:]), h3))
            index += 1
            continue
        if line.startswith("|") and index + 1 < len(lines) and "---" in lines[index + 1]:
            rows = []
            while index < len(lines) and lines[index].strip().startswith("|"):
                current = lines[index].strip()
                index += 1
                if "---" in current:
                    continue
                cells = [cell.strip() for cell in current.split("|")[1:-1]]
                rows.append([Paragraph(inline_markup(cell), body) for cell in cells])
            widths = [22 * mm, 75 * mm, 45 * mm, 28 * mm][: len(rows[0])]
            table = Table(rows, colWidths=widths, repeatRows=1)
            table.setStyle(table_style)
            story.append(table)
            story.append(Spacer(1, 4 * mm))
            continue
        style = bib if in_bibliography and re.match(r"\[\d+\]", line) else body
        story.append(Paragraph(inline_markup(line), style))
        index += 1

    doc = SimpleDocTemplate(
        str(OUTPUT),
        pagesize=A4,
        rightMargin=18 * mm,
        leftMargin=18 * mm,
        topMargin=17 * mm,
        bottomMargin=18 * mm,
        title="DeltaServe-style vLLM 单副本原型评估",
        author="CLIF prototype research",
    )
    doc.build(story, onFirstPage=footer, onLaterPages=footer)
    print(OUTPUT)


if __name__ == "__main__":
    build()
