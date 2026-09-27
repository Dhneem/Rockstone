"""Build MANUAL.pdf from MANUAL.md (dev tool, run from the project root).

Parses the Markdown manual and lays it out with ReportLab: heading
hierarchy, bullet/numbered lists, pipe tables, code blocks and inline
code/emphasis formatting, with clickable table of contents links.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.platypus import (BaseDocTemplate, Frame, KeepTogether,
                                PageTemplate, Paragraph, Spacer, Table,
                                TableStyle)

ROOT = Path(__file__).resolve().parent
MANUAL_MD = ROOT.parent / "MANUAL.md"
MANUAL_PDF = ROOT.parent / "MANUAL.pdf"

ACCENT = colors.HexColor("#2e6b4f")   # rockstone green
CODE_BG = colors.HexColor("#f4f4f2")
RULE = colors.HexColor("#cccccc")


# ---------------------------------------------------------------- styles --

def _styles() -> dict:
    s = {}
    s["title"] = ParagraphStyle(
        "title", fontName="Helvetica-Bold", fontSize=22, leading=26,
        textColor=ACCENT, spaceAfter=2)
    s["subtitle"] = ParagraphStyle(
        "subtitle", fontName="Helvetica", fontSize=10.5, leading=14,
        textColor=colors.HexColor("#555555"), spaceAfter=14)
    s["h2"] = ParagraphStyle(
        "h2", fontName="Helvetica-Bold", fontSize=14.5, leading=18,
        textColor=ACCENT, spaceBefore=16, spaceAfter=6)
    s["h3"] = ParagraphStyle(
        "h3", fontName="Helvetica-Bold", fontSize=11.5, leading=15,
        textColor=colors.HexColor("#333333"), spaceBefore=10, spaceAfter=4)
    s["body"] = ParagraphStyle(
        "body", fontName="Helvetica", fontSize=9.5, leading=13.5,
        spaceAfter=5)
    s["bullet"] = ParagraphStyle(
        "bullet", parent=s["body"], leftIndent=14, bulletIndent=4,
        spaceAfter=2.5)
    s["code"] = ParagraphStyle(
        "code", fontName="Courier", fontSize=8.2, leading=11,
        backColor=CODE_BG, borderPadding=(4, 6, 4, 6), spaceAfter=6,
        leftIndent=4)
    s["cell"] = ParagraphStyle(
        "cell", fontName="Helvetica", fontSize=8.6, leading=11.5)
    s["cellhead"] = ParagraphStyle(
        "cellhead", parent=s["cell"], fontName="Helvetica-Bold")
    s["toc"] = ParagraphStyle(
        "toc", fontName="Helvetica", fontSize=9.5, leading=15,
        leftIndent=6)
    return s


# -------------------------------------------------------- inline parsing --

def _inline(text: str) -> str:
    """Markdown inline syntax -> ReportLab paragraph markup."""
    # escape XML specials first
    text = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    # inline code (protect content from later replacements)
    codes: list[str] = []

    def _stash(m):
        codes.append(m.group(1))
        return f"\x00{len(codes) - 1}\x00"

    text = re.sub(r"`([^`]+)`", _stash, text)
    text = re.sub(r"\*\*([^*]+)\*\*", r"<b>\1</b>", text)
    text = re.sub(r"(?<!\*)\*([^*]+)\*(?!\*)", r"<i>\1</i>", text)
    text = re.sub(r"\[([^\]]+)\]\(#[^)]*\)", r"\1", text)  # internal links
    # restore code with a tinted background
    for i, c in enumerate(codes):
        text = text.replace(
            f"\x00{i}\x00",
            f'<font face="Courier" backColor="#e8ede9">{c}</font>')
    # links [text](url) -> plain text (no external fonts needed)
    text = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", r"\1", text)
    return text


# ------------------------------------------------------------ block model --

def _parse_blocks(md: str) -> list[dict]:
    """Group the Markdown into typed blocks for the layout engine."""
    blocks: list[dict] = []
    lines = md.splitlines()
    i = 0
    while i < len(lines):
        line = lines[i]
        if not line.strip():
            i += 1
            continue
        if line.startswith("## "):
            blocks.append({"type": "h2", "text": line[3:].strip()})
            i += 1
        elif line.startswith("### "):
            blocks.append({"type": "h3", "text": line[4:].strip()})
            i += 1
        elif line.startswith("# "):
            blocks.append({"type": "title", "text": line[2:].strip()})
            i += 1
        elif line.strip() == "---":
            i += 1  # horizontal rules handled by spacing
        elif line.startswith("```"):
            code = []
            i += 1
            while i < len(lines) and not lines[i].startswith("```"):
                code.append(lines[i])
                i += 1
            i += 1  # closing fence
            blocks.append({"type": "code", "text": "\n".join(code)})
        elif line.lstrip().startswith("|") and i + 1 < len(lines) \
                and set(lines[i + 1].replace("|", "").replace("-", "")
                        .strip()) <= {":", " ", ""} \
                and lines[i + 1].lstrip().startswith("|"):
            # pipe table with a separator row
            rows = []
            while i < len(lines) and lines[i].lstrip().startswith("|"):
                rows.append(lines[i])
                i += 1
            blocks.append({"type": "table", "rows": rows})
        elif re.match(r"^\s*[-*] ", line):
            items = []
            while i < len(lines) and re.match(r"^\s*[-*] ", lines[i]):
                items.append(re.sub(r"^\s*[-*] ", "", lines[i]))
                i += 1
            blocks.append({"type": "bullets", "items": items})
        elif re.match(r"^\s*\d+\. ", line):
            items = []
            while i < len(lines) and re.match(r"^\s*\d+\. ", lines[i]):
                items.append(re.sub(r"^\s*\d+\. ", "", lines[i]))
                i += 1
            blocks.append({"type": "numbers", "items": items})
        else:
            stop_prefixes = ("|", "#", "```", "- ", "* ")
            para = []
            while i < len(lines) and lines[i].strip() \
                    and not lines[i].lstrip().startswith(stop_prefixes) \
                    and not re.match(r"^\s*\d+\. ", lines[i]) \
                    and not lines[i].strip() == "---":
                para.append(lines[i].strip())
                i += 1
            if para:
                blocks.append({"type": "para", "text": " ".join(para)})
            else:
                i += 1
    return blocks


# ------------------------------------------------------------- tables --

def _split_row(row: str) -> list[str]:
    row = row.strip()
    if row.startswith("|"):
        row = row[1:]
    if row.endswith("|"):
        row = row[:-1]
    return [c.strip() for c in row.split("|")]


def _table_flowable(block: dict, styles: dict) -> Table:
    header = _split_row(block["rows"][0])
    body = [_split_row(r) for r in block["rows"][2:]]
    data = [[Paragraph(_inline(c), styles["cellhead"]) for c in header]]
    for row in body:
        data.append([Paragraph(_inline(c), styles["cell"]) for c in row])
    ncols = len(header)
    col_w = [None] * ncols
    if ncols == 2:               # two-column tables: name / description
        col_w = [38 * mm, None]
    elif ncols == 3:
        col_w = [30 * mm, None]
    t = Table(data, colWidths=col_w, repeatRows=1, hAlign="LEFT")
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e9efe9")),
        ("GRID", (0, 0), (-1, -1), 0.4, RULE),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 5),
        ("RIGHTPADDING", (0, 0), (-1, -1), 5),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
    ]))
    return t


# --------------------------------------------------------------- build --

def build() -> Path:
    styles = _styles()
    md = MANUAL_MD.read_text(encoding="utf-8")
    blocks = _parse_blocks(md)

    story: list = []
    h2_counter = 0

    for block in blocks:
        kind = block["type"]
        if kind == "title":
            story.append(Paragraph(_inline(block["text"]), styles["title"]))
        elif kind == "h2":
            h2_counter += 1
            if h2_counter == 1:
                # first ## block ends the title area -> add subtitle line
                story.append(Paragraph(
                    "User manual — generated from MANUAL.md",
                    styles["subtitle"]))
            story.append(Paragraph(_inline(block["text"]), styles["h2"]))
        elif kind == "h3":
            story.append(Paragraph(_inline(block["text"]), styles["h3"]))
        elif kind == "para":
            story.append(Paragraph(_inline(block["text"]), styles["body"]))
        elif kind == "bullets":
            for item in block["items"]:
                story.append(Paragraph(_inline(item), styles["bullet"],
                                       bulletText="•"))
            story.append(Spacer(1, 3))
        elif kind == "numbers":
            for n, item in enumerate(block["items"], start=1):
                story.append(Paragraph(_inline(item), styles["bullet"],
                                       bulletText=f"{n}."))
            story.append(Spacer(1, 3))
        elif kind == "code":
            text = block["text"].replace("&", "&amp;") \
                .replace("<", "&lt;").replace(">", "&gt;")
            for line in text.splitlines():
                story.append(Paragraph(line or " ", styles["code"]))
        elif kind == "table":
            story.append(_table_flowable(block, styles))
            story.append(Spacer(1, 6))

    def on_page(canvas, doc):
        canvas.saveState()
        canvas.setFont("Helvetica", 8)
        canvas.setFillColor(colors.HexColor("#888888"))
        canvas.drawString(18 * mm, 12 * mm, "Rockstone — User Manual")
        canvas.drawRightString(A4[0] - 18 * mm, 12 * mm,
                               f"Page {doc.page}")
        canvas.setStrokeColor(RULE)
        canvas.setLineWidth(0.5)
        canvas.line(18 * mm, 15 * mm, A4[0] - 18 * mm, 15 * mm)
        canvas.restoreState()

    doc = BaseDocTemplate(
        str(MANUAL_PDF), pagesize=A4,
        leftMargin=20 * mm, rightMargin=20 * mm,
        topMargin=18 * mm, bottomMargin=20 * mm,
        title="Rockstone — User Manual", author="Rockstone")
    frame = Frame(doc.leftMargin, doc.bottomMargin, doc.width, doc.height,
                  id="main")
    doc.addPageTemplates([PageTemplate(id="page", frames=[frame],
                                       onPage=on_page)])
    doc.build(story)
    return MANUAL_PDF


if __name__ == "__main__":
    out = build()
    print(f"written {out} ({out.stat().st_size / 1024:.0f} kB)")
    sys.exit(0)
