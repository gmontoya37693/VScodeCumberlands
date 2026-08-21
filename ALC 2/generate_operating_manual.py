#!/usr/bin/env python3
"""Render the ALC operating manual markdown to PDF.

The PDF should use a single document title, not a cover title plus the
markdown heading, so the first page does not repeat the same title twice.
"""

from __future__ import annotations

from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import HRFlowable, Paragraph, SimpleDocTemplate, Spacer


BASE_DIR = Path(__file__).resolve().parent
MD_PATH = BASE_DIR / "ALC_operating_manual.md"
PDF_PATH = BASE_DIR / "ALC_operating_manual.pdf"


def build_story(markdown_lines: list[str]):
    styles = getSampleStyleSheet()
    styles.add(
        ParagraphStyle(
            name="TitleALC",
            parent=styles["Title"],
            fontName="Helvetica-Bold",
            fontSize=20,
            leading=24,
            textColor=colors.HexColor("#153E75"),
            spaceAfter=12,
            alignment=TA_LEFT,
        )
    )
    styles.add(
        ParagraphStyle(
            name="H2ALC",
            parent=styles["Heading2"],
            fontName="Helvetica-Bold",
            fontSize=13,
            leading=16,
            textColor=colors.HexColor("#1F4E79"),
            spaceBefore=10,
            spaceAfter=6,
        )
    )
    styles.add(
        ParagraphStyle(
            name="H3ALC",
            parent=styles["Heading3"],
            fontName="Helvetica-Bold",
            fontSize=11,
            leading=14,
            textColor=colors.HexColor("#305D8A"),
            spaceBefore=8,
            spaceAfter=4,
        )
    )
    styles.add(
        ParagraphStyle(
            name="BodyALC",
            parent=styles["BodyText"],
            fontName="Helvetica",
            fontSize=10.2,
            leading=13,
            spaceAfter=4,
        )
    )
    styles.add(
        ParagraphStyle(
            name="BulletALC",
            parent=styles["BodyText"],
            fontName="Helvetica",
            fontSize=10.2,
            leading=13,
            leftIndent=14,
            firstLineIndent=0,
            spaceAfter=2,
        )
    )

    story = []

    for line in markdown_lines:
        line = line.rstrip()
        if not line:
            story.append(Spacer(1, 0.08 * inch))
            continue
        if line.startswith("# "):
            story.append(Paragraph(line[2:].strip(), styles["TitleALC"]))
            story.append(
                HRFlowable(
                    width="100%",
                    thickness=1,
                    color=colors.HexColor("#C9D6E3"),
                    spaceBefore=4,
                    spaceAfter=10,
                )
            )
        elif line.startswith("## "):
            story.append(Paragraph(line[3:].strip(), styles["H2ALC"]))
        elif line.startswith("### "):
            story.append(Paragraph(line[4:].strip(), styles["H3ALC"]))
        elif line.startswith("- "):
            story.append(Paragraph("• " + line[2:].strip(), styles["BulletALC"]))
        else:
            story.append(Paragraph(line, styles["BodyALC"]))

    return story


def add_page_number(canvas, doc):
    canvas.saveState()
    canvas.setFont("Helvetica", 9)
    canvas.setFillColor(colors.HexColor("#666666"))
    canvas.drawRightString(letter[0] - 0.75 * inch, 0.55 * inch, f"Page {doc.page}")
    canvas.drawString(0.75 * inch, 0.55 * inch, "ALC Item Sheet Operating Manual")
    canvas.restoreState()


def main() -> None:
    markdown_lines = MD_PATH.read_text(encoding="utf-8").splitlines()
    doc = SimpleDocTemplate(
        str(PDF_PATH),
        pagesize=letter,
        leftMargin=0.75 * inch,
        rightMargin=0.75 * inch,
        topMargin=0.8 * inch,
        bottomMargin=0.8 * inch,
        title="ALC Item Sheet Operating Manual",
    )
    doc.build(build_story(markdown_lines), onFirstPage=add_page_number, onLaterPages=add_page_number)
    print(f"Wrote {PDF_PATH}")


if __name__ == "__main__":
    main()