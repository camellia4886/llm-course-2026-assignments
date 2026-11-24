"""Render the course survey and resource ledger with embedded Windows CJK fonts."""
import os
import re
from pathlib import Path
from xml.sax.saxutils import escape

from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

BASE = Path(__file__).parent
NAVY = colors.HexColor("#183046")
INK = colors.HexColor("#283747")
TEAL = colors.HexColor("#2D6370")
PALE = colors.HexColor("#EAF1F4")
MUTED = colors.HexColor("#607081")
MARGIN = 49
WIDTH = A4[0] - 2 * MARGIN


def fonts():
    directory = Path(os.environ.get("WINDIR", "C:/Windows")) / "Fonts"
    pdfmetrics.registerFont(TTFont("Course", str(directory / "msyh.ttc"), subfontIndex=0))
    pdfmetrics.registerFont(TTFont("CourseBold", str(directory / "msyhbd.ttc"), subfontIndex=0))
    pdfmetrics.registerFontFamily("Course", normal="Course", bold="CourseBold",
                                 italic="Course", boldItalic="CourseBold")


def markup(text):
    text = escape(text).replace("&lt;!-- page --&gt;", "")
    superscripts = str.maketrans("⁰¹²³⁴⁵⁶⁷⁸⁹⁻", "0123456789-")
    text = re.sub(r"[⁰¹²³⁴⁵⁶⁷⁸⁹⁻]+",
                  lambda match: "<super>" + match.group().translate(superscripts) + "</super>",
                  text)
    return re.sub(r"\*\*(.*?)\*\*", r"<b>\1</b>", text)


def stylesheet(compact):
    size = 10 if compact else 11
    leading = 16 if compact else 18
    body = {"fontName": "Course", "textColor": INK, "alignment": TA_LEFT, "wordWrap": "CJK"}
    return {
        "title": ParagraphStyle("title", **{**body, "fontName": "CourseBold", "textColor": NAVY},
                                fontSize=25, leading=33, spaceAfter=8),
        "subtitle": ParagraphStyle("sub", **{**body, "textColor": TEAL},
                                   fontSize=10, leading=15, spaceAfter=17),
        "section": ParagraphStyle("section", **{**body, "fontName": "CourseBold", "textColor": NAVY},
                                  fontSize=20, leading=27, spaceBefore=5, spaceAfter=17),
        "subsection": ParagraphStyle("subsection", **{**body, "fontName": "CourseBold", "textColor": TEAL},
                                     fontSize=12, leading=17, spaceBefore=6, spaceAfter=6),
        "body": ParagraphStyle("body", **body, fontSize=size, leading=leading, spaceAfter=9),
        "cell": ParagraphStyle("cell", **body, fontSize=9.4, leading=15),
        "header": ParagraphStyle("header", **{**body, "fontName": "CourseBold", "textColor": colors.white},
                                 fontSize=9.4, leading=15),
    }


def make_table(rows, styles):
    ncols = len(rows[0])
    if ncols == 2:
        widths = [0.24, 0.76]
    elif ncols == 3:
        widths = [0.18, 0.45, 0.37]
    elif ncols == 4:
        widths = [0.16, 0.20, 0.31, 0.33]
    else:
        widths = [1 / ncols] * ncols
    cells = [[Paragraph(markup(text), styles["header" if i == 0 else "cell"])
              for text in row] for i, row in enumerate(rows)]
    table = Table(cells, colWidths=[WIDTH * fraction for fraction in widths], hAlign="LEFT", repeatRows=1)
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), TEAL),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [PALE, colors.white]),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ("LINEBELOW", (0, -1), (-1, -1), 0.5, PALE),
    ]))
    return table


def content(text, styles):
    items = []
    for page_no, page in enumerate(text.split("<!-- page -->")):
        if page_no:
            items.append(PageBreak())
        lines = iter(page.strip().splitlines())
        for line in lines:
            if not line.strip():
                continue
            if line.startswith("|"):
                headers = [part.strip() for part in line.strip("|").split("|")]
                rows = [headers]
                for row in lines:
                    if not row.startswith("|"):
                        if row.strip():
                            raise ValueError("Markdown tables require a blank line after their final row")
                        break
                    if re.fullmatch(r"[|:\-\s]+", row):
                        continue
                    rows.append([part.strip() for part in row.strip("|").split("|")])
                items.extend((make_table(rows, styles), Spacer(1, 10)))
                continue
            if line.startswith("# "):
                items.append(Paragraph(markup(line[2:]), styles["title"]))
                items.append(Paragraph(markup(next(lines)), styles["subtitle"]))
            elif line.startswith("## "):
                items.append(Paragraph(markup(line[3:]), styles["section"]))
            elif line.startswith("### "):
                items.append(Paragraph(markup(line[4:]), styles["subsection"]))
            else:
                body = "•　" + line[2:] if line.startswith("- ") else line
                items.append(Paragraph(markup(body), styles["body"]))
    return items


def render(source, output, label, compact=False):
    def footer(canvas, doc):
        canvas.setStrokeColor(PALE)
        canvas.line(MARGIN, 34, A4[0] - MARGIN, 34)
        canvas.setFillColor(MUTED)
        canvas.setFont("Course", 8)
        canvas.drawString(MARGIN, 21, label)
        canvas.drawRightString(A4[0] - MARGIN, 21, str(doc.page))

    document = SimpleDocTemplate(str(output), pagesize=A4, topMargin=46, bottomMargin=43,
                                 leftMargin=MARGIN, rightMargin=MARGIN,
                                 title=source.stem, author="课程作业")
    document.build(content(source.read_text(encoding="utf-8"), stylesheet(compact)),
                   onFirstPage=footer, onLaterPages=footer)


if __name__ == "__main__":
    fonts()
    render(BASE / "后训练综述.md", BASE / "后训练综述.pdf", "大语言模型与应用实践　·　第三讲研究报告")
    ledger = BASE.parent / "task2_ledger"
    render(ledger / "一页资源账本.md", ledger / "一页资源账本.pdf",
           "大语言模型与应用实践　·　第二讲资源账本", compact=True)
