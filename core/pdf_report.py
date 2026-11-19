"""One-click PDF summary of an analysis (text + tables)."""
from __future__ import annotations

import io
from datetime import date

import pandas as pd
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle


def _esc(s: str) -> str:
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace("**", "")


def build_pdf(title: str, subtitle: str, sections: list[tuple[str, list]], disclaimer: str) -> bytes:
    """sections: [(heading, [str | DataFrame, ...]), ...]. Strings become paragraphs/bullets, DataFrames tables."""
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4, leftMargin=2 * cm, rightMargin=2 * cm, topMargin=2 * cm, bottomMargin=2 * cm,
                            title=title, author="Portfolio Lab")
    ss = getSampleStyleSheet()
    small = ParagraphStyle("small", parent=ss["BodyText"], fontSize=8, textColor=colors.grey)
    story = [Paragraph(_esc(title), ss["Title"]), Paragraph(_esc(subtitle), ss["BodyText"]),
             Paragraph(f"Generated {date.today():%d %b %Y}", small), Spacer(1, 12)]
    for heading, items in sections:
        story.append(Paragraph(_esc(heading), ss["Heading2"]))
        for it in items:
            if isinstance(it, pd.DataFrame):
                data = [[""] + [str(c) for c in it.columns]] + [[str(i)] + [str(v) for v in row] for i, row in zip(it.index, it.values)]
                t = Table(data, repeatRows=1)
                t.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e2e8f0")),
                                       ("GRID", (0, 0), (-1, -1), 0.25, colors.lightgrey),
                                       ("FONTSIZE", (0, 0), (-1, -1), 8), ("ALIGN", (1, 0), (-1, -1), "RIGHT")]))
                story += [t, Spacer(1, 8)]
            else:
                story.append(Paragraph(_esc(str(it)), ss["BodyText"]))
        story.append(Spacer(1, 6))
    story += [Spacer(1, 12), Paragraph(_esc(disclaimer), small)]
    doc.build(story)
    return buf.getvalue()
