"""
report_generator.py
-------------------
Generate downloadable CSV and PDF evaluation reports.
Uses reportlab for PDF creation (no fpdf2 dependency).
"""

from __future__ import annotations

import io
import datetime
import pandas as pd
from typing import List, Dict, Any

try:
    from reportlab.lib.pagesizes import letter, A4
    from reportlab.lib import colors
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import inch
    from reportlab.platypus import (
        SimpleDocTemplate, Paragraph, Spacer, Table,
        TableStyle, HRFlowable, PageBreak,
    )
    _RL_AVAILABLE = True
except ImportError:
    _RL_AVAILABLE = False


# ── CSV ───────────────────────────────────────────────────────────────────────

def generate_csv_report(
    compare_df: pd.DataFrame,
    results: List[Dict[str, Any]],
) -> bytes:
    """
    Export model comparison + per-model classification details to CSV.
    Returns UTF-8 bytes.
    """
    buf = io.StringIO()

    buf.write("# FRAUD DETECTION SYSTEM — EVALUATION REPORT\n")
    buf.write(f"# Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}\n\n")

    buf.write("## MODEL COMPARISON\n")
    compare_df.to_csv(buf, index=False)
    buf.write("\n")

    for r in results:
        buf.write(f"## {r['name'].upper()} — CLASSIFICATION REPORT\n")
        buf.write(f"Threshold: {r['threshold']}\n")
        buf.write(r.get("report_str", "") + "\n\n")

    return buf.getvalue().encode("utf-8")


# ── PDF ───────────────────────────────────────────────────────────────────────

def generate_pdf_report(
    compare_df: pd.DataFrame,
    results: List[Dict[str, Any]],
    dataset_info: Dict[str, Any] | None = None,
) -> bytes:
    """
    Build a styled PDF evaluation report using reportlab.

    Parameters
    ----------
    compare_df   : DataFrame from evaluation.compare_models()
    results      : list of evaluate_model() result dicts
    dataset_info : optional dict with keys 'total', 'fraud', 'legit'

    Returns
    -------
    bytes  – PDF file content
    """
    if not _RL_AVAILABLE:
        raise RuntimeError("reportlab is not installed. Run: pip install reportlab")

    buf    = io.BytesIO()
    doc    = SimpleDocTemplate(buf, pagesize=A4,
                               leftMargin=0.75*inch, rightMargin=0.75*inch,
                               topMargin=0.75*inch, bottomMargin=0.75*inch)
    styles = getSampleStyleSheet()
    story  = []

    # ── Title ──────────────────────────────────────────────────────────────
    title_style = ParagraphStyle(
        "CustomTitle",
        parent=styles["Title"],
        fontSize=22,
        textColor=colors.HexColor("#1E88E5"),
        spaceAfter=6,
    )
    story.append(Paragraph("Credit Card Fraud Detection", title_style))
    story.append(Paragraph("Evaluation Report", styles["Heading2"]))
    story.append(Paragraph(
        f"Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}",
        styles["Normal"],
    ))
    story.append(HRFlowable(width="100%", thickness=1.5,
                            color=colors.HexColor("#1E88E5")))
    story.append(Spacer(1, 12))

    # ── Dataset Summary ────────────────────────────────────────────────────
    if dataset_info:
        story.append(Paragraph("Dataset Summary", styles["Heading2"]))
        ds_data = [
            ["Metric", "Value"],
            ["Total Transactions", f"{dataset_info.get('total', 'N/A'):,}"],
            ["Legitimate", f"{dataset_info.get('legit', 'N/A'):,}"],
            ["Fraud",      f"{dataset_info.get('fraud', 'N/A'):,}"],
            ["Fraud Rate",
             f"{dataset_info.get('fraud', 0) / max(dataset_info.get('total', 1), 1) * 100:.2f}%"],
        ]
        t = Table(ds_data, colWidths=[3*inch, 3*inch])
        t.setStyle(_table_style())
        story.append(t)
        story.append(Spacer(1, 14))

    # ── Model Comparison Table ─────────────────────────────────────────────
    story.append(Paragraph("Model Comparison", styles["Heading2"]))
    comp_data = [compare_df.columns.tolist()] + compare_df.values.tolist()
    comp_table = Table(comp_data, repeatRows=1)
    comp_table.setStyle(_table_style(header_color="#1E88E5"))
    story.append(comp_table)
    story.append(Spacer(1, 14))

    # ── Per-model detail ───────────────────────────────────────────────────
    story.append(Paragraph("Per-Model Details", styles["Heading2"]))

    for r in results:
        story.append(Paragraph(r["name"], styles["Heading3"]))
        detail_data = [
            ["Metric", "Value"],
            ["Threshold",     f"{r['threshold']:.2f}"],
            ["Precision",     f"{r['precision']:.4f}"],
            ["Recall",        f"{r['recall']:.4f}"],
            ["F1-Score",      f"{r['f1']:.4f}"],
            ["ROC-AUC",       f"{r['auc_roc']:.4f}"],
            ["Avg Precision", f"{r['avg_precision']:.4f}"],
        ]
        dt = Table(detail_data, colWidths=[3*inch, 3*inch])
        dt.setStyle(_table_style())
        story.append(dt)
        story.append(Spacer(1, 6))

        # Classification report as pre-formatted text
        cr_style = ParagraphStyle(
            "Mono", parent=styles["Code"], fontSize=7.5, leading=10,
        )
        story.append(Paragraph(
            r.get("report_str", "").replace("\n", "<br/>"),
            cr_style,
        ))
        story.append(Spacer(1, 10))

    # ── Footer note ────────────────────────────────────────────────────────
    story.append(HRFlowable(width="100%", thickness=1, color=colors.grey))
    story.append(Spacer(1, 6))
    story.append(Paragraph(
        "This report was generated automatically by the AI-Powered Fraud Detection System.",
        styles["Italic"],
    ))

    doc.build(story)
    return buf.getvalue()


def _table_style(header_color: str = "#37474F"):
    """Return a consistent TableStyle."""
    hc = colors.HexColor(header_color)
    return TableStyle([
        ("BACKGROUND",   (0, 0), (-1, 0),  hc),
        ("TEXTCOLOR",    (0, 0), (-1, 0),  colors.white),
        ("FONTNAME",     (0, 0), (-1, 0),  "Helvetica-Bold"),
        ("FONTSIZE",     (0, 0), (-1, 0),  9),
        ("ALIGN",        (0, 0), (-1, -1), "CENTER"),
        ("FONTSIZE",     (0, 1), (-1, -1), 8),
        ("ROWBACKGROUNDS",(0, 1), (-1, -1),
         [colors.HexColor("#F5F5F5"), colors.white]),
        ("GRID",         (0, 0), (-1, -1), 0.4, colors.grey),
        ("TOPPADDING",   (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING",(0, 0), (-1, -1), 4),
    ])
