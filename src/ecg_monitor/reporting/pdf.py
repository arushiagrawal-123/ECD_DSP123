"""
PDF Summary Generator.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle


def export_summary_pdf(
    out_path: Path,
    session_id: str,
    record_id: str,
    metrics: dict[str, Any],
    alert_counts: dict[str, int],
) -> None:
    """
    Generate a one-page PDF summary of the ECG processing session.

    WARNING: This output is strictly for educational and research prototyping.
    It does not constitute a medical diagnosis.
    """
    doc = SimpleDocTemplate(
        str(out_path),
        pagesize=letter,
        rightMargin=inch,
        leftMargin=inch,
        topMargin=inch,
        bottomMargin=inch,
    )

    styles = getSampleStyleSheet()
    title_style = styles["Heading1"]
    subtitle_style = styles["Heading2"]
    normal_style = styles["Normal"]

    # Custom footer style for the strict disclaimer
    footer_style = ParagraphStyle(
        "Footer",
        parent=styles["Normal"],
        fontSize=8,
        textColor=colors.red,
        alignment=1,  # Center
    )

    elements = []

    # Header
    elements.append(Paragraph("ECG Monitor — Session Summary", title_style))
    elements.append(Paragraph(f"Session ID: {session_id} | Record: {record_id}", normal_style))
    elements.append(Spacer(1, 0.25 * inch))

    # Metrics Table
    elements.append(Paragraph("Performance & Quality Metrics", subtitle_style))

    data = [["Metric Group", "Metric", "Value"]]

    # Quality
    data.append(["Signal Quality", "SNR (dB)", f"{metrics.get('snr_db', 'N/A')}"])
    data.append(["", "RMSE", f"{metrics.get('rmse', 'N/A')}"])
    data.append(["", "Label", f"{metrics.get('quality_label', 'N/A')}"])

    # Detection
    data.append(["Detection", "Sensitivity", f"{metrics.get('sensitivity', 'N/A')}"])
    data.append(["", "PPV", f"{metrics.get('ppv', 'N/A')}"])
    data.append(["", "F1 Score", f"{metrics.get('f1', 'N/A')}"])
    data.append(["", "Timing Error (ms)", f"{metrics.get('timing_error_mean_ms', 'N/A')}"])

    # Heart Rate
    data.append(["Heart Rate", "Average HR (BPM)", f"{metrics.get('hr_average_bpm', 'N/A')}"])
    data.append(["", "HRV SDNN (ms)", f"{metrics.get('hrv_sdnn_ms', 'N/A')}"])
    data.append(["", "Total Candidate Beats", f"{metrics.get('total_beats', 0)}"])

    t = Table(data, colWidths=[1.5 * inch, 2.0 * inch, 2.0 * inch])
    t.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.grey),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
                ("ALIGN", (0, 0), (-1, -1), "LEFT"),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("BOTTOMPADDING", (0, 0), (-1, 0), 12),
                ("BACKGROUND", (0, 1), (-1, -1), colors.beige),
                ("GRID", (0, 0), (-1, -1), 1, colors.black),
                # Span the group columns
                ("SPAN", (0, 1), (0, 3)),
                ("SPAN", (0, 4), (0, 7)),
                ("SPAN", (0, 8), (0, 10)),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ]
        )
    )
    elements.append(t)
    elements.append(Spacer(1, 0.25 * inch))

    # Alerts Table
    elements.append(Paragraph("Rule-Based Prototype Alerts", subtitle_style))
    alert_data = [["Alert Type (Candidate)", "Count"]]
    for a_type, count in alert_counts.items():
        alert_data.append([a_type, str(count)])

    if len(alert_data) == 1:
        alert_data.append(["No alerts triggered", "0"])

    t_alerts = Table(alert_data, colWidths=[3.5 * inch, 2.0 * inch])
    t_alerts.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.grey),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
                ("ALIGN", (0, 0), (-1, -1), "LEFT"),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("BOTTOMPADDING", (0, 0), (-1, 0), 12),
                ("BACKGROUND", (0, 1), (-1, -1), colors.lightgrey),
                ("GRID", (0, 0), (-1, -1), 1, colors.black),
            ]
        )
    )
    elements.append(t_alerts)

    elements.append(Spacer(1, 1.0 * inch))

    # Disclaimer
    disclaimer_text = (
        "EDUCATIONAL PROTOTYPE — NOT FOR MEDICAL USE<br/>"
        "This system and report are rule-based prototypes designed strictly for educational and "
        "research purposes. This is NOT a medical device. It does not provide any clinical "
        "evaluation or medical interpretation. Any flagged candidate events require human review "
        "by qualified personnel. Do not base any health decisions on this output."
    )
    elements.append(Paragraph(disclaimer_text, footer_style))

    doc.build(elements)
