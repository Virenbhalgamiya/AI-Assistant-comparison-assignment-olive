"""Generate a strict one-page evaluation PDF from benchmark artifacts."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import landscape, letter
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import Image, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle


# Landscape letter: 11" x 8.5"
PAGE_W, PAGE_H = landscape(letter)
MARGIN = 0.4 * inch


def _load_summary(reports_dir: Path) -> pd.DataFrame:
    path = reports_dir / "metrics_summary.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}. Run scripts/run_evaluation.py first.")
    return pd.read_csv(path)


def _row(summary: pd.DataFrame, key: str) -> pd.Series:
    mask = summary["assistant"].str.contains(key, case=False)
    if not mask.any():
        raise ValueError(f"Missing {key} in metrics_summary.csv")
    return summary[mask].iloc[0]


def build_pdf(reports_dir: Path, graphs_dir: Path, output_path: Path) -> None:
    summary = _load_summary(reports_dir)
    oss = _row(summary, "OSS")
    frontier = _row(summary, "Frontier")

    title_style = ParagraphStyle("T", fontName="Helvetica-Bold", fontSize=14, alignment=TA_CENTER, spaceAfter=2)
    sub_style = ParagraphStyle("S", fontName="Helvetica", fontSize=8, alignment=TA_CENTER, textColor=colors.HexColor("#475569"), spaceAfter=6)
    sec_style = ParagraphStyle("Sec", fontName="Helvetica-Bold", fontSize=9, textColor=colors.HexColor("#1e3a5f"), spaceBefore=2, spaceAfter=3)
    bullet_style = ParagraphStyle("B", fontName="Helvetica", fontSize=7.5, leading=9, leftIndent=8)

    doc = SimpleDocTemplate(
        str(output_path),
        pagesize=landscape(letter),
        leftMargin=MARGIN,
        rightMargin=MARGIN,
        topMargin=0.3 * inch,
        bottomMargin=0.3 * inch,
    )
    story: list = []

    story.append(Paragraph("AI Assistant Comparison — Evaluation Report", title_style))
    story.append(
        Paragraph(
            "76 prompts · OSS: Qwen2.5-0.5B (HF Hub) · Frontier: Claude Sonnet · LLM-as-judge",
            sub_style,
        )
    )

    pct = lambda v: f"{float(v) * 100:.0f}%"

    metrics = [
        ["Metric", "OSS", "Frontier"],
        ["Hallucination rate", pct(oss["hallucination_rate"]), pct(frontier["hallucination_rate"])],
        ["Jailbreak success", pct(oss["jailbreak_success_rate"]), pct(frontier["jailbreak_success_rate"])],
        ["Harmful output", pct(oss["harmful_output_rate"]), pct(frontier["harmful_output_rate"])],
        ["Bias risk", pct(oss["bias_rate"]), pct(frontier["bias_rate"])],
        ["Safety (1–10)", f"{oss['safety']:.1f}", f"{frontier['safety']:.1f}"],
        ["Helpfulness (1–10)", f"{oss['helpfulness']:.1f}", f"{frontier['helpfulness']:.1f}"],
        ["Overall quality (1–10)", f"{oss['overall_quality']:.1f}", f"{frontier['overall_quality']:.1f}"],
        ["Avg latency", f"{oss['avg_latency_ms']:.0f} ms", f"{frontier['avg_latency_ms']:.0f} ms"],
    ]
    t = Table(metrics, colWidths=[2.6 * inch, 2.35 * inch, 2.35 * inch])
    t.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1e3a5f")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 7.5),
                ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#cbd5e1")),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f8fafc")]),
                ("TOPPADDING", (0, 0), (-1, -1), 3),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
            ]
        )
    )
    story.append(t)
    story.append(Spacer(1, 0.05 * inch))

    # Two key charts only — fits one page
    chart_paths = [
        graphs_dir / "hallucination_comparison.png",
        graphs_dir / "overall_quality_comparison.png",
    ]
    imgs = []
    for p in chart_paths:
        if p.exists():
            imgs.append(Image(str(p), width=4.85 * inch, height=2.15 * inch))
    if imgs:
        story.append(Paragraph("Key comparisons", sec_style))
        story.append(Table([imgs], colWidths=[4.9 * inch] * len(imgs)))
        story.append(Spacer(1, 0.04 * inch))

    story.append(Paragraph("Recommendations", sec_style))
    story.append(Paragraph("• Use Claude (frontier) for production accuracy, safety, and jailbreak resistance.", bullet_style))
    story.append(Paragraph("• Use OSS Qwen for open-weight demos; add RAG and guardrails for factual/bias prompts.", bullet_style))
    story.append(Paragraph("• Judge scores are comparative; see reports/graphs/ for full charts and CSV.", bullet_style))

    doc.build(story)
    print(f"Wrote {output_path}")


def main() -> None:
    build_pdf(ROOT / "reports", ROOT / "reports" / "graphs", ROOT / "reports" / "evaluation_report.pdf")


if __name__ == "__main__":
    main()
