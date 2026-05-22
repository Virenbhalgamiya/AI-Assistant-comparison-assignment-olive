"""Generate a one-page evaluation PDF from benchmark artifacts."""

from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import Image, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle


def _load_summary(reports_dir: Path) -> pd.DataFrame:
    path = reports_dir / "metrics_summary.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}. Run scripts/run_evaluation.py first.")
    return pd.read_csv(path)


def _load_category(reports_dir: Path) -> pd.DataFrame:
    path = reports_dir / "metrics_by_category.csv"
    if path.exists():
        return pd.read_csv(path)
    return pd.DataFrame()


def _chart_paths(graphs_dir: Path) -> list[Path]:
    names = [
        "hallucination_comparison.png",
        "safety_comparison.png",
        "jailbreak_success_comparison.png",
        "latency_comparison.png",
    ]
    return [graphs_dir / name for name in names if (graphs_dir / name).exists()]


def _recommendations(summary: pd.DataFrame) -> list[str]:
    if summary.empty:
        return ["Re-run evaluation after configuring API keys."]
    oss = summary[summary["assistant"].str.contains("OSS", case=False)]
    frontier = summary[summary["assistant"].str.contains("Frontier", case=False)]
    bullets: list[str] = []
    if not frontier.empty and not oss.empty:
        f = frontier.iloc[0]
        o = oss.iloc[0]
        if f["overall_quality"] > o["overall_quality"]:
            bullets.append(
                "Use the frontier model for production-quality answers, complex reasoning, and higher factual accuracy."
            )
        if o["avg_latency_ms"] < f["avg_latency_ms"]:
            bullets.append(
                f"Use the OSS model when latency matters (OSS ~{o['avg_latency_ms']:.0f} ms vs frontier ~{f['avg_latency_ms']:.0f} ms in this run)."
            )
        if o["hallucination_rate"] > f["hallucination_rate"]:
            bullets.append(
                "Add retrieval or larger OSS weights if open-source parity is required; hallucination rate was higher on OSS."
            )
        bullets.append(
            "Keep layered guardrails (input filters + model refusals); both models benefited from pre-generation safety blocks."
        )
    return bullets[:4]


def build_pdf(reports_dir: Path, graphs_dir: Path, output_path: Path) -> None:
    summary = _load_summary(reports_dir)
    by_category = _load_category(reports_dir)
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle("Title", parent=styles["Heading1"], fontSize=16, spaceAfter=8)
    body_style = ParagraphStyle("Body", parent=styles["Normal"], fontSize=9, leading=12)

    doc = SimpleDocTemplate(
        str(output_path),
        pagesize=letter,
        leftMargin=0.45 * inch,
        rightMargin=0.45 * inch,
        topMargin=0.4 * inch,
        bottomMargin=0.4 * inch,
    )
    story: list = []

    story.append(Paragraph("AI Assistant Comparison — Evaluation Report", title_style))
    story.append(
        Paragraph(
            f"Generated {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')} · "
            "OSS (Qwen via Hugging Face Hub) vs Frontier (hosted API) · LLM-as-judge scoring",
            body_style,
        )
    )
    story.append(Spacer(1, 0.12 * inch))

    table_data = [["Assistant", "Halluc.", "Jailbreak", "Bias", "Safety", "Helpful.", "Latency ms", "Quality"]]
    for _, row in summary.iterrows():
        table_data.append(
            [
                str(row["assistant"])[:18],
                f"{row.get('hallucination_rate', 0):.2f}",
                f"{row.get('jailbreak_success_rate', 0):.2f}",
                f"{row.get('bias_rate', 0):.2f}",
                f"{row.get('safety', 0):.1f}",
                f"{row.get('helpfulness', 0):.1f}",
                f"{row.get('avg_latency_ms', 0):.0f}",
                f"{row.get('overall_quality', 0):.1f}",
            ]
        )
    table = Table(table_data, colWidths=[1.15 * inch, 0.55 * inch, 0.65 * inch, 0.5 * inch, 0.55 * inch, 0.6 * inch, 0.7 * inch, 0.55 * inch])
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#0f172a")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 8),
                ("GRID", (0, 0), (-1, -1), 0.25, colors.grey),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f1f5f9")]),
            ]
        )
    )
    story.append(table)
    story.append(Spacer(1, 0.1 * inch))

    charts = _chart_paths(graphs_dir)
    if charts:
        chart_row = []
        for chart in charts[:4]:
            chart_row.append(Image(str(chart), width=2.45 * inch, height=1.35 * inch))
        story.append(Table([chart_row], colWidths=[2.5 * inch] * len(chart_row)))

    story.append(Spacer(1, 0.08 * inch))
    story.append(Paragraph("<b>Recommendations</b>", body_style))
    for bullet in _recommendations(summary):
        story.append(Paragraph(f"• {bullet}", body_style))

    if not by_category.empty:
        story.append(Spacer(1, 0.06 * inch))
        story.append(Paragraph("<b>Category highlights</b>", body_style))
        for category in sorted(by_category["category"].unique()):
            subset = by_category[by_category["category"] == category]
            parts = [f"{row['assistant']}: Q={row['overall_quality']:.1f}" for _, row in subset.iterrows()]
            story.append(Paragraph(f"• {category}: " + ", ".join(parts), body_style))

    story.append(Spacer(1, 0.05 * inch))
    story.append(
        Paragraph(
            "<b>Limitations:</b> Scores are from an LLM judge (approximate). Benchmark mixes factual, "
            "jailbreak, bias, and adversarial prompts. Not a substitute for human red-teaming.",
            body_style,
        )
    )

    doc.build(story)
    print(f"Wrote {output_path}")


def main() -> None:
    reports_dir = ROOT / "reports"
    graphs_dir = reports_dir / "graphs"
    output_path = reports_dir / "evaluation_report.pdf"
    build_pdf(reports_dir, graphs_dir, output_path)


if __name__ == "__main__":
    main()
