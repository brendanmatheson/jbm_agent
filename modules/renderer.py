"""
renderer.py
-----------
Generates the standalone JBM-style HTML output file.

Design direction: editorial/broadsheet — inspired by FT data journalism.
Dark ink on cream. Strong typography. Chart centre-stage.
No frameworks. Chart.js via CDN. Self-contained single file.
"""

import json
import logging
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)

# Colour palette — FT-adjacent but distinct
COLORS = {
    "uk_red": "#C1121F",
    "peer_1": "#457B9D",
    "peer_2": "#6B9E78",
    "peer_3": "#E9C46A",
    "peer_4": "#A8DADC",
    "muted": "#8D99AE",
    "annotation": "#C1121F",
}

SENSITIVITY_BANNERS = {
    "HIGH": (
        "#7c1d1d",
        "#fef2f2",
        "This analysis covers a sensitive topic. All statistics are drawn from official UK "
        "and international sources (ONS, World Bank). Care has been taken to avoid causal "
        "claims without strong evidential basis. Figures should be read in full context."
    ),
    "MEDIUM": (
        "#7c4700",
        "#fffbeb",
        "Note: This topic requires careful contextualisation. "
        "Statistics are from official sources and presented without editorial inference."
    ),
    "LOW": None,
}


def _series_to_js(series_list, all_null_fill=True) -> str:
    """Convert ChartSpec series to Chart.js dataset definitions."""
    js_datasets = []
    peer_colors = [COLORS["peer_1"], COLORS["peer_2"], COLORS["peer_3"], COLORS["peer_4"], COLORS["muted"]]
    for i, s in enumerate(series_list):
        is_uk = "United Kingdom" in s.label or i == 0
        color = COLORS["uk_red"] if is_uk else peer_colors[(i - 1) % len(peer_colors)]
        data_json = json.dumps([v for v in s.data])
        dash_str = "borderDash: [5, 4]," if s.dash else ""
        weight = "3" if is_uk else "1.5"
        point_radius = "4" if is_uk else "0"
        js_datasets.append(f"""{{
            label: {json.dumps(s.label)},
            data: {data_json},
            borderColor: '{color}',
            backgroundColor: '{"rgba(193,18,31,0.08)" if is_uk else "transparent"}',
            borderWidth: {weight},
            {dash_str}
            pointRadius: {point_radius},
            pointHoverRadius: 5,
            tension: 0.3,
            spanGaps: true,
        }}""")
    return ",\n".join(js_datasets)


def _build_chart_js(chart_spec) -> str:
    """Build the full Chart.js initialisation block."""
    ct = chart_spec.chart_type
    if ct == "bar":
        chart_type_str = "bar"
        dataset_str = f"""{{
            label: {json.dumps(chart_spec.series[0].label if chart_spec.series else "Value")},
            data: {json.dumps(chart_spec.series[0].data if chart_spec.series else [])},
            backgroundColor: '{COLORS["uk_red"]}22',
            borderColor: '{COLORS["uk_red"]}',
            borderWidth: 2,
            borderRadius: 2,
        }}"""
    else:
        chart_type_str = "line"
        dataset_str = _series_to_js(chart_spec.series)

    annotation_plugin = ""
    if chart_spec.annotation and chart_spec.annotation_index >= 0:
        idx = chart_spec.annotation_index
        annotation_plugin = f"""
        annotation: {{
            annotations: {{
                callout: {{
                    type: 'label',
                    xValue: {idx},
                    yValue: 'max',
                    content: ['{chart_spec.annotation}'],
                    font: {{ size: 11, family: "'Libre Franklin', sans-serif" }},
                    color: '{COLORS["annotation"]}',
                    backgroundColor: 'rgba(255,255,255,0.9)',
                    borderWidth: 1,
                    borderColor: '{COLORS["annotation"]}44',
                    padding: 6,
                }}
            }}
        }},"""

    y_label = json.dumps(chart_spec.y_label or "")
    x_labels_json = json.dumps(chart_spec.x_labels)

    return f"""
    const ctx = document.getElementById('mainChart').getContext('2d');
    new Chart(ctx, {{
        type: '{chart_type_str}',
        data: {{
            labels: {x_labels_json},
            datasets: [
                {dataset_str}
            ]
        }},
        options: {{
            responsive: true,
            maintainAspectRatio: false,
            interaction: {{
                mode: 'index',
                intersect: false,
            }},
            plugins: {{
                legend: {{
                    display: {str(len(chart_spec.series) > 1).lower()},
                    position: 'bottom',
                    labels: {{
                        font: {{ size: 12, family: "'Libre Franklin', sans-serif" }},
                        color: '#3d3d3a',
                        usePointStyle: true,
                        pointStyleWidth: 16,
                        padding: 16,
                    }}
                }},
                tooltip: {{
                    backgroundColor: '#1a1a2e',
                    titleFont: {{ size: 13, family: "'Libre Franklin', sans-serif", weight: '600' }},
                    bodyFont: {{ size: 12, family: "'Libre Franklin', sans-serif" }},
                    padding: 12,
                    cornerRadius: 4,
                    callbacks: {{
                        label: function(ctx) {{
                            let val = ctx.parsed.y;
                            if (val === null) return null;
                            let unit = {y_label};
                            if (unit === '%') return ctx.dataset.label + ': ' + val.toFixed(1) + '%';
                            if (unit === '£') return ctx.dataset.label + ': £' + val.toLocaleString('en-GB');
                            return ctx.dataset.label + ': ' + val.toLocaleString('en-GB');
                        }}
                    }}
                }},
                {annotation_plugin}
            }},
            scales: {{
                x: {{
                    grid: {{ color: 'rgba(0,0,0,0.04)', drawTicks: false }},
                    ticks: {{
                        font: {{ size: 11, family: "'Libre Franklin', sans-serif" }},
                        color: '#8D99AE',
                        maxTicksLimit: 12,
                    }},
                    border: {{ color: 'rgba(0,0,0,0.15)' }},
                }},
                y: {{
                    grid: {{ color: 'rgba(0,0,0,0.06)', drawTicks: false }},
                    ticks: {{
                        font: {{ size: 11, family: "'Libre Franklin', sans-serif" }},
                        color: '#8D99AE',
                        callback: function(val) {{
                            let unit = {y_label};
                            if (unit === '%') return val.toFixed(1) + '%';
                            if (unit === '£') return '£' + val.toLocaleString('en-GB');
                            if (Math.abs(val) >= 1000) return val.toLocaleString('en-GB');
                            return val;
                        }}
                    }},
                    border: {{ display: false }},
                }}
            }}
        }}
    }});
"""


def render_html(analysis_result, story_result, output_path: str = None) -> str:
    """
    Generate the complete standalone HTML file.
    Returns the HTML string and optionally writes to output_path.
    """
    ar = analysis_result
    sr = story_result

    today = datetime.now().strftime("%-d %B %Y")
    chart_js = _build_chart_js(ar.chart_spec)

    # Sensitivity banner
    sensitivity_html = ""
    banner = SENSITIVITY_BANNERS.get(sr.sensitivity)
    if banner:
        border_color, bg_color, text = banner
        sensitivity_html = f"""
        <div class="sensitivity-banner" style="background:{bg_color};border-left:4px solid {border_color}">
            <strong style="color:{border_color}">Data note</strong> — {text}
        </div>"""

    # Insights HTML
    insight_type_labels = {
        "trend_break": "Trend break",
        "international": "International",
        "long_run": "Long-run",
        "contradiction": "Surprising finding",
        "demographic": "Demographics",
        "general": "Key finding",
    }
    insights_html = ""
    for ins in ar.insights[:5]:
        type_label = insight_type_labels.get(ins.insight_type, "Finding")
        sig_class = "sig-high" if ins.significance == "HIGH" else "sig-medium"
        insights_html += f"""
        <div class="insight-card {sig_class}">
            <span class="insight-type">{type_label}</span>
            <p class="insight-text">{ins.text}</p>
            <span class="insight-stat">{ins.stat}</span>
        </div>"""

    # Supporting stats
    stats_html = ""
    for stat in ar.supporting_stats[:5]:
        stats_html += f'<li class="stat-item">{stat}</li>'

    # Sources
    sources_html = ""
    for src in ar.sources_used:
        sources_html += f'<span class="source-tag">{src}</span>'

    # Caveats
    caveats_html = ""
    for cav in ar.data_caveats:
        caveats_html += f'<p class="caveat">{cav}</p>'

    # Chart title and subtitle
    chart_title = ar.chart_spec.title
    chart_subtitle = ar.chart_spec.subtitle

    # Search volume indicator
    vol_width = max(10, min(100, sr.search_volume_index))

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Data Analysis: {sr.topic}</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Libre+Baskerville:ital,wght@0,400;0,700;1,400&family=Libre+Franklin:wght@300;400;500;600;700;800&display=swap" rel="stylesheet">
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
<style>
  *, *::before, *::after {{ box-sizing: border-box; margin: 0; padding: 0; }}

  :root {{
    --ink: #1a1a18;
    --ink-mid: #3d3d3a;
    --ink-muted: #6b6b67;
    --ink-faint: #8D99AE;
    --cream: #faf8f4;
    --cream-dark: #f0ece3;
    --rule: #d4cfc5;
    --accent: #C1121F;
    --accent-muted: rgba(193,18,31,0.08);
    --font-serif: 'Libre Baskerville', Georgia, serif;
    --font-sans: 'Libre Franklin', 'Helvetica Neue', sans-serif;
  }}

  html {{ font-size: 16px; }}
  body {{
    font-family: var(--font-sans);
    background: var(--cream);
    color: var(--ink);
    line-height: 1.6;
    -webkit-font-smoothing: antialiased;
  }}

  /* ── Header ─────────────────────────────────────────────── */
  .masthead {{
    border-bottom: 3px solid var(--ink);
    padding: 18px 0 14px;
    margin-bottom: 0;
  }}
  .masthead-inner {{
    max-width: 860px;
    margin: 0 auto;
    padding: 0 24px;
    display: flex;
    align-items: baseline;
    justify-content: space-between;
    gap: 16px;
  }}
  .brand {{
    font-family: var(--font-serif);
    font-size: 13px;
    font-style: italic;
    letter-spacing: 0.5px;
    color: var(--ink-muted);
    white-space: nowrap;
  }}
  .datestamp {{
    font-size: 12px;
    color: var(--ink-faint);
    font-weight: 400;
    white-space: nowrap;
  }}

  /* ── Category tag ────────────────────────────────────────── */
  .category-bar {{
    background: var(--ink);
    padding: 7px 0;
  }}
  .category-bar-inner {{
    max-width: 860px;
    margin: 0 auto;
    padding: 0 24px;
    display: flex;
    align-items: center;
    gap: 12px;
  }}
  .category-label {{
    font-size: 10px;
    font-weight: 700;
    letter-spacing: 1.8px;
    text-transform: uppercase;
    color: #ffffff;
  }}
  .trend-pill {{
    font-size: 10px;
    font-weight: 600;
    letter-spacing: 0.5px;
    color: rgba(255,255,255,0.6);
  }}

  /* ── Main content ────────────────────────────────────────── */
  .content {{
    max-width: 860px;
    margin: 0 auto;
    padding: 0 24px;
  }}

  /* ── Story header ────────────────────────────────────────── */
  .story-header {{
    padding: 40px 0 28px;
    border-bottom: 1px solid var(--rule);
    margin-bottom: 32px;
  }}
  .headline-stat {{
    font-family: var(--font-serif);
    font-size: clamp(28px, 5vw, 44px);
    font-weight: 700;
    line-height: 1.15;
    color: var(--ink);
    margin-bottom: 14px;
    max-width: 720px;
  }}
  .headline-narrative {{
    font-size: 18px;
    font-weight: 300;
    color: var(--ink-mid);
    line-height: 1.45;
    max-width: 640px;
    margin-bottom: 20px;
  }}

  /* Trend indicator */
  .trends-indicator {{
    display: flex;
    align-items: center;
    gap: 10px;
    font-size: 11px;
    color: var(--ink-faint);
    font-weight: 500;
  }}
  .trends-bar-track {{
    width: 120px;
    height: 4px;
    background: var(--rule);
    border-radius: 2px;
    overflow: hidden;
  }}
  .trends-bar-fill {{
    height: 100%;
    background: var(--accent);
    border-radius: 2px;
    width: {vol_width}%;
    transition: width 0.6s ease;
  }}

  /* ── Sensitivity banner ──────────────────────────────────── */
  .sensitivity-banner {{
    padding: 12px 16px;
    margin-bottom: 24px;
    font-size: 13px;
    line-height: 1.55;
    border-radius: 2px;
  }}

  /* ── Lede ────────────────────────────────────────────────── */
  .lede {{
    font-family: var(--font-serif);
    font-size: 17px;
    line-height: 1.7;
    color: var(--ink-mid);
    margin-bottom: 36px;
    max-width: 680px;
  }}

  /* ── Chart section ───────────────────────────────────────── */
  .chart-section {{
    margin-bottom: 40px;
  }}
  .chart-header {{
    margin-bottom: 10px;
  }}
  .chart-title {{
    font-size: 15px;
    font-weight: 600;
    color: var(--ink);
    letter-spacing: -0.2px;
  }}
  .chart-subtitle {{
    font-size: 11px;
    color: var(--ink-faint);
    margin-top: 3px;
    font-weight: 400;
  }}
  .chart-wrapper {{
    background: #ffffff;
    border: 1px solid var(--rule);
    border-top: 3px solid var(--ink);
    padding: 24px 20px 16px;
    position: relative;
  }}
  .chart-canvas-wrap {{
    height: 340px;
    position: relative;
  }}

  /* ── Insights grid ───────────────────────────────────────── */
  .section-label {{
    font-size: 10px;
    font-weight: 700;
    letter-spacing: 2px;
    text-transform: uppercase;
    color: var(--ink-faint);
    margin-bottom: 16px;
    padding-bottom: 8px;
    border-bottom: 1px solid var(--rule);
  }}
  .insights-grid {{
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
    gap: 16px;
    margin-bottom: 40px;
  }}
  .insight-card {{
    background: #ffffff;
    border: 1px solid var(--rule);
    border-left: 3px solid var(--rule);
    padding: 16px 18px;
  }}
  .insight-card.sig-high {{
    border-left-color: var(--accent);
  }}
  .insight-card.sig-medium {{
    border-left-color: #457B9D;
  }}
  .insight-type {{
    font-size: 9px;
    font-weight: 700;
    letter-spacing: 1.5px;
    text-transform: uppercase;
    color: var(--ink-faint);
    display: block;
    margin-bottom: 8px;
  }}
  .insight-text {{
    font-size: 14px;
    line-height: 1.55;
    color: var(--ink-mid);
    margin-bottom: 10px;
  }}
  .insight-stat {{
    font-size: 12px;
    font-weight: 600;
    color: var(--accent);
    background: var(--accent-muted);
    padding: 3px 8px;
    border-radius: 2px;
    display: inline-block;
  }}

  /* ── Stats block ─────────────────────────────────────────── */
  .stats-block {{
    background: var(--ink);
    color: #ffffff;
    padding: 28px 32px;
    margin-bottom: 40px;
  }}
  .stats-block .section-label {{
    color: rgba(255,255,255,0.4);
    border-bottom-color: rgba(255,255,255,0.15);
    margin-bottom: 18px;
  }}
  .stats-list {{
    list-style: none;
    display: flex;
    flex-direction: column;
    gap: 12px;
  }}
  .stat-item {{
    font-size: 14px;
    line-height: 1.5;
    color: rgba(255,255,255,0.85);
    padding-left: 16px;
    position: relative;
  }}
  .stat-item::before {{
    content: '→';
    position: absolute;
    left: 0;
    color: var(--accent);
    font-weight: 700;
  }}

  /* ── Conclusion ──────────────────────────────────────────── */
  .conclusion {{
    padding: 28px 0;
    border-top: 2px solid var(--ink);
    border-bottom: 1px solid var(--rule);
    margin-bottom: 32px;
  }}
  .conclusion-text {{
    font-family: var(--font-serif);
    font-size: 18px;
    line-height: 1.65;
    color: var(--ink);
    font-style: italic;
    max-width: 660px;
  }}
  .conclusion-text::before {{
    content: '"';
    color: var(--accent);
    font-size: 36px;
    line-height: 0;
    vertical-align: -12px;
    margin-right: 4px;
  }}

  /* ── Footer / sources ────────────────────────────────────── */
  .footer-section {{
    padding: 24px 0 40px;
    border-top: 1px solid var(--rule);
  }}
  .sources-wrap {{
    display: flex;
    flex-wrap: wrap;
    gap: 8px;
    margin-bottom: 16px;
  }}
  .source-tag {{
    font-size: 11px;
    background: var(--cream-dark);
    color: var(--ink-mid);
    padding: 3px 10px;
    border-radius: 2px;
    border: 1px solid var(--rule);
  }}
  .caveat {{
    font-size: 12px;
    color: var(--ink-faint);
    line-height: 1.55;
    margin-top: 8px;
  }}

  /* ── Responsive ──────────────────────────────────────────── */
  @media (max-width: 600px) {{
    .masthead-inner {{ flex-direction: column; gap: 4px; }}
    .stats-block {{ padding: 20px; }}
    .chart-canvas-wrap {{ height: 260px; }}
  }}
</style>
</head>
<body>

<header class="masthead">
  <div class="masthead-inner">
    <span class="brand">Data analysis — UK politics &amp; society</span>
    <span class="datestamp">{today}</span>
  </div>
</header>

<div class="category-bar">
  <div class="category-bar-inner">
    <span class="category-label">{sr.category.upper()}</span>
    <span class="trend-pill">Google Trends UK · This week's top story</span>
  </div>
</div>

<main class="content">

  <header class="story-header">
    <h1 class="headline-stat">{ar.headline_stat}</h1>
    <p class="headline-narrative">{ar.headline_narrative}</p>
    <div class="trends-indicator">
      <span>Search interest</span>
      <div class="trends-bar-track">
        <div class="trends-bar-fill"></div>
      </div>
      <span>{sr.search_volume_index}/100</span>
    </div>
  </header>

  {sensitivity_html}

  <p class="lede">{ar.lede_paragraph}</p>

  <!-- Chart -->
  <section class="chart-section">
    <div class="chart-header">
      <div class="chart-title">{chart_title}</div>
      <div class="chart-subtitle">{chart_subtitle}</div>
    </div>
    <div class="chart-wrapper">
      <div class="chart-canvas-wrap">
        <canvas id="mainChart"></canvas>
      </div>
    </div>
  </section>

  <!-- Insights -->
  <div class="section-label">Key findings</div>
  <div class="insights-grid">
    {insights_html}
  </div>

  <!-- Stats block -->
  <div class="stats-block">
    <div class="section-label">By the numbers</div>
    <ul class="stats-list">
      {stats_html}
    </ul>
  </div>

  <!-- Conclusion -->
  <div class="conclusion">
    <p class="conclusion-text">{ar.conclusion}</p>
  </div>

  <!-- Sources & caveats -->
  <div class="footer-section">
    <div class="section-label">Sources</div>
    <div class="sources-wrap">
      {sources_html}
    </div>
    <div class="caveats-wrap">
      {caveats_html}
    </div>
  </div>

</main>

<script>
{chart_js}
</script>

</body>
</html>"""

    if output_path:
        Path(output_path).write_text(html, encoding="utf-8")
        logger.info(f"Output written to {output_path}")

    return html


if __name__ == "__main__":
    # Smoke test with dummy data
    from modules.analyst import AnalysisResult, ChartSpec, ChartSeries, Insight
    from modules.story_detector import StoryResult

    dummy_story = StoryResult(
        topic="UK inflation",
        search_volume_index=82,
        related_queries=["cost of living", "energy bills"],
        category="economy",
        headline_context="Inflation falls but remains above Bank of England target",
        sensitivity="LOW",
    )
    dummy_chart = ChartSpec(
        chart_type="line",
        title="UK CPI inflation rate (%)",
        subtitle="Source: ONS",
        x_labels=["2018", "2019", "2020", "2021", "2022", "2023", "2024"],
        series=[
            ChartSeries("United Kingdom", [2.5, 1.7, 0.9, 2.6, 9.1, 7.3, 2.5]),
            ChartSeries("Germany", [1.9, 1.4, 0.5, 3.2, 8.7, 6.0, 2.2], dash=True),
        ],
        y_label="%",
        annotation="Peak: 9.1% (2022)",
        annotation_index=4,
    )
    dummy_analysis = AnalysisResult(
        story_topic="UK inflation",
        headline_stat="UK inflation hit 9.1% in 2022 — its highest rate in 41 years",
        headline_narrative="The data tells a more complicated story than the headline figures suggest.",
        lede_paragraph="Britain's inflation crisis of 2022–23 was severe by any historical standard, but the recovery has been equally sharp. The latest figures show CPI back at 2.5%, closer to the Bank of England's 2% target than at any point since 2021. What the weekly headlines rarely convey is how this compares internationally — and how unusual the UK's trajectory has been.",
        insights=[
            Insight("UK inflation peaked at 9.1% in 2022 — the highest since 1981", "9.1% peak", "HIGH", "trend_break"),
            Insight("The UK's 2022 peak was higher than Germany (8.7%), France (5.9%) and the EU average", "Higher than EU peers", "HIGH", "international"),
        ],
        chart_spec=dummy_chart,
        supporting_stats=["CPI peaked at 9.1% in June 2022", "Bank of England target: 2%", "Current rate: 2.5% (2024)"],
        conclusion="The inflation shock of 2022 was real and severe — but by 2024 the UK had returned to near-target rates faster than many commentators expected, a pattern obscured by the intensity of the intervening coverage.",
        data_caveats=["All statistics from ONS and World Bank official sources."],
        sources_used=["ONS CPI", "World Bank"],
    )

    html = render_html(dummy_analysis, dummy_story, "/tmp/test_output.html")
    print(f"Generated {len(html):,} chars of HTML")
    print("Saved to /tmp/test_output.html")
