"""
analyst.py
----------
The analytical brain of the agent. Takes a DataPackage and generates
JBM-style insights: surprising findings, international comparisons,
trend breaks, and a strong lede statistic.

Designed around how JBM actually works:
  1. Find the thing the news story implies
  2. Check whether the data supports or contradicts it
  3. Find the international context that reframes it
  4. Find the long-run trend that surprises
  5. Build a narrative arc

Returns: AnalysisResult with headline_stat, insights, chart_spec, narrative_draft
"""

import logging
import os
import json
from dataclasses import dataclass, field
from typing import Optional
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class ChartSeries:
    label: str
    data: list[float]
    color: str = "#e63946"
    dash: bool = False


@dataclass
class ChartSpec:
    chart_type: str          # "line", "bar", "grouped_bar"
    title: str
    subtitle: str
    x_labels: list[str]
    series: list[ChartSeries]
    x_label: str = ""
    y_label: str = ""
    annotation: str = ""     # key callout text on chart
    annotation_index: int = -1
    source_label: str = ""


@dataclass
class Insight:
    text: str
    stat: str                # the key number
    significance: str        # "HIGH" / "MEDIUM" / "LOW"
    insight_type: str        # "trend_break", "international", "demographic", "contradiction", "long_run"


@dataclass
class AnalysisResult:
    story_topic: str
    headline_stat: str           # The one number that opens the piece
    headline_narrative: str      # One sentence that frames the piece
    lede_paragraph: str          # Opening paragraph, JBM style
    insights: list[Insight]
    chart_spec: ChartSpec
    supporting_stats: list[str]  # Bullet-point stats block
    conclusion: str              # Surprising closing observation
    data_caveats: list[str]      # Validation notes
    sources_used: list[str]


# ── Statistical helpers ──────────────────────────────────────────────────────

def _detect_trend_break(series: pd.Series, dates: pd.Series) -> Optional[dict]:
    """
    Detect a meaningful break from the historical trend.
    Returns dict with break_year, pre_avg, post_value, change_pct
    """
    if len(series) < 5:
        return None
    values = series.dropna().values
    if len(values) < 5:
        return None
    # Compare last value to rolling 5-year average
    pre_avg = float(np.mean(values[-6:-1]))
    last_val = float(values[-1])
    if pre_avg == 0:
        return None
    change_pct = ((last_val - pre_avg) / abs(pre_avg)) * 100
    if abs(change_pct) > 10:
        return {
            "pre_avg": pre_avg,
            "post_value": last_val,
            "change_pct": change_pct,
            "break_year": int(dates.iloc[-1]),
            "direction": "risen" if change_pct > 0 else "fallen",
        }
    return None


def _find_extreme_year(df: pd.DataFrame) -> dict:
    """Find the year with the max and min value."""
    if df.empty or "value" not in df.columns:
        return {}
    max_row = df.loc[df["value"].idxmax()]
    min_row = df.loc[df["value"].idxmin()]
    return {
        "max_year": int(max_row["date"]),
        "max_val": float(max_row["value"]),
        "min_year": int(min_row["date"]),
        "min_val": float(min_row["value"]),
    }


def _uk_vs_peers(comp_df: pd.DataFrame, latest_year: Optional[int] = None) -> Optional[dict]:
    """
    Compare UK to peer countries in the most recent shared year.
    Returns rank, n_countries, uk_value, best_country, worst_country.
    """
    if comp_df is None or comp_df.empty:
        return None
    if latest_year is None:
        latest_year = comp_df["date"].max()
    snapshot = comp_df[comp_df["date"] == latest_year].copy()
    if snapshot.empty:
        # Try previous year
        latest_year = comp_df["date"].max() - 1
        snapshot = comp_df[comp_df["date"] == latest_year].copy()
    if snapshot.empty or "country" not in snapshot.columns:
        return None
    snapshot = snapshot.dropna(subset=["value"])
    if len(snapshot) < 2:
        return None
    snapshot = snapshot.sort_values("value", ascending=False).reset_index(drop=True)
    uk_rows = snapshot[snapshot["country"].str.contains("United Kingdom|UK|Britain", case=False)]
    if uk_rows.empty:
        return None
    uk_idx = uk_rows.index[0]
    uk_val = float(uk_rows.iloc[0]["value"])
    rank = int(uk_idx) + 1
    return {
        "year": int(latest_year),
        "uk_value": uk_val,
        "rank": rank,
        "n_countries": len(snapshot),
        "best_country": snapshot.iloc[0]["country"],
        "best_value": float(snapshot.iloc[0]["value"]),
        "worst_country": snapshot.iloc[-1]["country"],
        "worst_value": float(snapshot.iloc[-1]["value"]),
        "all_countries": snapshot[["country", "value"]].to_dict("records"),
    }


def _calculate_change(df: pd.DataFrame, years_back: int = 10) -> Optional[dict]:
    """Calculate absolute and % change over N years."""
    if df.empty:
        return None
    df_sorted = df.sort_values("date")
    recent = df_sorted.tail(1)
    historical = df_sorted[df_sorted["date"] <= df_sorted["date"].max() - years_back]
    if historical.empty:
        historical = df_sorted.head(1)
    old_val = float(historical.iloc[-1]["value"])
    new_val = float(recent.iloc[-1]["value"])
    old_year = int(historical.iloc[-1]["date"])
    new_year = int(recent.iloc[-1]["date"])
    if old_val == 0:
        return None
    pct_change = ((new_val - old_val) / abs(old_val)) * 100
    return {
        "old_year": old_year,
        "new_year": new_year,
        "old_value": old_val,
        "new_value": new_val,
        "abs_change": new_val - old_val,
        "pct_change": pct_change,
    }


def _format_value(val: float, unit: str) -> str:
    """Format a number for display."""
    if unit == "£":
        if val >= 1_000_000:
            return f"£{val/1_000_000:.1f}m"
        elif val >= 1000:
            return f"£{val:,.0f}"
        else:
            return f"£{val:.0f}"
    elif unit == "%":
        return f"{val:.1f}%"
    elif unit == "USD":
        if val >= 1000:
            return f"${val:,.0f}"
        return f"${val:.0f}"
    elif abs(val) >= 1_000_000:
        return f"{val/1_000_000:.1f}m"
    elif abs(val) >= 1000:
        return f"{val:,.0f}"
    else:
        return f"{val:.1f}"


def _build_chart_spec(datasets, category: str, topic: str) -> ChartSpec:
    """Build the primary chart specification — the JBM hero chart."""
    if not datasets:
        return ChartSpec("line", topic, "", [], [])

    # Pick the most interesting dataset for the hero chart
    # Priority: one with comparison data > one with trend break > first available
    hero = None
    for ds in datasets:
        if ds.comparison_data is not None and not ds.comparison_data.empty:
            hero = ds
            break
    if hero is None:
        hero = datasets[0]

    df = hero.data.sort_values("date")
    # Use last 15 years max
    df = df[df["date"] >= df["date"].max() - 15]

    x_labels = [str(int(d)) for d in df["date"].tolist()]
    uk_values = [round(v, 2) for v in df["value"].tolist()]

    # Determine chart type
    if hero.comparison_data is not None and not hero.comparison_data.empty:
        chart_type = "line"
        series = []
        comp = hero.comparison_data.copy()
        # Show UK prominently
        series.append(ChartSeries(
            label="United Kingdom",
            data=uk_values,
            color="#e63946",
        ))
        # Add comparison countries (muted)
        peer_colors = ["#adb5bd", "#6c757d", "#495057", "#868e96", "#ced4da"]
        countries = comp["country"].unique()
        uk_countries = [c for c in countries if "United Kingdom" not in c and "UK" not in c]
        for i, country in enumerate(uk_countries[:4]):
            c_df = comp[comp["country"] == country].sort_values("date")
            c_df = c_df[c_df["date"] >= int(x_labels[0])]
            # Align to same x axis
            c_vals = []
            for yr in [int(x) for x in x_labels]:
                row = c_df[c_df["date"] == yr]
                c_vals.append(round(float(row["value"].iloc[0]), 2) if not row.empty else None)
            series.append(ChartSeries(
                label=country,
                data=c_vals,
                color=peer_colors[i % len(peer_colors)],
                dash=True,
            ))
    else:
        chart_type = "bar" if len(df) <= 12 else "line"
        series = [ChartSeries(
            label=hero.name,
            data=uk_values,
            color="#e63946",
        )]

    # Find annotation point — biggest deviation or most recent notable value
    annotation = ""
    annotation_index = len(uk_values) - 1
    tb = _detect_trend_break(df["value"], df["date"])
    if tb:
        annotation = f"{tb['direction'].capitalize()} {abs(tb['change_pct']):.0f}% above trend"

    return ChartSpec(
        chart_type=chart_type,
        title=hero.name,
        subtitle=f"Source: {hero.source}",
        x_labels=x_labels,
        series=series,
        y_label=hero.unit,
        annotation=annotation,
        annotation_index=annotation_index,
        source_label=f"Source: {hero.source}",
    )


def _call_claude_for_narrative(
    topic: str,
    category: str,
    stats_block: str,
    sensitivity: str,
    sensitivity_notes: list[str],
    headline_context: str,
) -> dict:
    """
    Use Claude claude-sonnet-4-6 to write the JBM-style narrative.
    Falls back to template if API key not available.
    """
    try:
        import anthropic
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("No API key")

        client = anthropic.Anthropic(api_key=api_key)

        sensitivity_instruction = ""
        if sensitivity == "HIGH":
            sensitivity_instruction = (
                "\n\nSENSITIVITY WARNING: This topic is high-sensitivity. "
                "You MUST: (1) anchor every claim to specific cited statistics, "
                "(2) avoid any causal language that could imply blame on a group, "
                "(3) present contradictory evidence where it exists, "
                "(4) use passive/systemic framing rather than attributing outcomes to individuals. "
                f"Notes: {'; '.join(sensitivity_notes)}"
            )
        elif sensitivity == "MEDIUM":
            sensitivity_instruction = (
                "\n\nNote: This topic requires care. Ground all claims in data and avoid overreach."
            )

        prompt = f"""You are a data journalist in the style of John Burn-Murdoch at the Financial Times.
        
Your task: Write a compelling data-driven analysis piece about this UK political story: "{topic}"

News context: {headline_context}

Here are the key statistics from public data sources:
{stats_block}

Write in John Burn-Murdoch's distinctive style:
- Open with ONE surprising statistic that reframes the story (the lede stat)
- Write a sharp, confident lede paragraph (3-4 sentences) that establishes the data angle
- Find 4-5 key insights that are genuinely surprising or counter-intuitive
- Include international comparisons to benchmark the UK
- Find the long-run trend that most people don't know about
- End with a conclusion that surprises — something the news coverage is missing
- Be precise: use actual numbers, percentages, year comparisons
- Tone: authoritative but accessible, never preachy, data-first
- Do NOT editorialize politically — let the numbers speak{sensitivity_instruction}

Respond ONLY as a JSON object with these exact keys:
{{
  "headline_stat": "The single most striking statistic in one sentence",
  "headline_narrative": "One sentence framing the data angle",
  "lede_paragraph": "Opening paragraph (3-4 sentences)",
  "insights": [
    {{"text": "insight text", "stat": "key number", "type": "trend_break|international|long_run|contradiction"}},
    ...
  ],
  "supporting_stats": ["stat 1", "stat 2", "stat 3", "stat 4", "stat 5"],
  "conclusion": "The surprising closing observation",
  "chart_annotation": "Short annotation for the key chart moment (max 8 words)"
}}"""

        response = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=2000,
            messages=[{"role": "user", "content": prompt}],
        )
        text = response.content[0].text.strip()
        # Extract JSON
        if "```json" in text:
            text = text.split("```json")[1].split("```")[0].strip()
        elif "```" in text:
            text = text.split("```")[1].split("```")[0].strip()
        return json.loads(text)

    except Exception as e:
        logger.warning(f"Claude API call failed: {e} — using template fallback")
        return None


def analyse(data_package, story_result) -> AnalysisResult:
    """
    Main analysis entry point.
    Takes a DataPackage and StoryResult, returns a full AnalysisResult.
    """
    topic = story_result.topic
    category = story_result.category
    sensitivity = story_result.sensitivity
    datasets = data_package.datasets

    # ── Step 1: Extract raw statistics ──────────────────────────────────────
    insights_raw = []
    stats_lines = []
    sources_used = []

    for ds in datasets:
        if ds.data.empty:
            continue
        df = ds.data.sort_values("date")
        unit = ds.unit
        sources_used.append(f"{ds.source}: {ds.name}")

        # Latest value
        latest = df.iloc[-1]
        latest_str = _format_value(latest["value"], unit)
        stats_lines.append(f"- {ds.name}: {latest_str} ({int(latest['date'])})")

        # Trend break
        tb = _detect_trend_break(df["value"], df["date"])
        if tb:
            val_str = _format_value(tb["post_value"], unit)
            avg_str = _format_value(tb["pre_avg"], unit)
            insight_text = (
                f"{ds.name} reached {val_str} in {tb['break_year']}, "
                f"compared to a 5-year average of {avg_str} — "
                f"a {abs(tb['change_pct']):.0f}% {'increase' if tb['change_pct'] > 0 else 'decrease'}"
            )
            insights_raw.append(Insight(
                text=insight_text,
                stat=f"{abs(tb['change_pct']):.0f}% {'above' if tb['change_pct'] > 0 else 'below'} recent trend",
                significance="HIGH",
                insight_type="trend_break",
            ))
            stats_lines.append(f"  → Trend break: {abs(tb['change_pct']):.0f}% from 5yr average")

        # 10-year change
        ch = _calculate_change(df, years_back=10)
        if ch:
            old_str = _format_value(ch["old_value"], unit)
            new_str = _format_value(ch["new_value"], unit)
            insight_text = (
                f"{ds.name} has {'risen' if ch['pct_change'] > 0 else 'fallen'} "
                f"from {old_str} in {ch['old_year']} to {new_str} in {ch['new_year']} "
                f"({abs(ch['pct_change']):.0f}% {'increase' if ch['pct_change'] > 0 else 'decrease'})"
            )
            insights_raw.append(Insight(
                text=insight_text,
                stat=f"{abs(ch['pct_change']):.0f}% change over {ch['new_year'] - ch['old_year']} years",
                significance="MEDIUM",
                insight_type="long_run",
            ))
            stats_lines.append(f"  → 10yr change: {ch['old_year']} {old_str} → {ch['new_year']} {new_str} ({ch['pct_change']:+.1f}%)")

        # International comparison
        if ds.comparison_data is not None:
            peer = _uk_vs_peers(ds.comparison_data)
            if peer:
                uk_str = _format_value(peer["uk_value"], unit)
                insight_text = (
                    f"On {ds.name}, the UK ranks {peer['rank']} of {peer['n_countries']} comparable nations "
                    f"at {uk_str} — compared to {peer['best_country']} "
                    f"({_format_value(peer['best_value'], unit)}) at the top"
                )
                insights_raw.append(Insight(
                    text=insight_text,
                    stat=f"Rank {peer['rank']}/{peer['n_countries']}",
                    significance="HIGH",
                    insight_type="international",
                ))
                stats_lines.append(f"  → International rank: {peer['rank']}/{peer['n_countries']} ({peer['year']})")
                for c in peer.get("all_countries", [])[:5]:
                    stats_lines.append(f"    {c['country']}: {_format_value(c['value'], unit)}")

        # Extremes
        ext = _find_extreme_year(df)
        if ext and ext.get("max_year") != int(latest["date"]):
            stats_lines.append(
                f"  → Peak: {_format_value(ext['max_val'], unit)} ({ext['max_year']}), "
                f"Low: {_format_value(ext['min_val'], unit)} ({ext['min_year']})"
            )

    stats_block = "\n".join(stats_lines)

    # ── Step 2: Try Claude for narrative, fall back to template ─────────────
    claude_result = _call_claude_for_narrative(
        topic=topic,
        category=category,
        stats_block=stats_block,
        sensitivity=sensitivity,
        sensitivity_notes=story_result.sensitivity_notes,
        headline_context=story_result.headline_context,
    )

    # ── Step 3: Build final result ───────────────────────────────────────────
    if claude_result:
        headline_stat = claude_result.get("headline_stat", "")
        headline_narrative = claude_result.get("headline_narrative", "")
        lede = claude_result.get("lede_paragraph", "")
        supporting = claude_result.get("supporting_stats", [])
        conclusion = claude_result.get("conclusion", "")

        # Merge Claude insights with statistical insights
        for ci in claude_result.get("insights", []):
            insights_raw.insert(0, Insight(
                text=ci.get("text", ""),
                stat=ci.get("stat", ""),
                significance="HIGH",
                insight_type=ci.get("type", "general"),
            ))
    else:
        # Template fallback — build rich narrative from actual computed stats
        # Find the most striking insight of each type
        trend_breaks = [i for i in insights_raw if i.insight_type == "trend_break"]
        international = [i for i in insights_raw if i.insight_type == "international"]
        long_run = [i for i in insights_raw if i.insight_type == "long_run"]

        # Build headline from the most significant trend break or long-run change
        if trend_breaks:
            top = trend_breaks[0]
            headline_stat = top.text.split(" — ")[0] if " — " in top.text else top.text[:80]
        elif long_run:
            top = long_run[0]
            headline_stat = top.text.split(" — ")[0] if " — " in top.text else top.text[:80]
        elif insights_raw:
            top = insights_raw[0]
            headline_stat = top.text[:80]
        else:
            headline_stat = f"New official data published on {topic}"

        # Build headline narrative from international comparison if available
        if international:
            headline_narrative = (
                f"The UK's position among comparable nations tells a story "
                f"more nuanced than this week's coverage suggests: {international[0].stat.lower()}"
            )
        elif trend_breaks:
            direction = "upward" if "risen" in trend_breaks[0].text or "increase" in trend_breaks[0].text else "downward"
            headline_narrative = (
                f"A sustained {direction} shift in the underlying data "
                f"predates this week's story — and changes what it means"
            )
        else:
            headline_narrative = (
                f"The long-run trajectory of {topic.lower()} "
                f"reveals patterns that rarely surface in weekly news coverage"
            )

        # Lede: open with the most striking number, contextualise, hint at surprise
        lede_sentences = []
        if datasets and not datasets[0].data.empty:
            ds0 = datasets[0]
            latest = ds0.data.sort_values("date").iloc[-1]
            earliest = ds0.data.sort_values("date").iloc[0]
            lede_sentences.append(
                f"The latest official figures put {ds0.name.lower()} at "
                f"{_format_value(latest['value'], ds0.unit)} — "
                f"a figure that looks very different when set against "
                f"the {int(latest['date']) - int(earliest['date'])}-year record."
            )
        if trend_breaks:
            lede_sentences.append(trend_breaks[0].text + ".")
        if international:
            lede_sentences.append(
                f"Internationally, {international[0].text.lower()}"
                if not international[0].text[0].islower() else international[0].text + "."
            )
        if not lede_sentences:
            lede_sentences = [
                f"A data analysis of {topic} drawing on ONS and World Bank official statistics "
                f"reveals a more complex picture than this week's headlines suggest.",
                "The long-run trajectory matters as much as the most recent number.",
            ]
        lede = " ".join(lede_sentences[:3])

        supporting = [i.text for i in insights_raw[:5]]

        # Conclusion: find the most counter-intuitive finding
        if long_run:
            lr = long_run[0]
            direction = "risen" if "risen" in lr.text or "increase" in lr.text else "fallen"
            opposite = "fallen" if direction == "risen" else "risen"
            conclusion = (
                f"Perhaps the most striking finding from the long-run data is not what has changed recently, "
                f"but what changed well before this week's story. The structural shift was already underway — "
                f"the current headlines are arriving late to a trend that official statistics have been "
                f"recording for years. {lr.text}"
            )
        elif international:
            conclusion = (
                f"The international comparison cuts against the most common framing of this story. "
                f"Where the news coverage tends to treat the UK's situation as exceptional, "
                f"the cross-country data tells a different story: {international[0].text.lower()}"
            )
        else:
            conclusion = (
                f"The data on {topic} rewards patience: the weekly signal is noisy, "
                f"but the multi-year trend is clear. Understanding which one this week's "
                f"story is actually about is the crucial analytical question the headline figures alone cannot answer."
            )

    # ── Step 4: Build chart ──────────────────────────────────────────────────
    chart_spec = _build_chart_spec(datasets, category, topic)
    if claude_result and claude_result.get("chart_annotation"):
        chart_spec.annotation = claude_result["chart_annotation"]

    # ── Step 5: Validation caveats ───────────────────────────────────────────
    caveats = []
    if data_package.fetch_errors:
        caveats.append(f"Data retrieval issues: {len(data_package.fetch_errors)} sources unavailable")
    if sensitivity != "LOW":
        caveats.append(
            f"Sensitivity level {sensitivity}: statistics on this topic require careful "
            "contextualisation. All figures sourced from official statistics only."
        )
    if any(ds.source == "fallback" for ds in datasets if hasattr(ds, 'source')):
        caveats.append("Some datasets use curated recent figures where live API data was unavailable.")
    caveats.append("All statistics reflect most recently published official data. Methodological notes available at source.")

    return AnalysisResult(
        story_topic=topic,
        headline_stat=headline_stat,
        headline_narrative=headline_narrative,
        lede_paragraph=lede,
        insights=insights_raw[:6],
        chart_spec=chart_spec,
        supporting_stats=supporting[:5] if supporting else [i.text for i in insights_raw[:5]],
        conclusion=conclusion,
        data_caveats=caveats,
        sources_used=list(set(sources_used)),
    )
