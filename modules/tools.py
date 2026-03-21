"""
tools.py
--------
Exposes all data-fetching and analysis capabilities as clean tool
definitions for the ReAct agent.

Each tool has:
  - name: used by Claude to call it
  - description: what it does and when to use it (Claude reads this)
  - parameters: JSON-schema style input spec
  - fn: the actual Python callable

The agent sees the descriptions and schemas; it never sees the implementation.
"""

import logging
import time
from typing import Any
import pandas as pd

logger = logging.getLogger(__name__)

# ── Import fetch functions from data_router ──────────────────────────────────
from modules.data_router import (
    _fetch_worldbank,
    _fetch_worldbank_comparison,
    _fetch_ons_timeseries,
    _fetch_fred,
    _ons_search_fallback,
    _make_fallback_dataset,
    _infer_unit,
    FALLBACK_DATA,
    COMPARISON_COUNTRIES,
    Dataset,
)


# ── Tool result types ────────────────────────────────────────────────────────

class ToolResult:
    """Returned by every tool call. Agent reads summary; code uses data."""
    def __init__(self, success: bool, summary: str, data: Any = None, error: str = ""):
        self.success = success
        self.summary = summary   # what the agent sees in its scratchpad
        self.data = data         # raw data for downstream use
        self.error = error

    def __repr__(self):
        return f"ToolResult(success={self.success}, summary={self.summary[:80]!r})"


# ── Individual tool implementations ─────────────────────────────────────────

def tool_fetch_fred(series_id: str, label: str) -> ToolResult:
    """
    Fetch a UK economic time series from FRED (Federal Reserve Economic Data).
    FRED is the most reliable source — try this first for economic indicators.

    Good series IDs:
      CPGRLE01GBM659N  — UK CPI inflation YoY (%)
      LRHUTTTTGBM156S  — UK unemployment rate (%)
      QGBR628BIS       — UK real house price index
      LCEAMN01GBM189S  — UK average monthly earnings
      IRLTLT01GBM156N  — UK long-term interest rates
    """
    df = _fetch_fred(series_id, label)
    if df is not None and len(df) >= 3:
        latest = df.iloc[-1]
        oldest = df.iloc[0]
        unit = _infer_unit(label)
        return ToolResult(
            success=True,
            summary=(
                f"FRED '{label}': {len(df)} annual data points "
                f"({int(oldest['date'])}–{int(latest['date'])}). "
                f"Latest value: {latest['value']:.2f}{unit}."
            ),
            data=Dataset(
                name=label, source="FRED", series_id=series_id,
                description=label, data=df, unit=unit,
                source_url=f"https://fred.stlouisfed.org/series/{series_id}",
            ),
        )
    # Try fallback
    fb = _make_fallback_dataset(label)
    if fb:
        return ToolResult(
            success=True,
            summary=f"FRED unavailable for '{label}' — using curated fallback ({len(fb.data)} points).",
            data=fb,
        )
    return ToolResult(success=False, summary=f"No data found for FRED series {series_id}.", error="fetch_failed")


def tool_fetch_worldbank(indicator: str, label: str, country: str = "GBR") -> ToolResult:
    """
    Fetch a World Bank indicator for the UK (or another country).
    Best for long-run series: inequality, health spend, life expectancy, CO2.

    Useful indicators:
      NY.GDP.PCAP.CD    — GDP per capita (USD)
      SH.XPD.CHEX.GD.ZS — Health expenditure % GDP
      SH.MED.BEDS.ZS    — Hospital beds per 1,000
      SP.DYN.LE00.IN    — Life expectancy at birth
      SI.POV.GINI       — Gini inequality index
      EN.ATM.CO2E.PC    — CO2 emissions per capita
      EG.ELC.RNEW.ZS    — Renewable electricity %
      SM.POP.NETM       — Net migration
      SE.XPD.TOTL.GD.ZS — Education spend % GDP
    """
    series_id = f"{country}/{indicator}"
    df = _fetch_worldbank(series_id, label)
    if df is not None and len(df) >= 3:
        latest = df.iloc[-1]
        unit = _infer_unit(label)
        return ToolResult(
            success=True,
            summary=(
                f"World Bank '{label}' ({country}): {len(df)} data points. "
                f"Latest: {latest['value']:.2f}{unit} ({int(latest['date'])})."
            ),
            data=Dataset(
                name=label, source="World Bank", series_id=series_id,
                description=label, data=df, unit=unit,
                source_url=f"https://data.worldbank.org/indicator/{indicator}",
            ),
        )
    fb = _make_fallback_dataset(label)
    if fb:
        return ToolResult(
            success=True,
            summary=f"World Bank unavailable for '{label}' — using fallback ({len(fb.data)} points).",
            data=fb,
        )
    return ToolResult(success=False, summary=f"No World Bank data for {indicator}.", error="fetch_failed")


def tool_fetch_comparison(indicator: str, label: str, countries: list[str] = None) -> ToolResult:
    """
    Fetch a World Bank indicator for multiple countries simultaneously.
    Use this to build the international comparison chart — JBM's signature move.
    Returns data for UK plus peer countries for side-by-side benchmarking.

    Use after fetching UK data to add comparative context.
    Default peer group: UK, Germany, France, USA, Sweden, Italy.
    """
    if countries is None:
        countries = ["GBR", "DEU", "FRA", "USA", "SWE", "ITA"]
    df = _fetch_worldbank_comparison(indicator, countries)
    if df is not None and not df.empty:
        n_countries = df["country"].nunique()
        latest_year = int(df["date"].max())
        snapshot = df[df["date"] == latest_year].sort_values("value", ascending=False)
        top3 = ", ".join(
            f"{row['country']} ({row['value']:.1f})"
            for _, row in snapshot.head(3).iterrows()
        )
        return ToolResult(
            success=True,
            summary=(
                f"Comparison data for '{label}': {n_countries} countries, "
                f"latest year {latest_year}. Top 3: {top3}."
            ),
            data=df,
        )
    return ToolResult(success=False, summary=f"Comparison data unavailable for {indicator}.", error="fetch_failed")


def tool_fetch_ons(dataset_series: str, label: str) -> ToolResult:
    """
    Fetch a time series from the ONS (Office for National Statistics) API.
    Format: 'DATASET/SERIES_ID' e.g. 'LMS/MGSX'

    Verified working series:
      LMS/MGSX    — UK unemployment rate (%)
      EARN/KAB9   — Average weekly earnings (£)
      MM23/D7G7   — CPI inflation rate (%)
      HPM1/UKMHPSA — UK average house price (£)
    """
    df = _fetch_ons_timeseries(dataset_series, label)
    if df is not None and len(df) >= 3:
        latest = df.iloc[-1]
        unit = _infer_unit(label)
        return ToolResult(
            success=True,
            summary=(
                f"ONS '{label}': {len(df)} data points. "
                f"Latest: {latest['value']:.2f}{unit} ({int(latest['date'])})."
            ),
            data=Dataset(
                name=label, source="ONS", series_id=dataset_series,
                description=label, data=df, unit=unit,
                source_url="https://www.ons.gov.uk",
            ),
        )
    fb = _make_fallback_dataset(label)
    if fb:
        return ToolResult(
            success=True,
            summary=f"ONS API unavailable for '{label}' — using fallback ({len(fb.data)} points).",
            data=fb,
        )
    return ToolResult(success=False, summary=f"No ONS data for {dataset_series}.", error="fetch_failed")


def tool_summarise_dataset(dataset: Dataset) -> ToolResult:
    """
    Compute key statistics from a fetched dataset: trend direction,
    10-year change, peak/trough, recent acceleration.
    Call this after fetching a dataset to extract the analytical signal.
    """
    df = dataset.data.sort_values("date")
    if df.empty or len(df) < 3:
        return ToolResult(success=False, summary="Dataset too small to summarise.", error="insufficient_data")

    unit = dataset.unit
    latest = df.iloc[-1]
    oldest = df.iloc[0]

    def fmt(v):
        if unit == "%": return f"{v:.1f}%"
        if unit == "£": return f"£{v:,.0f}"
        if abs(v) >= 1000: return f"{v:,.0f}"
        return f"{v:.2f}"

    # 10-year change
    decade_df = df[df["date"] >= df["date"].max() - 10]
    decade_start = decade_df.iloc[0]
    abs_change = latest["value"] - decade_start["value"]
    pct_change = (abs_change / max(abs(decade_start["value"]), 0.001)) * 100

    # 5-year avg vs latest (trend break detection)
    recent_5 = df[df["date"] >= df["date"].max() - 5]["value"].mean()
    pre_5 = df[df["date"] < df["date"].max() - 5]["value"].mean()
    trend_direction = "rising" if latest["value"] > recent_5 else "falling"

    # Peak and trough
    peak_row = df.loc[df["value"].idxmax()]
    trough_row = df.loc[df["value"].idxmin()]

    summary = (
        f"'{dataset.name}': "
        f"latest {fmt(latest['value'])} ({int(latest['date'])}), "
        f"10yr change {pct_change:+.1f}% ({fmt(decade_start['value'])} in {int(decade_start['date'])}), "
        f"trend {trend_direction}. "
        f"Peak: {fmt(peak_row['value'])} ({int(peak_row['date'])}), "
        f"trough: {fmt(trough_row['value'])} ({int(trough_row['date'])})."
    )
    return ToolResult(success=True, summary=summary, data={
        "latest_value": float(latest["value"]),
        "latest_year": int(latest["date"]),
        "pct_change_10yr": float(pct_change),
        "abs_change_10yr": float(abs_change),
        "decade_start_value": float(decade_start["value"]),
        "decade_start_year": int(decade_start["date"]),
        "trend_direction": trend_direction,
        "recent_5yr_avg": float(recent_5),
        "pre_5yr_avg": float(pre_5),
        "peak_value": float(peak_row["value"]),
        "peak_year": int(peak_row["date"]),
        "trough_value": float(trough_row["value"]),
        "trough_year": int(trough_row["date"]),
    })


def tool_summarise_comparison(comp_df: pd.DataFrame, label: str, unit: str = "") -> ToolResult:
    """
    Summarise an international comparison dataset — find the UK's rank,
    identify outliers, and surface the most interesting cross-country pattern.
    Call this after tool_fetch_comparison to extract the insight.
    """
    if comp_df is None or comp_df.empty:
        return ToolResult(success=False, summary="No comparison data to summarise.", error="no_data")

    latest_year = int(comp_df["date"].max())
    snapshot = comp_df[comp_df["date"] == latest_year].dropna(subset=["value"])
    if snapshot.empty:
        latest_year -= 1
        snapshot = comp_df[comp_df["date"] == latest_year].dropna(subset=["value"])
    if snapshot.empty:
        return ToolResult(success=False, summary="Cannot find shared year for comparison.", error="no_data")

    snapshot = snapshot.sort_values("value", ascending=False).reset_index(drop=True)
    uk_rows = snapshot[snapshot["country"].str.contains("United Kingdom|UK|Britain", case=False, na=False)]
    if uk_rows.empty:
        return ToolResult(success=False, summary="UK not found in comparison data.", error="no_uk_data")

    uk_rank = int(uk_rows.index[0]) + 1
    uk_val = float(uk_rows.iloc[0]["value"])
    n = len(snapshot)
    best = snapshot.iloc[0]
    worst = snapshot.iloc[-1]

    def fmt(v):
        if unit == "%": return f"{v:.1f}%"
        if unit == "£": return f"£{v:,.0f}"
        if abs(v) >= 1000: return f"{v:,.1f}"
        return f"{v:.2f}"

    all_vals = [f"{r['country']} {fmt(r['value'])}" for _, r in snapshot.iterrows()]

    summary = (
        f"'{label}' international comparison ({latest_year}): "
        f"UK ranks {uk_rank}/{n} at {fmt(uk_val)}. "
        f"Highest: {best['country']} ({fmt(best['value'])}). "
        f"Lowest: {worst['country']} ({fmt(worst['value'])}). "
        f"All: {'; '.join(all_vals)}."
    )
    return ToolResult(success=True, summary=summary, data={
        "year": latest_year,
        "uk_value": uk_val,
        "uk_rank": uk_rank,
        "n_countries": n,
        "best_country": best["country"],
        "best_value": float(best["value"]),
        "worst_country": worst["country"],
        "worst_value": float(worst["value"]),
        "all": [{"country": r["country"], "value": float(r["value"])} for _, r in snapshot.iterrows()],
    })


def tool_finish(reasoning: str) -> ToolResult:
    """
    Signal that the agent has gathered sufficient data and reasoning
    to write the final JBM-style analysis. Call this when you have:
      - At least 2 datasets with computed statistics
      - At least 1 international comparison (if available)
      - A clear headline finding and a surprising insight
    Pass a full reasoning summary that will be handed to the writer.
    """
    return ToolResult(success=True, summary=f"FINISH: {reasoning}", data=reasoning)


# ── Tool registry ─────────────────────────────────────────────────────────────

TOOLS = {
    "fetch_fred": {
        "fn": tool_fetch_fred,
        "description": tool_fetch_fred.__doc__,
        "params": ["series_id: str", "label: str"],
    },
    "fetch_worldbank": {
        "fn": tool_fetch_worldbank,
        "description": tool_fetch_worldbank.__doc__,
        "params": ["indicator: str", "label: str", "country: str = 'GBR'"],
    },
    "fetch_comparison": {
        "fn": tool_fetch_comparison,
        "description": tool_fetch_comparison.__doc__,
        "params": ["indicator: str", "label: str", "countries: list[str] = None"],
    },
    "fetch_ons": {
        "fn": tool_fetch_ons,
        "description": tool_fetch_ons.__doc__,
        "params": ["dataset_series: str", "label: str"],
    },
    "summarise_dataset": {
        "fn": tool_summarise_dataset,
        "description": tool_summarise_dataset.__doc__,
        "params": ["dataset: Dataset"],
    },
    "summarise_comparison": {
        "fn": tool_summarise_comparison,
        "description": tool_summarise_comparison.__doc__,
        "params": ["comp_df: DataFrame", "label: str", "unit: str = ''"],
    },
    "finish": {
        "fn": tool_finish,
        "description": tool_finish.__doc__,
        "params": ["reasoning: str"],
    },
}


def describe_tools() -> str:
    """Return a formatted description of all tools for the system prompt."""
    lines = []
    for name, spec in TOOLS.items():
        params = ", ".join(spec["params"])
        # Take first paragraph of docstring as the short description
        short_desc = (spec["description"] or "").strip().split("\n\n")[0].strip()
        lines.append(f"- {name}({params})\n  {short_desc}")
    return "\n\n".join(lines)
