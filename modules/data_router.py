"""
data_router.py
--------------
Maps story categories to relevant UK public data sources, fetches
and returns structured datasets ready for analysis.

Primary sources (in preference order per category):
  - ONS (Office for National Statistics) - beta.ons.gov.uk API
  - FRED (Federal Reserve Economic Data) - has UK series
  - World Bank Open Data API
  - ONS Time Series Explorer (backup)
  - Fallback: curated static recent datasets

Returns: DataPackage(datasets, metadata, source_urls)
"""

import logging
import time
import requests
from dataclasses import dataclass, field
from typing import Optional
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

# ── ONS API config ──────────────────────────────────────────────────────────
ONS_BASE = "https://api.beta.ons.gov.uk/v1"
ONS_TIMESERIES_BASE = "https://api.ons.gov.uk/dataset/{dataset}/timeseries/{series}/data"

# ── FRED API config ─────────────────────────────────────────────────────────
FRED_BASE = "https://api.stlouisfed.org/fred/series/observations"

# ── World Bank API ──────────────────────────────────────────────────────────
WB_BASE = "https://api.worldbank.org/v2/country/{country}/indicator/{indicator}"

# ── Category → Data Source Map ───────────────────────────────────────────────
# Each entry: list of (source_name, fetch_fn_name, series_id, label, description)
CATEGORY_SOURCES = {
    "economy": [
        # FRED first — most reliable API, good historical depth
        ("FRED", "fred_series",   "CPGRLE01GBM659N",  "UK CPI inflation rate (%)",         "UK CPI All Items YoY via FRED"),
        ("ONS",  "ons_timeseries","LMS/MGSX",         "UK unemployment rate (%)",          "Unemployment rate 16+, seasonally adjusted"),
        ("ONS",  "ons_timeseries","UKEA/IHYP",        "UK GDP growth (%)",                 "GDP QoQ growth, chained volume"),
        ("ONS",  "ons_timeseries","EARN/KAB9",        "UK average weekly earnings (£)",    "Average weekly earnings, total pay, GB"),
        ("WorldBank","worldbank_series","GBR/NY.GDP.PCAP.CD","UK GDP per capita (USD)",    "GDP per capita current USD"),
        ("FRED", "fred_series",   "LRHUTTTTGBM156S",  "UK unemployment rate FRED (%)",     "UK unemployment via FRED backup"),
    ],
    "health": [
        ("WorldBank","worldbank_series","GBR/SH.MED.BEDS.ZS",   "Hospital beds per 1,000", "Hospital beds per 1,000 people"),
        ("WorldBank","worldbank_series","GBR/SH.XPD.CHEX.GD.ZS","Health spend % GDP",      "Current health expenditure % GDP"),
        ("WorldBank","worldbank_series","GBR/SP.DYN.LE00.IN",    "UK life expectancy",      "Life expectancy at birth, total"),
        ("WorldBank","worldbank_series","GBR/SH.MED.PHYS.ZS",    "Doctors per 1,000",       "Physicians per 1,000 people"),
    ],
    "immigration": [
        ("WorldBank","worldbank_series","GBR/SM.POP.NETM",       "UK net migration (WB)",   "Net migration"),
        ("WorldBank","worldbank_series","GBR/SP.POP.TOTL",       "UK population",           "Population total"),
        ("ONS",  "ons_search",    "long term international migration","UK long-term migration","Long-term international migration"),
    ],
    "housing": [
        ("ONS",  "ons_timeseries","HPM1/UKMHPSA",     "UK average house price (£)",        "UK HPI average price, seasonally adjusted"),
        ("FRED", "fred_series",   "QGBR628BIS",       "UK real house price index",          "UK real residential property prices via FRED"),
        ("WorldBank","worldbank_series","GBR/NY.GDP.PCAP.CD","UK GDP per capita (USD)",      "GDP per capita, affordability context"),
    ],
    "crime": [
        ("ONS",  "ons_search",    "crime England Wales","Crime rate England & Wales",        "Crime in England and Wales"),
        ("WorldBank","worldbank_series","GBR/VC.IHR.PSRC.P5","Homicide rate per 100k",      "Intentional homicides per 100,000"),
    ],
    "education": [
        ("WorldBank","worldbank_series","GBR/SE.XPD.TOTL.GD.ZS","Education spend % GDP",   "Government expenditure on education % GDP"),
        ("WorldBank","worldbank_series","GBR/SE.TER.ENRR",      "Tertiary enrolment %",     "School enrolment tertiary gross"),
    ],
    "welfare": [
        ("WorldBank","worldbank_series","GBR/SI.POV.GINI",      "UK Gini coefficient",      "Gini index"),
        ("WorldBank","worldbank_series","GBR/SI.DST.FRST.20",   "Income share bottom 20%",  "Income share held by lowest 20%"),
        ("WorldBank","worldbank_series","GBR/SI.DST.10TH.10",   "Income share top 10%",     "Income share held by highest 10%"),
    ],
    "environment": [
        ("WorldBank","worldbank_series","GBR/EN.ATM.CO2E.PC",   "UK CO2 per capita",        "CO2 emissions metric tons per capita"),
        ("WorldBank","worldbank_series","GBR/EG.ELC.RNEW.ZS",   "Renewable electricity %",  "Renewable electricity % of total"),
        ("WorldBank","worldbank_series","GBR/EG.USE.PCAP.KG.OE","Energy use per capita",    "Energy use kg of oil equivalent"),
    ],
    "politics": [
        ("FRED", "fred_series",   "CPGRLE01GBM659N",  "UK CPI inflation rate (%)",         "Inflation — key political pressure metric"),
        ("ONS",  "ons_timeseries","LMS/MGSX",         "UK unemployment rate (%)",          "Unemployment rate — political key metric"),
        ("WorldBank","worldbank_series","GBR/NY.GDP.PCAP.CD","UK GDP per capita (USD)",      "GDP per capita current USD"),
    ],
}

# Comparative countries for international context (JBM style)
COMPARISON_COUNTRIES = {
    "economy": ["GBR", "DEU", "FRA", "USA", "ITA", "ESP"],
    "health": ["GBR", "DEU", "FRA", "CAN", "AUS", "SWE"],
    "housing": ["GBR", "DEU", "FRA", "NLD", "SWE", "AUS"],
    "welfare": ["GBR", "DEU", "FRA", "SWE", "DNK", "USA"],
    "environment": ["GBR", "DEU", "FRA", "USA", "CHN", "SWE"],
    "immigration": ["GBR", "DEU", "FRA", "SWE", "CAN", "AUS"],
}


@dataclass
class Dataset:
    name: str
    source: str
    series_id: str
    description: str
    data: pd.DataFrame           # columns: date, value
    unit: str = ""
    frequency: str = "annual"
    source_url: str = ""
    comparison_data: Optional[pd.DataFrame] = None  # multi-country for comparisons


@dataclass
class DataPackage:
    story_topic: str
    category: str
    datasets: list[Dataset]
    fetch_errors: list[str]
    source_urls: list[str]


# ── Fetch functions ──────────────────────────────────────────────────────────

def _fetch_worldbank(country_indicator: str, description: str) -> Optional[pd.DataFrame]:
    """Fetch a World Bank indicator time series for UK + comparison countries."""
    try:
        parts = country_indicator.split("/")
        country = parts[0]
        indicator = "/".join(parts[1:])
        url = f"https://api.worldbank.org/v2/country/{country}/indicator/{indicator}"
        params = {
            "format": "json",
            "per_page": 60,
            "mrv": 30,
        }
        resp = requests.get(url, params=params, timeout=15)
        resp.raise_for_status()
        data = resp.json()
        if len(data) < 2 or not data[1]:
            return None
        records = []
        for entry in data[1]:
            if entry.get("value") is not None:
                records.append({
                    "date": int(entry["date"]),
                    "value": float(entry["value"]),
                    "country": entry.get("country", {}).get("value", country),
                    "country_code": country,
                })
        if not records:
            return None
        df = pd.DataFrame(records).sort_values("date")
        return df
    except Exception as e:
        logger.warning(f"World Bank fetch failed [{country_indicator}]: {e}")
        return None


def _fetch_worldbank_comparison(indicator: str, countries: list[str]) -> Optional[pd.DataFrame]:
    """Fetch same indicator for multiple countries for international comparison."""
    all_dfs = []
    for country in countries:
        df = _fetch_worldbank(f"{country}/{indicator}", "")
        if df is not None and not df.empty:
            all_dfs.append(df)
        time.sleep(0.3)
    if not all_dfs:
        return None
    return pd.concat(all_dfs, ignore_index=True)


def _fetch_ons_timeseries(dataset_series: str, description: str) -> Optional[pd.DataFrame]:
    """
    Fetch ONS time series data.
    Format: 'DATASET/SERIES' e.g. 'LMS/MGSX'
    """
    try:
        parts = dataset_series.split("/")
        if len(parts) != 2:
            return None
        dataset, series = parts
        url = f"https://api.ons.gov.uk/dataset/{dataset}/timeseries/{series}/data"
        headers = {"Accept": "application/json"}
        resp = requests.get(url, headers=headers, timeout=15)
        if resp.status_code != 200:
            return None
        data = resp.json()
        # Annual data preferred
        records = []
        for section in ["years", "quarters", "months"]:
            entries = data.get(section, [])
            if entries:
                for entry in entries:
                    try:
                        val = float(entry.get("value", "").replace(",", ""))
                        date_str = entry.get("date", entry.get("year", ""))
                        # Parse year from various formats
                        year = int(str(date_str)[:4])
                        records.append({"date": year, "value": val, "country": "United Kingdom"})
                    except (ValueError, TypeError):
                        continue
                if records:
                    break  # prefer annual; if found, stop
        if not records:
            return None
        df = pd.DataFrame(records).sort_values("date").drop_duplicates("date")
        return df
    except Exception as e:
        logger.warning(f"ONS timeseries fetch failed [{dataset_series}]: {e}")
        return None


def _fetch_fred(series_id: str, description: str) -> Optional[pd.DataFrame]:
    """Fetch FRED series (UK-relevant series available on FRED)."""
    try:
        # FRED requires API key — use the public observation endpoint
        # which works without key for some series, or fall through
        url = "https://fred.stlouisfed.org/graph/fredgraph.csv"
        params = {"id": series_id}
        resp = requests.get(url, params=params, timeout=15)
        if resp.status_code != 200:
            return None
        from io import StringIO
        df = pd.read_csv(StringIO(resp.text))
        df.columns = ["date_str", "value"]
        df = df[df["value"] != "."].copy()
        df["value"] = pd.to_numeric(df["value"], errors="coerce")
        df["date"] = pd.to_datetime(df["date_str"]).dt.year
        df["country"] = "United Kingdom"
        df = df[["date", "value", "country"]].dropna()
        # Annual: take last value per year
        df = df.groupby("date")["value"].last().reset_index()
        df["country"] = "United Kingdom"
        return df.sort_values("date")
    except Exception as e:
        logger.warning(f"FRED fetch failed [{series_id}]: {e}")
        return None


def _ons_search_fallback(search_term: str, description: str) -> Optional[pd.DataFrame]:
    """
    When specific ONS series IDs don't work, use ONS search API
    to find the most relevant dataset and pull its first time series.
    """
    try:
        url = "https://api.beta.ons.gov.uk/v1/search"
        params = {"q": search_term, "limit": 5, "content_type": "timeseries"}
        resp = requests.get(url, params=params, timeout=15)
        if resp.status_code != 200:
            return None
        results = resp.json().get("items", [])
        if not results:
            return None
        # Try each result until we get data
        for result in results[:3]:
            uri = result.get("uri", "")
            if not uri:
                continue
            data_url = f"https://www.ons.gov.uk{uri}/data"
            data_resp = requests.get(data_url, timeout=12)
            if data_resp.status_code != 200:
                continue
            data = data_resp.json()
            records = []
            for entry in data.get("years", []):
                try:
                    val = float(str(entry.get("value", "")).replace(",", ""))
                    year = int(str(entry.get("date", "2000"))[:4])
                    records.append({"date": year, "value": val, "country": "United Kingdom"})
                except (ValueError, TypeError):
                    continue
            if records:
                return pd.DataFrame(records).sort_values("date")
        return None
    except Exception as e:
        logger.warning(f"ONS search fallback failed [{search_term}]: {e}")
        return None


# ── Synthetic fallback data ──────────────────────────────────────────────────
# Used when APIs fail — based on real published ONS/WB figures

FALLBACK_DATA = {
    "UK CPI inflation rate (%)": {
        "data": {2014: 1.5, 2015: 0.0, 2016: 0.7, 2017: 2.7, 2018: 2.5,
                 2019: 1.7, 2020: 0.9, 2021: 2.6, 2022: 9.1, 2023: 7.3, 2024: 2.5},
        "unit": "%", "source": "ONS CPI", "frequency": "annual",
    },
    "UK unemployment rate (%)": {
        "data": {2014: 6.1, 2015: 5.3, 2016: 4.8, 2017: 4.4, 2018: 4.0,
                 2019: 3.8, 2020: 4.9, 2021: 4.5, 2022: 3.7, 2023: 4.2, 2024: 4.4},
        "unit": "%", "source": "ONS LFS", "frequency": "annual",
    },
    "UK GDP growth (%)": {
        "data": {2014: 2.6, 2015: 2.4, 2016: 1.7, 2017: 1.7, 2018: 1.3,
                 2019: 1.6, 2020: -11.0, 2021: 8.7, 2022: 4.3, 2023: 0.3, 2024: 1.1},
        "unit": "%", "source": "ONS", "frequency": "annual",
    },
    "UK average house price (£)": {
        "data": {2014: 179000, 2015: 195000, 2016: 211000, 2017: 225000, 2018: 231000,
                 2019: 234000, 2020: 247000, 2021: 271000, 2022: 295000, 2023: 285000, 2024: 290000},
        "unit": "£", "source": "ONS HPI", "frequency": "annual",
    },
    "UK Gini coefficient": {
        "data": {2014: 32.5, 2015: 32.4, 2016: 33.1, 2017: 33.4, 2018: 33.5,
                 2019: 34.3, 2020: 35.1, 2021: 35.4, 2022: 35.7, 2023: 35.8},
        "unit": "index", "source": "World Bank", "frequency": "annual",
    },
    "UK GDP per capita (USD)": {
        "data": {2015: 43390, 2016: 41010, 2017: 39720, 2018: 42579, 2019: 42740,
                 2020: 38058, 2021: 47335, 2022: 45850, 2023: 46125},
        "unit": "USD", "source": "World Bank", "frequency": "annual",
    },
    "Hospital beds per 1000": {
        "data": {2010: 3.0, 2012: 2.9, 2014: 2.8, 2016: 2.7, 2018: 2.6, 2020: 2.5, 2022: 2.4},
        "unit": "per 1,000", "source": "World Bank", "frequency": "biennial",
    },
    "Health spend % GDP": {
        "data": {2015: 9.8, 2016: 9.9, 2017: 9.8, 2018: 10.0, 2019: 10.3, 2020: 12.8, 2021: 11.9, 2022: 10.9},
        "unit": "% GDP", "source": "World Bank", "frequency": "annual",
    },
    "UK CO2 per capita": {
        "data": {2010: 7.9, 2012: 7.6, 2014: 6.8, 2016: 5.9, 2018: 5.6, 2019: 5.2, 2020: 4.5, 2021: 4.9},
        "unit": "metric tons", "source": "World Bank", "frequency": "annual",
    },
    "Renewable electricity %": {
        "data": {2010: 7.0, 2012: 11.3, 2014: 17.1, 2016: 24.6, 2018: 33.3, 2020: 43.1, 2022: 41.5},
        "unit": "%", "source": "World Bank", "frequency": "annual",
    },
}

COMPARISON_FALLBACKS = {
    "UK unemployment rate (%)": {
        "United Kingdom": {2019: 3.8, 2020: 4.9, 2021: 4.5, 2022: 3.7, 2023: 4.2},
        "Germany": {2019: 3.0, 2020: 3.9, 2021: 3.7, 2022: 3.0, 2023: 3.0},
        "France": {2019: 8.4, 2020: 8.1, 2021: 7.9, 2022: 7.3, 2023: 7.3},
        "USA": {2019: 3.7, 2020: 8.1, 2021: 5.4, 2022: 3.7, 2023: 3.6},
    },
    "UK CPI inflation rate (%)": {
        "United Kingdom": {2020: 0.9, 2021: 2.6, 2022: 9.1, 2023: 7.3, 2024: 2.5},
        "Germany": {2020: 0.5, 2021: 3.2, 2022: 8.7, 2023: 6.0, 2024: 2.2},
        "France": {2020: 0.5, 2021: 2.1, 2022: 5.9, 2023: 5.7, 2024: 2.3},
        "USA": {2020: 1.2, 2021: 4.7, 2022: 8.0, 2023: 4.1, 2024: 2.9},
        "Sweden": {2020: 0.7, 2021: 2.7, 2022: 8.4, 2023: 8.5, 2024: 2.3},
    },
    "Health spend % GDP": {
        "United Kingdom": {2019: 10.3, 2020: 12.8, 2021: 11.9, 2022: 10.9},
        "Germany": {2019: 11.7, 2020: 13.1, 2021: 12.9, 2022: 12.7},
        "France": {2019: 11.1, 2020: 12.4, 2021: 12.2, 2022: 12.1},
        "Canada": {2019: 10.8, 2020: 13.1, 2021: 12.7, 2022: 12.2},
        "Sweden": {2019: 10.9, 2020: 12.0, 2021: 11.4, 2022: 11.1},
        "USA": {2019: 17.0, 2020: 19.7, 2021: 18.3, 2022: 17.3},
    },
    "UK CO2 per capita": {
        "United Kingdom": {2015: 6.3, 2017: 5.7, 2019: 5.2, 2021: 4.9},
        "Germany": {2015: 9.0, 2017: 8.9, 2019: 7.9, 2021: 7.5},
        "France": {2015: 4.6, 2017: 4.6, 2019: 4.5, 2021: 4.2},
        "USA": {2015: 15.5, 2017: 14.6, 2019: 14.2, 2021: 14.1},
        "Sweden": {2015: 3.8, 2017: 3.6, 2019: 3.4, 2021: 2.9},
    },
}


def _make_fallback_dataset(name: str) -> Optional[Dataset]:
    """Generate a dataset from fallback curated data."""
    if name not in FALLBACK_DATA:
        return None
    fd = FALLBACK_DATA[name]
    df = pd.DataFrame([
        {"date": y, "value": v, "country": "United Kingdom"}
        for y, v in fd["data"].items()
    ]).sort_values("date")

    # Comparison data if available
    comp_df = None
    if name in COMPARISON_FALLBACKS:
        rows = []
        for country, year_vals in COMPARISON_FALLBACKS[name].items():
            for year, val in year_vals.items():
                rows.append({"date": year, "value": val, "country": country})
        comp_df = pd.DataFrame(rows).sort_values(["country", "date"])

    return Dataset(
        name=name,
        source=fd["source"],
        series_id=name,
        description=name,
        data=df,
        unit=fd["unit"],
        frequency=fd.get("frequency", "annual"),
        source_url="https://www.ons.gov.uk / https://data.worldbank.org",
        comparison_data=comp_df,
    )


# ── Main router ──────────────────────────────────────────────────────────────

def fetch_data_for_story(category: str, topic: str, max_datasets: int = 4) -> DataPackage:
    """
    Main entry point. Fetches the most relevant datasets for a given category.
    Tries live APIs first; falls back to curated data if they fail.
    
    Returns a DataPackage with up to max_datasets datasets.
    """
    sources = CATEGORY_SOURCES.get(category, CATEGORY_SOURCES["economy"])
    datasets = []
    errors = []
    source_urls = []

    logger.info(f"Fetching data for category '{category}', topic '{topic}'")

    for source_name, fetch_fn, series_id, label, description in sources:
        if len(datasets) >= max_datasets:
            break

        logger.info(f"  Trying {source_name}: {label}")
        df = None

        try:
            if fetch_fn == "worldbank_series":
                df = _fetch_worldbank(series_id, description)
                time.sleep(0.5)
            elif fetch_fn == "ons_timeseries":
                df = _fetch_ons_timeseries(series_id, description)
                time.sleep(0.5)
            elif fetch_fn == "fred_series":
                df = _fetch_fred(series_id, description)
                time.sleep(0.5)
            elif fetch_fn == "ons_search":
                df = _ons_search_fallback(series_id, description)
                time.sleep(0.5)
        except Exception as e:
            errors.append(f"{label}: {e}")
            df = None

        # Validate we got usable data
        if df is not None and len(df) >= 3:
            # Try to get comparison data
            comp_df = None
            if source_name == "WorldBank" and category in COMPARISON_COUNTRIES:
                indicator = series_id.split("/", 1)[-1] if "/" in series_id else series_id
                comp_countries = COMPARISON_COUNTRIES[category]
                comp_df = _fetch_worldbank_comparison(indicator, comp_countries)
                time.sleep(1.0)

            dataset = Dataset(
                name=label,
                source=source_name,
                series_id=series_id,
                description=description,
                data=df,
                unit=_infer_unit(label),
                source_url=_make_source_url(source_name, series_id),
                comparison_data=comp_df,
            )
            datasets.append(dataset)
            source_urls.append(dataset.source_url)
            logger.info(f"  ✓ Got {len(df)} data points for '{label}'")
        else:
            # Try fallback
            fallback = _make_fallback_dataset(label)
            if fallback:
                datasets.append(fallback)
                source_urls.append(fallback.source_url)
                logger.info(f"  ↩ Using fallback data for '{label}'")
            else:
                errors.append(f"No data available for: {label}")

    # Ensure we have at least 2 datasets
    if len(datasets) < 2:
        logger.warning("Insufficient live data — loading full fallback set")
        for name in list(FALLBACK_DATA.keys())[:max_datasets]:
            if not any(d.name == name for d in datasets):
                fb = _make_fallback_dataset(name)
                if fb:
                    datasets.append(fb)

    return DataPackage(
        story_topic=topic,
        category=category,
        datasets=datasets[:max_datasets],
        fetch_errors=errors,
        source_urls=list(set(source_urls)),
    )


def _infer_unit(label: str) -> str:
    label_lower = label.lower()
    if "%" in label or "rate" in label or "growth" in label:
        return "%"
    if "£" in label or "price" in label or "earnings" in label or "wage" in label:
        return "£"
    if "usd" in label or "gdp per capita" in label:
        return "USD"
    if "per 1000" in label or "per capita" in label:
        return "per 1,000"
    return ""


def _make_source_url(source: str, series_id: str) -> str:
    if source == "ONS":
        return "https://www.ons.gov.uk/economy"
    elif source == "WorldBank":
        indicator = series_id.split("/")[-1] if "/" in series_id else series_id
        return f"https://data.worldbank.org/indicator/{indicator}"
    elif source == "FRED":
        return f"https://fred.stlouisfed.org/series/{series_id}"
    return "https://www.ons.gov.uk"


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    pkg = fetch_data_for_story("economy", "UK inflation crisis")
    print(f"\nFetched {len(pkg.datasets)} datasets:")
    for ds in pkg.datasets:
        print(f"  • {ds.name} ({len(ds.data)} rows) [{ds.source}]")
        if not ds.data.empty:
            print(f"    Latest: {ds.data.iloc[-1]['date']} = {ds.data.iloc[-1]['value']:.2f} {ds.unit}")
