# JBM Data Journalism Agent

A Python agent that replicates the analytical approach of **John Burn-Murdoch** (FT data journalist) — finding the top UK political story of the week via Google Trends, pulling relevant public data, surfacing surprising insights, and generating a standalone HTML report with a chart and JBM-style narrative.

---

## What it does

1. **Story detection** — Monitors Google Trends UK for the week's top political story
2. **Safety screening** — Classifies sensitivity (LOW/MEDIUM/HIGH) and applies appropriate data-handling constraints
3. **Data routing** — Maps the story's policy category to the most relevant public datasets (ONS, World Bank, FRED)
4. **Statistical analysis** — Detects trend breaks, long-run changes, international comparisons (JBM's signature moves)
5. **Narrative generation** — With an Anthropic API key: Claude generates JBM-style prose. Without one: template-based narrative from the computed statistics
6. **HTML output** — A self-contained broadsheet-styled report with Chart.js visualisation

---

## Setup

```bash
# Install dependencies
pip install pytrends anthropic requests beautifulsoup4 pandas numpy

# Clone / download this folder, then:
cd jbm_agent
```

---

## Usage

```bash
# Auto-detect top UK political story from Google Trends
python agent.py

# Override with a specific topic
python agent.py --topic "UK NHS waiting times"
python agent.py --topic "UK housing crisis"
python agent.py --topic "UK cost of living"
python agent.py --topic "UK immigration"

# Custom output file
python agent.py --topic "UK economy" --output report.html

# Verbose logging
python agent.py --verbose

# With AI-powered narrative (strongly recommended)
ANTHROPIC_API_KEY=sk-ant-... python agent.py --topic "UK economy"
```

---

## Adding your Anthropic API key

The agent works without a key (template narrative), but with one it generates genuinely JBM-quality prose. Set the environment variable before running:

```bash
export ANTHROPIC_API_KEY=sk-ant-your-key-here
python agent.py --topic "UK economy"
```

Or inline:
```bash
ANTHROPIC_API_KEY=sk-ant-... python agent.py
```

Get a key at: https://console.anthropic.com

---

## Data sources

| Source | What it provides | Access |
|--------|-----------------|--------|
| **ONS API** | UK inflation, unemployment, GDP, earnings, housing | Free, no key |
| **World Bank API** | International comparisons, long-run series | Free, no key |
| **FRED** | UK economic series via Federal Reserve | Free, no key |
| **Google Trends** | Story detection via pytrends | Free, rate-limited |
| **BBC News** | Headline context scraping | Public |

All data falls back to curated recent figures (sourced from published ONS/WB reports) if live APIs are unavailable.

---

## Story categories supported

| Category | Primary datasets |
|----------|-----------------|
| `economy` | CPI inflation, unemployment, GDP, earnings |
| `health` | NHS waiting times, beds per 1000, health spend % GDP |
| `housing` | House price index, HPI, private rents |
| `immigration` | Net migration, immigration statistics |
| `crime` | Crime rates, homicide rate |
| `education` | Education spend, enrolment rates |
| `welfare` | Gini coefficient, poverty, income shares |
| `environment` | CO₂ per capita, renewable electricity % |
| `politics` | Polling, GDP, population |

---

## Output

The agent generates a self-contained `.html` file saved to `output/`. Open it in any browser — no server needed.

**The report includes:**
- A headline statistic (the single most striking number)
- A framing narrative
- A Google Trends search interest bar
- A Chart.js visualisation with international comparisons where available
- Up to 5 insight cards (trend breaks, international comparisons, long-run changes)
- A "by the numbers" stats block
- A closing observation (the thing the news coverage is missing)
- Full source attribution and data caveats

---

## Architecture

```
agent.py                    ← Orchestrator
modules/
  story_detector.py         ← Google Trends + sensitivity screening
  data_router.py            ← Source mapping + API fetching + fallbacks
  analyst.py                ← Statistical analysis + Claude narrative
  renderer.py               ← HTML/CSS/Chart.js output generation
output/
  jbm_<topic>_<timestamp>.html
```

---

## Limitations & known constraints

**What this does well:**
- Statistical insight generation (trend breaks, international comparisons)
- Data sourcing logic and fallback handling
- JBM-style output formatting
- Sensitivity classification and safe data handling

**What requires manual attention:**
- Google Trends detection can surface non-political trending topics — use `--topic` override when the auto-detected story isn't right
- ONS API series IDs change; the fallback dataset covers the most important series with real published figures
- "Surprising" is partly editorial judgment — the agent finds statistical outliers, but genuine editorial instinct is harder to systematise
- Chart polish: output is functional but not publication-ready without Datawrapper-level tooling

**API rate limits:**
- pytrends is rate-limited by Google; repeated rapid runs may get blocked temporarily
- World Bank API is permissive; ONS API is more restrictive
- The agent includes sleep delays between requests to stay within limits

---

## Extending it

**Add a new data source:** Add an entry to `CATEGORY_SOURCES` in `data_router.py` and a corresponding `_fetch_*` function.

**Add a new category:** Add an entry to `CATEGORY_SOURCES` and `COMPARISON_COUNTRIES`, and update `_classify_category` in `story_detector.py`.

**Customise the output style:** The HTML template is in `renderer.py` — all styling is vanilla CSS with CSS variables. No framework dependencies.

**Schedule it:** Run weekly via cron:
```bash
# Every Monday at 8am
0 8 * * 1 ANTHROPIC_API_KEY=sk-... /usr/bin/python3 /path/to/jbm_agent/agent.py
```

---

## Inspiration

- John Burn-Murdoch's data journalism at the FT: https://www.ft.com/john-burn-murdoch
- Agno investment team agent pattern: https://github.com/agno-agi/investment-team
- ONS Data Explorer: https://www.ons.gov.uk/explore-local-statistics
- World Bank Open Data: https://data.worldbank.org

---

*Built with Python, Chart.js, pytrends, and Anthropic Claude.*
