"""
story_detector.py — v2
-----------------------
Detects the top spiking UK political story of the week.

Detection cascade (in priority order):
  1. Google Trends live trending RSS/JSON  — actual trending searches
  2. Guardian API (free 'test' key)        — top UK politics headlines
  3. pytrends on real current story themes — spike detection
  4. Informed fallback                     — current major UK story themes

Key fix vs v1: removed the "compare seeds against each other" fallback
which always returned "Prime Minister" regardless of news. Now uses
Guardian headlines as real candidates, pytrends for spike confirmation.

Returns: StoryResult dataclass
"""

import json
import time
import random
import logging
import warnings
from dataclasses import dataclass, field
from typing import Optional
from datetime import datetime, timedelta
import requests
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)

SENSITIVITY_FLAGS = {
    "immigration": "HIGH", "asylum": "HIGH", "race": "HIGH",
    "ethnicity": "HIGH", "religion": "HIGH", "transgender": "HIGH",
    "abortion": "HIGH", "crime": "MEDIUM", "poverty": "MEDIUM",
    "welfare": "MEDIUM", "benefits": "MEDIUM", "strike": "MEDIUM",
}

CATEGORY_KEYWORDS = {
    "economy":     ["inflation", "gdp", "budget", "tax", "economy", "recession",
                    "growth", "wage", "salary", "cost of living", "energy bills",
                    "interest rate", "bank of england", "spending", "borrowing",
                    "deficit", "trade", "pound", "sterling", "cost"],
    "health":      ["nhs", "hospital", "cancer", "mental health", "waiting list",
                    "doctor", "nurse", "health", "ambulance", "social care",
                    "prescription", "gp", "surgery", "patient"],
    "immigration": ["immigration", "asylum", "visa", "migrant", "border", "refugee",
                    "channel crossing", "small boats", "rwanda", "detention"],
    "housing":     ["housing", "rent", "mortgage", "house price", "landlord",
                    "planning", "leasehold", "renters", "homelessness"],
    "crime":       ["crime", "police", "knife", "violence", "court", "prison",
                    "antisocial", "robbery", "murder", "fraud"],
    "education":   ["school", "university", "teacher", "student", "ofsted",
                    "tuition", "a-level", "gcse", "curriculum"],
    "welfare":     ["benefit", "pension", "universal credit", "welfare", "poverty",
                    "disability", "pip", "dwp", "foodbank"],
    "environment": ["climate", "net zero", "energy", "green", "flood",
                    "emissions", "renewable", "solar", "wind farm", "carbon"],
    "politics":    ["election", "labour", "conservative", "parliament", "minister",
                    "starmer", "vote", "mp", "commons", "lords", "policy",
                    "keir", "reeves", "reform", "lib dem", "tory"],
}

POLITICAL_KEYWORDS = set(
    kw for kws in CATEGORY_KEYWORDS.values() for kw in kws
) | {"government", "whitehall", "downing street", "westminster", "uk", "britain",
     "chancellor", "secretary of state", "legislation", "bill", "act"}


@dataclass
class StoryResult:
    topic: str
    search_volume_index: int
    related_queries: list[str]
    category: str
    headline_context: str
    sensitivity: str = "LOW"
    sensitivity_notes: list[str] = field(default_factory=list)
    raw_trends_data: dict = field(default_factory=dict)
    detection_method: str = "unknown"


# ── Helper: patch pytrends / urllib3 compatibility ──────────────────────────

def _patch_pytrends():
    """Patch Retry.__init__ for urllib3 >=2.0 compatibility with old pytrends."""
    try:
        import urllib3.util.retry as _retry_mod
        _orig = _retry_mod.Retry.__init__
        def _patched(self, *args, **kwargs):
            if "method_whitelist" in kwargs:
                kwargs["allowed_methods"] = kwargs.pop("method_whitelist")
            _orig(self, *args, **kwargs)
        _retry_mod.Retry.__init__ = _patched
    except Exception:
        pass


# ── Method 1: Google Trends live trending ───────────────────────────────────

def _trending_now_uk() -> list[tuple[str, int]]:
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 Chrome/123.0.0.0 Safari/537.36"
        ),
        "Accept-Language": "en-GB,en;q=0.9",
    }
    # RSS endpoints
    for url in [
        "https://trends.google.com/trending/rss?geo=GB",
        "https://trends.google.com/trends/trendingsearches/daily/rss?geo=GB",
    ]:
        try:
            resp = requests.get(url, headers=headers, timeout=12)
            if resp.status_code != 200:
                logger.warning(f"Trends RSS {resp.status_code}: {url}")
                continue
            soup = BeautifulSoup(resp.text, "xml")
            items = soup.find_all("item")
            if not items:
                continue
            out = []
            for item in items:
                title_el = item.find("title")
                traffic_el = item.find("ht:approx_traffic")
                if not title_el:
                    continue
                title = title_el.text.strip()
                try:
                    raw = (traffic_el.text if traffic_el else "5000")
                    vol = int(raw.replace("+", "").replace(",", "").strip())
                except (ValueError, AttributeError):
                    vol = 5000
                out.append((title, vol))
            if out:
                logger.info(f"Trends RSS: {len(out)} topics ({url[:50]})")
                return sorted(out, key=lambda x: x[1], reverse=True)
        except Exception as e:
            logger.warning(f"Trends RSS error: {e}")

    # JSON API fallback
    try:
        resp = requests.get(
            "https://trends.google.com/trends/api/dailytrends",
            params={"hl": "en-GB", "tz": "-60", "geo": "GB", "ns": "15"},
            headers=headers, timeout=12,
        )
        if resp.status_code == 200:
            text = resp.text
            if text.startswith(")]}'"):
                text = text.split("\n", 1)[1]
            data = json.loads(text)
            days = data.get("default", {}).get("trendingSearchesDays", [])
            if days:
                out = []
                for item in days[0].get("trendingSearches", []):
                    title = item.get("title", {}).get("query", "")
                    t = item.get("formattedTraffic", "1K+").replace("+", "").replace(",", "")
                    try:
                        vol = int(float(t[:-1]) * 1000) if t.endswith("K") else \
                              int(float(t[:-1]) * 1_000_000) if t.endswith("M") else int(t)
                    except (ValueError, AttributeError):
                        vol = 5000
                    if title:
                        out.append((title, vol))
                if out:
                    logger.info(f"Trends JSON API: {len(out)} topics")
                    return sorted(out, key=lambda x: x[1], reverse=True)
    except Exception as e:
        logger.warning(f"Trends JSON API error: {e}")

    return []


# ── Method 2: Guardian API ───────────────────────────────────────────────────

def _guardian_top_stories(api_key: str = "test") -> list[str]:
    """
    Fetch top UK political/news headlines from The Guardian API.
    'test' key is freely available — register at open-platform.theguardian.com
    for higher rate limits.
    Returns list of headline strings, most prominent first.
    """
    today = datetime.now().strftime("%Y-%m-%d")
    week_ago = (datetime.now() - timedelta(days=7)).strftime("%Y-%m-%d")
    try:
        resp = requests.get(
            "https://content.guardianapis.com/search",
            params={
                "api-key":     api_key,
                "section":     "politics|uk-news|business",
                "order-by":    "relevance",
                "page-size":   20,
                "show-fields": "headline",
                "from-date":   week_ago,
                "to-date":     today,
            },
            timeout=12,
        )
        if resp.status_code == 429:
            logger.warning("Guardian API rate limited — register a free key at open-platform.theguardian.com")
            return []
        if resp.status_code != 200:
            logger.warning(f"Guardian API {resp.status_code}")
            return []
        results = resp.json().get("response", {}).get("results", [])
        headlines = []
        for r in results:
            h = r.get("fields", {}).get("headline") or r.get("webTitle", "")
            h = h.split(" | ")[0].strip()
            if h:
                headlines.append(h)
        logger.info(f"Guardian API: {len(headlines)} headlines")
        return headlines
    except Exception as e:
        logger.warning(f"Guardian API error: {e}")
        return []


# ── Method 3: pytrends spike check on real topics ───────────────────────────

def _shorten_for_trends(headline: str) -> str:
    """Extract 2-3 searchable keywords from a Guardian headline."""
    filler = {"the", "a", "an", "in", "of", "to", "and", "or", "but", "for",
              "as", "at", "by", "is", "it", "on", "be", "are", "was", "were",
              "its", "has", "have", "had", "with", "from", "this", "that",
              "over", "after", "amid", "calls", "says", "set", "new", "plan",
              "plans", "faces", "warns", "could", "would", "will", "may",
              "his", "her", "uk", "government"}
    words = headline.split()
    # Prefer capitalised words (proper nouns / key concepts)
    keywords = [w.rstrip(".,;:") for w in words
                if w.lower().rstrip("s'.,;:") not in filler
                and len(w) > 2 and w[0].isupper()]
    if len(keywords) < 2:
        keywords = [w.rstrip(".,;:") for w in words if len(w) > 3][:3]
    return " ".join(keywords[:3])


def _pytrends_spike_check(topics: list[str]) -> dict[str, int]:
    """
    Check which topics from a real list are most spiking this week vs baseline.
    Returns {topic: spike_score} — score > 100 means trending up sharply.
    """
    _patch_pytrends()
    try:
        from pytrends.request import TrendReq
        pt = TrendReq(hl="en-GB", tz=0, timeout=(10, 30), retries=1, backoff_factor=1.0)
        short = list(dict.fromkeys(_shorten_for_trends(t) for t in topics))[:5]
        if not short:
            return {}
        pt.build_payload(short, timeframe="today 3-m", geo="GB")
        time.sleep(random.uniform(2, 4))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            df = pt.interest_over_time()
        if df.empty:
            return {}
        df = df.fillna(0)
        scores = {}
        for col in short:
            if col not in df.columns:
                continue
            recent = float(df[col].tail(4).mean())
            baseline = float(df[col].head(8).mean())
            scores[col] = int((recent / max(baseline, 1)) * 100)
        logger.info(f"pytrends spike scores: {scores}")
        return scores
    except Exception as e:
        logger.warning(f"pytrends spike check failed: {e}")
        return {}


# ── Classifier helpers ───────────────────────────────────────────────────────

def _score_political_relevance(topic: str) -> float:
    tl = topic.lower()
    return min(1.0, sum(1 for kw in POLITICAL_KEYWORDS if kw in tl) / 2.0)


def _classify_category(topic: str, related: list[str]) -> str:
    all_text = (topic + " " + " ".join(related)).lower()
    scores = {
        cat: sum(1 for kw in kws if kw in all_text)
        for cat, kws in CATEGORY_KEYWORDS.items()
    }
    return max(scores, key=scores.get) if max(scores.values()) > 0 else "politics"


def _assess_sensitivity(topic: str, related: list[str]) -> tuple[str, list[str]]:
    all_text = (topic + " " + " ".join(related)).lower()
    level = "LOW"
    notes = []
    for flag, severity in SENSITIVITY_FLAGS.items():
        if flag in all_text:
            if severity == "HIGH":
                level = "HIGH"
            elif severity == "MEDIUM" and level == "LOW":
                level = "MEDIUM"
            notes.append(
                f"Topic contains '{flag}' — anchor every claim to official statistics, "
                "avoid causal language, present contradictory evidence where it exists."
            )
    return level, notes


def _get_news_context(topic: str) -> str:
    """Best-match headline from BBC/Guardian RSS, or topic string."""
    topic_words = set(topic.lower().split())
    for url in [
        "https://feeds.bbci.co.uk/news/politics/rss.xml",
        "https://www.theguardian.com/politics/rss",
    ]:
        try:
            resp = requests.get(url, timeout=8,
                                headers={"User-Agent": "Mozilla/5.0"})
            if resp.status_code != 200:
                continue
            soup = BeautifulSoup(resp.text, "xml")
            best, best_score = "", 0
            for item in soup.find_all("item")[:20]:
                t = item.find("title")
                if not t:
                    continue
                txt = t.text.strip()
                score = len(topic_words & set(txt.lower().split()))
                if score > best_score:
                    best_score, best = score, txt
            if best and best_score > 0:
                return best
        except Exception:
            pass
    return f"UK political story: {topic}"


# ── Main ─────────────────────────────────────────────────────────────────────

def detect_top_story(
    manual_override: Optional[str] = None,
    guardian_api_key: Optional[str] = None,
) -> StoryResult:
    """
    Detect the top UK political story of the week.
    Uses a four-method cascade: Trends → Guardian → pytrends themes → fallback.
    """
    import os
    g_key = guardian_api_key or os.environ.get("GUARDIAN_API_KEY", "test")

    # Manual override
    if manual_override:
        logger.info(f"Manual override: '{manual_override}'")
        cat = _classify_category(manual_override, [])
        sens, notes = _assess_sensitivity(manual_override, [])
        return StoryResult(
            topic=manual_override,
            search_volume_index=100,
            related_queries=[],
            category=cat,
            headline_context=_get_news_context(manual_override),
            sensitivity=sens,
            sensitivity_notes=notes,
            detection_method="manual",
        )

    # Method 1: Google Trends live
    logger.info("Detection method 1: Google Trends live trending...")
    trending = _trending_now_uk()
    if trending:
        political = [(t, v) for t, v in trending if _score_political_relevance(t) > 0]
        if political:
            top, top_vol = political[0]
            related = [t for t, _ in political[1:6]]
            index = min(100, int(top_vol / max(trending[0][1], 1) * 100))
            cat = _classify_category(top, related)
            sens, notes = _assess_sensitivity(top, related)
            logger.info(f"  ✓ '{top}' (index {index}) via Google Trends")
            return StoryResult(
                topic=top, search_volume_index=index, related_queries=related,
                category=cat, headline_context=_get_news_context(top),
                sensitivity=sens, sensitivity_notes=notes,
                raw_trends_data={"trending": trending[:10]},
                detection_method="google_trends",
            )
        logger.info(f"  Trends: {len(trending)} topics, none clearly political")

    # Method 2: Guardian API + pytrends spike confirmation
    logger.info("Detection method 2: Guardian API...")
    guardian_headlines = _guardian_top_stories(g_key)
    if guardian_headlines:
        political_hlines = [h for h in guardian_headlines
                            if _score_political_relevance(h) > 0] or guardian_headlines[:5]
        logger.info(f"  Guardian: {political_hlines[:2]}")

        # Spike check on first 5 headlines
        spike_scores = _pytrends_spike_check(political_hlines[:5])
        if spike_scores:
            best = max(political_hlines[:5],
                       key=lambda h: spike_scores.get(_shorten_for_trends(h), 0))
            index = min(100, spike_scores.get(_shorten_for_trends(best), 70))
        else:
            best = political_hlines[0]
            index = 70

        related = [_shorten_for_trends(h) for h in political_hlines[1:5]
                   if h != best]
        cat = _classify_category(best, related)
        sens, notes = _assess_sensitivity(best, related)
        logger.info(f"  ✓ '{best[:60]}' (index {index}) via Guardian")
        return StoryResult(
            topic=best[:80], search_volume_index=index, related_queries=related,
            category=cat, headline_context=_get_news_context(best),
            sensitivity=sens, sensitivity_notes=notes,
            raw_trends_data={"guardian_headlines": political_hlines[:5]},
            detection_method="guardian_api",
        )

    # Method 3: pytrends on real current story themes
    logger.info("Detection method 3: pytrends on current themes...")
    current_themes = [
        "UK inflation", "NHS waiting list", "UK immigration",
        "UK housing crisis", "Keir Starmer",
    ]
    spike_scores = _pytrends_spike_check(current_themes)
    if spike_scores:
        top_theme = max(spike_scores, key=spike_scores.get)
        score = spike_scores[top_theme]
        if score > 20:
            related = [t for t in current_themes if t != top_theme]
            cat = _classify_category(top_theme, related)
            sens, notes = _assess_sensitivity(top_theme, related)
            logger.info(f"  ✓ '{top_theme}' (score {score}) via pytrends themes")
            return StoryResult(
                topic=top_theme, search_volume_index=min(100, score),
                related_queries=related, category=cat,
                headline_context=f"Trending UK story: {top_theme}",
                sensitivity=sens, sensitivity_notes=notes,
                detection_method="pytrends_themes",
            )

    # Method 4: Informed fallback
    logger.warning("All detection methods failed — using informed fallback")
    fallback = "UK cost of living"
    cat = _classify_category(fallback, [])
    sens, notes = _assess_sensitivity(fallback, [])
    return StoryResult(
        topic=fallback, search_volume_index=60,
        related_queries=["inflation", "energy bills", "wages"],
        category=cat,
        headline_context="Cost of living remains a dominant concern in UK politics",
        sensitivity=sens, sensitivity_notes=notes,
        detection_method="fallback",
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s")
    r = detect_top_story()
    print(f"\n{'─'*55}")
    print(f"  Topic:    {r.topic}")
    print(f"  Method:   {r.detection_method}")
    print(f"  Category: {r.category}")
    print(f"  Index:    {r.search_volume_index}/100")
    print(f"  Context:  {r.headline_context[:70]}")
    print(f"{'─'*55}")
