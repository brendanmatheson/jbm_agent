"""
agent.py
--------
JBM Data Journalism Agent — main orchestrator.

Usage:
    python agent.py                          # auto-detect top UK political story
    python agent.py --topic "UK housing"     # override with specific topic
    python agent.py --output my_report.html  # custom output filename
    python agent.py --verbose                # detailed logging

Requires:
    pip install pytrends anthropic requests beautifulsoup4 pandas numpy

Optional:
    ANTHROPIC_API_KEY environment variable for AI-powered narrative generation.
    Without it, the agent uses template-based narrative fallback.
"""

import argparse
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path

# Ensure modules directory is importable
sys.path.insert(0, str(Path(__file__).parent))

from modules.story_detector import detect_top_story
from modules.data_router import fetch_data_for_story
from modules.analyst import analyse
from modules.renderer import render_html


def setup_logging(verbose: bool = False):
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def print_banner():
    print("\n" + "─" * 58)
    print("  JBM Data Journalism Agent")
    print("  UK Political Story Analysis — FT/JBM Style")
    print("─" * 58)


def print_progress(step: int, total: int, message: str):
    bar_len = 30
    filled = int(bar_len * step / total)
    bar = "█" * filled + "░" * (bar_len - filled)
    print(f"\r  [{bar}] {step}/{total}  {message:<40}", end="", flush=True)
    if step == total:
        print()


def run_agent(
    topic_override: str = None,
    output_path: str = None,
    verbose: bool = False,
) -> str:
    """
    Run the full agent pipeline.
    Returns the path to the generated HTML file.
    """
    setup_logging(verbose)
    logger = logging.getLogger("agent")

    print_banner()
    total_steps = 5

    # ── Step 1: Story detection ──────────────────────────────────────────────
    print_progress(0, total_steps, "Detecting top UK political story...")
    try:
        story = detect_top_story(manual_override=topic_override)
        method_tag = getattr(story, 'detection_method', '?')
        print_progress(1, total_steps, f"Story: '{story.topic[:32]}' [{story.category}] via {method_tag}")
        logger.info(f"Story: {story.topic} | Category: {story.category} | Method: {method_tag} | Sensitivity: {story.sensitivity}")
    except Exception as e:
        logger.error(f"Story detection failed: {e}")
        print(f"\n  ✗ Story detection failed: {e}")
        print("  → Using fallback topic: UK economy")
        from modules.story_detector import StoryResult
        story = StoryResult(
            topic="UK economy",
            search_volume_index=70,
            related_queries=["inflation", "cost of living"],
            category="economy",
            headline_context="UK economic outlook remains uncertain",
            sensitivity="LOW",
        )

    # Safety gate
    if story.sensitivity == "HIGH":
        print(f"\n  ⚠  Sensitivity: HIGH")
        print(f"     Notes:")
        for note in story.sensitivity_notes[:2]:
            print(f"       • {note[:70]}...")
        print(f"     Proceeding with enhanced data-anchoring constraints.\n")
        time.sleep(1)

    # ── Step 2: Data fetching ────────────────────────────────────────────────
    print_progress(1, total_steps, "Fetching public data sources...")
    try:
        data_package = fetch_data_for_story(
            category=story.category,
            topic=story.topic,
            max_datasets=4,
        )
        n_datasets = len(data_package.datasets)
        print_progress(2, total_steps, f"Fetched {n_datasets} datasets")
        logger.info(f"Datasets: {[d.name for d in data_package.datasets]}")
        if data_package.fetch_errors:
            for err in data_package.fetch_errors[:3]:
                logger.warning(f"Data fetch warning: {err}")
    except Exception as e:
        logger.error(f"Data fetch failed: {e}")
        print(f"\n  ✗ Data fetch failed: {e}")
        from modules.data_router import DataPackage
        data_package = DataPackage(
            story_topic=story.topic,
            category=story.category,
            datasets=[],
            fetch_errors=[str(e)],
            source_urls=[],
        )

    # ── Step 3: Analysis ─────────────────────────────────────────────────────
    print_progress(2, total_steps, "Analysing patterns & generating insights...")

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if api_key:
        print_progress(2, total_steps, "Calling Claude for narrative (JBM style)...")
    else:
        logger.info("No ANTHROPIC_API_KEY found — using template narrative")

    try:
        analysis = analyse(data_package, story)
        print_progress(3, total_steps, "Analysis complete")
        logger.info(f"Headline stat: {analysis.headline_stat[:60]}")
        logger.info(f"Insights generated: {len(analysis.insights)}")
    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        print(f"\n  ✗ Analysis failed: {e}")
        raise

    # ── Step 4: Validation ───────────────────────────────────────────────────
    print_progress(3, total_steps, "Validating output...")

    issues = []
    if not analysis.headline_stat:
        issues.append("Missing headline statistic")
    if not analysis.lede_paragraph:
        issues.append("Missing lede paragraph")
    if not analysis.chart_spec.series:
        issues.append("No chart data")
    if len(analysis.insights) == 0:
        issues.append("No insights generated")
    if not analysis.sources_used:
        issues.append("No sources attributed")

    if issues:
        logger.warning(f"Validation issues: {issues}")
        print(f"\n  ⚠  Validation warnings: {', '.join(issues)}")
    else:
        logger.info("Validation passed")

    print_progress(4, total_steps, "Rendering HTML output...")

    # ── Step 5: Render ───────────────────────────────────────────────────────
    if output_path is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M")
        slug = story.topic.lower().replace(" ", "_")[:30]
        slug = "".join(c for c in slug if c.isalnum() or c == "_")
        output_dir = Path(__file__).parent / "output"
        output_dir.mkdir(exist_ok=True)
        output_path = str(output_dir / f"jbm_{slug}_{timestamp}.html")

    try:
        html = render_html(analysis, story, output_path=output_path)
        print_progress(5, total_steps, "Done")
    except Exception as e:
        logger.error(f"Render failed: {e}")
        raise

    # ── Summary ──────────────────────────────────────────────────────────────
    print("\n" + "─" * 58)
    print(f"  ✓  Report generated successfully")
    print(f"     Topic:    {story.topic}")
    print(f"     Category: {story.category}")
    print(f"     Datasets: {len(data_package.datasets)}")
    print(f"     Insights: {len(analysis.insights)}")
    print(f"     Output:   {output_path}")
    print(f"     Size:     {len(html) / 1024:.1f} KB")
    print("─" * 58 + "\n")

    return output_path


def main():
    parser = argparse.ArgumentParser(
        description="JBM Data Journalism Agent — UK political story analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python agent.py
  python agent.py --topic "UK NHS waiting times"
  python agent.py --topic "UK housing crisis" --output housing_report.html
  ANTHROPIC_API_KEY=sk-... python agent.py --topic "UK inflation" --verbose

Set ANTHROPIC_API_KEY environment variable for AI-powered narrative generation.
Without it, the agent uses template-based narrative (still produces full output).
        """,
    )
    parser.add_argument(
        "--topic",
        type=str,
        default=None,
        help="Override story topic (skips Google Trends detection)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output HTML file path (default: output/jbm_<topic>_<timestamp>.html)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable detailed logging",
    )

    args = parser.parse_args()

    try:
        output_file = run_agent(
            topic_override=args.topic,
            output_path=args.output,
            verbose=args.verbose,
        )
        print(f"Open in browser: file://{Path(output_file).resolve()}\n")
        return 0
    except KeyboardInterrupt:
        print("\n\n  Interrupted.\n")
        return 1
    except Exception as e:
        print(f"\n  ✗ Agent failed: {e}\n")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
