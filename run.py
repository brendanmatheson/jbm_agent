#!/usr/bin/env python3
"""
run.py
------
Main entry point for the JBM Data Journalism Agent — ReAct version.

Usage:
    python run.py                              # auto-detect story from Google Trends / Guardian
    python run.py --topic "UK NHS"             # manual story override
    python run.py --topic "UK housing" --output housing.html
    python run.py --verbose                    # show full scratchpad

Requires: ANTHROPIC_API_KEY environment variable.
"""

import argparse
import logging
import os
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from modules.story_detector import detect_top_story
from modules.react_agent import run_react_agent
from modules.renderer import render_html


def setup_logging(verbose: bool):
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def print_banner():
    print("\n" + "─" * 60)
    print("  JBM Data Journalism Agent  ·  ReAct Edition")
    print("─" * 60)


def run(topic_override=None, output_path=None, verbose=False):
    setup_logging(verbose)
    logger = logging.getLogger("run")
    print_banner()

    # ── Check API key ─────────────────────────────────────────────────────────
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("\n  ✗  ANTHROPIC_API_KEY not set.")
        print("     Export it before running:")
        print("     export ANTHROPIC_API_KEY=sk-ant-...")
        print("\n     (The old pipeline agent.py works without a key)\n")
        return None

    # ── Step 1: Story detection ───────────────────────────────────────────────
    print("\n  [1/3]  Detecting story...", end="", flush=True)
    try:
        story = detect_top_story(manual_override=topic_override)
        print(f"\r  [1/3]  Story: '{story.topic}'  [{story.category}]  via {story.detection_method}")
        if story.sensitivity != "LOW":
            print(f"         Sensitivity: {story.sensitivity}")
    except Exception as e:
        print(f"\n  ✗  Story detection failed: {e}")
        logger.exception("Story detection error")
        return None

    # ── Step 2: ReAct agent loop ──────────────────────────────────────────────
    print(f"\n  [2/3]  ReAct agent running (max 12 steps)...")

    step_log = []

    def on_step(step_num, thought, action, observation):
        # Truncate for display
        thought_short = thought[:70] + "..." if len(thought) > 70 else thought
        obs_short = observation[:70] + "..." if len(observation) > 70 else observation
        print(f"         step {step_num:02d}  {action:<22}  {obs_short}")
        step_log.append((step_num, thought, action, observation))

    try:
        analysis = run_react_agent(story, on_step=on_step)
        n_steps = len(step_log)
        n_datasets = len(analysis.sources_used)
        print(f"\r  [2/3]  Agent complete: {n_steps} steps, {n_datasets} sources")
    except ValueError as e:
        print(f"\n  ✗  {e}\n")
        return None
    except Exception as e:
        print(f"\n  ✗  Agent error: {e}")
        logger.exception("ReAct agent error")
        return None

    # ── Step 3: Render ────────────────────────────────────────────────────────
    print(f"\n  [3/3]  Rendering HTML...", end="", flush=True)

    if output_path is None:
        ts = datetime.now().strftime("%Y%m%d_%H%M")
        slug = "".join(c if c.isalnum() else "_" for c in story.topic.lower())[:28]
        out_dir = Path(__file__).parent / "output"
        out_dir.mkdir(exist_ok=True)
        output_path = str(out_dir / f"jbm_react_{slug}_{ts}.html")

    try:
        html = render_html(analysis, story, output_path=output_path)
        print(f"\r  [3/3]  Rendered  ({len(html) / 1024:.1f} KB)")
    except Exception as e:
        print(f"\n  ✗  Render failed: {e}")
        logger.exception("Render error")
        return None

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n" + "─" * 60)
    print(f"  ✓  Done")
    print(f"     Headline: {analysis.headline_stat[:65]}")
    print(f"     Insights: {len(analysis.insights)}")
    print(f"     Output:   {output_path}")
    print("─" * 60)

    if verbose and step_log:
        print("\n  Full scratchpad:\n")
        for step_num, thought, action, obs in step_log:
            print(f"  [{step_num:02d}] THOUGHT: {thought[:100]}")
            print(f"       ACTION:  {action}")
            print(f"       OBS:     {obs[:100]}\n")

    print(f"\n  Open: file://{Path(output_path).resolve()}\n")
    return output_path


def main():
    parser = argparse.ArgumentParser(
        description="JBM Data Journalism Agent — ReAct edition",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run.py
  python run.py --topic "UK NHS waiting times"
  python run.py --topic "UK housing crisis" --output housing.html
  python run.py --verbose

Requires ANTHROPIC_API_KEY environment variable.
        """,
    )
    parser.add_argument("--topic",   type=str, default=None)
    parser.add_argument("--output",  type=str, default=None)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    result = run(
        topic_override=args.topic,
        output_path=args.output,
        verbose=args.verbose,
    )
    return 0 if result else 1


if __name__ == "__main__":
    sys.exit(main())
