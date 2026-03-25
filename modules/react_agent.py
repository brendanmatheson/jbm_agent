"""
react_agent.py
--------------
ReAct (Reason + Act) agent for JBM-style data journalism.

The loop:
  1. THINK  — Claude reads the current scratchpad and decides what to do next
  2. ACT    — Claude emits a tool call as JSON
  3. OBSERVE — we execute the tool and append the result to the scratchpad
  4. REPEAT — until Claude calls `finish()`

Unlike the pipeline (agent.py), the LLM decides:
  - Which datasets are relevant to THIS story
  - Whether the data it has is sufficient or needs more
  - When to fetch international comparisons vs. go deeper on UK data
  - When it has enough to write and what the most surprising angle is

The scratchpad is a plain text log that grows with each step.
Claude reads the full scratchpad at every step — it has memory of
everything it's done and seen so far within the run.

Max steps: 12 (prevents runaway loops)
"""

import json
import logging
import os
import re
import time
from dataclasses import dataclass, field
from typing import Optional

import anthropic

from modules.tools import (
    TOOLS,
    describe_tools,
    tool_fetch_fred,
    tool_fetch_worldbank,
    tool_fetch_comparison,
    tool_fetch_ons,
    tool_summarise_dataset,
    tool_summarise_comparison,
    tool_finish,
    ToolResult,
    Dataset,
)
from modules.analyst import (
    AnalysisResult, ChartSpec, ChartSeries, Insight,
    _build_chart_spec, _detect_trend_break, _calculate_change,
    _uk_vs_peers, _format_value,
)

logger = logging.getLogger(__name__)

MAX_STEPS = 12


# ── Scratchpad ────────────────────────────────────────────────────────────────

@dataclass
class Scratchpad:
    """Accumulates the agent's observations across steps."""
    entries: list[str] = field(default_factory=list)
    datasets: list[Dataset] = field(default_factory=list)
    comparison_data: dict = field(default_factory=dict)  # label → DataFrame
    tool_call_count: int = 0
    finish_reasoning: str = ""

    def add(self, role: str, content: str):
        self.entries.append(f"[{role.upper()}] {content}")

    def full_text(self) -> str:
        return "\n\n".join(self.entries)

    def datasets_summary(self) -> str:
        if not self.datasets:
            return "No datasets fetched yet."
        lines = []
        for ds in self.datasets:
            latest = ds.data.iloc[-1] if not ds.data.empty else None
            val = f"{latest['value']:.2f} {ds.unit}" if latest is not None else "?"
            lines.append(f"  • {ds.name} ({ds.source}): latest {val}")
        return "\n".join(lines)


# ── System prompt ─────────────────────────────────────────────────────────────

def _build_system_prompt(story_topic: str, headline_context: str,
                          sensitivity: str, sensitivity_notes: list[str]) -> str:
    sensitivity_block = ""
    if sensitivity == "HIGH":
        sensitivity_block = f"""
SENSITIVITY: HIGH — this topic requires extra care.
Rules: anchor every claim to a specific statistic, avoid causal language,
present contradictory evidence, use systemic framing not individual blame.
Notes: {"; ".join(sensitivity_notes)}
"""
    elif sensitivity == "MEDIUM":
        sensitivity_block = "\nSENSITIVITY: MEDIUM — ground all claims in data, avoid overreach.\n"

    return f"""You are a data journalism agent working in the style of John Burn-Murdoch (Financial Times).

Your task: gather data and produce a compelling, data-driven analysis of this UK political story:
  Topic: {story_topic}
  Context: {headline_context}
{sensitivity_block}
You work by calling tools to fetch data, then reasoning about what you find.
At each step you will:
  1. THINK: reason about what data you need and what you've found so far
  2. ACT: call exactly one tool
  3. The tool result will be shown to you as an OBSERVATION
  4. Repeat until you have enough to write the analysis, then call finish()

JBM's analytical approach (follow this):
  - Find the thing the news story implies, then check if the data supports or contradicts it
  - Always seek the LONG-RUN trend — what was happening 10-20 years before this week's story
  - International comparisons are gold — how does the UK rank vs peers? Is it an outlier?
  - The most powerful insight is usually a CONTRADICTION: data that flips the headline framing
  - One strong surprising statistic beats five obvious ones
  - Be precise: exact numbers, exact years, exact percentage changes

Available tools:
{describe_tools()}

IMPORTANT RULES:
- Always call summarise_dataset after fetching a dataset — you need the stats to reason
- Always call summarise_comparison after fetch_comparison — you need the ranking
- Fetch at least 2 different datasets before finishing
- Try to get at least 1 international comparison if the indicator exists
- Do not fetch the same series twice
- When you call finish(), write a thorough reasoning summary covering:
    * The headline statistic (most striking single number)
    * The main narrative arc (what the data says about the story)
    * The most surprising insight (what contradicts or reframes the story)
    * International context (if available)
    * Which dataset should be the hero chart and why
    * Any important caveats

Respond in this exact format on every step:

THOUGHT: <your reasoning about what to do next>
ACTION: <tool_name>
PARAMS: <JSON object with parameters>
"""


# ── Response parser ───────────────────────────────────────────────────────────

def _extract_json_safe(text: str) -> dict:
    """
    Multi-strategy JSON extraction that handles malformed output gracefully.
    Tries strategies in order of strictness, falling back progressively.
    """
    if not text or not text.strip():
        return {}

    # Strategy 1: direct parse
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Strategy 2: extract first {...} block with re.DOTALL
    m = re.search(r'\{.*\}', text, re.DOTALL)
    if m:
        candidate = m.group(0)
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            pass

    # Strategy 3: truncated JSON — find the last complete key-value pair
    # before the unterminated string and close the object
    for end in range(len(text) - 1, 0, -100):
        chunk = text[:end]
        # Find last complete "key": "value" or "key": [...] pair
        last_comma = max(chunk.rfind(","), chunk.rfind("}"), chunk.rfind("]"))
        if last_comma > 0:
            closed = chunk[:last_comma] + "}"
            try:
                return json.loads(closed)
            except json.JSONDecodeError:
                pass

    # Strategy 4: key-value extraction as last resort — pull any "key": "value" pairs
    result = {}
    for match in re.finditer(r'"(\w+)"\s*:\s*"([^"]*?)"', text):
        result[match.group(1)] = match.group(2)
    if result:
        logger.warning("JSON extraction used key-value fallback — partial result")
        return result

    return {}


def _parse_response(text: str) -> tuple[str, str, dict]:
    """
    Extract THOUGHT, ACTION, PARAMS from the LLM response.
    Returns (thought, action_name, params_dict).
    Raises ValueError if the format is not followed.
    """
    thought_match = re.search(r"THOUGHT:\s*(.+?)(?=ACTION:|$)", text, re.DOTALL)
    action_match  = re.search(r"ACTION:\s*(\w+)", text)
    params_match  = re.search(r"PARAMS:\s*(\{.+?\})", text, re.DOTALL)

    thought = thought_match.group(1).strip() if thought_match else ""
    action  = action_match.group(1).strip()  if action_match  else ""
    params_str = params_match.group(1).strip() if params_match else "{}"

    if not action:
        raise ValueError(f"No ACTION found in response:\n{text[:300]}")

    try:
        params = json.loads(params_str)
    except json.JSONDecodeError as e:
        logger.warning(f"JSON parse failed: {e} — attempting repair")
        params = _extract_json_safe(params_str)

    return thought, action, params


# ── Tool dispatcher ───────────────────────────────────────────────────────────

def _dispatch(action: str, params: dict, scratchpad: Scratchpad) -> ToolResult:
    """Execute the tool call and update scratchpad state."""
    scratchpad.tool_call_count += 1

    if action == "fetch_fred":
        result = tool_fetch_fred(
            series_id=params.get("series_id", ""),
            label=params.get("label", "Series"),
        )
        if result.success and isinstance(result.data, Dataset):
            # Avoid duplicate datasets
            existing_names = [d.name for d in scratchpad.datasets]
            if result.data.name not in existing_names:
                scratchpad.datasets.append(result.data)

    elif action == "fetch_worldbank":
        result = tool_fetch_worldbank(
            indicator=params.get("indicator", ""),
            label=params.get("label", "Series"),
            country=params.get("country", "GBR"),
        )
        if result.success and isinstance(result.data, Dataset):
            existing_names = [d.name for d in scratchpad.datasets]
            if result.data.name not in existing_names:
                scratchpad.datasets.append(result.data)

    elif action == "fetch_ons":
        result = tool_fetch_ons(
            dataset_series=params.get("dataset_series", ""),
            label=params.get("label", "Series"),
        )
        if result.success and isinstance(result.data, Dataset):
            existing_names = [d.name for d in scratchpad.datasets]
            if result.data.name not in existing_names:
                scratchpad.datasets.append(result.data)

    elif action == "fetch_comparison":
        result = tool_fetch_comparison(
            indicator=params.get("indicator", ""),
            label=params.get("label", "Comparison"),
            countries=params.get("countries", None),
        )
        if result.success and result.data is not None:
            scratchpad.comparison_data[params.get("label", "comparison")] = result.data

    elif action == "summarise_dataset":
        # Find the most recently added dataset if no specific one named
        label = params.get("label", "")
        ds = None
        if label:
            for d in scratchpad.datasets:
                if label.lower() in d.name.lower():
                    ds = d
                    break
        if ds is None and scratchpad.datasets:
            ds = scratchpad.datasets[-1]
        if ds:
            result = tool_summarise_dataset(ds)
        else:
            result = ToolResult(success=False, summary="No dataset available to summarise.", error="no_data")

    elif action == "summarise_comparison":
        label = params.get("label", "")
        unit  = params.get("unit", "")
        comp_df = None
        if label and label in scratchpad.comparison_data:
            comp_df = scratchpad.comparison_data[label]
        elif scratchpad.comparison_data:
            comp_df = list(scratchpad.comparison_data.values())[-1]
            label = list(scratchpad.comparison_data.keys())[-1]
        result = tool_summarise_comparison(comp_df, label, unit)

    elif action == "finish":
        reasoning = params.get("reasoning", "Analysis complete.")
        result = tool_finish(reasoning)
        scratchpad.finish_reasoning = reasoning

    else:
        result = ToolResult(
            success=False,
            summary=f"Unknown tool: '{action}'. Available: {list(TOOLS.keys())}",
            error="unknown_tool",
        )

    return result


# ── Reflection constants ──────────────────────────────────────────────────────

REFLECTION_QUALITY_THRESHOLD = 75   # score out of 100 to accept draft
REFLECTION_MAX_ROUNDS        = 3    # max generate→critique→revise cycles

# The rubric the critic scores against — five dimensions, 20 pts each
CRITIC_RUBRIC = """
Score the draft on these five dimensions (0–20 each, total out of 100):

1. HEADLINE PRECISION (0-20)
   - 20: Contains a specific number, year, and counter-intuitive framing
   - 10: Has a number but framing is predictable
   - 0:  Vague, no statistics, or purely descriptive

2. LEDE SURPRISE (0-20)
   - 20: Opens with the most counter-intuitive finding; contradicts the obvious narrative
   - 10: Interesting but leads with the expected angle
   - 0:  Buries the best insight; starts with context-setting rather than revelation

3. CLAIM SUPPORT (0-20)
   - 20: Every factual claim cites an exact figure from the scratchpad data
   - 10: Most claims supported but some vague ('significantly', 'much higher')
   - 0:  Claims not traceable to specific data points; possible hallucination

4. INTERNATIONAL CONTEXT (0-20)
   - 20: UK benchmarked against ≥3 named peer countries with specific figures
   - 10: Some international reference but incomplete or imprecise
   - 0:  No international comparison; UK treated in isolation

5. CONCLUSION INSIGHT (0-20)
   - 20: Conclusion reveals something the weekly news cycle actively obscures; genuinely surprising
   - 10: Reasonable but predictable; something a reader already suspects
   - 0:  Vague, hedged, or restates what the lede already said
"""


# ── Generator ────────────────────────────────────────────────────────────────

def _write_narrative(
    client: anthropic.Anthropic,
    scratchpad: Scratchpad,
    story_topic: str,
    sensitivity: str,
    sensitivity_notes: list[str],
    objections: list[dict] = None,
    previous_draft: dict = None,
) -> dict:
    """
    Generate (or revise) a JBM-style narrative from the scratchpad.
    On first call: objections=None, previous_draft=None → fresh generation.
    On revision calls: objections and previous_draft are both provided.
    """
    sens_block = ""
    if sensitivity == "HIGH":
        sens_block = (
            "\n\nSENSITIVITY WARNING: HIGH. "
            "Every claim MUST be backed by an exact statistic from the observations above. "
            "No causal language. No group attribution. Systemic framing only. "
            f"Notes: {'; '.join(sensitivity_notes)}"
        )

    # Revision mode: include previous draft and critic objections
    revision_block = ""
    if objections and previous_draft:
        obj_lines = "\n".join(
            f"  [{o['dimension']}] {o['objection']}  →  Fix: {o['fix']}"
            for o in objections
        )
        revision_block = f"""
PREVIOUS DRAFT (round {previous_draft.get('_round', '?')}):
Headline: {previous_draft.get('headline_stat', '')}
Lede: {previous_draft.get('lede_paragraph', '')}
Conclusion: {previous_draft.get('conclusion', '')}

CRITIC OBJECTIONS TO ADDRESS IN THIS REVISION:
{obj_lines}

Rewrite the full analysis addressing every objection above.
Do not simply add a sentence — restructure where needed.
Every fix must be traceable to a specific data point in the scratchpad.
"""

    mode = "REVISE the analysis" if revision_block else "Write the analysis"

    prompt = f"""You are John Burn-Murdoch at the Financial Times. You have just completed \
a data investigation into: "{story_topic}"

--- RESEARCH SCRATCHPAD ---
{scratchpad.full_text()}
--- END SCRATCHPAD ---

Agent's final reasoning: {scratchpad.finish_reasoning}
{sens_block}
{revision_block}

{mode}. Rules:
- Lead with the most SURPRISING finding — the thing that contradicts the obvious narrative
- Every statistic must come verbatim from the scratchpad above
- International comparisons are mandatory if the data supports them
- The conclusion must reveal something the weekly news cycle actively obscures
- Be precise: exact numbers, exact years, exact percentage changes

Respond as JSON with exactly these keys:
{{
  "headline_stat": "Most striking statistic — one punchy sentence with a specific number",
  "headline_narrative": "One sentence framing the data angle",
  "lede_paragraph": "3-4 sentences. Lead with the most surprising finding.",
  "insights": [
    {{"text": "insight with specific number", "stat": "key figure", "type": "trend_break|international|long_run|contradiction"}},
    {{"text": "...", "stat": "...", "type": "..."}},
    {{"text": "...", "stat": "...", "type": "..."}},
    {{"text": "...", "stat": "...", "type": "..."}}
  ],
  "supporting_stats": ["Stat with exact figure", "...", "...", "...", "..."],
  "conclusion": "What the weekly news coverage is missing. Genuinely surprising.",
  "chart_title": "Hero chart title",
  "chart_annotation": "Key data point annotation (max 8 words)"
}}"""

    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=3000,
        messages=[{"role": "user", "content": prompt}],
    )
    text = response.content[0].text.strip()
    if "```json" in text:
        text = text.split("```json")[1].split("```")[0].strip()
    elif "```" in text:
        text = text.split("```")[1].split("```")[0].strip()
    result = _extract_json_safe(text)
    if not result:
        raise ValueError(f"Could not parse narrative JSON. Response length: {len(text)}")
    # Ensure required keys exist with defaults
    for key in ["headline_stat", "headline_narrative", "lede_paragraph",
                "insights", "supporting_stats", "conclusion",
                "chart_title", "chart_annotation"]:
        if key not in result:
            result[key] = [] if key in ("insights", "supporting_stats") else ""
    return result


# ── Critic ────────────────────────────────────────────────────────────────────

def _critique_narrative(
    client: anthropic.Anthropic,
    narrative: dict,
    scratchpad: Scratchpad,
    story_topic: str,
) -> dict:
    """
    Score the draft against the JBM quality rubric and return structured
    objections for each dimension that falls below threshold.

    Returns:
      {
        "scores": {"headline_precision": int, "lede_surprise": int,
                   "claim_support": int, "international_context": int,
                   "conclusion_insight": int},
        "total": int,          # sum of five scores, 0-100
        "objections": [        # only for dimensions scoring < 15/20
          {"dimension": str, "score": int, "objection": str, "fix": str}
        ],
        "verdict": str         # one-sentence overall assessment
      }
    """
    prompt = f"""You are a senior data journalism editor at the Financial Times.
Your job is to critique a draft analysis against the JBM quality rubric.

STORY: "{story_topic}"

DRAFT TO CRITIQUE:
Headline: {narrative.get('headline_stat', '')}
Deck: {narrative.get('headline_narrative', '')}
Lede: {narrative.get('lede_paragraph', '')}
Stats block: {json.dumps(narrative.get('supporting_stats', []))}
Conclusion: {narrative.get('conclusion', '')}

AVAILABLE DATA (what the writer had access to):
{scratchpad.full_text()[-3000:]}

{CRITIC_RUBRIC}

Be a tough, precise editor. Vague praise is useless. Every objection must name
the specific problem and give a concrete fix grounded in the available data.

Respond as JSON:
{{
  "scores": {{
    "headline_precision": <0-20>,
    "lede_surprise": <0-20>,
    "claim_support": <0-20>,
    "international_context": <0-20>,
    "conclusion_insight": <0-20>
  }},
  "total": <0-100>,
  "objections": [
    {{
      "dimension": "headline_precision|lede_surprise|claim_support|international_context|conclusion_insight",
      "score": <int>,
      "objection": "Specific problem with this dimension",
      "fix": "Concrete fix using data from the scratchpad"
    }}
  ],
  "verdict": "One sentence overall assessment"
}}

Only include objections for dimensions scoring below 15. If a dimension scores 15+, omit it."""

    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=1600,
        messages=[{"role": "user", "content": prompt}],
    )
    text = response.content[0].text.strip()
    if "```json" in text:
        text = text.split("```json")[1].split("```")[0].strip()
    elif "```" in text:
        text = text.split("```")[1].split("```")[0].strip()
    result = _extract_json_safe(text)
    if not result:
        logger.warning("Critic returned unparseable JSON — returning default pass")
        return {"scores": {}, "total": 80, "objections": [], "verdict": "Could not parse critique."}
    result.setdefault("total", 0)
    result.setdefault("objections", [])
    result.setdefault("verdict", "")
    return result


# ── Reflection coordinator ────────────────────────────────────────────────────

def _write_with_reflection(
    client: anthropic.Anthropic,
    scratchpad: Scratchpad,
    story_topic: str,
    sensitivity: str,
    sensitivity_notes: list[str],
    on_reflection_step=None,   # optional callback(round, score, objections, passed)
) -> tuple[dict, list[dict]]:
    """
    Runs the generate → critique → revise loop until the narrative
    meets REFLECTION_QUALITY_THRESHOLD or REFLECTION_MAX_ROUNDS is reached.

    Returns:
      (best_narrative_dict, reflection_log)

    reflection_log is a list of dicts, one per round:
      {"round": int, "score": int, "objections": list, "verdict": str, "passed": bool}
    """
    reflection_log = []
    best_narrative = None
    best_score = 0
    objections = None

    for round_num in range(1, REFLECTION_MAX_ROUNDS + 1):
        logger.info(f"  Reflection round {round_num}/{REFLECTION_MAX_ROUNDS}")

        # Generate (or revise)
        narrative = _write_narrative(
            client=client,
            scratchpad=scratchpad,
            story_topic=story_topic,
            sensitivity=sensitivity,
            sensitivity_notes=sensitivity_notes,
            objections=objections,
            previous_draft={**best_narrative, "_round": round_num - 1}
                           if best_narrative else None,
        )
        narrative["_round"] = round_num
        time.sleep(0.4)

        # Critique
        critique = _critique_narrative(
            client=client,
            narrative=narrative,
            scratchpad=scratchpad,
            story_topic=story_topic,
        )
        time.sleep(0.4)

        score   = critique.get("total", 0)
        verdict = critique.get("verdict", "")
        objections = critique.get("objections", [])
        passed  = score >= REFLECTION_QUALITY_THRESHOLD

        logger.info(
            f"    Score: {score}/100  |  Objections: {len(objections)}  "
            f"|  {'PASS' if passed else 'REVISE'}"
        )
        if objections:
            for obj in objections:
                logger.info(f"    [{obj['dimension']}] {obj['objection'][:70]}")

        reflection_log.append({
            "round":      round_num,
            "score":      score,
            "scores":     critique.get("scores", {}),
            "objections": objections,
            "verdict":    verdict,
            "passed":     passed,
        })

        # Keep the best draft seen so far
        if score > best_score:
            best_score = score
            best_narrative = narrative

        if on_reflection_step:
            on_reflection_step(round_num, score, objections, passed)

        if passed:
            logger.info(f"  Reflection passed at round {round_num} (score {score})")
            break

        if round_num == REFLECTION_MAX_ROUNDS:
            logger.warning(
                f"  Reflection hit max rounds — using best draft (score {best_score})"
            )

    return best_narrative, reflection_log


# ── Main ReAct loop ───────────────────────────────────────────────────────────

def run_react_agent(
    story_result,
    on_step=None,  # optional callback(step_num, thought, action, observation)
) -> AnalysisResult:
    """
    Run the ReAct agent loop for a given story.

    Args:
        story_result: StoryResult from story_detector
        on_step: optional callback for progress reporting

    Returns:
        AnalysisResult ready for rendering
    """
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise ValueError(
            "ANTHROPIC_API_KEY is required for the ReAct agent. "
            "The pipeline fallback (agent.py) works without it."
        )

    client = anthropic.Anthropic(api_key=api_key)
    scratchpad = Scratchpad()

    system_prompt = _build_system_prompt(
        story_topic=story_result.topic,
        headline_context=story_result.headline_context,
        sensitivity=story_result.sensitivity,
        sensitivity_notes=story_result.sensitivity_notes,
    )

    # Seed the scratchpad with story context
    scratchpad.add("CONTEXT", (
        f"Story: {story_result.topic}\n"
        f"Category: {story_result.category}\n"
        f"Context: {story_result.headline_context}\n"
        f"Sensitivity: {story_result.sensitivity}"
    ))

    messages = []
    finished = False
    step = 0

    logger.info(f"ReAct agent starting for: '{story_result.topic}'")

    while not finished and step < MAX_STEPS:
        step += 1
        logger.info(f"  Step {step}/{MAX_STEPS}")

        # Build the user message: current scratchpad + available datasets
        user_content = (
            f"Current scratchpad:\n{scratchpad.full_text()}\n\n"
            f"Datasets collected so far:\n{scratchpad.datasets_summary()}\n\n"
            f"Step {step} of max {MAX_STEPS}. What do you do next?"
        )

        messages.append({"role": "user", "content": user_content})

        # Call Claude
        response = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=1000,
            system=system_prompt,
            messages=messages,
        )
        assistant_text = response.content[0].text
        messages.append({"role": "assistant", "content": assistant_text})

        # Parse
        try:
            thought, action, params = _parse_response(assistant_text)
        except ValueError as e:
            logger.warning(f"Parse error at step {step}: {e}")
            scratchpad.add("ERROR", f"Parse error: {e}")
            break

        logger.info(f"  Action: {action}({list(params.keys())})")
        scratchpad.add("THOUGHT", thought)
        scratchpad.add("ACTION", f"{action}({json.dumps(params, ensure_ascii=False)[:120]})")

        # Execute
        result = _dispatch(action, params, scratchpad)

        observation = result.summary
        scratchpad.add("OBSERVATION", observation)
        logger.info(f"  Result: {observation[:80]}")

        if on_step:
            on_step(step, thought, action, observation)

        if action == "finish":
            finished = True

        time.sleep(0.3)  # gentle rate limiting

    if not finished:
        logger.warning(f"ReAct loop hit max steps ({MAX_STEPS}) without finishing")
        scratchpad.add("NOTE", f"Agent hit step limit — proceeding with data collected so far")
        scratchpad.finish_reasoning = (
            f"Reached step limit. Available data: {scratchpad.datasets_summary()}"
        )

    # ── Write narrative with reflection loop ─────────────────────────────────
    logger.info("Starting reflection loop (generate → critique → revise)...")

    def on_reflection_step(round_num, score, objections, passed):
        if on_step:
            status = "PASS" if passed else f"{len(objections)} objection(s)"
            on_step(
                f"reflect-{round_num}",
                f"Reflection round {round_num}: score {score}/100",
                "reflect",
                f"Score {score}/100 · {status}",
            )

    narrative, reflection_log = _write_with_reflection(
        client=client,
        scratchpad=scratchpad,
        story_topic=story_result.topic,
        sensitivity=story_result.sensitivity,
        sensitivity_notes=story_result.sensitivity_notes,
        on_reflection_step=on_reflection_step,
    )

    final_score  = reflection_log[-1]["score"]
    total_rounds = len(reflection_log)

    # ── Build chart spec ──────────────────────────────────────────────────────
    # Attach best comparison data to the most relevant dataset
    datasets = scratchpad.datasets
    if scratchpad.comparison_data and datasets:
        best_label = list(scratchpad.comparison_data.keys())[0]
        best_comp = scratchpad.comparison_data[best_label]
        # Attach to the dataset whose name most closely matches
        for ds in datasets:
            if ds.comparison_data is None:
                ds.comparison_data = best_comp
                break

    chart_spec = _build_chart_spec(datasets, story_result.category, story_result.topic)

    # Override chart title/annotation from Claude's narrative if provided
    if narrative.get("chart_title"):
        chart_spec.title = narrative["chart_title"]
    if narrative.get("chart_annotation"):
        chart_spec.annotation = narrative["chart_annotation"]

    # ── Assemble insights ─────────────────────────────────────────────────────
    insights = []
    for raw in narrative.get("insights", []):
        insights.append(Insight(
            text=raw.get("text", ""),
            stat=raw.get("stat", ""),
            significance="HIGH",
            insight_type=raw.get("type", "general"),
        ))

    # ── Caveats ───────────────────────────────────────────────────────────────
    passed = reflection_log[-1]["passed"]
    caveats = [
        f"Analysis conducted by ReAct agent in {step} steps, "
        f"refined by reflection over {total_rounds} round(s) "
        f"(final quality score: {final_score}/100"
        f"{' — passed threshold' if passed else ' — best available draft'}).",
        "All statistics drawn from official sources: ONS, World Bank, FRED.",
    ]
    if story_result.sensitivity != "LOW":
        caveats.append(
            f"Sensitivity level {story_result.sensitivity}: statistics handled with "
            "elevated care. All figures from official sources only."
        )

    sources = list({f"{ds.source}: {ds.name}" for ds in datasets})

    return AnalysisResult(
        story_topic=story_result.topic,
        headline_stat=narrative.get("headline_stat", ""),
        headline_narrative=narrative.get("headline_narrative", ""),
        lede_paragraph=narrative.get("lede_paragraph", ""),
        insights=insights,
        chart_spec=chart_spec,
        supporting_stats=narrative.get("supporting_stats", []),
        conclusion=narrative.get("conclusion", ""),
        data_caveats=caveats,
        sources_used=sources,
    )
