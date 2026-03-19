"""Dynamic Rubrics Evaluation Pipeline for Writing Coach V2.

Two-stage async LLM pipeline:
  Stage 1 — Rubrics Generator: produces rubric items from (query, input, route)
  Stage 2 — Rubrics Judge: scores each rubric item against the system output

Uses LangChain AzureChatOpenAI for async LLM calls with built-in retry,
and writes results incrementally as CSV + JSONL for resume support.

Usage:
  python -m src.evaluation.eval_pipeline --input final_data/all_results.csv --limit 5
  python -m src.evaluation.eval_pipeline --deployment gpt-4o --concurrency 3
  python -m src.evaluation.eval_pipeline --routes RESEARCH RESPOND --limit 10
"""

import asyncio
import csv
import json
import logging
import os
import re
import argparse
import time
import statistics
import mlflow
from pathlib import Path
from datetime import datetime, timezone
from typing import Any
from collections import Counter  # noqa: E402

import pandas as pd
from dotenv import load_dotenv
from langchain_openai import AzureChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

from src.evaluation.prompt_loader import (
    build_generator_prompts,
    build_judge_prompts,
    format_metrics_definition,
)
from src.constants.metrics_definitions import CORRECTNESS_METRIC_NAME, METRICS_DEFINITION

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

load_dotenv()

# Disable LangSmith tracing — avoids SSL noise when no LangSmith access is configured
os.environ.setdefault("LANGCHAIN_TRACING_V2", "false")

logger = logging.getLogger("eval_pipeline")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
# Suppress noisy third-party loggers
logging.getLogger("langsmith").setLevel(logging.ERROR)
logging.getLogger("langchain").setLevel(logging.WARNING)

DEFAULT_DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4o")
DEFAULT_TEMPERATURE = 0.3
DEFAULT_MAX_TOKENS = 4096
DEFAULT_CONCURRENCY = 5
MAX_RETRIES = 3  # passed to LangChain AzureChatOpenAI max_retries
DEFAULT_INPUT = "final_data/all_results.csv"

# All columns written to CSV for each evaluated row
EVAL_CSV_FIELDNAMES = [
    "row_id",
    "route",
    "status",
    "timestamp",
    "output_relevancy_score",
    "completeness_score",
    "correctness_score",
    "overall_notes",
    "generator_raw_response",
    "rubrics_json",
    "rubrics_reasoning",
    "evaluation_items_json",  # evaluation_items extracted per metric from rubrics
    "judge_raw_response",
    "verdicts_json",
]

METRIC_SCORE_FIELDS = {
    "output relevancy": "output_relevancy_score",
    "completeness": "completeness_score",
    "correctness": "correctness_score",
}

SCORE_FIELD_LABELS = {
    "output_relevancy_score": "Output Relevancy",
    "completeness_score": "Completeness",
    "correctness_score": CORRECTNESS_METRIC_NAME,
}

# ---------------------------------------------------------------------------
# LangChain Azure OpenAI model (cached per parameter set)
# ---------------------------------------------------------------------------

_model: AzureChatOpenAI | None = None
_model_key: tuple | None = None


def get_model(
    deployment: str = DEFAULT_DEPLOYMENT,
    temperature: float = DEFAULT_TEMPERATURE,
    max_tokens: int = DEFAULT_MAX_TOKENS,
) -> AzureChatOpenAI:
    """Get or create a cached LangChain AzureChatOpenAI model.

    Re-creates the model only when parameters change.
    """
    global _model, _model_key
    key = (deployment, temperature, max_tokens)
    if _model is None or _model_key != key:
        _model = AzureChatOpenAI(
            azure_deployment=deployment,
            azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
            api_key=os.environ["AZURE_OPENAI_API_KEY"],
            api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2025-01-01-preview"),
            temperature=temperature,
            max_tokens=max_tokens,
            max_retries=MAX_RETRIES,
        )
        _model_key = key
        logger.info(
            "Created AzureChatOpenAI model: deployment=%s, temperature=%.2f, max_tokens=%d",
            deployment, temperature, max_tokens,
        )
    return _model


# ---------------------------------------------------------------------------
# LLM helpers
# ---------------------------------------------------------------------------


async def call_llm(
    deployment: str,
    system_prompt: str,
    user_prompt: str,
    temperature: float = DEFAULT_TEMPERATURE,
    max_tokens: int = DEFAULT_MAX_TOKENS,
) -> str:
    """Call Azure OpenAI via LangChain and return the text response.

    LangChain handles retries internally (max_retries on the model).

    Args:
        deployment: Azure OpenAI deployment name (e.g., 'gpt-4o').
        system_prompt: System message content.
        user_prompt: User message content.
        temperature: Sampling temperature.
        max_tokens: Maximum tokens in the response.

    Returns:
        The assistant's response text.

    Raises:
        Exception: If all retries are exhausted.
    """
    model = get_model(deployment, temperature, max_tokens)
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt),
    ]
    response = await model.ainvoke(messages)
    return response.content


def parse_json_response(text: str) -> dict | None:
    """Try to parse a JSON object from an LLM response.

    Handles common issues like markdown code fences wrapping the JSON.
    Falls back to ``parse_markdown_rubrics`` if the response looks like the
    structured Markdown rubrics format produced by rubrics_prompt.txt.
    Returns None if all parsing strategies fail.
    """
    # Strip markdown code fences if present
    cleaned = text.strip()
    match = re.search(r"```(?:json)?\s*([\s\S]*?)```", cleaned)
    if match:
        cleaned = match.group(1).strip()

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    # Fallback: try to parse the structured Markdown rubrics format
    return parse_markdown_rubrics(text)


def parse_markdown_rubrics(text: str) -> dict | None:
    """Parse the structured Markdown rubrics format produced by the generator.

    Supports both the legacy ``### Criterion N: [Name]`` format and the newer
    ``### Metric Rubric N: [Name]`` format. Returns the normalized
    ``{"rubrics": [...]}`` structure expected by the rest of the pipeline.

    Returns None if the text does not look like rubrics Markdown.
    """
    header_pattern = r"###\s+(?:Criterion|Metric Rubric)\s+\d+:\s*(.+)"

    # Must have at least one rubric header to be considered rubrics Markdown
    rubric_headers = re.findall(header_pattern, text, re.IGNORECASE)
    if not rubric_headers:
        return None

    rubrics = []
    # Split on rubric headers to process each block individually
    blocks = re.split(
        r"(?=###\s+(?:Criterion|Metric Rubric)\s+\d+:)",
        text,
        flags=re.IGNORECASE,
    )

    for block in blocks:
        header_match = re.match(header_pattern, block, re.IGNORECASE)
        if not header_match:
            continue

        rubric_name = header_match.group(1).strip().strip("*")

        description = ""
        desc_match = re.search(r"\*\*Description\*\*:\s*(.+?)(?=\n\*\*|\n###|$)", block, re.DOTALL)
        if desc_match:
            description = desc_match.group(1).strip()

        linked_to = ""
        linked_match = re.search(r"\*\*Linked to\*\*:\s*(.+)", block)
        if linked_match:
            linked_to = linked_match.group(1).strip()

        weight = ""
        weight_match = re.search(r"\*\*(?:Weight|Metric Importance)\*\*:\s*(.+)", block)
        if weight_match:
            weight = weight_match.group(1).strip()

        evaluation_items: list[dict[str, str]] = []
        items_match = re.search(
            r"\*\*Evaluation Items\*\*:\s*(.+?)(?=\n\|\s*Level\s*\||\n###|$)",
            block,
            re.DOTALL,
        )
        if items_match:
            items_block = items_match.group(1)
            for line in items_block.splitlines():
                stripped = line.strip()
                if not stripped.startswith("-"):
                    continue

                item_text = stripped[1:].strip()
                item_importance = ""
                # Extract importance: handles patterns like "(Importance: High)", "(Importance: High).", etc.
                importance_match = re.search(r"\(Importance:\s*([^)]+)\)", item_text)
                if importance_match:
                    item_importance = importance_match.group(1).strip()
                    # Remove the "(Importance: ...)" pattern and any trailing punctuation
                    item_text = re.sub(r"\s*\(Importance:\s*[^)]+\)[.,;:]?\s*$", "", item_text).strip()

                evaluation_items.append({
                    "item": item_text,
                    "importance": item_importance,
                })

        # Extract table rows: | Level | Score | Description | Indicators |
        # Level cells may be bold-formatted and use either the legacy 1-4
        # labels or the newer metric-specific 0-2 labels.
        levels = []
        table_rows = re.findall(
            r"\|\s*\*{0,2}([^|]+?)\*{0,2}\s*\|\s*(\d+)\s*\|([^|]+)\|([^|]+)\|",
            block,
            re.IGNORECASE,
        )
        for level, score, desc, indicators in table_rows:
            if level.strip().lower() == "level":
                continue
            levels.append({
                "level": level.strip(),
                "score": int(score.strip()),
                "description": desc.strip(),
                "indicators": indicators.strip(),
            })

        rubrics.append({
            "metric": rubric_name,
            "description": description,
            "linked_to": linked_to,
            "metric_importance": weight,
            "evaluation_items": evaluation_items,
            "levels": levels,
        })

    if not rubrics:
        return None

    # Extract any overall reasoning/assessment block
    reasoning = ""
    reasoning_match = re.search(
        r"###\s+Overall Assessment Guidelines(.+?)(?=##|$)", text, re.DOTALL | re.IGNORECASE
    )
    if reasoning_match:
        reasoning = reasoning_match.group(1).strip()

    logger.debug("parse_markdown_rubrics: extracted %d rubrics", len(rubrics))
    return {"rubrics": rubrics, "reasoning": reasoning}


def _normalize_metric_name(name: str | None) -> str:
    """Normalize metric names for reliable matching across prompt outputs."""
    if not name:
        return ""
    return re.sub(r"\s+", " ", name).strip().lower()


def _extract_metric_scores(evaluation: list[dict[str, Any]]) -> dict[str, int | None]:
    """Extract per-metric 0-2 scores from judge evaluation entries."""
    scores = {field_name: None for field_name in METRIC_SCORE_FIELDS.values()}

    for verdict in evaluation:
        metric_name = verdict.get("metric_name") or verdict.get("criterion_name")
        field_name = METRIC_SCORE_FIELDS.get(_normalize_metric_name(metric_name))
        if field_name is None:
            continue

        score = verdict.get("score")
        if isinstance(score, int):
            scores[field_name] = score

    return scores


def _required_score_fields(has_suggestions: bool) -> list[str]:
    """Return the metric score fields expected for a given row."""
    required = ["output_relevancy_score", "completeness_score"]
    if has_suggestions:
        required.append("correctness_score")
    return required


def _avg_std(scores: list) -> tuple[float, float]:
    """Return (mean, stdev) of scores, or (nan, 0.0) for an empty list."""
    if not scores:
        return float("nan"), 0.0
    avg = sum(scores) / len(scores)
    return avg, (statistics.stdev(scores) if len(scores) >= 2 else 0.0)


def _compute_score_stats(ok_results: list[dict[str, Any]]) -> dict[str, Any]:
    """Compute per-route, micro, and macro score statistics from OK-status results."""
    route_buckets: dict[str, dict[str, list]] = {}
    for r in ok_results:
        b = route_buckets.setdefault(
            r["route"],
            {"row_count": 0, **{field: [] for field in SCORE_FIELD_LABELS}},
        )
        b["row_count"] += 1
        for field in SCORE_FIELD_LABELS:
            if r.get(field) is not None:
                b[field].append(r[field])

    per_route = {
        route: {
            "n": b["row_count"],
            "counts": {field: len(b[field]) for field in SCORE_FIELD_LABELS},
            "output_relevancy": _avg_std(b["output_relevancy_score"]),
            "completeness": _avg_std(b["completeness_score"]),
            "correctness": _avg_std(b["correctness_score"]),
        }
        for route, b in route_buckets.items()
    }
    all_rel = [s for b in route_buckets.values() for s in b["output_relevancy_score"]]
    all_comp = [s for b in route_buckets.values() for s in b["completeness_score"]]
    all_correctness = [s for b in route_buckets.values() for s in b["correctness_score"]]
    route_rel_avgs = [_avg_std(b["output_relevancy_score"])[0] for b in route_buckets.values() if b["output_relevancy_score"]]
    route_comp_avgs = [_avg_std(b["completeness_score"])[0] for b in route_buckets.values() if b["completeness_score"]]
    route_correctness_avgs = [_avg_std(b["correctness_score"])[0] for b in route_buckets.values() if b["correctness_score"]]
    return {
        "per_route": per_route,
        "micro": {
            "n": len(ok_results),
            "counts": {
                "output_relevancy_score": len(all_rel),
                "completeness_score": len(all_comp),
                "correctness_score": len(all_correctness),
            },
            "output_relevancy": _avg_std(all_rel),
            "completeness": _avg_std(all_comp),
            "correctness": _avg_std(all_correctness),
        },
        "macro": {
            "n_routes": len(route_buckets),
            "counts": {
                "output_relevancy_score": len(route_rel_avgs),
                "completeness_score": len(route_comp_avgs),
                "correctness_score": len(route_correctness_avgs),
            },
            "output_relevancy": _avg_std(route_rel_avgs),
            "completeness": _avg_std(route_comp_avgs),
            "correctness": _avg_std(route_correctness_avgs),
        },
    }


# ---------------------------------------------------------------------------
# Resume support
# ---------------------------------------------------------------------------


def load_processed_ids(details_jsonl: Path) -> set:
    """Return set of row_ids already written to the JSONL output file."""
    processed = set()
    if details_jsonl.exists():
        with open(details_jsonl, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        processed.add(json.loads(line)["row_id"])
                    except (json.JSONDecodeError, KeyError):
                        pass
    return processed


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_input_data(
    input_path: str,
    routes: list[str] | None = None,
    limit: int | None = None,
    route_column: str = "intended",
    data_origin: str = "all",
) -> list[dict[str, Any]]:
    """Load and filter the input dataset.

    Parses the ``output`` JSON column from all_results.csv to extract:
    - ``response_text``: the ``output["response"]`` field — for RESPOND/RESEARCH this is
      the full output to evaluate; for REVISE routes this is a short explanatory note
      that accompanies the suggestions.
    - ``suggestions``: list of revision suggestion dicts (original_text, transformed_text, explanation)
    - ``has_suggestions``: True when at least one suggestion has a non-empty transformed_text

    Args:
        input_path: Path to all_results.csv (must have an ``output`` JSON column).
        routes: Optional list of routes to filter by (e.g., ['RESEARCH', 'RESPOND']).
        limit: Optional max number of rows to process.
        route_column: Which route column to use: ``'intended'`` (default) for ``route_intended``,
            or ``'orchestrator'`` for ``route_orch``. If the selected column is missing,
            falls back to the other column.
        data_origin: Filter rows by data origin. ``'all'`` (default) keeps all rows.
            ``'synthetic'`` keeps rows whose ``dataset_source`` starts with ``'Synthetic'``
            or equals ``'extra_respond_alex'``. ``'natural'`` keeps all other rows.

    Returns:
        List of row dicts with keys: row_id, query, input, route, response_text,
        suggestions, has_suggestions.
    """
    df = pd.read_csv(input_path, encoding="utf-8-sig")

    # Determine which route column to use
    if route_column == "intended":
        preferred_col = "route_intended"
        fallback_col = "route_orch"
    else:  # orchestrator
        preferred_col = "route_orch"
        fallback_col = "route_intended"

    # Use preferred column if present, otherwise fall back
    if preferred_col in df.columns:
        route_col = preferred_col
    elif fallback_col in df.columns:
        logger.info("Preferred column '%s' not found, falling back to '%s'", preferred_col, fallback_col)
        route_col = fallback_col
    else:
        logger.error("Neither '%s' nor '%s' column found in input CSV", preferred_col, fallback_col)
        return []

    if routes:
        routes_upper = [r.upper() for r in routes]
        df = df[df[route_col].str.upper().isin(routes_upper)]
        logger.info("Filtered to routes %s: %d rows", routes_upper, len(df))

    if data_origin != "all":
        if "dataset_source" not in df.columns:
            logger.warning("'dataset_source' column not found — data_origin filter '%s' ignored", data_origin)
        else:
            is_synthetic = (
                df["dataset_source"].str.startswith("Synthetic", na=False)
                | (df["dataset_source"] == "extra_respond_alex")
            )
            if data_origin == "synthetic":
                df = df[is_synthetic]
            else:  # natural
                df = df[~is_synthetic]
            logger.info("Filtered to data_origin='%s': %d rows", data_origin, len(df))

    if limit:
        df = df.head(limit)

    rows = []
    for _, row in df.iterrows():
        # Parse the output JSON column
        output_data: dict = {}
        raw_output = row.get("output", "")
        if pd.notna(raw_output) and raw_output:
            try:
                output_data = json.loads(str(raw_output))
            except (json.JSONDecodeError, ValueError):
                logger.warning("Row %s: Failed to parse output JSON", row.get("row_id"))

        response_text = str(output_data.get("response", "")).strip() if output_data else ""

        # Keep only suggestions with a non-empty transformed_text
        raw_suggestions = output_data.get("suggestions", []) if output_data else []
        suggestions = [
            s for s in (raw_suggestions or [])
            if isinstance(s, dict) and (s.get("transformed_text") or "").strip()
        ]

        rows.append({
            "row_id": int(row["row_id"]),
            "query": str(row.get("query", "")),
            "input": str(row.get("input", "")),
            "route": str(row.get(route_col, "UNKNOWN")),
            "response_text": response_text,
            "suggestions": suggestions,
            "has_suggestions": len(suggestions) > 0,
        })

    logger.info("Loaded %d rows from %s", len(rows), input_path)
    return rows


# ---------------------------------------------------------------------------
# Output formatting helpers
# ---------------------------------------------------------------------------


def _format_output_for_judge(response_text: str, suggestions: list[dict]) -> str:
    """Format the writing coach output for the judge LLM.

    For routes with no suggestions (RESPOND, RESEARCH), returns response_text directly —
    it is the full output to be evaluated.
    For routes with suggestions (REVISE_SIMPLE, REVISE_RESEARCH), returns a structured block
    containing the brief response note followed by numbered suggestions.  Each suggestion
    shows the original passage, the proposed revision, and an explanatory note.

    Args:
        response_text: The output["response"] field. This is the full response for
            RESPOND/RESEARCH routes, or a short explanatory note for REVISE routes.
        suggestions: List of suggestion dicts with keys: original_text, transformed_text,
            explanation.  Only suggestions with a non-empty transformed_text are expected here.

    Returns:
        A formatted string suitable for the judge_prompt ``{output_text}`` slot.
    """
    if not suggestions:
        return response_text

    parts: list[str] = []

    if response_text.strip():
        parts.append("### Writing Coach Note\n" + response_text.strip())

    parts.append(
        f"### Revision Suggestions ({len(suggestions)} total)\n"
        "Each suggestion shows the original passage and the proposed revision. "
        "The user can accept or reject each suggestion independently."
    )

    for i, s in enumerate(suggestions, 1):
        original = (s.get("original_text") or "").strip()
        transformed = (s.get("transformed_text") or "").strip()
        explanation = (s.get("explanation") or "").strip()
        char_start = s.get("char_start")
        char_end = s.get("char_end")

        block_lines = [f"#### Suggestion {i}"]
        if explanation:
            block_lines.append(f"**Purpose**: {explanation}")
        if isinstance(char_start, int) and isinstance(char_end, int):
            block_lines.append(f"**Span**: char_start={char_start}, char_end={char_end}")
        if original:
            block_lines.append(f"**Original Text**:\n{original}")
        else:
            block_lines.append("**Original Text**: (insertion — no existing text is replaced)")
        block_lines.append(f"**Proposed Revision**:\n{transformed}")

        parts.append("\n".join(block_lines))

    return "\n\n---\n\n".join(parts)


# ---------------------------------------------------------------------------
# Core pipeline: process a single row
# ---------------------------------------------------------------------------


async def process_row(
    row: dict[str, Any],
    deployment: str,
    temperature: float,
    max_tokens: int,
    semaphore: asyncio.Semaphore,
    generator_only: bool = False,
    rubrics_mode: str = "combined",
) -> dict[str, Any]:
    """Run Stage 1 (generate rubrics) and optionally Stage 2 (judge rubrics) for one row.

    Args:
        row: Dict with row_id, query, input, route, response_text, output_data.
        deployment: Azure OpenAI deployment name.
        temperature: LLM temperature.
        max_tokens: Max tokens per LLM call.
        semaphore: Concurrency limiter.
        generator_only: If True, stop after Stage 1 and skip the judge.
        rubrics_mode: ``'combined'`` uses a single LLM call with rubrics_prompt.txt
            (both metrics together). ``'split'`` makes two separate LLM calls — one
            per metric — using the metric-specific prompt files, then merges the
            resulting rubrics before judging.

    Returns:
        Dict with full evaluation results for this row.
    """
    row_id = row["row_id"]
    result = {
        "row_id": row_id,
        "route": row["route"],
        "status": "OK",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        # Stage 1 outputs
        "generator_raw_response": None,
        "rubrics": None,
        "rubrics_reasoning": None,
        # Stage 2 outputs
        "judge_raw_response": None,
        "verdicts": None,
        "overall_notes": None,
        # Scores
        "output_relevancy_score": None,
        "completeness_score": None,
        "correctness_score": None,
    }

    async with semaphore:
        try:
            logger.info("Row %d: processing (route=%s)", row_id, row["route"])

            # ---- Stage 1: Generate Rubrics ----
            if rubrics_mode == "split":
                # Two separate LLM calls, one per metric, then merge rubrics.
                or_metrics_def = format_metrics_definition(
                    {k: v for k, v in METRICS_DEFINITION.items() if k == "Output Relevancy"}
                )
                c_metrics_def = format_metrics_definition(
                    {k: v for k, v in METRICS_DEFINITION.items() if k == "Completeness"}
                )

                gen_system_or, gen_user_or = build_generator_prompts(
                    user_query=row["query"],
                    input_text=row["input"],
                    route=row["route"],
                    metrics_definition=or_metrics_def,
                    prompt_file="rubrics_prompt_output_relevancy.txt",
                )
                gen_system_c, gen_user_c = build_generator_prompts(
                    user_query=row["query"],
                    input_text=row["input"],
                    route=row["route"],
                    metrics_definition=c_metrics_def,
                    prompt_file="rubrics_prompt_completeness.txt",
                )

                gen_response_or, gen_response_c = await asyncio.gather(
                    call_llm(deployment, gen_system_or, gen_user_or, temperature, max_tokens),
                    call_llm(deployment, gen_system_c, gen_user_c, temperature, max_tokens),
                )
                result["generator_raw_response"] = json.dumps(
                    {"output_relevancy": gen_response_or, "completeness": gen_response_c},
                    ensure_ascii=False,
                )

                gen_parsed_or = parse_json_response(gen_response_or)
                gen_parsed_c  = parse_json_response(gen_response_c)

                if gen_parsed_or is None:
                    logger.warning("Row %d: Failed to parse generator response (Output Relevancy)", row_id)
                    result["status"] = "GENERATOR_PARSE_ERROR"
                    return result
                if gen_parsed_c is None:
                    logger.warning("Row %d: Failed to parse generator response (Completeness)", row_id)
                    result["status"] = "GENERATOR_PARSE_ERROR"
                    return result

                rubrics = (
                    gen_parsed_or.get("rubrics", [])
                    + gen_parsed_c.get("rubrics", [])
                )
                result["rubrics_reasoning"] = json.dumps(
                    {
                        "output_relevancy": gen_parsed_or.get("reasoning", ""),
                        "completeness": gen_parsed_c.get("reasoning", ""),
                    },
                    ensure_ascii=False,
                )
            else:
                # combined mode: single LLM call with both metrics
                gen_system, gen_user = build_generator_prompts(
                    user_query=row["query"],
                    input_text=row["input"],
                    route=row["route"],
                )
                gen_response = await call_llm(
                    deployment=deployment,
                    system_prompt=gen_system,
                    user_prompt=gen_user,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
                result["generator_raw_response"] = gen_response

                gen_parsed = parse_json_response(gen_response)
                if gen_parsed is None:
                    logger.warning("Row %d: Failed to parse generator response", row_id)
                    result["status"] = "GENERATOR_PARSE_ERROR"
                    return result

                rubrics = gen_parsed.get("rubrics", [])
                result["rubrics_reasoning"] = gen_parsed.get("reasoning", "")

            result["rubrics"] = rubrics

            if not rubrics:
                logger.warning("Row %d: Generator produced zero rubrics", row_id)
                result["status"] = "NO_RUBRICS"
                return result

            # Stop here if only Stage 1 was requested
            if generator_only:
                result["status"] = "GENERATOR_ONLY"
                return result

            # Check if we have anything to judge
            if not row["response_text"] and not row.get("has_suggestions"):
                logger.warning("Row %d: No response text or suggestions to judge", row_id)
                result["status"] = "NO_OUTPUT_TEXT"
                return result

            # ---- Stage 2: Judge Rubrics ----

            rubrics_str = json.dumps(rubrics, indent=2, ensure_ascii=False)

            # Format output: plain text for RESPOND/RESEARCH, structured
            # suggestions block for REVISE routes (detected by has_suggestions).
            output_for_judge = _format_output_for_judge(
                response_text=row["response_text"],
                suggestions=row.get("suggestions", []),
            )

            judge_system, judge_user = build_judge_prompts(
                user_query=row["query"],
                input_text=row["input"],
                output_text=output_for_judge,
                rubrics=rubrics_str,
                include_correctness=bool(row.get("has_suggestions")),
            )

            judge_response = await call_llm(
                deployment=deployment,
                system_prompt=judge_system,
                user_prompt=judge_user,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            result["judge_raw_response"] = judge_response

            judge_parsed = parse_json_response(judge_response)
            if judge_parsed is None:
                logger.warning("Row %d: Failed to parse judge response", row_id)
                result["status"] = "JUDGE_PARSE_ERROR"
                return result

            # Process per-metric evaluation entries (score 0-2 per metric)
            evaluation = judge_parsed.get("evaluation", [])
            result["verdicts"] = evaluation
            result["overall_notes"] = judge_parsed.get("summary") or (
                judge_parsed.get("overall_assessment", {}).get("summary", "")
            )

            metric_scores = _extract_metric_scores(evaluation)
            result.update(metric_scores)

            required_score_fields = _required_score_fields(bool(row.get("has_suggestions")))
            if any(result.get(field) is None for field in required_score_fields):
                logger.warning("Row %d: Judge response missing one or more metric scores", row_id)
                result["status"] = "JUDGE_INCOMPLETE"
                return result

            logger.info(
                "Row %d: done — Output Relevancy=%s, Completeness=%s, Correctness=%s",
                row_id,
                result["output_relevancy_score"],
                result["completeness_score"],
                result["correctness_score"],
            )

        except Exception as e:
            logger.error("Row %d: Error — %s", row_id, e)
            result["status"] = "ERROR"
            result["error"] = str(e)

    return result


# ---------------------------------------------------------------------------
# Output writing (thread-safe with asyncio.Lock)
# ---------------------------------------------------------------------------


class OutputWriter:
    """Thread-safe incremental writer for CSV + JSONL evaluation outputs."""

    def __init__(self, output_dir: Path, run_name: str):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.results_csv = output_dir / f"{run_name}_results.csv"
        self.details_jsonl = output_dir / f"{run_name}_details.jsonl"
        self._lock = asyncio.Lock()

        # Open files in append mode
        csv_is_new = not self.results_csv.exists() or self.results_csv.stat().st_size == 0
        self._csv_file = open(self.results_csv, "a", newline="", encoding="utf-8")
        self._jsonl_file = open(self.details_jsonl, "a", encoding="utf-8")

        self._csv_writer = csv.DictWriter(self._csv_file, fieldnames=EVAL_CSV_FIELDNAMES)
        if csv_is_new:
            self._csv_writer.writeheader()
            self._csv_file.flush()

    async def write_result(self, result: dict[str, Any]) -> None:
        """Write one evaluation result to both CSV and JSONL."""
        async with self._lock:
            # CSV — all result fields (nested dicts/lists as JSON strings)
            self._csv_writer.writerow({
                "row_id": result["row_id"],
                "route": result["route"],
                "status": result["status"],
                "timestamp": result["timestamp"],
                "output_relevancy_score": result["output_relevancy_score"],
                "completeness_score": result["completeness_score"],
                "correctness_score": result["correctness_score"],
                "overall_notes": result["overall_notes"],
                "generator_raw_response": result["generator_raw_response"],
                "rubrics_json": json.dumps(result["rubrics"], ensure_ascii=False)
                    if result["rubrics"] is not None else "",
                "rubrics_reasoning": result["rubrics_reasoning"] if isinstance(result["rubrics_reasoning"], str)
                    else json.dumps(result["rubrics_reasoning"], ensure_ascii=False)
                    if result["rubrics_reasoning"] is not None else "",
                "evaluation_items_json": json.dumps(
                    (
                        [
                            {
                                "metric": v.get("metric_name") or v.get("criterion_name"),
                                "evaluation_items": v.get("evaluation_items", []),
                            }
                            for v in (result["verdicts"] or [])
                        ]
                        if result["verdicts"] is not None
                        else [
                            {"metric": r.get("metric"), "evaluation_items": r.get("evaluation_items", [])}
                            for r in (result["rubrics"] or [])
                        ]
                    ),
                    ensure_ascii=False,
                ) if result["rubrics"] is not None else "",
                "judge_raw_response": result["judge_raw_response"],
                "verdicts_json": json.dumps(result["verdicts"], ensure_ascii=False)
                    if result["verdicts"] is not None else "",
            })
            self._csv_file.flush()

            # JSONL — full details
            self._jsonl_file.write(json.dumps(result, ensure_ascii=False) + "\n")
            self._jsonl_file.flush()

    def close(self) -> None:
        """Close output files."""
        self._csv_file.close()
        self._jsonl_file.close()

    @property
    def details_path(self) -> Path:
        return self.details_jsonl

    @property
    def results_path(self) -> Path:
        return self.results_csv


# ---------------------------------------------------------------------------
# Main pipeline orchestrator
# ---------------------------------------------------------------------------


async def run_pipeline(
    input_path: str,
    output_dir: str = "data_outputs/eval",
    deployment: str = DEFAULT_DEPLOYMENT,
    temperature: float = DEFAULT_TEMPERATURE,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    concurrency: int = DEFAULT_CONCURRENCY,
    routes: list[str] | None = None,
    limit: int | None = None,
    resume: bool = True,
    run_name: str | None = None,
    generator_only: bool = False,
    rubrics_mode: str = "combined",
    route_column: str = "intended",
    save_local: bool = False,
    data_origin: str = "all",
) -> None:
    """Run the full dynamic rubrics evaluation pipeline.

    Args:
        input_path: Path to the input CSV (all_results.csv).
        output_dir: Directory for evaluation output files.
        deployment: Azure OpenAI deployment name (e.g., 'gpt-4o').
        temperature: LLM temperature for both generator and judge.
        max_tokens: Max tokens per LLM call.
        concurrency: Max parallel LLM calls.
        routes: Optional list of routes to filter by.
        limit: Optional max rows to process.
        resume: If True, skip rows already in the output JSONL.
        run_name: Optional name for output files. Defaults to timestamp-based name.
        generator_only: If True, run only Stage 1 (rubrics generation) and skip
            the judge. Useful for testing or inspecting generated rubrics.
        rubrics_mode: ``'combined'`` (default) uses a single generator LLM call
            with both metrics in one prompt.  ``'split'`` makes two parallel calls,
            one per metric, using the metric-specific prompt files, then merges the
            rubrics before judging.
        route_column: Which route column to use: ``'intended'`` (default) for
            ``route_intended``, or ``'orchestrator'`` for ``route_orch``.
        data_origin: Filter rows by data origin. ``'all'`` (default) keeps all rows.
            ``'synthetic'`` or ``'natural'`` filters on the ``dataset_source`` column.
    """
    if run_name is None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_name = f"eval_{ts}"

    logger.info("=" * 80)
    logger.info("DYNAMIC RUBRICS EVALUATION PIPELINE")
    logger.info("=" * 80)
    logger.info("Deployment:  %s", deployment)
    logger.info("Concurrency: %d", concurrency)
    logger.info("Temperature: %.2f", temperature)
    logger.info("Input:       %s", input_path)
    logger.info("Output dir:  %s", output_dir)
    logger.info("Run name:    %s", run_name)
    if generator_only:
        logger.info("Mode:        GENERATOR ONLY (Stage 1, no judge)")
    logger.info("Rubrics mode: %s", rubrics_mode)
    logger.info("Route column: %s", route_column)
    logger.info("Data origin:  %s", data_origin)
    logger.info("=" * 80)

    # Load data
    rows = load_input_data(input_path, routes=routes, limit=limit, route_column=route_column, data_origin=data_origin)
    if not rows:
        logger.warning("No rows to process. Exiting.")
        return

    # Set up output writer
    writer = OutputWriter(Path(output_dir), run_name)

    # Resume support
    if resume:
        processed_ids = load_processed_ids(writer.details_path)
        if processed_ids:
            before = len(rows)
            rows = [r for r in rows if r["row_id"] not in processed_ids]
            logger.info(
                "Resume: %d rows already processed, %d remaining",
                before - len(rows),
                len(rows),
            )

    if not rows:
        logger.info("All rows already processed. Nothing to do.")
        writer.close()
        return

    # Set up concurrency control
    semaphore = asyncio.Semaphore(concurrency)

    # Create tasks for all rows
    async def process_and_write(row: dict) -> dict:
        result = await process_row(
            row, deployment, temperature, max_tokens, semaphore,
            generator_only=generator_only,
            rubrics_mode=rubrics_mode,
        )
        await writer.write_result(result)
        return result

    logger.info("Processing %d rows with concurrency=%d...", len(rows), concurrency)
    start_time = time.time()

    tasks = [process_and_write(row) for row in rows]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    elapsed = time.time() - start_time

    # Handle any exceptions from gather
    actual_results = []
    for i, r in enumerate(results):
        if isinstance(r, Exception):
            logger.error("Row task %d raised exception: %s", i, r)
        else:
            actual_results.append(r)

    writer.close()

    # Build enriched all_results copy with eval columns appended
    enriched_paths = build_enriched_output(
        input_path=input_path,
        details_jsonl=writer.details_path,
        output_dir=Path(output_dir),
        run_name=run_name,
    )

    # Print summary
    _print_summary(actual_results, elapsed, writer, enriched_paths, save_local=save_local)

    # Log to MLflow
    _log_to_mlflow(
        results=actual_results,
        elapsed=elapsed,
        enriched_paths=enriched_paths,
        results_csv=writer.results_path,
        details_jsonl=writer.details_path,
        save_local=save_local,
        params={
            "deployment": deployment,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "concurrency": concurrency,
            "rubrics_mode": rubrics_mode,
            "route_column": route_column,
            "input_path": input_path,
            "routes": routes,
            "limit": limit,
            "run_name": run_name,
            "data_origin": data_origin,
        },
    )


def build_enriched_output(
    input_path: str,
    details_jsonl: Path,
    output_dir: Path,
    run_name: str,
) -> tuple[Path, Path] | None:
    """Copy the evaluated rows from the input file and append eval result columns.

    Reads ``all_results_with_final_text`` (the input), filters to only the rows
    that were evaluated in this run, merges eval results by ``row_id``, and
    writes two enriched files in ``output_dir``:

    - ``{run_name}_all_results_enriched.csv``
    - ``{run_name}_all_results_enriched.jsonl``

    Args:
        input_path: Path to all_results_with_final_text.csv used as pipeline input.
        details_jsonl: Path to the eval details JSONL for this run.
        output_dir: Directory where enriched files will be written.
        run_name: Run name prefix for output files.

    Returns:
        Tuple of (enriched_csv_path, enriched_jsonl_path), or None on failure.
    """
    # Load eval results indexed by row_id
    eval_by_id: dict[int, dict] = {}
    if details_jsonl.exists():
        with open(details_jsonl, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        rec = json.loads(line)
                        eval_by_id[int(rec["row_id"])] = rec
                    except (json.JSONDecodeError, KeyError):
                        pass

    if not eval_by_id:
        logger.warning("No eval results found in %s — skipping enriched output", details_jsonl)
        return None

    # Load input CSV and keep only the rows that were evaluated
    df = pd.read_csv(input_path, encoding="utf-8-sig")
    df = df[df["row_id"].isin(eval_by_id.keys())].copy()
    df.reset_index(drop=True, inplace=True)

    # Append eval columns
    _scalar_map = {
        "eval_status": "status", "eval_timestamp": "timestamp",
        "eval_output_relevancy_score": "output_relevancy_score",
        "eval_completeness_score": "completeness_score",
        "eval_correctness_score": "correctness_score",
        "eval_overall_notes": "overall_notes",
        "eval_generator_raw": "generator_raw_response",
        "eval_judge_raw": "judge_raw_response",
    }
    _json_map = {
        "eval_rubrics_json": "rubrics",
        "eval_verdicts_json": "verdicts",
    }
    for col in [*_scalar_map, *_json_map]:
        df[col] = None

    def _to_json_str(val: Any) -> str | None:
        if val is None:
            return None
        if isinstance(val, str):
            return val
        return json.dumps(val, ensure_ascii=False)

    for idx, row in df.iterrows():
        ev = eval_by_id[int(row["row_id"])]
        for col, key in _scalar_map.items():
            df.at[idx, col] = ev.get(key)
        for col, key in _json_map.items():
            df.at[idx, col] = _to_json_str(ev.get(key))

    logger.info("Enriched output: %d rows merged", len(df))

    enriched_csv = output_dir / f"{run_name}_all_results_enriched.csv"
    df.to_csv(enriched_csv, index=False, encoding="utf-8")
    logger.info("Enriched CSV written: %s", enriched_csv)

    enriched_jsonl = output_dir / f"{run_name}_all_results_enriched.jsonl"
    with open(enriched_jsonl, "w", encoding="utf-8") as f:
        for _, row in df.iterrows():
            rec = row.to_dict()
            # Parse the JSON string columns back into objects for the JSONL
            for col in ("eval_rubrics_json", "eval_verdicts_json"):
                v = rec.get(col)
                if v and isinstance(v, str):
                    try:
                        rec[col] = json.loads(v)
                    except json.JSONDecodeError:
                        pass
            f.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")
    logger.info("Enriched JSONL written: %s", enriched_jsonl)

    return enriched_csv, enriched_jsonl


def _print_summary(
    results: list[dict[str, Any]],
    elapsed: float,
    writer: OutputWriter,
    enriched_paths: tuple[Path, Path] | None = None,
    save_local: bool = False,
) -> None:
    """Print evaluation summary to console."""
    print("\n" + "=" * 80)
    print("EVALUATION SUMMARY")
    print("=" * 80)
    print(f"Total rows evaluated: {len(results)}")
    print(f"Elapsed time: {elapsed:.1f}s")
    print()

    # Status breakdown

    status_counts = Counter(r["status"] for r in results)
    print("Status breakdown:")
    for status, count in status_counts.most_common():
        print(f"  {status:30} {count:4}")

    # Score summary by route
  

    ok_results = [r for r in results if r["status"] == "OK"]
    if ok_results:
        stats = _compute_score_stats(ok_results)
        print()
        print("Scores by route:")
        for route in sorted(stats["per_route"]):
            s = stats["per_route"][route]
            rel_avg, rel_std = s["output_relevancy"]
            comp_avg, comp_std = s["completeness"]
            corr_avg, corr_std = s["correctness"]
            correctness_text = (
                f"  Correctness={corr_avg:.3f} (±{corr_std:.3f})"
                if s["counts"]["correctness_score"] > 0
                else "  Correctness=N/A"
            )
            print(
                f"  {route:20} Output Relevancy={rel_avg:.3f} (±{rel_std:.3f})  "
                f"Completeness={comp_avg:.3f} (±{comp_std:.3f})"
                f"{correctness_text}  n={s['n']}"
            )

        print()
        micro, macro = stats["micro"], stats["macro"]
        rel_avg, rel_std = micro["output_relevancy"]
        comp_avg, comp_std = micro["completeness"]
        corr_avg, corr_std = micro["correctness"]
        micro_correctness_text = (
            f"  Correctness={corr_avg:.3f} (±{corr_std:.3f})"
            if micro["counts"]["correctness_score"] > 0
            else "  Correctness=N/A"
        )
        print(
            f"  {'OVERALL (micro)':20} Output Relevancy={rel_avg:.3f} (±{rel_std:.3f})  "
            f"Completeness={comp_avg:.3f} (±{comp_std:.3f})"
            f"{micro_correctness_text}  n={micro['n']}"
        )
        rel_avg, rel_std = macro["output_relevancy"]
        comp_avg, comp_std = macro["completeness"]
        corr_avg, corr_std = macro["correctness"]
        macro_correctness_text = (
            f"  Correctness={corr_avg:.3f} (±{corr_std:.3f})"
            if macro["counts"]["correctness_score"] > 0
            else "  Correctness=N/A"
        )
        print(
            f"  {'OVERALL (macro)':20} Output Relevancy={rel_avg:.3f} (±{rel_std:.3f})  "
            f"Completeness={comp_avg:.3f} (±{comp_std:.3f})"
            f"{macro_correctness_text}  n_routes={macro['n_routes']}"
        )

    print()
    if save_local:
        print("Outputs saved locally:")
        print(f"  Summary  : {writer.results_path}")
        print(f"  Details  : {writer.details_path}")
        if enriched_paths:
            print(f"  Enriched CSV  : {enriched_paths[0]}")
            print(f"  Enriched JSONL: {enriched_paths[1]}")
    else:
        print("Outputs uploaded to MLflow (local files will be removed).")
    print("=" * 80)


def _log_to_mlflow(
    results: list[dict[str, Any]],
    elapsed: float,
    params: dict[str, Any],
    enriched_paths: tuple[Path, Path] | None = None,
    results_csv: Path | None = None,
    details_jsonl: Path | None = None,
    save_local: bool = False,
) -> None:
    """Log evaluation run parameters and metrics to MLflow.

    Tracking URI is read from the ``MLFLOW_TRACKING_URI`` environment variable
    (defaults to a local ``mlruns/`` folder if not set). Experiment name is
    derived from ``rubrics_mode`` so that combined-prompt and split-prompt runs
    are stored in separate MLflow experiment folders.
    """
    rubrics_mode = params.get("rubrics_mode", "combined")
    route_source = params.get("route_column", "intended")  # "intended" or "orchestrator"
    n_total = len(results)
    ts = datetime.now().strftime("%m%d_%H%M")
    mlflow_run_name = f"{ts}_{route_source}_{n_total}rows"

    experiment_name = f"wc-eval-rubrics-{rubrics_mode}"
    mlflow.set_experiment(experiment_name)

    with mlflow.start_run(run_name=mlflow_run_name):
        # ---- Parameters ----
        mlflow.log_param("deployment", params.get("deployment"))
        mlflow.log_param("temperature", params.get("temperature"))
        mlflow.log_param("max_tokens", params.get("max_tokens"))
        mlflow.log_param("concurrency", params.get("concurrency"))
        mlflow.log_param("rubrics_mode", rubrics_mode)
        mlflow.log_param("route_source", route_source)
        mlflow.log_param("data_origin", params.get("data_origin", "all"))
        mlflow.log_param("input_path", params.get("input_path"))
        mlflow.log_param(
            "routes_filter",
            json.dumps(params["routes"]) if params.get("routes") else "all",
        )
        mlflow.log_param("limit", params.get("limit"))

        # ---- Input dataset ----
        try:
            input_path = params.get("input_path", "")
            if input_path:
                evaluated_ids = {r["row_id"] for r in results}
                input_df = pd.read_csv(input_path, encoding="utf-8-sig")
                input_df_eval = input_df[input_df["row_id"].isin(evaluated_ids)]
                dataset = mlflow.data.from_pandas(
                    input_df_eval,
                    source=input_path,
                    name="eval_input",
                )
                mlflow.log_input(dataset, context="evaluation")
        except Exception as exc:
            logger.warning("Failed to log dataset to MLflow: %s", exc)

        # ---- Run-level counters ----
        status_counts = Counter(r["status"] for r in results)
        mlflow.log_metric("total_rows", len(results))
        mlflow.log_metric("ok_rows", status_counts.get("OK", 0))
        mlflow.log_metric("elapsed_seconds", round(elapsed, 1))
        for status, count in status_counts.items():
            mlflow.log_metric(f"status_{status.lower()}_count", count)

        # ---- Per-route and aggregate scores ----
        ok_results = [r for r in results if r["status"] == "OK"]
        if ok_results:
            stats = _compute_score_stats(ok_results)
            for route in sorted(stats["per_route"]):
                s = stats["per_route"][route]
                prefix = route.lower()
                mlflow.log_metric(f"{prefix}_n", s["n"])
                for metric in ("output_relevancy", "completeness", "correctness"):
                    score_field = f"{metric}_score"
                    if s["counts"].get(score_field, 0) > 0:
                        avg, std = s[metric]
                        mlflow.log_metric(f"{prefix}_{metric}_avg", avg)
                        mlflow.log_metric(f"{prefix}_{metric}_std", std)

            for level, n_key in (("micro", "n"), ("macro", "n_routes")):
                ls = stats[level]
                if ls[n_key] > 0:
                    for metric in ("output_relevancy", "completeness", "correctness"):
                        score_field = f"{metric}_score"
                        if ls["counts"].get(score_field, 0) == 0:
                            continue
                        avg, std = ls[metric]
                        mlflow.log_metric(f"{level}_{metric}_avg", avg)
                        mlflow.log_metric(f"{level}_{metric}_std", std)

        # ---- Artifacts: actual result data ----
        artifact_paths = [results_csv, details_jsonl, *(enriched_paths or [])]
        for path in artifact_paths:
            if path and Path(path).exists():
                mlflow.log_artifact(str(path), artifact_path="results")

        # ---- Cleanup local files (only after successful upload) ----
        # Note: keep `details_jsonl` locally to support pipeline resume and debugging.
        cleanup_paths = [results_csv, *(enriched_paths or [])]
        if not save_local:
            for path in cleanup_paths:
                if path and Path(path).exists():
                    try:
                        Path(path).unlink()
                    except Exception as exc:
                        logger.warning("Failed to delete local file %s: %s", path, exc)

    logger.info("MLflow run logged to experiment '%s'", experiment_name)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Dynamic Rubrics Evaluation Pipeline for Writing Coach V2",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python -m src.evaluation.eval_pipeline --input final_data/all_results.csv --limit 5\n"
            "  python -m src.evaluation.eval_pipeline --deployment gpt-4o --concurrency 3\n"
            "  python -m src.evaluation.eval_pipeline --routes RESEARCH RESPOND --limit 10\n"
            "  python -m src.evaluation.eval_pipeline --run-name my_experiment\n"
            "\nEnvironment variables (Azure OpenAI):\n"
            "  AZURE_OPENAI_ENDPOINT    — Azure OpenAI endpoint URL\n"
            "  AZURE_OPENAI_API_KEY     — Azure OpenAI API key\n"
            "  AZURE_OPENAI_API_VERSION — API version (default: 2025-01-01-preview)\n"
            "  AZURE_OPENAI_DEPLOYMENT  — Default deployment name (default: gpt-4o)\n"
        ),
    )
    parser.add_argument(
        "--input",
        default=DEFAULT_INPUT,
        help=f"Path to input CSV with system outputs (default: {DEFAULT_INPUT})",
    )
    parser.add_argument(
        "--output-dir",
        default="data_outputs/eval",
        help="Output directory for evaluation results (default: data_outputs/eval)",
    )
    parser.add_argument(
        "--deployment",
        default=DEFAULT_DEPLOYMENT,
        help=f"Azure OpenAI deployment name (default: {DEFAULT_DEPLOYMENT})",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=DEFAULT_TEMPERATURE,
        help=f"LLM temperature (default: {DEFAULT_TEMPERATURE})",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=DEFAULT_MAX_TOKENS,
        help=f"Max tokens per LLM call (default: {DEFAULT_MAX_TOKENS})",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=DEFAULT_CONCURRENCY,
        help=f"Max parallel LLM calls (default: {DEFAULT_CONCURRENCY})",
    )
    parser.add_argument(
        "--routes",
        nargs="+",
        default=None,
        help="Only evaluate rows with these routes (e.g., RESEARCH RESPOND)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Max number of rows to process (for testing)",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Disable resume — reprocess all rows even if already in output",
    )
    parser.add_argument(
        "--generator-only",
        action="store_true",
        help="Run Stage 1 only (rubrics generation) — skip the judge. Useful for testing.",
    )
    parser.add_argument(
        "--rubrics-mode",
        choices=["combined", "split"],
        default="combined",
        help=(
            "How to run the rubrics generator. "
            "'combined' (default): single LLM call with both metrics in one prompt (rubrics_prompt.txt). "
            "'split': two parallel LLM calls, one per metric, using the metric-specific prompt files."
        ),
    )
    parser.add_argument(
        "--route-column",
        choices=["intended", "orchestrator"],
        default="intended",
        help=(
            "Which route column to use from the input CSV. "
            "'intended' (default): use route_intended (the ground truth route). "
            "'orchestrator': use route_orch (the orchestrator's predicted route). "
            "Falls back to the other column if the preferred one is missing."
        ),
    )
    parser.add_argument(
        "--run-name",
        default=None,
        help="Name for output files (default: eval_YYYYMMDD_HHMMSS)",
    )
    parser.add_argument(
        "--save-local",
        action="store_true",
        default=False,
        help=(
            "Keep output files on disk after the run (default: False). "
            "By default, files are uploaded to MLflow and then deleted locally."
        ),
    )
    parser.add_argument(
        "--data-origin",
        choices=["all", "natural", "synthetic"],
        default="all",
        help=(
            "Filter rows by data origin (default: all). "
            "'synthetic': rows where dataset_source starts with 'Synthetic' or equals 'extra_respond_alex'. "
            "'natural': all other rows."
        ),
    )

    args = parser.parse_args()

    asyncio.run(
        run_pipeline(
            input_path=args.input,
            output_dir=args.output_dir,
            deployment=args.deployment,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            concurrency=args.concurrency,
            routes=args.routes,
            limit=args.limit,
            resume=not args.no_resume,
            run_name=args.run_name,
            generator_only=args.generator_only,
            rubrics_mode=args.rubrics_mode,
            route_column=args.route_column,
            save_local=args.save_local,
            data_origin=args.data_origin,
        )
    )


if __name__ == "__main__":
    main()
