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
import sys
import time
from pathlib import Path
from datetime import datetime, timezone
from typing import Any

import pandas as pd
from dotenv import load_dotenv
from langchain_openai import AzureChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

from src.evaluation.prompt_loader import build_generator_prompts, build_judge_prompts

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

# Scalar summary columns written once per evaluated row
EVAL_CSV_FIELDNAMES = [
    "row_id",
    "route",
    "status",
    "timestamp",
    "output_relevancy_score",
    "completeness_score",
    "overall_notes",
    # Structured fields packed as valid JSON strings
    "rubrics_json",    # generated rubrics array
    "verdicts_json",   # judge evaluation array
    "meta_verdicts_json",
]

METRIC_SCORE_FIELDS = {
    "output relevancy": "output_relevancy_score",
    "completeness": "completeness_score",
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
                importance_match = re.search(r"\(Importance:\s*([^)]+)\)\s*$", item_text)
                if importance_match:
                    item_importance = importance_match.group(1).strip()
                    item_text = re.sub(r"\s*\(Importance:\s*[^)]+\)\s*$", "", item_text).strip()

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

    Returns:
        List of row dicts with keys: row_id, query, input, route, response_text,
        suggestions, has_suggestions.
    """
    df = pd.read_csv(input_path, encoding="utf-8-sig")

    # Use route_intended as the gold route (fallback to route_orch if missing)
    route_col = "route_intended" if "route_intended" in df.columns else "route_orch"

    if routes:
        routes_upper = [r.upper() for r in routes]
        df = df[df[route_col].str.upper().isin(routes_upper)]
        logger.info("Filtered to routes %s: %d rows", routes_upper, len(df))

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

        block_lines = [f"#### Suggestion {i}"]
        if explanation:
            block_lines.append(f"**Purpose**: {explanation}")
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
) -> dict[str, Any]:
    """Run Stage 1 (generate rubrics) and optionally Stage 2 (judge rubrics) for one row.

    Args:
        row: Dict with row_id, query, input, route, response_text, output_data.
        deployment: Azure OpenAI deployment name.
        temperature: LLM temperature.
        max_tokens: Max tokens per LLM call.
        semaphore: Concurrency limiter.
        generator_only: If True, stop after Stage 1 and skip the judge.

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
        "meta_verdicts": None,
        "overall_notes": None,
        # Scores
        "output_relevancy_score": None,
        "completeness_score": None,
    }

    async with semaphore:
        try:
            logger.info("Row %d: processing (route=%s)", row_id, row["route"])

            # ---- Stage 1: Generate Rubrics ----
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
            result["rubrics"] = rubrics
            result["rubrics_reasoning"] = gen_parsed.get("reasoning", "")

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

            if any(score is None for score in metric_scores.values()):
                logger.warning("Row %d: Judge response missing one or more metric scores", row_id)
                result["status"] = "JUDGE_INCOMPLETE"
                return result

            # Process meta-verdicts
            meta_verdicts = judge_parsed.get("meta_verdicts", [])
            result["meta_verdicts"] = meta_verdicts

            logger.info(
                "Row %d: done — Output Relevancy=%s, Completeness=%s",
                row_id,
                result["output_relevancy_score"],
                result["completeness_score"],
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
            # CSV — scalar columns + structured fields as JSON strings
            self._csv_writer.writerow({
                "row_id": result["row_id"],
                "route": result["route"],
                "status": result["status"],
                "timestamp": result["timestamp"],
                "output_relevancy_score": result["output_relevancy_score"],
                "completeness_score": result["completeness_score"],
                "overall_notes": result["overall_notes"],
                "rubrics_json": json.dumps(result["rubrics"], ensure_ascii=False)
                    if result["rubrics"] is not None else "",
                "verdicts_json": json.dumps(result["verdicts"], ensure_ascii=False)
                    if result["verdicts"] is not None else "",
                "meta_verdicts_json": json.dumps(result["meta_verdicts"], ensure_ascii=False)
                    if result["meta_verdicts"] is not None else "",
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
    logger.info("=" * 80)

    # Load data
    rows = load_input_data(input_path, routes=routes, limit=limit)
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
    _print_summary(actual_results, elapsed, writer, enriched_paths)


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
    eval_scalar_cols = [
        "eval_status", "eval_timestamp",
        "eval_output_relevancy_score", "eval_completeness_score", "eval_overall_notes",
        "eval_rubrics_json", "eval_verdicts_json", "eval_meta_verdicts_json",
        "eval_generator_raw", "eval_judge_raw",
    ]
    for col in eval_scalar_cols:
        df[col] = None

    def _to_json_str(val: Any) -> str | None:
        if val is None:
            return None
        if isinstance(val, str):
            return val
        return json.dumps(val, ensure_ascii=False)

    for idx, row in df.iterrows():
        ev = eval_by_id[int(row["row_id"])]
        df.at[idx, "eval_status"] = ev.get("status")
        df.at[idx, "eval_timestamp"] = ev.get("timestamp")
        df.at[idx, "eval_output_relevancy_score"] = ev.get("output_relevancy_score")
        df.at[idx, "eval_completeness_score"] = ev.get("completeness_score")
        df.at[idx, "eval_overall_notes"] = ev.get("overall_notes")
        df.at[idx, "eval_rubrics_json"] = _to_json_str(ev.get("rubrics"))
        df.at[idx, "eval_verdicts_json"] = _to_json_str(ev.get("verdicts"))
        df.at[idx, "eval_meta_verdicts_json"] = _to_json_str(ev.get("meta_verdicts"))
        df.at[idx, "eval_generator_raw"] = ev.get("generator_raw_response")
        df.at[idx, "eval_judge_raw"] = ev.get("judge_raw_response")

    logger.info("Enriched output: %d rows merged", len(df))

    enriched_csv = output_dir / f"{run_name}_all_results_enriched.csv"
    df.to_csv(enriched_csv, index=False, encoding="utf-8")
    logger.info("Enriched CSV written: %s", enriched_csv)

    enriched_jsonl = output_dir / f"{run_name}_all_results_enriched.jsonl"
    with open(enriched_jsonl, "w", encoding="utf-8") as f:
        for _, row in df.iterrows():
            rec = row.to_dict()
            # Parse the JSON string columns back into objects for the JSONL
            for col in ("eval_rubrics_json", "eval_verdicts_json", "eval_meta_verdicts_json"):
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
) -> None:
    """Print evaluation summary to console."""
    print("\n" + "=" * 80)
    print("EVALUATION SUMMARY")
    print("=" * 80)
    print(f"Total rows evaluated: {len(results)}")
    print(f"Elapsed time: {elapsed:.1f}s")
    print()

    # Status breakdown
    from collections import Counter  # noqa: E402

    status_counts = Counter(r["status"] for r in results)
    print("Status breakdown:")
    for status, count in status_counts.most_common():
        print(f"  {status:30} {count:4}")

    # Score summary by route
    ok_results = [r for r in results if r["status"] == "OK"]
    if ok_results:
        print()
        print("Scores by route:")
        route_scores: dict[str, dict[str, list[int]]] = {}
        for r in ok_results:
            route_bucket = route_scores.setdefault(
                r["route"],
                {"output_relevancy_score": [], "completeness_score": []},
            )
            if r.get("output_relevancy_score") is not None:
                route_bucket["output_relevancy_score"].append(r["output_relevancy_score"])
            if r.get("completeness_score") is not None:
                route_bucket["completeness_score"].append(r["completeness_score"])

        for route in sorted(route_scores):
            relevancy_scores = route_scores[route]["output_relevancy_score"]
            completeness_scores = route_scores[route]["completeness_score"]
            relevancy_avg = sum(relevancy_scores) / len(relevancy_scores) if relevancy_scores else float("nan")
            completeness_avg = sum(completeness_scores) / len(completeness_scores) if completeness_scores else float("nan")
            print(
                f"  {route:20} Output Relevancy={relevancy_avg:.3f}  "
                f"Completeness={completeness_avg:.3f}  n={len(relevancy_scores)}"
            )

    print()
    print(f"Outputs saved:")
    print(f"  Summary  : {writer.results_path}")
    print(f"  Details  : {writer.details_path}")
    if enriched_paths:
        print(f"  Enriched CSV  : {enriched_paths[0]}")
        print(f"  Enriched JSONL: {enriched_paths[1]}")
    print("=" * 80)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    import argparse

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
        "--run-name",
        default=None,
        help="Name for output files (default: eval_YYYYMMDD_HHMMSS)",
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
        )
    )


if __name__ == "__main__":
    main()
