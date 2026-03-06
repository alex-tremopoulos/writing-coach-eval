"""Synthesize 10 additional RESPOND-route evaluation cases.

Reads the existing eval results from data_outputs/whole_input/respond_only/,
filters rows where route == "RESPOND", and uses those as few-shot exemplars
to generate 10 new (query, input) pairs that should reliably trigger RESPOND.

The prompts, constraints, and LLM config are taken directly from
synthesize_domain_data.py to ensure consistency.

Usage:
    python src/dataset_handling/add_respond_cases.py

Environment variables (or .env file in repo root):
    AZURE_OPENAI_ENDPOINT    – e.g. https://my-resource.openai.azure.com/
    AZURE_OPENAI_API_KEY     – secret key
    AZURE_OPENAI_API_VERSION – e.g. 2025-01-01-preview
    AZURE_OPENAI_DEPLOYMENT  – deployment name
"""

from __future__ import annotations

import json
import logging
import os
import re
import sys
import time
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from openai import AzureOpenAI

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

start_time = time.time()

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[2]

INPUT_CSV = REPO_ROOT / "data_outputs" / "whole_input" / "respond_only" / "data_routes_expanded_RESPOND_results.csv"
OUTPUT_CSV = REPO_ROOT / "data_outputs" / "whole_input" / "respond_only" / "data_routes_expanded_RESPOND_extra.csv"
OUTPUT_JSONL = REPO_ROOT / "data_outputs" / "whole_input" / "respond_only" / "data_routes_expanded_RESPOND_extra.jsonl"
CHECKPOINT_FILE = REPO_ROOT / "data" / ".respond_extra_checkpoint.json"

N_TO_GENERATE = 10

# Azure OpenAI settings — identical to synthesize_domain_data.py
AZURE_ENDPOINT   = os.getenv("AZURE_OPENAI_ENDPOINT", "")
AZURE_API_KEY    = os.getenv("AZURE_OPENAI_API_KEY", "")
AZURE_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION", "2025-01-01-preview")
AZURE_DEPLOYMENT  = os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-51-chat")

API_CALL_DELAY = 1.5
MAX_RETRIES    = 5

INPUT_PRICE_PER_1M  = float(os.getenv("INPUT_PRICE_PER_1M",  "1.25"))
OUTPUT_PRICE_PER_1M = float(os.getenv("OUTPUT_PRICE_PER_1M", "10.00"))
MAX_BUDGET_USD      = float(os.getenv("MAX_BUDGET_USD",       "10.0"))
DRY_RUN             = os.getenv("DRY_RUN", "false").lower() == "true"

_total_input_tokens:  int = 0
_total_output_tokens: int = 0

# ---------------------------------------------------------------------------
# Route description — copied verbatim from synthesize_domain_data.py
# ---------------------------------------------------------------------------

RESPOND_ROUTE_DESC = (
    "The user wants to summarise, compare, or analyse information already provided in the conversation."
    "The user asks to reformat or reorganise findings from previous turns."
    "The user references specific earlier turns or findings without requesting new research or text edits."
    "The user asks general questions about their document or approach."
    "The user makes meta-commentary ('What do you think?', 'Is this approach reasonable?')."
    "The user asks about the assistant's capabilities or how to use it."
    "The message is a follow-up question about previously found papers that doesn't require new search."
    "Examples: 'Summarise what you found', 'Compare those papers', 'What do you think?', 'Is this approach reasonable?'"
    "Trigger verbs: summarise, compare, explain, what did you mean, create a table from, organise, recap, what do you think, is this, how should I"
)

# ---------------------------------------------------------------------------
# Prompt template — copied verbatim from synthesize_domain_data.py
# ---------------------------------------------------------------------------

GENERIC_GENERATION_SYSTEM = """\
You are an expert writer who produces realistic text across many everyday \
and scientific domains.  You will generate {count} new evaluation examples \
for a writing-assistant routing benchmark.

Each example consists of a "query" (what the user says to the assistant) and \
an "input" (the text the user is currently working on).

Route for ALL examples: {route}
Route meaning: {route_desc}

*** HARD REQUIREMENT — INPUT LENGTH ***
Every "input" field MUST contain at least 700 characters.  Most inputs \
should be 1000–1800 characters.  Distribute the {count} examples roughly \
equally: ~1/3 at 700–1000 chars, ~1/3 at 1000–1500 chars, ~1/3 at 1500+ \
chars.  Short inputs (under 700 chars) are INVALID and should be rejected. \
The few-shot examples below may be short — ignore their length if it is short \
and always produce substantially longer texts.

Rules:
1. Every query MUST unambiguously trigger the route "{route}" according to \
   these routing rules.  Use the appropriate trigger verbs and phrasing.
2. Topics must be diverse — spread across different domains such as medicine, \
   environmental science, architecture, psychology, law, marine biology, \
   nutrition, climate policy, personal finance, engineering, agriculture, etc. \
   Topics CAN be scientific, but must NOT be specifically about Computer \
   Science, Physics, Biochemistry, or Humanities & Social Sciences.
3. Do NOT include citation markers like [0], [1], [2].
4. If the route involves revising or editing text (e.g. REVISE_SIMPLE, \
   REVISE_RESEARCH), the input text MUST already contain the specific \
   problems the query asks to fix.  Examples: \
   - query asks to fix grammar → input must contain grammatical errors; \
   - query asks to simplify → input must use unnecessarily complex language; \
   - query asks to shorten → input must be verbose and repetitive; \
   - query asks to fix tone → input must have an inappropriate tone; \
   - query asks to convert to bullets → input must be dense prose. \
   A query that asks to "fix X" is incoherent if the input has no X to fix — \
   always ensure the intent is grounded in the input.
5. Before finalising each example, mentally count the characters in the \
   "input" field.  If it is under 700 characters, expand it before returning.
6. Queries should be similar in style and brevity to these examples:
{examples}
7. Return ONLY a valid JSON array of objects: \
   [{{"query": "...", "input": "..."}}, ...]
"""

# ---------------------------------------------------------------------------
# Azure OpenAI client
# ---------------------------------------------------------------------------


def get_client() -> AzureOpenAI:
    if not AZURE_ENDPOINT or not AZURE_API_KEY:
        log.error(
            "Set AZURE_OPENAI_ENDPOINT and AZURE_OPENAI_API_KEY env vars "
            "(or in a .env file)."
        )
        sys.exit(1)
    return AzureOpenAI(
        azure_endpoint=AZURE_ENDPOINT,
        api_key=AZURE_API_KEY,
        api_version=AZURE_API_VERSION,
    )


# ---------------------------------------------------------------------------
# LLM helpers — copied verbatim from synthesize_domain_data.py
# ---------------------------------------------------------------------------


def call_llm(
    client: AzureOpenAI, system: str, user: str, *, temperature: float = 0.9
) -> str:
    """Call Azure OpenAI with retry logic. Returns raw content string."""
    global _total_input_tokens, _total_output_tokens

    if DRY_RUN:
        log.info("[DRY RUN] Skipping API call")
        return '[{"query": "DRY_RUN_QUERY", "input": "DRY_RUN_INPUT ' + 'x' * 700 + '"}]'

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = client.chat.completions.create(
                model=AZURE_DEPLOYMENT,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user",   "content": user},
                ],
                temperature=temperature,
                max_tokens=4096,
            )
            usage = resp.usage
            if usage:
                _total_input_tokens  += usage.prompt_tokens
                _total_output_tokens += usage.completion_tokens
                cost = (
                    _total_input_tokens  / 1_000_000 * INPUT_PRICE_PER_1M
                    + _total_output_tokens / 1_000_000 * OUTPUT_PRICE_PER_1M
                )
                log.info(
                    "Cumulative cost: $%.4f  (in=%d, out=%d tokens)",
                    cost, _total_input_tokens, _total_output_tokens,
                )
                if cost > MAX_BUDGET_USD:
                    raise RuntimeError(
                        f"Budget cap of ${MAX_BUDGET_USD:.2f} exceeded "
                        f"(current: ${cost:.4f}). "
                        "Raise MAX_BUDGET_USD env var to continue."
                    )
            return resp.choices[0].message.content.strip()

        except RuntimeError:
            raise
        except Exception as exc:
            wait = 2 ** attempt
            log.warning(
                "API call failed (attempt %d/%d): %s — retrying in %ds",
                attempt, MAX_RETRIES, exc, wait,
            )
            time.sleep(wait)
    raise RuntimeError(f"LLM call failed after {MAX_RETRIES} retries")


def parse_json(raw: str) -> dict | list:
    """Extract JSON from an LLM response (handles markdown fences)."""
    cleaned = re.sub(r"^```(?:json)?\s*", "", raw, flags=re.MULTILINE)
    cleaned = re.sub(r"```\s*$", "", cleaned, flags=re.MULTILINE)
    return json.loads(cleaned.strip())


def strip_citation_markers(text: str) -> str:
    """Remove citation markers like [0], [1], [2] etc."""
    return re.sub(r"\[\d+\]", "", text).strip()


# ---------------------------------------------------------------------------
# Checkpoint helpers — same pattern as synthesize_domain_data.py
# ---------------------------------------------------------------------------


def load_checkpoint() -> list[dict]:
    if CHECKPOINT_FILE.exists():
        return json.loads(CHECKPOINT_FILE.read_text(encoding="utf-8"))
    return []


def save_checkpoint(rows: list[dict]) -> None:
    CHECKPOINT_FILE.write_text(
        json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8"
    )


# ---------------------------------------------------------------------------
# Validation — copied verbatim from synthesize_domain_data.py
# ---------------------------------------------------------------------------


def validate_rows(rows: list[dict], label: str) -> None:
    """Run sanity checks on generated rows."""
    issues = 0
    for i, r in enumerate(rows):
        tag = f"{label}[{i}]"
        if not r.get("query"):
            log.warning("%s: empty query", tag)
            issues += 1
        if not r.get("input"):
            log.warning("%s: empty input", tag)
            issues += 1
        if len(r.get("input", "")) < 100:
            log.warning("%s: very short input (%d chars)", tag, len(r.get("input", "")))
            issues += 1
        for field in ("query", "input"):
            if re.search(r"\[\d+\]", r.get(field, "")):
                log.warning("%s: citation marker found in %s — stripping", tag, field)
                r[field] = strip_citation_markers(r[field])
    log.info("%s validation: %d issues in %d rows", label, issues, len(rows))


# ---------------------------------------------------------------------------
# Build few-shot exemplar string from actual RESPOND rows
# ---------------------------------------------------------------------------


def build_exemplars(respond_df: pd.DataFrame) -> str:
    """Format RESPOND rows as few-shot examples for the generation prompt."""
    lines = []
    for _, r in respond_df.iterrows():
        lines.append(f'  query: "{r["query"]}"')
        inp = str(r["input"])[:2000]
        lines.append(f'  input (preview): "{inp}…"')
        lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main generation
# ---------------------------------------------------------------------------


def generate_respond_rows(client: AzureOpenAI, exemplars: str) -> list[dict]:
    """Generate N_TO_GENERATE new RESPOND examples using the existing rows as exemplars."""
    existing = load_checkpoint()
    if len(existing) >= N_TO_GENERATE:
        log.info("Checkpoint already has %d rows — skipping generation", len(existing))
        return existing

    remaining = N_TO_GENERATE - len(existing)
    rows = list(existing)

    log.info("Generating %d new RESPOND rows (%d already done)", remaining, len(existing))

    batch_size = min(10, remaining)
    generated = 0
    MAX_BATCH_RETRIES = 3

    while generated < remaining:
        batch_count = min(batch_size, remaining - generated)
        log.info("  sub-batch: %d rows (total so far: %d/%d)", batch_count, generated + len(existing), N_TO_GENERATE)

        sys_prompt = GENERIC_GENERATION_SYSTEM.format(
            count=batch_count,
            route="RESPOND",
            route_desc=RESPOND_ROUTE_DESC,
            examples=exemplars,
        )
        usr_prompt = (
            f"Generate exactly {batch_count} diverse examples for route "
            f'"RESPOND".  Return ONLY a JSON array.'
        )

        parsed: list = []
        for attempt in range(1, MAX_BATCH_RETRIES + 1):
            raw = call_llm(client, sys_prompt, usr_prompt)
            try:
                parsed = parse_json(raw)
                if not isinstance(parsed, list):
                    parsed = [parsed]
                break
            except json.JSONDecodeError as exc:
                log.error(
                    "Failed to parse batch (attempt %d/%d): %s\nRaw:\n%s",
                    attempt, MAX_BATCH_RETRIES, exc, raw[:500],
                )
                if attempt < MAX_BATCH_RETRIES:
                    time.sleep(API_CALL_DELAY)
                else:
                    log.error("Skipping sub-batch after %d failed parse attempts", MAX_BATCH_RETRIES)

        for item in parsed[:batch_count]:
            record = {
                "target": "TBD",
                "route": "RESPOND",
                "input": strip_citation_markers(item.get("input", "")),
                "dataset": "Synthetic-RespondExtra",
                "query": strip_citation_markers(item.get("query", "")),
            }
            rows.append(record)
            generated += 1

        save_checkpoint(rows)
        time.sleep(API_CALL_DELAY)

    return rows


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    log.info("Loading RESPOND eval results from %s", INPUT_CSV)
    df = pd.read_csv(INPUT_CSV)
    log.info("Total rows: %d", len(df))

    respond_df = df[df["route"] == "RESPOND"].copy()
    log.info("RESPOND rows found (correct predictions): %d", len(respond_df))

    if respond_df.empty:
        log.error("No rows with route == 'RESPOND' found. Cannot build exemplars.")
        sys.exit(1)

    exemplars = build_exemplars(respond_df)
    log.info("Built exemplars from %d RESPOND rows", len(respond_df))

    client = get_client()

    new_rows = generate_respond_rows(client, exemplars)
    validate_rows(new_rows, "respond_extra")

    new_df = pd.DataFrame(new_rows)

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    new_df.to_csv(OUTPUT_CSV, index=False)
    log.info("Saved %d new rows to %s", len(new_df), OUTPUT_CSV)

    new_df.to_json(OUTPUT_JSONL, orient="records", lines=True)
    log.info("Saved %d new rows to %s", len(new_df), OUTPUT_JSONL)

    if CHECKPOINT_FILE.exists():
        CHECKPOINT_FILE.unlink()
        log.info("Removed checkpoint file")

    log.info("Done! Generated %d new RESPOND cases in %.1fs", len(new_rows), time.time() - start_time)


if __name__ == "__main__":
    main()
