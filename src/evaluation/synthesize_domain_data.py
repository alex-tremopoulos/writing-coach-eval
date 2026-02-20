"""Synthesize multi-domain evaluation data for the Writing Coach routing evaluator.

Uses Azure OpenAI (GPT-5.1 chat) to generate:
  - 84 domain-adapted rows (28 CS rows × 3 domains: Physics, Biochemistry, Humanities & Social Sciences)
  - 72 generic rows (12 RESPOND + 24 REVISE_RESEARCH + 36 REVISE_SIMPLE)
  → Total: 156 new rows.  Merged with the original 52 → 208 rows (52 per route).

Usage:
    python src/evaluation/synthesize_domain_data.py

Environment variables (or .env file in repo root):
    AZURE_OPENAI_ENDPOINT   – e.g. https://my-resource.openai.azure.com/
    AZURE_OPENAI_API_KEY    – secret key
    AZURE_OPENAI_API_VERSION – e.g. 2025-01-01-preview
    AZURE_OPENAI_DEPLOYMENT  – deployment name for GPT-5.1 chat
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

load_dotenv()  # reads .env from cwd (repo root)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[2]
ORIGINAL_CSV = REPO_ROOT / "data" / "data_routes_original.csv"
OUTPUT_EXPANDED = REPO_ROOT / "data" / "data_routes_expanded.csv"
OUTPUT_SYNTHETIC = REPO_ROOT / "data" / "data_routes_synthetic.csv"
CHECKPOINT_FILE = REPO_ROOT / "data" / ".synthesis_checkpoint.json"

# CS-oriented row indices (0-based) in the original CSV
CS_INDICES: list[int] = list(range(0, 22)) + list(range(26, 31)) + [40]

# Target domains for domain-adapted rows
DOMAINS: list[str] = ["Physics", "Biochemistry", "Humanities and Social Sciences"]

# Route descriptions (derived from orchestrator_prompts.py) used in prompts
ROUTE_DESCRIPTIONS: dict[str, str] = {
    "RESEARCH": (
        "The user wants the assistant to search for academic papers, explore "
        "literature, or discover what is known on a topic.  Typical trigger "
        "verbs: find, search, retrieve, list, explore, verify, validate."
    ),
    "RESPOND": (
        "The user wants the assistant to summarise, compare, analyse, or "
        "discuss information already provided in the conversation.  They are "
        "NOT requesting new research or text edits.  Typical trigger verbs: "
        "summarise, compare, explain, what do you think, is this reasonable."
    ),
    "REVISE_RESEARCH": (
        "The user asks to improve, strengthen, or enhance existing text AND "
        "the revision requires finding new external evidence.  Typical "
        "trigger verbs: strengthen with evidence, add citations, add sources, "
        "provide more evidence, add a definition."
    ),
    "REVISE_SIMPLE": (
        "The user asks for mechanical text changes that do NOT require new "
        "research: grammar, spelling, paraphrasing, reformatting, tone "
        "changes, making text concise, etc.  Typical trigger verbs: fix "
        "grammar, simplify, shorten, rephrase, reformat, paraphrase."
    ),
}

# How many generic rows to create per route (to reach 52 per route total)
GENERIC_ROUTE_COUNTS: dict[str, int] = {
    # RESEARCH already at 52 after domain expansion → 0 needed
    "RESPOND": 12,
    "REVISE_RESEARCH": 24,
    "REVISE_SIMPLE": 36,
}

# Azure OpenAI settings
AZURE_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT", "")
AZURE_API_KEY = os.getenv("AZURE_OPENAI_API_KEY", "")
AZURE_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION", "2025-01-01-preview")
AZURE_DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-51-chat")

# Delay between API calls (seconds)
API_CALL_DELAY = 1.5
# Max retries on transient failures
MAX_RETRIES = 5

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
# Prompt templates
# ---------------------------------------------------------------------------

DOMAIN_ADAPTATION_SYSTEM = """\
You are an expert academic writer who can produce realistic scholarly text \
in any scientific or humanities domain.  You will be given a query and an \
input text from a computer-science research-writing context, along with the \
"route" label that categorises the user intent.

Your task:  produce an EQUIVALENT query and input in the domain "{domain}".

Rules:
1. The new query MUST preserve the same kind of user intent so that the \
   route label "{route}" still applies.  Keep the same trigger verbs and \
   instruction style — only change the *topic*.
   Route meaning: {route_desc}
2. The new query should be of similar length and specificity as the original.
3. The new input must be realistic scholarly text of roughly the same length \
   and paragraph structure as the original — but about a plausible topic in \
   {domain}.
4. Do NOT include citation markers like [0], [1], [2], etc.  Instead, \
   reference authors or works inline naturally (e.g. "Smith et al. showed…").
5. Use real, plausible {domain} terminology and topics.
6. Return ONLY valid JSON: {{"query": "...", "input": "..."}}
"""

DOMAIN_ADAPTATION_USER = """\
Original route: {route}
Original query: {query}
Original input (first 2000 chars):
{input_text}

Now produce the equivalent in the domain "{domain}".  Return ONLY JSON.
"""

GENERIC_GENERATION_SYSTEM = """\
You are an expert writer who produces realistic text across many everyday \
and scientific domains.  You will generate {count} new evaluation examples \
for a writing-assistant routing benchmark.

Each example consists of a "query" (what the user says to the assistant) and \
an "input" (the text the user is currently working on).

Route for ALL examples: {route}
Route meaning: {route_desc}

Rules:
1. Every query MUST unambiguously trigger the route "{route}" according to \
   these routing rules.  Use the appropriate trigger verbs and phrasing.
2. Topics must be diverse — spread across different domains such as cooking, \
   travel, sports, medicine, environmental science, architecture, psychology, \
   law, marine biology, nutrition, climate policy, music, personal finance, \
   engineering, agriculture, etc.  Topics CAN be scientific, but must NOT \
   be specifically about Computer Science, Physics, Biochemistry, or \
   Humanities & Social Sciences.
3. Do NOT include citation markers like [0], [1], [2].
4. Vary input length: some short (50-200 chars), some medium (200-600 chars), \
   some long (600+ chars).
5. Queries should be similar in style and brevity to these examples:
{examples}
6. Return ONLY a valid JSON array of objects: \
   [{{"query": "...", "input": "..."}}, ...]
"""


# ---------------------------------------------------------------------------
# LLM call helper
# ---------------------------------------------------------------------------


def call_llm(
    client: AzureOpenAI, system: str, user: str, *, temperature: float = 0.9
) -> str:
    """Call Azure OpenAI with retry logic.  Returns raw content string."""
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = client.chat.completions.create(
                model=AZURE_DEPLOYMENT,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                temperature=temperature,
                max_tokens=4096,
            )
            return resp.choices[0].message.content.strip()
        except Exception as exc:
            wait = 2**attempt
            log.warning(
                "API call failed (attempt %d/%d): %s — retrying in %ds",
                attempt,
                MAX_RETRIES,
                exc,
                wait,
            )
            time.sleep(wait)
    raise RuntimeError(f"LLM call failed after {MAX_RETRIES} retries")


def parse_json(raw: str) -> dict | list:
    """Extract JSON from an LLM response (handles markdown fences)."""
    # Strip markdown code fences if present
    cleaned = re.sub(r"^```(?:json)?\s*", "", raw, flags=re.MULTILINE)
    cleaned = re.sub(r"```\s*$", "", cleaned, flags=re.MULTILINE)
    return json.loads(cleaned.strip())


def strip_citation_markers(text: str) -> str:
    """Remove citation markers like [0], [1], [2] etc."""
    return re.sub(r"\[\d+\]", "", text).strip()


# ---------------------------------------------------------------------------
# Checkpoint helpers (save progress so we can resume on failure)
# ---------------------------------------------------------------------------


def load_checkpoint() -> dict:
    if CHECKPOINT_FILE.exists():
        return json.loads(CHECKPOINT_FILE.read_text(encoding="utf-8"))
    return {"domain_rows": [], "generic_rows": []}


def save_checkpoint(data: dict) -> None:
    CHECKPOINT_FILE.write_text(
        json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"
    )


# ---------------------------------------------------------------------------
# Domain adaptation (84 rows)
# ---------------------------------------------------------------------------


def generate_domain_rows(client: AzureOpenAI, df: pd.DataFrame) -> list[dict]:
    """Generate domain-adapted rows for every CS row × 3 domains."""
    checkpoint = load_checkpoint()
    existing = checkpoint.get("domain_rows", [])

    # Build a set of already-done (orig_idx, domain) pairs
    done = {(r["orig_idx"], r["domain"]) for r in existing}
    rows = list(existing)

    cs_rows = df.iloc[CS_INDICES]
    total = len(CS_INDICES) * len(DOMAINS)
    completed = len(done)

    for idx in CS_INDICES:
        row = df.iloc[idx]
        route = row["route"]
        query = row["query"]
        input_text = str(row["input"])[:2000]

        for domain in DOMAINS:
            if (idx, domain) in done:
                continue

            completed += 1
            log.info(
                "[Domain %d/%d]  idx=%d  domain=%s  route=%s",
                completed,
                total,
                idx,
                domain,
                route,
            )

            sys_prompt = DOMAIN_ADAPTATION_SYSTEM.format(
                domain=domain,
                route=route,
                route_desc=ROUTE_DESCRIPTIONS[route],
            )
            usr_prompt = DOMAIN_ADAPTATION_USER.format(
                route=route,
                query=query,
                input_text=input_text,
                domain=domain,
            )

            raw = call_llm(client, sys_prompt, usr_prompt)
            try:
                parsed = parse_json(raw)
                new_query = strip_citation_markers(parsed["query"])
                new_input = strip_citation_markers(parsed["input"])
            except (json.JSONDecodeError, KeyError) as exc:
                log.error(
                    "Failed to parse response for idx=%d domain=%s: %s\nRaw:\n%s",
                    idx,
                    domain,
                    exc,
                    raw[:500],
                )
                # Use raw text as fallback
                new_query = f"[PARSE_ERROR] idx={idx} domain={domain}"
                new_input = raw[:1000]

            # Map domain to dataset label
            domain_label = domain.replace(" ", "").replace("&", "And")
            record = {
                "target": "TBD",
                "route": route,
                "input": new_input,
                "dataset": f"Synthetic-{domain_label}",
                "query": new_query,
                "orig_idx": idx,
                "domain": domain,
            }
            rows.append(record)
            done.add((idx, domain))

            # Checkpoint after every row
            checkpoint["domain_rows"] = rows
            save_checkpoint(checkpoint)

            time.sleep(API_CALL_DELAY)

    return rows


# ---------------------------------------------------------------------------
# Generic row generation (72 rows)
# ---------------------------------------------------------------------------


def _get_generic_exemplars(df: pd.DataFrame, route: str) -> str:
    """Return formatted few-shot examples for a given route from existing generics."""
    # Generic indices (non-CS rows)
    generic_mask = ~df.index.isin(CS_INDICES)
    subset = df.loc[generic_mask & (df["route"] == route)]
    lines = []
    for _, r in subset.iterrows():
        lines.append(f'  query: "{r["query"]}"')
        inp = str(r["input"])[:200]
        lines.append(f'  input (preview): "{inp}…"')
        lines.append("")
    return "\n".join(lines)


def generate_generic_rows(client: AzureOpenAI, df: pd.DataFrame) -> list[dict]:
    """Generate generic (domain-agnostic) rows to fill each route to 52."""
    checkpoint = load_checkpoint()
    existing = checkpoint.get("generic_rows", [])

    # Count how many we already have per route
    done_counts: dict[str, int] = {}
    for r in existing:
        done_counts[r["route"]] = done_counts.get(r["route"], 0) + 1

    rows = list(existing)

    for route, needed in GENERIC_ROUTE_COUNTS.items():
        already = done_counts.get(route, 0)
        remaining = needed - already
        if remaining <= 0:
            log.info("Generic %s: already have %d/%d — skipping", route, already, needed)
            continue

        log.info("Generic %s: generating %d rows (%d already done)", route, remaining, already)

        exemplars = _get_generic_exemplars(df, route)

        # Generate in sub-batches of at most 12
        batch_size = min(12, remaining)
        generated_so_far = 0

        while generated_so_far < remaining:
            batch_count = min(batch_size, remaining - generated_so_far)
            log.info(
                "  sub-batch: %d rows (total so far: %d/%d)",
                batch_count,
                generated_so_far + already,
                needed,
            )

            sys_prompt = GENERIC_GENERATION_SYSTEM.format(
                count=batch_count,
                route=route,
                route_desc=ROUTE_DESCRIPTIONS[route],
                examples=exemplars,
            )
            usr_prompt = (
                f"Generate exactly {batch_count} diverse examples for route "
                f'"{route}".  Return ONLY a JSON array.'
            )

            raw = call_llm(client, sys_prompt, usr_prompt)
            try:
                parsed = parse_json(raw)
                if not isinstance(parsed, list):
                    parsed = [parsed]
            except json.JSONDecodeError as exc:
                log.error("Failed to parse generic batch: %s\nRaw:\n%s", exc, raw[:500])
                time.sleep(API_CALL_DELAY)
                continue  # retry this sub-batch implicitly on next iteration

            for item in parsed[:batch_count]:
                record = {
                    "target": "TBD",
                    "route": route,
                    "input": strip_citation_markers(item.get("input", "")),
                    "dataset": "Synthetic-Generic",
                    "query": strip_citation_markers(item.get("query", "")),
                }
                rows.append(record)
                generated_so_far += 1

            checkpoint["generic_rows"] = rows
            save_checkpoint(checkpoint)

            time.sleep(API_CALL_DELAY)

    return rows


# ---------------------------------------------------------------------------
# Post-processing validation
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
        if len(r.get("input", "")) < 30:
            log.warning("%s: very short input (%d chars)", tag, len(r.get("input", "")))
            issues += 1
        # Check for leftover citation markers
        for field in ("query", "input"):
            if re.search(r"\[\d+\]", r.get(field, "")):
                log.warning("%s: citation marker found in %s — stripping", tag, field)
                r[field] = strip_citation_markers(r[field])
    log.info("%s validation: %d issues in %d rows", label, issues, len(rows))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def apply_baseline_patches(df: pd.DataFrame) -> pd.DataFrame:
    """Apply the same row modifications as modify_public_datasets.py."""
    df.loc[39, "target"] = (
        "In distributed systems, data consistency is often difficult to "
        "maintain because nodes communicate over an unreliable network, and "
        "messages can be delayed or lost. Many algorithms attempt to address "
        "this by introducing consensus protocols such as Paxos or Raft, which "
        "ensure that all replicas agree on the same value even if some of them "
        "fail. However, implementing these protocols is complex and requires "
        "careful reasoning about concurrency, since race conditions and "
        "deadlocks can easily occur. Furthermore, scaling the system "
        "horizontally introduces additional challenges, such as partition "
        "tolerance and load balancing, that are not trivial to address in "
        "practice."
    )
    df.loc[39, "input"] = (
        "In distributed systems the consistency of data are often difficult "
        "to maintain because nodes communicates over unreliable network and "
        "messages can delayed or lost. Many algorithm tries to solve this by "
        "introducing consensus protocols such as Paxos or Raft, which ensures "
        "that all replicas agree on a same value even if some of them fails. "
        "However, the implementation of these protocols is complex and "
        "requires careful reasoning about concurrency, since race condition "
        "and deadlocks can easily occurs. Furthermore, scaling the system "
        "horizontally introduce additional challenges, like partition "
        "tolerance and load balancing, that is not trivial to address in "
        "practice."
    )
    df.loc[39, "query"] = "Fix fluency in this text"
    df.loc[39, "dataset"] = "ChatGPT5.2"

    df.loc[44, "query"] = (
        "Are there any other existing methods for leveraging information "
        "in knowledge bases for task-specific language model fine-tuning?"
    )
    return df


def main() -> None:
    log.info("Loading original CSV from %s", ORIGINAL_CSV)
    df = pd.read_csv(ORIGINAL_CSV)
    df = apply_baseline_patches(df)
    log.info("Original dataset: %d rows", len(df))
    log.info("Route distribution:\n%s", df["route"].value_counts().to_string())

    client = get_client()

    # --- Phase 1: Domain-adapted rows (84) ---
    log.info("=" * 60)
    log.info("PHASE 1: Domain-adapted rows (28 CS × 3 domains = 84)")
    log.info("=" * 60)
    domain_rows = generate_domain_rows(client, df)
    validate_rows(domain_rows, "domain")

    # --- Phase 2: Generic rows (72) ---
    log.info("=" * 60)
    log.info("PHASE 2: Generic rows (12 RESPOND + 24 REVISE_RESEARCH + 36 REVISE_SIMPLE = 72)")
    log.info("=" * 60)
    generic_rows = generate_generic_rows(client, df)
    validate_rows(generic_rows, "generic")

    # --- Phase 3: Merge and save ---
    log.info("=" * 60)
    log.info("PHASE 3: Merge and save")
    log.info("=" * 60)

    # Drop helper columns before merging
    for r in domain_rows:
        r.pop("orig_idx", None)
        r.pop("domain", None)

    synthetic_df = pd.DataFrame(domain_rows + generic_rows)
    # Ensure column order matches original
    col_order = ["target", "route", "input", "dataset", "query"]
    synthetic_df = synthetic_df[col_order]

    merged_df = pd.concat([df, synthetic_df], ignore_index=True)

    # Final route count assertion
    route_counts = merged_df["route"].value_counts()
    log.info("Final route distribution:\n%s", route_counts.to_string())
    log.info("Total rows: %d", len(merged_df))

    for route in ["RESEARCH", "RESPOND", "REVISE_RESEARCH", "REVISE_SIMPLE"]:
        count = route_counts.get(route, 0)
        if count != 52:
            log.warning("Route %s has %d rows (expected 52)", route, count)
        else:
            log.info("Route %s: 52 ✓", route)

    # Save outputs
    merged_df.to_csv(OUTPUT_EXPANDED, index=False)
    log.info("Saved expanded dataset to %s", OUTPUT_EXPANDED)

    synthetic_df.to_csv(OUTPUT_SYNTHETIC, index=False)
    log.info("Saved synthetic-only dataset to %s", OUTPUT_SYNTHETIC)

    # Clean up checkpoint
    if CHECKPOINT_FILE.exists():
        CHECKPOINT_FILE.unlink()
        log.info("Removed checkpoint file")

    log.info("Done!  %d original + %d synthetic = %d total rows",
             len(df), len(synthetic_df), len(merged_df))


if __name__ == "__main__":
    main()
