"""Build ``returned_final_text`` for every row in final_data/.

For RESPOND and RESEARCH routes the value is simply the ``response`` field.
For REVISE_SIMPLE and REVISE_RESEARCH routes the value is the original
``input`` text with **all** suggestions applied (as if the user accepted
every suggestion).

Outputs
-------
- final_data/all_results_with_final_text.csv   — CSV with new column
- final_data/all_results_with_final_text.jsonl  — JSONL with new field

Usage
-----
    python scripts/build_final_text.py
"""

import csv
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
FINAL_DATA = BASE_DIR / "final_data"
INPUT_CSV = FINAL_DATA / "all_results.csv"
INPUT_JSONL = FINAL_DATA / "all_results.jsonl"
OUTPUT_CSV = FINAL_DATA / "all_results_with_final_text.csv"
OUTPUT_JSONL = FINAL_DATA / "all_results_with_final_text.jsonl"

REVISE_ROUTES = {"REVISE_SIMPLE", "REVISE_RESEARCH"}


# ---------------------------------------------------------------------------
# Core logic — apply all suggestions to original text
# ---------------------------------------------------------------------------

def _resolve_overlaps(ops: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Remove suggestions whose span is fully contained within another suggestion.

    All ``char_start``/``char_end`` values refer to the original input text.
    When two suggestions overlap, the outer (larger) one takes priority because
    applying it replaces the full region including the inner span.  Partial
    overlaps are preserved as-is; they are uncommon and the reverse-order
    application handles them as well as possible.
    """
    if len(ops) <= 1:
        return ops

    # Sort by start asc, then by span size desc so the larger span comes first
    # when two suggestions share the same start position.
    sorted_ops = sorted(
        ops,
        key=lambda o: (o["char_start"], -(o["char_end"] - o["char_start"])),
    )

    kept: List[Dict[str, Any]] = []
    for op in sorted_ops:
        is_inner = any(
            outer["char_start"] <= op["char_start"] and outer["char_end"] >= op["char_end"]
            for outer in kept
        )
        if not is_inner:
            kept.append(op)

    return kept


def apply_all_suggestions(original_text: str, suggestions: List[Dict[str, Any]]) -> str:
    """Return the text that results from accepting every suggestion.

    All ``char_start``/``char_end`` values are relative to the **original**
    input text.  Suggestions are applied in reverse ``char_start`` order so
    that earlier character offsets remain valid after later ones are applied.

    Overlapping (nested) suggestions are resolved first: the inner/smaller
    suggestion is dropped in favour of the outer/larger one.

    Two cases:
    1. **Replacement** — ``original_text`` in the suggestion is non-empty:
       replace ``original_text[char_start:char_end]`` with ``transformed_text``.
    2. **Insertion** — ``original_text`` is empty and ``char_start == char_end``:
       insert ``transformed_text`` at that position.
    """
    if not suggestions:
        return original_text

    # Build a list of operations with validated positions
    ops: List[Dict[str, Any]] = []
    for sug in suggestions:
        char_start = sug.get("char_start")
        char_end = sug.get("char_end")
        transformed = sug.get("transformed_text", "")

        # Skip suggestions without usable positions
        if char_start is None or char_end is None:
            continue
        if char_start < 0 or char_end < 0:
            continue

        ops.append({
            "char_start": char_start,
            "char_end": char_end,
            "transformed_text": transformed,
        })

    # Drop inner suggestions that are fully covered by an outer suggestion.
    ops = _resolve_overlaps(ops)

    # Sort by char_start descending so we apply from the end of the document
    # backwards — earlier character offsets stay valid across iterations.
    ops.sort(key=lambda o: o["char_start"], reverse=True)

    result = original_text
    for op in ops:
        start = op["char_start"]
        end = op["char_end"]
        result = result[:start] + op["transformed_text"] + result[end:]

    return result


def get_returned_final_text(row: Dict[str, Any]) -> str:
    """Compute ``returned_final_text`` for a single row."""
    route = row.get("route_orch", "")

    if route in REVISE_ROUTES:
        # Parse suggestions from the output JSON
        output_raw = row.get("output", "")
        if isinstance(output_raw, str):
            try:
                output = json.loads(output_raw)
            except (json.JSONDecodeError, TypeError):
                return ""
        else:
            output = output_raw

        suggestions = output.get("suggestions", [])
        input_text = row.get("input", "")

        if not suggestions:
            # No suggestions → the text stays unchanged
            return input_text

        return apply_all_suggestions(input_text, suggestions)

    # RESPOND, RESEARCH, or anything else → response as-is
    return row.get("response", "")


# ---------------------------------------------------------------------------
# Processing
# ---------------------------------------------------------------------------

def process() -> None:
    # Read all rows from JSONL (richer / safer than CSV for nested data)
    rows: List[Dict[str, Any]] = []
    with open(INPUT_JSONL, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))

    print(f"Read {len(rows)} rows from {INPUT_JSONL.name}")

    # Compute new field
    stats = {"RESPOND": 0, "RESEARCH": 0, "REVISE_SIMPLE": 0,
             "REVISE_RESEARCH": 0, "OTHER": 0}
    for row in rows:
        row["returned_final_text"] = get_returned_final_text(row)
        route = row.get("route_orch", "OTHER")
        stats[route] = stats.get(route, 0) + 1

    print("Route distribution:", {k: v for k, v in stats.items() if v})

    # --- Write JSONL ---
    with open(OUTPUT_JSONL, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"Wrote {OUTPUT_JSONL.name}")

    # --- Write CSV ---
    # Read original CSV headers, append new column
    with open(INPUT_CSV, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        csv_headers = list(reader.fieldnames or [])

    if "returned_final_text" not in csv_headers:
        csv_headers.append("returned_final_text")

    with open(OUTPUT_CSV, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=csv_headers, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    print(f"Wrote {OUTPUT_CSV.name}")

    # Quick sanity check
    revise_count = 0
    revise_changed = 0
    overlap_rows = 0
    for row in rows:
        if row.get("route_orch") in REVISE_ROUTES:
            revise_count += 1
            if row["returned_final_text"] != row.get("input", ""):
                revise_changed += 1
            # Count rows where overlaps were resolved
            output_raw = row.get("output", "")
            output = json.loads(output_raw) if isinstance(output_raw, str) else output_raw
            sugs = output.get("suggestions", [])
            ops = [s for s in sugs if s.get("char_start") is not None and s.get("char_start", -1) >= 0]
            if len(ops) != len(_resolve_overlaps(ops)):
                overlap_rows += 1
    print(f"\nREVISE rows: {revise_count} total, {revise_changed} had text changes from suggestions")
    if overlap_rows:
        print(f"Overlap-resolved rows: {overlap_rows} (inner nested suggestions dropped in favour of outer)")


if __name__ == "__main__":
    process()
