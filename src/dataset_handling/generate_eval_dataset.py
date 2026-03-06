"""Generate the Writing Coach V2 evaluation dataset.

Creates a CSV with 21 (input_text, query) combinations drawn from the 10
in-scope writing coach examples and 21 queries (7 original UI queries +
2 variants per original = 3 variants per intent family × 7 families).

Excluded examples (per spec):
  - water-scarcity-ar       (Arabic)
  - urban-mobility-fr       (French)
  - special-chars-stress    (Special characters)
  - gene-therapy-latex      (LaTeX)

Appearance counts:
  - 9 examples used twice
  - climate-change used three times
  → 9 × 2 + 3 = 21 rows total

Query coverage:
  - 21 unique queries (3 per intent family, each used exactly once)
  - Each of the 7 intent families contributes exactly 3 queries

Variance constraint:
  - When an example appears more than once, each occurrence uses a query
    from a DIFFERENT intent family (no repeated intent for the same example)

Usage:
    python src/dataset_handling/generate_eval_dataset.py
    python src/dataset_handling/generate_eval_dataset.py --output data_outputs/eval/wc2_eval_21.csv
"""

import argparse
import csv
import importlib.util
import sys
from pathlib import Path

# Make sure the project root is on the path when run directly
ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

# Import writing_coach_examples.py directly to avoid triggering the heavy
# app_src/__init__.py import chain (which requires uplink, search clients, etc.)
_examples_path = ROOT / "src" / "app_src_wc" / "writing_coach_examples.py"
_spec = importlib.util.spec_from_file_location("writing_coach_examples", _examples_path)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
EXAMPLES = _mod.EXAMPLES


# ---------------------------------------------------------------------------
# 21 queries: 7 intent families × 3 variants each
#
# Variant "a" is always the original predefined UI query.
# Variants "b" and "c" are similar in intent but phrased differently.
# ---------------------------------------------------------------------------

QUERIES = {
    # --- Family 1: Draft / introduce ---
    "draft_introduction_a": (
        "Draft an introduction for this document based on the content."
    ),
    "draft_introduction_b": (
        "Write a compelling opening paragraph that frames the key argument of this text."
    ),
    "draft_introduction_c": (
        "Create a concise abstract summarising the main contribution and scope of this document."
    ),

    # --- Family 2: Discover gaps ---
    "discover_gaps_a": (
        "What gaps exist in the evidence or argument? What's missing?"
    ),
    "discover_gaps_b": (
        "Which claims in this text lack sufficient supporting evidence or citations?"
    ),
    "discover_gaps_c": (
        "What important counterarguments or alternative perspectives are not addressed here?"
    ),

    # --- Family 3: Find literature ---
    "find_literature_a": (
        "What does the current literature say about this topic? "
        "What key perspectives or findings am I missing?"
    ),
    "find_literature_b": (
        "What are the most influential recent papers on this subject that should be cited here?"
    ),
    "find_literature_c": (
        "What foundational studies or seminal works does this text overlook?"
    ),

    # --- Family 4: Check evidence ---
    "check_evidence_a": (
        "Is this well-supported? What claims need stronger evidence?"
    ),
    "check_evidence_b": (
        "Identify the weakest claims in this text from an evidence standpoint."
    ),
    "check_evidence_c": (
        "How does the evidence in this text hold up against recent empirical findings in the field?"
    ),

    # --- Family 5: Strengthen argument ---
    "strengthen_argument_a": (
        "How should I revise this to make the argument more robust and precise?"
    ),
    "strengthen_argument_b": (
        "What specific revisions would improve the logical flow and persuasiveness of this text?"
    ),
    "strengthen_argument_c": (
        "Rewrite the weakest section to make it more rigorous and better supported by evidence."
    ),

    # --- Family 6: Make concise ---
    "make_concise_a": (
        "Make this more concise and direct without losing key points."
    ),
    "make_concise_b": (
        "Identify and remove redundant or overly verbose passages from this text."
    ),
    "make_concise_c": (
        "Condense the key argument of this document into a single clear, well-structured paragraph."
    ),

    # --- Family 7: Challenge / critique ---
    "challenge_this_a": (
        "Challenge this position. What would a skeptical reviewer argue against these claims?"
    ),
    "challenge_this_b": (
        "What are the most serious methodological weaknesses a peer reviewer would flag here?"
    ),
    "challenge_this_c": (
        "What evidence from the literature contradicts or complicates the main claims in this document?"
    ),
}

# Map each query key to its intent family (for variance validation)
QUERY_FAMILY = {qk: qk.rsplit("_", 1)[0] for qk in QUERIES}


# ---------------------------------------------------------------------------
# 21 combinations
#
# Design:
#   - 9 examples × 2 rows + climate-change × 3 rows = 21 rows
#   - Each of the 21 queries used exactly once
#   - Each repeated example uses queries from different intent families
#
# Family assignments per example:
#   neuroplasticity          → F3 (find_lit_a),    F1 (draft_b)
#   climate-change           → F4 (check_ev_a),    F5 (strengthen_b),    F7 (challenge_c)
#   ai-ethics                → F6 (concise_a),     F2 (gaps_b)
#   gut-brain-html           → F1 (draft_a),       F7 (challenge_a)
#   sleep-myths              → F4 (check_ev_b),    F3 (find_lit_b)
#   social-media-gaps        → F2 (gaps_a),        F5 (strengthen_a)
#   protein-structure        → F6 (concise_b),     F1 (draft_c)
#   crispr-car-t-intro       → F3 (find_lit_c),    F2 (gaps_c)
#   stroke-rehab-discussion  → F7 (challenge_b),   F4 (check_ev_c)
#   obvious-discoveries      → F5 (strengthen_c),  F6 (concise_c)
# ---------------------------------------------------------------------------

COMBINATIONS = [
    # row  example_key                query_key
    ( 1,  "neuroplasticity",          "find_literature_a"),
    ( 2,  "climate-change",           "check_evidence_a"),
    ( 3,  "ai-ethics",                "make_concise_a"),
    ( 4,  "gut-brain-html",           "draft_introduction_a"),
    ( 5,  "sleep-myths",              "check_evidence_b"),
    ( 6,  "social-media-gaps",        "discover_gaps_a"),
    ( 7,  "protein-structure",        "make_concise_b"),
    ( 8,  "crispr-car-t-intro",       "find_literature_c"),
    ( 9,  "stroke-rehab-discussion",  "challenge_this_b"),
    (10,  "obvious-discoveries",      "strengthen_argument_c"),
    (11,  "neuroplasticity",          "draft_introduction_b"),     # F3 → F1
    (12,  "climate-change",           "strengthen_argument_b"),    # F4 → F5
    (13,  "ai-ethics",                "discover_gaps_b"),          # F6 → F2
    (14,  "gut-brain-html",           "challenge_this_a"),         # F1 → F7
    (15,  "sleep-myths",              "find_literature_b"),        # F4 → F3
    (16,  "social-media-gaps",        "strengthen_argument_a"),    # F2 → F5
    (17,  "protein-structure",        "draft_introduction_c"),     # F6 → F1
    (18,  "crispr-car-t-intro",       "discover_gaps_c"),          # F3 → F2
    (19,  "stroke-rehab-discussion",  "check_evidence_c"),         # F7 → F4
    (20,  "obvious-discoveries",      "make_concise_c"),           # F5 → F6
    (21,  "climate-change",           "challenge_this_c"),         # F4,F5 → F7
]


def build_rows() -> list[dict]:
    """Return the 21 evaluation rows as a list of dicts."""
    rows = []
    for row_num, example_key, query_key in COMBINATIONS:
        example = EXAMPLES[example_key]
        rows.append({
            "row_id":        row_num,
            "example_key":   example_key,
            "example_title": example["title"],
            "query_key":     query_key,
            "query_family":  QUERY_FAMILY[query_key],
            "query":         QUERIES[query_key],
            "input":         example["text"],
        })
    return rows


def validate_combinations(rows: list[dict]) -> None:
    """Assert all design constraints are satisfied."""
    from collections import Counter, defaultdict

    # Each query used exactly once
    query_counts = Counter(r["query_key"] for r in rows)
    for qk, count in query_counts.items():
        assert count == 1, f"Query '{qk}' used {count} times (expected 1)"
    assert len(query_counts) == 21, f"Expected 21 distinct queries, got {len(query_counts)}"

    # All 21 queries must appear
    missing = set(QUERIES) - set(query_counts)
    assert not missing, f"Unused queries: {missing}"

    # climate-change used 3 times, all others exactly 2
    example_counts = Counter(r["example_key"] for r in rows)
    for ek, count in example_counts.items():
        expected = 3 if ek == "climate-change" else 2
        assert count == expected, f"Example '{ek}' used {count} times (expected {expected})"

    # Repeated examples must use different intent families across occurrences
    example_families: dict[str, list[str]] = defaultdict(list)
    for r in rows:
        example_families[r["example_key"]].append(r["query_family"])
    for ek, families in example_families.items():
        assert len(families) == len(set(families)), (
            f"Example '{ek}' reuses intent family: {families}"
        )

    print(f"  Validation passed: {len(rows)} rows, "
          f"{len(query_counts)} unique queries, "
          f"{len(example_counts)} unique examples.")


def write_csv(rows: list[dict], output_path: Path) -> None:
    """Write the evaluation dataset to a CSV file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["row_id", "example_key", "example_title",
                  "query_key", "query_family", "query", "input"]
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"  Written {len(rows)} rows → {output_path}")


def print_summary(rows: list[dict]) -> None:
    """Print a human-readable summary table."""
    from collections import Counter
    example_counts = Counter(r["example_key"] for r in rows)

    print("\n  Row  Example                              Family               Query variant")
    print("  " + "-" * 90)
    for r in rows:
        count = example_counts[r["example_key"]]
        marker = f" (×{count})" if count > 1 else "      "
        print(
            f"  {r['row_id']:>3}  "
            f"{r['example_title']:<32}{marker}  "
            f"{r['query_family']:<24}  "
            f"{r['query_key'].split('_')[-1]}"
        )


def main():
    parser = argparse.ArgumentParser(
        description="Generate the Writing Coach V2 21-row evaluation dataset."
    )
    parser.add_argument(
        "--output",
        default="data_outputs/eval/wc2_eval_21.csv",
        help="Output CSV path (default: data_outputs/eval/wc2_eval_21.csv)",
    )
    args = parser.parse_args()

    output_path = ROOT / args.output

    print("Generating Writing Coach V2 evaluation dataset...")
    rows = build_rows()

    print("Validating combinations...")
    validate_combinations(rows)

    print_summary(rows)

    write_csv(rows, output_path)
    print("\nDone.")
    print(f"\nTo run the batch evaluation:")
    print(f"  python -m src.store_output {output_path} \\")
    print(f"    --results-csv data_outputs/eval/wc2_eval_21_results.csv \\")
    print(f"    --details-jsonl data_outputs/eval/wc2_eval_21_details.jsonl")


if __name__ == "__main__":
    main()
