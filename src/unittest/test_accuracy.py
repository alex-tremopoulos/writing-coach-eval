"""
Unit tests for accuracy.py.

Covers:
  - _ids_in_bracket
  - _find_consecutive_groups
  - preprocess_references
  - load_and_preprocess_revise_rows  (RESEARCH, REVISE_RESEARCH, unknown routes,
                                       .jsonl and .csv inputs)
"""

import json
import re
import warnings
import unittest
import tempfile
from pathlib import Path

import jsonlines
import pandas as pd

from ..metrics.accuracy import (
    _BRACKET_GROUP,
    _ids_in_bracket,
    _find_consecutive_groups,
    preprocess_references,
    load_and_preprocess_revise_rows,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_ref(ref_id: str, title: str = "") -> dict:
    return {"referenceId": ref_id, "title": title or ref_id}


def _make_research_row(
    query: str,
    response: str,
    references: list[dict],
    route: str = "RESEARCH",
) -> dict:
    return {
        "query": query,
        "route_orch": route,
        "output": {
            "response": response,
            "references": references,
            "suggestions": [],
        },
    }


def _make_revise_research_row(
    query: str,
    suggestions: list[dict],
    references: list[dict],
) -> dict:
    return {
        "query": query,
        "route_orch": "REVISE_RESEARCH",
        "output": {
            "response": "",
            "references": references,
            "suggestions": suggestions,
        },
    }


def _suggestion(transformed_text: str) -> dict:
    return {
        "original_text": "",
        "transformed_text": transformed_text,
        "explanation": "",
        "char_start": 0,
        "char_end": 0,
        "research_backing": {},
    }


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    with jsonlines.open(path, mode="w") as w:
        w.write_all(rows)


def _write_csv(path: Path, rows: list[dict]) -> None:
    # Serialise the nested `output` dict to JSON string, as the real CSV does
    serialised = [
        {k: (json.dumps(v) if isinstance(v, dict) else v) for k, v in row.items()}
        for row in rows
    ]
    pd.DataFrame(serialised).to_csv(path, index=False)


# ===========================================================================
# Tests for _ids_in_bracket
# ===========================================================================

class TestIdsInBracket(unittest.TestCase):

    def _match(self, text: str) -> re.Match:
        return _BRACKET_GROUP.search(text)

    def test_single_id(self):
        m = self._match("[C11-1-2-1]")
        self.assertEqual(_ids_in_bracket(m), ["C11-1-2-1"])

    def test_multiple_ids(self):
        m = self._match("[C11-1-2-1, C11-1-3-6, C11-1-2-5]")
        self.assertEqual(_ids_in_bracket(m), ["C11-1-2-1", "C11-1-3-6", "C11-1-2-5"])

    def test_ids_with_extra_whitespace(self):
        m = self._match("[C11-1-2-1,  C11-1-3-6]")
        self.assertEqual(_ids_in_bracket(m), ["C11-1-2-1", "C11-1-3-6"])


# ===========================================================================
# Tests for _find_consecutive_groups
# ===========================================================================

class TestFindConsecutiveGroups(unittest.TestCase):

    def _matches(self, text: str) -> list[re.Match]:
        return list(_BRACKET_GROUP.finditer(text))

    def test_empty(self):
        self.assertEqual(_find_consecutive_groups([]), [])

    def test_single_match(self):
        matches = self._matches("text [C1-1-1-1] end")
        groups = _find_consecutive_groups(matches)
        self.assertEqual(len(groups), 1)
        self.assertEqual(len(groups[0]), 1)

    def test_non_consecutive(self):
        matches = self._matches("A [C1-1-1-1] then B [C1-1-1-2] end")
        groups = _find_consecutive_groups(matches)
        self.assertEqual(len(groups), 2)

    def test_consecutive_comma_separated(self):
        matches = self._matches("A [C1-1-1-1], [C1-1-1-2] end")
        groups = _find_consecutive_groups(matches)
        self.assertEqual(len(groups), 1)
        self.assertEqual(len(groups[0]), 2)

    def test_consecutive_whitespace_only(self):
        matches = self._matches("A [C1-1-1-1]  [C1-1-1-2] end")
        groups = _find_consecutive_groups(matches)
        self.assertEqual(len(groups), 1)

    def test_three_consecutive(self):
        matches = self._matches("A [C1-1-1-1], [C1-1-1-2], [C1-1-1-3] end")
        groups = _find_consecutive_groups(matches)
        self.assertEqual(len(groups), 1)
        self.assertEqual(len(groups[0]), 3)

    def test_mixed_consecutive_and_separate(self):
        matches = self._matches("A [C1-1-1-1], [C1-1-1-2] text [C1-1-1-3] end")
        groups = _find_consecutive_groups(matches)
        self.assertEqual(len(groups), 2)
        self.assertEqual(len(groups[0]), 2)
        self.assertEqual(len(groups[1]), 1)


# ===========================================================================
# Tests for preprocess_references
# ===========================================================================

class TestPreprocessReferences(unittest.TestCase):

    # --- response rewriting -------------------------------------------------

    def test_single_ref_replaced(self):
        data = {
            "response": "Text [C1-1-1-1].",
            "references": [_make_ref("C1-1-1-1", "Ref A")],
        }
        result = preprocess_references(data)
        self.assertIn("[1]", result["response"])

    def test_multi_ref_bracket_replaced(self):
        data = {
            "response": "Text [C1-1-1-1, C1-1-1-2].",
            "references": [_make_ref("C1-1-1-1"), _make_ref("C1-1-1-2")],
        }
        result = preprocess_references(data)
        self.assertIn("[1, 2]", result["response"])

    def test_consecutive_separate_brackets_merged(self):
        data = {
            "response": "Text [C1-1-1-1], [C1-1-1-2].",
            "references": [_make_ref("C1-1-1-1"), _make_ref("C1-1-1-2")],
        }
        result = preprocess_references(data)
        self.assertIn("[1, 2]", result["response"])
        # Original separate brackets must be gone
        self.assertNotIn("[C1-1-1-1]", result["response"])
        self.assertNotIn("[C1-1-1-2]", result["response"])

    def test_non_consecutive_brackets_not_merged(self):
        data = {
            "response": "A [C1-1-1-1] text B [C1-1-1-2].",
            "references": [_make_ref("C1-1-1-1"), _make_ref("C1-1-1-2")],
        }
        result = preprocess_references(data)
        self.assertIn("[1]", result["response"])
        self.assertIn("[2]", result["response"])
        # Should not be merged into one bracket
        self.assertNotIn("[1, 2]", result["response"])

    def test_indexes_are_sequential_and_start_at_one(self):
        data = {
            "response": "A [C1-1-1-1, C1-1-1-2, C1-1-1-3].",
            "references": [
                _make_ref("C1-1-1-1"),
                _make_ref("C1-1-1-2"),
                _make_ref("C1-1-1-3"),
            ],
        }
        result = preprocess_references(data)
        self.assertIn("[1, 2, 3]", result["response"])

    def test_old_response_preserved(self):
        original = "Text [C1-1-1-1]."
        data = {"response": original, "references": [_make_ref("C1-1-1-1")]}
        result = preprocess_references(data)
        self.assertEqual(result["old_response"], original)

    # --- repeated references get independent indexes ------------------------

    def test_same_id_repeated_gets_new_index_each_time(self):
        data = {
            "response": "A [C1-1-1-1] and B [C1-1-1-1].",
            "references": [_make_ref("C1-1-1-1", "Ref A")],
        }
        result = preprocess_references(data)
        self.assertIn("[1]", result["response"])
        self.assertIn("[2]", result["response"])
        self.assertEqual(result["reference_mapping"]["C1-1-1-1"], [1, 2])

    def test_references_list_has_one_entry_per_occurrence(self):
        data = {
            "response": "A [C1-1-1-1] B [C1-1-1-1].",
            "references": [_make_ref("C1-1-1-1", "Ref A")],
        }
        result = preprocess_references(data)
        # Two occurrences → two entries in references
        self.assertEqual(len(result["references"]), 2)
        self.assertEqual(result["references"][0]["index"], 1)
        self.assertEqual(result["references"][1]["index"], 2)

    def test_index_is_integer_not_list(self):
        data = {
            "response": "A [C1-1-1-1].",
            "references": [_make_ref("C1-1-1-1")],
        }
        result = preprocess_references(data)
        self.assertIsInstance(result["references"][0]["index"], int)

    def test_references_sorted_by_index(self):
        data = {
            "response": "A [C1-1-1-2] then [C1-1-1-1].",
            "references": [_make_ref("C1-1-1-1"), _make_ref("C1-1-1-2")],
        }
        result = preprocess_references(data)
        indexes = [r["index"] for r in result["references"]]
        self.assertEqual(indexes, sorted(indexes))

    # --- reference_mapping --------------------------------------------------

    def test_reference_mapping_single_occurrence(self):
        data = {
            "response": "A [C1-1-1-1].",
            "references": [_make_ref("C1-1-1-1")],
        }
        result = preprocess_references(data)
        self.assertEqual(result["reference_mapping"], {"C1-1-1-1": [1]})

    def test_reference_mapping_multiple_occurrences(self):
        data = {
            "response": "A [C1-1-1-1] B [C1-1-1-2] C [C1-1-1-1].",
            "references": [_make_ref("C1-1-1-1"), _make_ref("C1-1-1-2")],
        }
        result = preprocess_references(data)
        self.assertEqual(result["reference_mapping"]["C1-1-1-1"], [1, 3])
        self.assertEqual(result["reference_mapping"]["C1-1-1-2"], [2])

    # --- no references in response ------------------------------------------

    def test_no_references_in_response(self):
        data = {"response": "Plain text with no refs.", "references": []}
        result = preprocess_references(data)
        self.assertEqual(result["response"], "Plain text with no refs.")
        self.assertEqual(result["references"], [])
        self.assertEqual(result["reference_mapping"], {})

    # --- unknown ref ID (not in references list) ----------------------------

    def test_unknown_ref_id_still_indexed(self):
        data = {
            "response": "A [C9-9-9-9].",
            "references": [],  # no matching entry
        }
        result = preprocess_references(data)
        self.assertIn("[1]", result["response"])
        self.assertEqual(result["references"][0]["index"], 1)

    # --- original reference fields preserved --------------------------------

    def test_original_ref_fields_preserved(self):
        ref = {"referenceId": "C1-1-1-1", "title": "My Paper", "doi": "10.1/x"}
        data = {"response": "A [C1-1-1-1].", "references": [ref]}
        result = preprocess_references(data)
        out_ref = result["references"][0]
        self.assertEqual(out_ref["title"], "My Paper")
        self.assertEqual(out_ref["doi"], "10.1/x")
        self.assertEqual(out_ref["referenceId"], "C1-1-1-1")


# ===========================================================================
# Tests for load_and_preprocess_revise_rows
# ===========================================================================

class TestLoadAndPreprocessReviseRows(unittest.TestCase):

    # -----------------------------------------------------------------------
    # Shared fixtures
    # -----------------------------------------------------------------------

    REFS = [
        _make_ref("C1-1-1-1", "Ref A"),
        _make_ref("C1-1-1-2", "Ref B"),
        _make_ref("C1-1-1-3", "Ref C"),
    ]

    RESEARCH_ROW = _make_research_row(
        query="What is urbanisation?",
        response="Cities grow [C1-1-1-1, C1-1-1-2]. Evidence [C1-1-1-1].",
        references=REFS,
    )

    REVISE_ROW = _make_revise_research_row(
        query="Revise my text",
        suggestions=[
            _suggestion("New text [C1-1-1-2, C1-1-1-3]. More [C1-1-1-2]."),
            _suggestion("Another suggestion [C1-1-1-1]."),
        ],
        references=REFS,
    )

    # -----------------------------------------------------------------------
    # Helper: write rows to a temp file and call the function
    # -----------------------------------------------------------------------

    def _run_jsonl(self, rows: list[dict]) -> list[dict]:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "data.jsonl"
            _write_jsonl(path, rows)
            return load_and_preprocess_revise_rows(path)

    def _run_csv(self, rows: list[dict]) -> list[dict]:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "data.csv"
            _write_csv(path, rows)
            return load_and_preprocess_revise_rows(path)

    # -----------------------------------------------------------------------
    # Output shape / key names
    # -----------------------------------------------------------------------

    def test_research_output_keys(self):
        results = self._run_jsonl([self.RESEARCH_ROW])
        self.assertEqual(len(results), 1)
        self.assertIn("query", results[0])
        self.assertIn("input_line", results[0])
        self.assertIn("summary", results[0])
        self.assertIn("reference", results[0])
        self.assertIn("old_response", results[0])
        self.assertIn("reference_mapping", results[0])

    def test_revise_research_output_keys(self):
        results = self._run_jsonl([self.REVISE_ROW])
        for r in results:
            self.assertIn("query", r)
            self.assertIn("input_line", r)
            self.assertIn("summary", r)
            self.assertIn("reference", r)

    def test_no_response_or_references_key_in_output(self):
        """The old key names must not appear at the top level."""
        results = self._run_jsonl([self.RESEARCH_ROW])
        self.assertNotIn("response", results[0])
        self.assertNotIn("references", results[0])

    # -----------------------------------------------------------------------
    # RESEARCH routing
    # -----------------------------------------------------------------------

    def test_research_query_propagated(self):
        results = self._run_jsonl([self.RESEARCH_ROW])
        self.assertEqual(results[0]["query"], "What is urbanisation?")

    def test_research_input_line_is_one_based(self):
        results = self._run_jsonl([self.RESEARCH_ROW])
        self.assertEqual(results[0]["input_line"], 1)

    def test_research_summary_has_integer_refs(self):
        results = self._run_jsonl([self.RESEARCH_ROW])
        self.assertNotIn("C1-1-1-1", results[0]["summary"])
        self.assertIn("[1, 2]", results[0]["summary"])

    def test_research_skipped_when_no_references(self):
        row = _make_research_row("q", "Text [C1-1-1-1].", references=[])
        results = self._run_jsonl([row])
        self.assertEqual(results, [])

    def test_research_reference_list_one_entry_per_occurrence(self):
        # C1-1-1-1 appears twice → two entries in reference
        results = self._run_jsonl([self.RESEARCH_ROW])
        indexes = [r["index"] for r in results[0]["reference"]]
        # index 1 (first occurrence of C1-1-1-1) and index 3 (second)
        self.assertIn(1, indexes)
        self.assertIn(3, indexes)

    # -----------------------------------------------------------------------
    # REVISE_RESEARCH routing
    # -----------------------------------------------------------------------

    def test_revise_research_produces_one_entry_per_suggestion(self):
        results = self._run_jsonl([self.REVISE_ROW])
        self.assertEqual(len(results), 2)

    def test_revise_research_query_propagated(self):
        results = self._run_jsonl([self.REVISE_ROW])
        for r in results:
            self.assertEqual(r["query"], "Revise my text")

    def test_revise_research_input_line_propagated(self):
        results = self._run_jsonl([self.REVISE_ROW])
        for r in results:
            self.assertEqual(r["input_line"], 1)

    def test_revise_research_summary_has_integer_refs(self):
        results = self._run_jsonl([self.REVISE_ROW])
        self.assertNotIn("C1-1-1-2", results[0]["summary"])
        self.assertNotIn("C1-1-1-3", results[0]["summary"])

    def test_revise_research_refs_filtered_to_suggestion(self):
        # First suggestion cites C1-1-1-2 and C1-1-1-3 only
        results = self._run_jsonl([self.REVISE_ROW])
        cited_ids = {r["referenceId"] for r in results[0]["reference"]}
        self.assertIn("C1-1-1-2", cited_ids)
        self.assertIn("C1-1-1-3", cited_ids)
        self.assertNotIn("C1-1-1-1", cited_ids)

    def test_revise_research_suggestion_skipped_when_no_cited_refs(self):
        row = _make_revise_research_row(
            query="q",
            suggestions=[_suggestion("No citations here.")],
            references=self.REFS,
        )
        results = self._run_jsonl([row])
        self.assertEqual(results, [])

    # -----------------------------------------------------------------------
    # Unknown route
    # -----------------------------------------------------------------------

    def test_unknown_route_skipped_with_warning(self):
        row = {
            "query": "q",
            "route_orch": "RESPOND",
            "output": {"response": "r", "references": self.REFS, "suggestions": []},
        }
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            results = self._run_jsonl([row])
        self.assertEqual(results, [])
        self.assertTrue(
            any("RESPOND" in str(w.message) for w in caught),
            "Expected a warning mentioning the unknown route value",
        )

    def test_missing_route_orch_skipped_silently(self):
        row = {
            "query": "q",
            "output": {"response": "r", "references": self.REFS, "suggestions": []},
        }
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            results = self._run_jsonl([row])
        self.assertEqual(results, [])
        self.assertFalse(any(issubclass(w.category, UserWarning) for w in caught))

    # -----------------------------------------------------------------------
    # Multiple rows and line numbering
    # -----------------------------------------------------------------------

    def test_input_line_increments_across_jsonl_rows(self):
        rows = [self.RESEARCH_ROW, self.RESEARCH_ROW]
        results = self._run_jsonl(rows)
        self.assertEqual(results[0]["input_line"], 1)
        self.assertEqual(results[1]["input_line"], 2)

    def test_input_line_csv_accounts_for_header(self):
        # Row 0 in DataFrame → line 2 in file (1 = header, 2 = first data row)
        results = self._run_csv([self.RESEARCH_ROW])
        self.assertEqual(results[0]["input_line"], 2)

    def test_input_line_csv_second_row_is_three(self):
        results = self._run_csv([self.RESEARCH_ROW, self.RESEARCH_ROW])
        self.assertEqual(results[0]["input_line"], 2)
        self.assertEqual(results[1]["input_line"], 3)

    def test_mixed_routes_only_valid_ones_processed(self):
        rows = [
            self.RESEARCH_ROW,
            {"query": "skip", "route_orch": "RESPOND",
             "output": {"response": "", "references": self.REFS, "suggestions": []}},
            self.REVISE_ROW,
        ]
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            results = self._run_jsonl(rows)
        # 1 RESEARCH + 2 suggestions from REVISE_RESEARCH
        self.assertEqual(len(results), 3)

    # -----------------------------------------------------------------------
    # CSV and JSONL parity
    # -----------------------------------------------------------------------

    def test_csv_and_jsonl_same_count(self):
        rows = [self.RESEARCH_ROW, self.REVISE_ROW]
        jsonl_results = self._run_jsonl(rows)
        csv_results = self._run_csv(rows)
        self.assertEqual(len(jsonl_results), len(csv_results))

    def test_csv_and_jsonl_same_summaries(self):
        rows = [self.RESEARCH_ROW]
        jsonl_results = self._run_jsonl(rows)
        csv_results = self._run_csv(rows)
        self.assertEqual(jsonl_results[0]["summary"], csv_results[0]["summary"])

    # -----------------------------------------------------------------------
    # Unsupported format
    # -----------------------------------------------------------------------

    def test_unsupported_format_raises(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "data.txt"
            path.write_text("hello")
            with self.assertRaises(ValueError):
                load_and_preprocess_revise_rows(path)


if __name__ == "__main__":
    unittest.main()
