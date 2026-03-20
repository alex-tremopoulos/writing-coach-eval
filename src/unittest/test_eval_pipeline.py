import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from ..evaluation.eval_pipeline import (
    _classify_suggestion_operation,
    _compute_removed_text,
    _format_output_for_judge,
    _is_removal_within_context,
    _is_whitespace_only_edit,
    _should_force_zero_correctness,
    load_input_data,
)


class TestOperationClassification(unittest.TestCase):
    """Verify that suggestions are correctly classified, especially removals."""

    def test_paragraph_minus_sentence_is_removal(self):
        """Row 276 pattern: transformed is original minus one sentence."""
        original = (
            "It is important to note that the performance of models using "
            "contrastive explanations can vary. The improvements in performance "
            "reported in the NLI task using contrastive explanations have been "
            "observed in terms of accuracy [4]. However, the performance can be "
            "influenced by factors such as the availability and quality."
        )
        transformed = (
            "It is important to note that the performance of models using "
            "contrastive explanations can vary. However, the performance can be "
            "influenced by factors such as the availability and quality."
        )
        sug = {"original_text": original, "transformed_text": transformed, "explanation": "remove requested sentence"}
        self.assertEqual(_classify_suggestion_operation(sug), "removal")

    def test_whitespace_only_edit(self):
        """Row 313 pattern: text identical, only empty lines removed."""
        original = "Sentence one.\n\nSentence two.\n\nSentence three."
        transformed = "Sentence one.\nSentence two.\nSentence three."
        sug = {"original_text": original, "transformed_text": transformed, "explanation": "remove empty lines"}
        self.assertEqual(_classify_suggestion_operation(sug), "whitespace_edit")

    def test_pure_deletion_empty_transformed(self):
        sug = {"original_text": "Delete this.", "transformed_text": "", "explanation": "delete"}
        self.assertEqual(_classify_suggestion_operation(sug), "deletion")

    def test_normal_replacement_stays_replacement(self):
        sug = {"original_text": "old text", "transformed_text": "completely new text here", "explanation": "rewrite"}
        self.assertEqual(_classify_suggestion_operation(sug), "replacement")

    def test_sentence_removed_from_context_pattern_b(self):
        """Row 276 actual data shape: original is the short sentence to remove,
        transformed is the surrounding paragraph after removal (longer)."""
        original = (
            "The improvements in performance reported in the NLI task using "
            "contrastive explanations have been observed in terms of accuracy [4]."
        )
        transformed = (
            "It is important to note that the performance of models using "
            "contrastive explanations can vary. However, the performance can be "
            "influenced by factors such as the availability and quality."
        )
        sug = {"original_text": original, "transformed_text": transformed, "explanation": "remove requested sentence"}
        # Pattern B requires is_removal_request=True because original < transformed
        self.assertEqual(_classify_suggestion_operation(sug, is_removal_request=True), "removal")


class TestRemovedTextComputation(unittest.TestCase):

    def test_compute_removed_text(self):
        original = "A B C D E F"
        transformed = "A B E F"
        removed = _compute_removed_text(original, transformed)
        self.assertIn("C D", removed)


class TestFormatOutputForJudge(unittest.TestCase):

    def test_removal_within_context_shows_annotation(self):
        """Paragraph-minus-sentence should show Removed Content annotation."""
        original = "First sentence. Second sentence to remove. Third sentence."
        transformed = "First sentence. Third sentence."
        formatted = _format_output_for_judge(
            response_text="Done.",
            suggestions=[{"original_text": original, "transformed_text": transformed, "explanation": "remove"}],
        )
        self.assertIn("**Operation Type**: removal", formatted)
        self.assertIn("**Removed Content**", formatted)
        self.assertIn("Second sentence to remove.", formatted)

    def test_whitespace_edit_shows_annotation(self):
        """Empty-line removal should show Whitespace-Only Edit annotation."""
        original = "A.\n\nB.\n\nC."
        transformed = "A.\nB.\nC."
        formatted = _format_output_for_judge(
            response_text="Done.",
            suggestions=[{"original_text": original, "transformed_text": transformed, "explanation": "remove empty lines"}],
        )
        self.assertIn("**Operation Type**: whitespace_edit", formatted)
        self.assertIn("Whitespace-Only Edit", formatted)

    def test_empty_deletion_still_works(self):
        formatted = _format_output_for_judge(
            response_text="Removed.",
            suggestions=[{"original_text": "Delete me.", "transformed_text": "", "explanation": "delete"}],
        )
        self.assertIn("**Operation Type**: deletion", formatted)
        self.assertIn("empty string — delete the original text", formatted)


class TestForceZeroCorrectness(unittest.TestCase):

    def test_force_zero_for_empty_non_removal(self):
        row = {"has_suggestions": True, "has_nonempty_transformed_text": False, "is_removal_request": False, "route": "REVISE_SIMPLE"}
        self.assertTrue(_should_force_zero_correctness(row))

    def test_no_force_for_valid_removal(self):
        row = {"has_suggestions": True, "has_nonempty_transformed_text": False, "is_removal_request": True, "route": "REVISE_SIMPLE"}
        self.assertFalse(_should_force_zero_correctness(row))

    def test_force_zero_for_revise_with_zero_suggestions_non_removal(self):
        """Row 290 pattern: REVISE route, 0 suggestions, not a removal."""
        row = {"has_suggestions": False, "has_nonempty_transformed_text": False, "is_removal_request": False, "route": "REVISE_SIMPLE"}
        self.assertTrue(_should_force_zero_correctness(row))

    def test_no_force_for_revise_with_zero_suggestions_removal(self):
        """Row 275/277 pattern: REVISE route, 0 suggestions, IS a removal."""
        row = {"has_suggestions": False, "has_nonempty_transformed_text": False, "is_removal_request": True, "route": "REVISE_SIMPLE"}
        self.assertFalse(_should_force_zero_correctness(row))

    def test_no_force_for_non_revise_route(self):
        row = {"has_suggestions": False, "has_nonempty_transformed_text": False, "is_removal_request": False, "route": "RESPOND"}
        self.assertFalse(_should_force_zero_correctness(row))


class TestLoadInputData(unittest.TestCase):

    def test_keeps_empty_transformed_deletion(self):
        row = {
            "row_id": 1,
            "query": "Remove the second sentence.",
            "input": "Sentence one. Sentence two. Sentence three.",
            "route_intended": "REVISE_SIMPLE",
            "output": json.dumps({
                "response": "I removed the requested sentence.",
                "suggestions": [{"original_text": "Sentence two.", "transformed_text": "", "explanation": "delete", "char_start": 14, "char_end": 27}],
            }),
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "rows.csv"
            pd.DataFrame([row]).to_csv(csv_path, index=False)
            rows = load_input_data(str(csv_path))
        self.assertEqual(len(rows), 1)
        loaded = rows[0]
        self.assertTrue(loaded["has_suggestions"])
        self.assertFalse(loaded["has_nonempty_transformed_text"])
        self.assertTrue(loaded["is_removal_request"])


if __name__ == "__main__":
    unittest.main()