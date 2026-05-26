import csv
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from ..scripts import store_output


class FakeCoach:
    version = "v3"

    def __init__(self):
        self.calls = []

    def run_query(self, row_id, query, document_text):
        self.calls.append((row_id, query, document_text))
        return {
            "row_id": row_id,
            "query": query,
            "input_preview": document_text[:200],
            "input": document_text,
            "route": "RESPOND",
            "intent": "conversation",
            "reasoning": f"handled {row_id}",
            "response": f"response for {row_id}",
            "suggestions": [],
            "references": [],
            "research_papers": [],
            "segments_count": 0,
            "tools_used": [],
        }


class TestStoreOutputCsv(unittest.TestCase):
    def test_write_csv_row_includes_response_text(self):
        result = {
            "row_id": 1,
            "query": "Summarize this",
            "input_preview": "Preview",
            "input": "Full input",
            "route": "RESPOND",
            "intent": "conversation",
            "reasoning": "Uses existing context.",
            "response": "Here is the summary.",
            "suggestions": [],
            "references": ["ref-1"],
            "research_papers": [],
            "segments_count": 0,
            "tools_used": ["tool_a", "tool_b"],
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "results.csv"
            with open(csv_path, "w", encoding="utf-8", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=store_output.CSV_FIELDNAMES)
                writer.writeheader()
                store_output._write_csv_row(writer, result)

            with open(csv_path, "r", encoding="utf-8", newline="") as f:
                rows = list(csv.DictReader(f))

        self.assertEqual(rows[0]["response"], "Here is the summary.")
        self.assertEqual(rows[0]["route_orch"], "RESPOND")
        self.assertEqual(rows[0]["tools_used"], "tool_a,tool_b")
        self.assertEqual(json.loads(rows[0]["suggestions"]), [])
        self.assertEqual(json.loads(rows[0]["references"]), ["ref-1"])
        self.assertEqual(json.loads(rows[0]["research_papers"]), [])

    def test_ensure_results_csv_schema_backfills_response_from_jsonl(self):
        legacy_fieldnames = [field for field in store_output.CSV_FIELDNAMES if field != "response"]
        legacy_row = {
            "row_id": 7,
            "query": "Rewrite this",
            "input_preview": "Preview",
            "input": "Input text",
            "route_orch": "REVISE_SIMPLE",
            "intent": "edit",
            "reasoning": "Direct rewrite request.",
            "segments_count": 0,
            "tools_used": "",
        }
        jsonl_record = {
            "row_id": 7,
            "query": "Rewrite this",
            "input_preview": "Preview",
            "input": "Input text",
            "route": "REVISE_SIMPLE",
            "intent": "edit",
            "reasoning": "Direct rewrite request.",
            "response": "Rewritten text",
            "suggestions": [{"original_text": "a", "transformed_text": "b"}],
            "references": [],
            "research_papers": [],
            "segments_count": 0,
            "tools_used": [],
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            results_csv = Path(tmpdir) / "legacy_results.csv"
            details_jsonl = Path(tmpdir) / "details.jsonl"

            with open(results_csv, "w", encoding="utf-8", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=legacy_fieldnames)
                writer.writeheader()
                writer.writerow(legacy_row)

            details_jsonl.write_text(json.dumps(jsonl_record) + "\n", encoding="utf-8")

            store_output._ensure_results_csv_schema(results_csv, details_jsonl)

            with open(results_csv, "r", encoding="utf-8", newline="") as f:
                rows = list(csv.DictReader(f))

        self.assertEqual(rows[0]["response"], "Rewritten text")
        self.assertEqual(rows[0]["route_orch"], "REVISE_SIMPLE")

    def test_process_csv_limit_applies_after_route_filter(self):
        fake_coach = FakeCoach()

        with tempfile.TemporaryDirectory() as tmpdir:
            input_csv = Path(tmpdir) / "input.csv"
            input_csv.write_text(
                "query,input,route\n"
                "q1,input one,RESPOND\n"
                "q2,input two,RESEARCH\n"
                "q3,input three,RESPOND\n",
                encoding="utf-8",
            )
            output_dir = Path(tmpdir) / "out"

            with mock.patch.object(store_output, "_load_environment"), \
                 mock.patch.object(store_output, "install_wc_app_path"), \
                 mock.patch.object(store_output, "create_writing_coach", return_value=fake_coach), \
                 mock.patch.object(store_output.time, "sleep"):
                store_output.process_csv(
                    input_csv=str(input_csv),
                    output_dir=str(output_dir),
                    filter_route="RESPOND",
                    limit=1,
                )

            self.assertEqual(fake_coach.calls, [
                (0, "What is machine learning?", mock.ANY),
                (1, "q1", "input one"),
            ])

            results_csv = output_dir / "input_RESPOND_results.csv"
            with open(results_csv, "r", encoding="utf-8", newline="") as f:
                rows = list(csv.DictReader(f))

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["query"], "q1")
        self.assertEqual(rows[0]["response"], "response for 1")

    def test_process_csv_provenance_columns_populated(self):
        """Verify folder_source, dataset_source, route_intended and metadata are written."""
        fake_coach = FakeCoach()

        with tempfile.TemporaryDirectory() as tmpdir:
            # Place input CSV inside a named sub-folder to test folder_source
            sub_dir = Path(tmpdir) / "my_dataset_folder"
            sub_dir.mkdir()
            input_csv = sub_dir / "run_batch.csv"
            input_csv.write_text(
                "query,input,route,route_intended,metadata\n"
                "q1,input one,RESPOND,RESPOND,{\"domain\": \"ml\"}\n"
                "q2,input two,RESPOND,,\n",
                encoding="utf-8",
            )
            output_dir = Path(tmpdir) / "out"

            with mock.patch.object(store_output, "_load_environment"), \
                 mock.patch.object(store_output, "install_wc_app_path"), \
                 mock.patch.object(store_output, "create_writing_coach", return_value=fake_coach), \
                 mock.patch.object(store_output.time, "sleep"):
                store_output.process_csv(
                    input_csv=str(input_csv),
                    output_dir=str(output_dir),
                )

            results_csv = output_dir / "run_batch_results.csv"
            with open(results_csv, "r", encoding="utf-8", newline="") as f:
                rows = list(csv.DictReader(f))

        self.assertEqual(len(rows), 2)

        # Row 1 — all provenance fields present in input
        self.assertEqual(rows[0]["folder_source"], "my_dataset_folder")
        self.assertEqual(rows[0]["dataset_source"], "run_batch")
        self.assertEqual(rows[0]["route_intended"], "RESPOND")
        self.assertEqual(rows[0]["metadata"], '{"domain": "ml"}')

        # Row 2 — optional fields absent in input should be empty
        self.assertEqual(rows[1]["folder_source"], "my_dataset_folder")
        self.assertEqual(rows[1]["dataset_source"], "run_batch")
        self.assertEqual(rows[1]["route_intended"], "")
        self.assertEqual(rows[1]["metadata"], "")

    def test_process_csv_route_intended_is_uppercased(self):
        fake_coach = FakeCoach()

        with tempfile.TemporaryDirectory() as tmpdir:
            input_csv = Path(tmpdir) / "input.csv"
            input_csv.write_text(
                "query,input,route_intended\n"
                "q1,input one,simple_revise\n",
                encoding="utf-8",
            )
            output_dir = Path(tmpdir) / "out"

            with mock.patch.object(store_output, "_load_environment"), \
                 mock.patch.object(store_output, "install_wc_app_path"), \
                 mock.patch.object(store_output, "create_writing_coach", return_value=fake_coach), \
                 mock.patch.object(store_output.time, "sleep"):
                store_output.process_csv(
                    input_csv=str(input_csv),
                    output_dir=str(output_dir),
                )

            results_csv = output_dir / "input_results.csv"
            with open(results_csv, "r", encoding="utf-8", newline="") as f:
                rows = list(csv.DictReader(f))

        self.assertEqual(rows[0]["route_intended"], "SIMPLE_REVISE")

    def test_process_csv_zero_limit_skips_warmup_and_writes_only_header(self):
        fake_coach = FakeCoach()

        with tempfile.TemporaryDirectory() as tmpdir:
            input_csv = Path(tmpdir) / "input.csv"
            input_csv.write_text(
                "query,input,route\n"
                "q1,input one,RESPOND\n",
                encoding="utf-8",
            )
            output_dir = Path(tmpdir) / "out"

            with mock.patch.object(store_output, "_load_environment"), \
                 mock.patch.object(store_output, "install_wc_app_path"), \
                 mock.patch.object(store_output, "create_writing_coach", return_value=fake_coach), \
                 mock.patch.object(store_output.time, "sleep"):
                store_output.process_csv(
                    input_csv=str(input_csv),
                    output_dir=str(output_dir),
                    limit=0,
                )

            self.assertEqual(fake_coach.calls, [])

            results_csv = output_dir / "input_results.csv"
            with open(results_csv, "r", encoding="utf-8", newline="") as f:
                rows = list(csv.DictReader(f))

        self.assertEqual(rows, [])

    def test_output_column_is_valid_json_with_agent_state_fields_only(self):
        """Verify output column contains JSON with only agent state fields."""
        result = {
            "row_id": 5,
            "query": "Analyze this",
            "input_preview": "Doc preview",
            "input": "Full document text",  # Should NOT be in output
            "route": "RESEARCH",
            "intent": "literature_review",
            "reasoning": "Need broader context.",
            "response": "Found 3 relevant papers.",
            "suggestions": ["suggestion 1"],
            "references": ["ref-1", "ref-2"],
            "research_papers": [{"title": "Paper 1"}, {"title": "Paper 2"}],
            "segments_count": 5,
            "tools_used": ["scholar", "arxiv"],
            "response_length": 28,
            "references_count": 2,
            "papers_count": 2,
            "suggestions_count": 1,
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "results.csv"
            with open(csv_path, "w", encoding="utf-8", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=store_output.CSV_FIELDNAMES)
                writer.writeheader()
                store_output._write_csv_row(writer, result)

            with open(csv_path, "r", encoding="utf-8", newline="") as f:
                rows = list(csv.DictReader(f))

        output_json_str = rows[0]["output"]
        output_dict = json.loads(output_json_str)

        # Verify exact fields present
        self.assertEqual(set(output_dict.keys()), {
            'route', 'intent', 'reasoning', 'response',
            'segments_count', 'tools_used', 'suggestions',
            'references', 'research_papers',
        })

        # Verify values match
        self.assertEqual(output_dict["route"], "RESEARCH")
        self.assertEqual(output_dict["intent"], "literature_review")
        self.assertEqual(output_dict["reasoning"], "Need broader context.")
        self.assertEqual(output_dict["response"], "Found 3 relevant papers.")
        self.assertEqual(output_dict["segments_count"], 5)
        self.assertEqual(output_dict["tools_used"], ["scholar", "arxiv"])
        self.assertEqual(output_dict["suggestions"], ["suggestion 1"])
        self.assertEqual(output_dict["references"], ["ref-1", "ref-2"])
        self.assertEqual(len(output_dict["research_papers"]), 2)

        # Flat columns should copy these values without removing them from nested output
        self.assertEqual(json.loads(rows[0]["suggestions"]), ["suggestion 1"])
        self.assertEqual(json.loads(rows[0]["references"]), ["ref-1", "ref-2"])
        self.assertEqual(
            json.loads(rows[0]["research_papers"]),
            [{"title": "Paper 1"}, {"title": "Paper 2"}],
        )

        # Verify input dataset fields are NOT in output
        self.assertNotIn("query", output_dict)
        self.assertNotIn("input", output_dict)
        self.assertNotIn("input_preview", output_dict)
        self.assertNotIn("row_id", output_dict)

    def test_output_column_includes_empty_fields(self):
        """Verify output column includes all fields even when empty."""
        result = {
            "row_id": 8,
            "query": "Summarize",
            "input_preview": "Preview",
            "input": "Input",
            "route": "RESPOND",
            "intent": "conversation",
            "reasoning": "Simple query.",
            "response": "Here is a summary.",
            # Empty/missing agent state fields:
            # suggestions, references, research_papers, tools_used not provided
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "results.csv"
            with open(csv_path, "w", encoding="utf-8", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=store_output.CSV_FIELDNAMES)
                writer.writeheader()
                store_output._write_csv_row(writer, result)

            with open(csv_path, "r", encoding="utf-8", newline="") as f:
                rows = list(csv.DictReader(f))

        output_dict = json.loads(rows[0]["output"])

        # All fields must be present even if empty
        self.assertIn("suggestions", output_dict)
        self.assertEqual(output_dict["suggestions"], [])
        self.assertIn("references", output_dict)
        self.assertEqual(output_dict["references"], [])
        self.assertIn("research_papers", output_dict)
        self.assertEqual(output_dict["research_papers"], [])
        self.assertIn("tools_used", output_dict)
        self.assertEqual(output_dict["tools_used"], [])
        self.assertIn("segments_count", output_dict)
        self.assertEqual(output_dict["segments_count"], 0)

    def test_ensure_results_csv_schema_backfills_output_from_jsonl(self):
        """Verify output column is backfilled when upgrading legacy CSVs."""
        # Legacy record without output column
        legacy_fieldnames = [field for field in store_output.CSV_FIELDNAMES
                            if field not in ["response", "output"]]
        legacy_row = {
            "row_id": 12,
            "query": "Test query",
            "input_preview": "Preview",
            "input": "Input",
            "route_orch": "RESEARCH",
            "intent": "explore",
            "reasoning": "Exploratory search.",
            "segments_count": 3,
            "tools_used": "tool_x",
        }

        # JSONL record with full agent state
        jsonl_record = {
            "row_id": 12,
            "query": "Test query",
            "input_preview": "Preview",
            "input": "Input",
            "route": "RESEARCH",
            "intent": "explore",
            "reasoning": "Exploratory search.",
            "response": "Found papers on topic.",
            "suggestions": [],
            "references": ["A", "B"],
            "research_papers": [{"title": "P1"}],
            "segments_count": 3,
            "tools_used": ["tool_x"],
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            results_csv = Path(tmpdir) / "legacy_results.csv"
            details_jsonl = Path(tmpdir) / "details.jsonl"

            with open(results_csv, "w", encoding="utf-8", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=legacy_fieldnames)
                writer.writeheader()
                writer.writerow(legacy_row)

            details_jsonl.write_text(json.dumps(jsonl_record) + "\n", encoding="utf-8")

            store_output._ensure_results_csv_schema(results_csv, details_jsonl)

            with open(results_csv, "r", encoding="utf-8", newline="") as f:
                rows = list(csv.DictReader(f))

        # Verify output column exists and is valid JSON
        self.assertIn("output", rows[0])
        output_dict = json.loads(rows[0]["output"])

        # Verify output contains correct agent state
        self.assertEqual(output_dict["route"], "RESEARCH")
        self.assertEqual(output_dict["intent"], "explore")
        self.assertEqual(output_dict["references"], ["A", "B"])
        self.assertEqual(len(output_dict["research_papers"]), 1)


if __name__ == "__main__":
    unittest.main()

