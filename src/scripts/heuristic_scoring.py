"""Compute heuristic metric scores from verdict item scores and compare them with LLM scores.

This script reconstructs metric-level scores from the per-item verdicts stored in
evaluation outputs. Each item's importance label is mapped to a numeric base
weight, then normalized so the coefficients for that metric sum to 1 for the
current example.

Default heuristic:
- Low = 1
- Medium = 2
- High = 3
- metric weighted average = sum(item_score_i * coeff_i)
- heuristic metric score = raw weighted average

Supported inputs:
- details JSONL files containing a `verdicts` field
- enriched CSV/JSONL files containing `eval_verdicts_json` or `verdicts_json`

Examples:
  python -m src.scripts.heuristic_scoring --input eval_data/one_prompt_metrics/route_intended/0320_1732/eval_20260320_173257_details.jsonl
  python -m src.scripts.heuristic_scoring --input eval_data/one_prompt_metrics/route_intended/0320_1732/eval_20260320_173257_details.jsonl --write-augmented
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any


METRIC_SCORE_FIELDS = {
	"Output Relevancy": ("output_relevancy_score", "eval_output_relevancy_score"),
	"Completeness": ("completeness_score", "eval_completeness_score"),
	"Correctness": ("correctness_score", "eval_correctness_score"),
}

VERDICTS_FIELDS = ("verdicts", "eval_verdicts_json", "verdicts_json")
IMPORTANCE_WEIGHTS = {
	"low": 1.0,
	"medium": 2.0,
	"high": 3.0,
}


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(
		description=(
			"Reconstruct metric scores from verdict item scores using fixed normalized importance weights and compare the raw weighted-average scores with the LLM-Judge scores."
		)
	)
	parser.add_argument("--input", type=Path, required=True, help="Input JSONL or CSV file")
	parser.add_argument(
		"--write-augmented",
		action="store_true",
		help="Write an augmented copy of the input rows with heuristic weighted-average score columns",
	)
	parser.add_argument(
		"--augmented-output",
		type=Path,
		default=None,
		help="Optional explicit path for the augmented output file",
	)
	parser.add_argument(
		"--summary-json",
		type=Path,
		default=None,
		help="Optional path to write the summary JSON",
	)
	return parser.parse_args()


def parse_json_maybe(value: Any) -> Any:
	if isinstance(value, (list, dict)):
		return value
	if value is None:
		return None
	if isinstance(value, float) and math.isnan(value):
		return None
	if not isinstance(value, str):
		return None
	stripped = value.strip()
	if not stripped:
		return None
	try:
		return json.loads(stripped)
	except json.JSONDecodeError:
		return None


def load_rows(input_path: Path) -> list[dict[str, Any]]:
	suffix = input_path.suffix.lower()
	if suffix == ".jsonl":
		rows: list[dict[str, Any]] = []
		with input_path.open("r", encoding="utf-8") as handle:
			for line in handle:
				stripped = line.strip()
				if stripped:
					rows.append(json.loads(stripped))
		return rows

	if suffix == ".csv":
		csv.field_size_limit(10 * 1024 * 1024)  # 10 MB
		with input_path.open("r", encoding="utf-8-sig", newline="") as handle:
			return list(csv.DictReader(handle))

	raise ValueError(f"Unsupported input format: {input_path}")


def extract_verdicts(row: dict[str, Any]) -> list[dict[str, Any]]:
	for field in VERDICTS_FIELDS:
		parsed = parse_json_maybe(row.get(field)) if field in row else None
		if isinstance(parsed, list):
			return [item for item in parsed if isinstance(item, dict)]
	return []


def extract_actual_metric_score(row: dict[str, Any], metric_name: str, verdict: dict[str, Any]) -> int | None:
	verdict_score = verdict.get("score")
	if isinstance(verdict_score, (int, float)):
		return int(verdict_score)

	for field in METRIC_SCORE_FIELDS[metric_name]:
		raw_value = row.get(field)
		if raw_value in (None, ""):
			continue
		try:
			return int(float(raw_value))
		except (TypeError, ValueError):
			continue
	return None


def normalize_importance(importance: Any) -> str:
	if not isinstance(importance, str):
		return "low"
	normalized = importance.strip().lower()
	if normalized in {"high", "medium", "low"}:
		return normalized
	return "low"


def compute_coefficients(
	items: list[dict[str, Any]],
	weight_map: dict[str, float],
) -> tuple[list[float], list[str]]:
	importances = [normalize_importance(item.get("importance")) for item in items]
	raw_weights = [weight_map.get(importance, 1.0) for importance in importances]
	total_weight = sum(raw_weights)
	if total_weight <= 0:
		raise ValueError("The total item weight must be positive")
	coefficients = [weight / total_weight for weight in raw_weights]
	return coefficients, importances


def compute_weighted_average(items: list[dict[str, Any]], coefficients: list[float]) -> float:
	weighted_average = 0.0
	for item, coefficient in zip(items, coefficients, strict=True):
		score = item.get("score", 0)
		try:
			numeric_score = float(score)
		except (TypeError, ValueError):
			numeric_score = 0.0
		weighted_average += numeric_score * coefficient
	return weighted_average


def build_augmented_output_path(input_path: Path) -> Path:
	if input_path.suffix.lower() == ".jsonl":
		return input_path.with_name(f"{input_path.stem}_heuristic.jsonl")
	return input_path.with_name(f"{input_path.stem}_heuristic.csv")


def write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
	suffix = path.suffix.lower()
	if suffix == ".jsonl":
		with path.open("w", encoding="utf-8") as handle:
			for row in rows:
				handle.write(json.dumps(row, ensure_ascii=False) + "\n")
		return

	if suffix == ".csv":
		fieldnames = sorted({key for row in rows for key in row.keys()})
		with path.open("w", encoding="utf-8", newline="") as handle:
			writer = csv.DictWriter(handle, fieldnames=fieldnames)
			writer.writeheader()
			writer.writerows(rows)
		return

	raise ValueError(f"Unsupported output format: {path}")


def analyze_rows(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
	augmented_rows: list[dict[str, Any]] = []

	metric_summary: dict[str, dict[str, Any]] = {
		metric_name: {
			"count": 0,
			"absolute_error_sum": 0.0,
			"signed_error_sum": 0.0,
			"llm_judge_score_distribution": Counter(),
			"item_count_distribution": Counter(),
			"importance_pattern_distribution": Counter(),
			"coefficient_patterns": Counter(),
			"heuristic_score_sum": 0.0,
			"llm_judge_score_sum": 0.0,
		}
		for metric_name in METRIC_SCORE_FIELDS
	}

	scored_rows = 0

	for row in rows:
		augmented_row = dict(row)
		verdicts = extract_verdicts(row)
		row_has_scores = False

		for verdict in verdicts:
			metric_name = verdict.get("metric_name")
			if not isinstance(metric_name, str) or metric_name not in METRIC_SCORE_FIELDS:
				continue

			items = verdict.get("evaluation_items") or []
			if not isinstance(items, list) or not items:
				continue

			actual_score = extract_actual_metric_score(row, metric_name, verdict)
			if actual_score is None:
				continue

			coefficients, importances = compute_coefficients(items, IMPORTANCE_WEIGHTS)
			heuristic_score = compute_weighted_average(items, coefficients)

			score_field = METRIC_SCORE_FIELDS[metric_name][0]
			heuristic_score_field = f"heuristic_{score_field}"
			heuristic_coefficients_field = f"heuristic_{score_field}_coefficients_json"

			augmented_row[heuristic_score_field] = round(heuristic_score, 6)
			augmented_row[heuristic_coefficients_field] = json.dumps(
				[round(coefficient, 6) for coefficient in coefficients]
			)

			row_has_scores = True

			summary_bucket = metric_summary[metric_name]
			summary_bucket["count"] += 1
			summary_bucket["absolute_error_sum"] += abs(heuristic_score - actual_score)
			summary_bucket["signed_error_sum"] += heuristic_score - actual_score
			summary_bucket["llm_judge_score_distribution"][actual_score] += 1
			summary_bucket["item_count_distribution"][len(items)] += 1
			summary_bucket["importance_pattern_distribution"][tuple(importances)] += 1
			summary_bucket["coefficient_patterns"][tuple(round(value, 6) for value in coefficients)] += 1
			summary_bucket["heuristic_score_sum"] += heuristic_score
			summary_bucket["llm_judge_score_sum"] += actual_score

		if row_has_scores:
			scored_rows += 1

		augmented_rows.append(augmented_row)

	summary: dict[str, Any] = {
		"rows_total": len(rows),
		"rows_scored": scored_rows,
		"weights": dict(IMPORTANCE_WEIGHTS),
		"scoring_method": "weighted_average",
		"metrics": {},
	}

	for metric_name, bucket in metric_summary.items():
		count = bucket["count"]
		coefficient_patterns = [
			{
				"coefficients": list(pattern),
				"count": occurrences,
			}
			for pattern, occurrences in bucket["coefficient_patterns"].most_common(5)
		]
		summary["metrics"][metric_name] = {
			"count": count,
			"mean_absolute_error": (bucket["absolute_error_sum"] / count if count else None),
			"mean_signed_error": (bucket["signed_error_sum"] / count if count else None),
			"heuristic_score_mean": (bucket["heuristic_score_sum"] / count if count else None),
			"llm_judge_score_mean": (bucket["llm_judge_score_sum"] / count if count else None),
			"llm_judge_score_distribution": dict(sorted(bucket["llm_judge_score_distribution"].items())),
			"item_count_distribution": dict(sorted(bucket["item_count_distribution"].items())),
			"importance_pattern_distribution": {
				str(list(pattern)): occurrences
				for pattern, occurrences in bucket["importance_pattern_distribution"].most_common()
			},
			"coefficient_patterns": coefficient_patterns,
			"has_stable_formula": len(bucket["coefficient_patterns"]) == 1,
		}

	return augmented_rows, summary


def print_summary(summary: dict[str, Any], input_path: Path) -> None:
	print(f"Input file: {input_path}")
	print(f"Rows total: {summary['rows_total']}")
	print(f"Rows scored: {summary['rows_scored']}")
	print(
		"Weights: "
		f"High={summary['weights']['high']}, "
		f"Medium={summary['weights']['medium']}, "
		f"Low={summary['weights']['low']}"
	)
	print("Scoring method: weighted average")

	for metric_name, metric_summary in summary["metrics"].items():
		print()
		print(f"[{metric_name}]")
		print(f"Count: {metric_summary['count']}")
		if metric_summary["mean_absolute_error"] is not None:
			print(
				"Heuristic score mean: "
				f"{metric_summary['heuristic_score_mean']:.3f}"
			)
			print(f"LLM-Judge score mean: {metric_summary['llm_judge_score_mean']:.3f}")
			print(f"Mean absolute error: {metric_summary['mean_absolute_error']:.3f}")
			print(f"Mean signed error: {metric_summary['mean_signed_error']:.3f}")
		print(f"Item counts: {metric_summary['item_count_distribution']}")
		print(f"LLM-Judge score distribution: {metric_summary['llm_judge_score_distribution']}")
		print(f"Stable formula: {metric_summary['has_stable_formula']}")
		if metric_summary["coefficient_patterns"]:
			top_pattern = metric_summary["coefficient_patterns"][0]
			print(f"Top coefficient pattern: {top_pattern['coefficients']} (n={top_pattern['count']})")


def main() -> None:
	args = parse_args()
	input_path = args.input
	rows = load_rows(input_path)
	augmented_rows, summary = analyze_rows(rows)

	print_summary(summary, input_path)

	if args.summary_json is not None:
		args.summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

	if args.write_augmented:
		output_path = args.augmented_output or build_augmented_output_path(input_path)
		write_rows(output_path, augmented_rows)
		print()
		print(f"Augmented output written to: {output_path}")


if __name__ == "__main__":
	main()
