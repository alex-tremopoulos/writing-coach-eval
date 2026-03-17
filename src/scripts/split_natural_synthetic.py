"""Split one enriched eval run into natural and synthetic subsets and recompute scores.

This script reuses the score aggregation logic from the evaluation pipeline,
but operates on an already-produced enriched eval file instead of rerunning the
LLM evaluation.

The expected input for this script is an enriched file that already contains a
single `route` column. Whether that route came from intended routing or
orchestrator routing is determined by which folder/file you pass in.

Supported inputs:
- ``*_all_results_enriched.csv``
- ``*_all_results_enriched.jsonl``

Examples:
  python -m src.scripts.split_natural_synthetic
  python -m src.scripts.split_natural_synthetic --input eval_data/one_prompt_metrics/combined_intended_all/eval_20260313_164718_all_results_enriched.csv
  python -m src.scripts.split_natural_synthetic --route-column route_orch --save-summary-json
  python -m src.scripts.split_natural_synthetic --write-splits --output-dir eval_data/one_prompt_metrics/combined_intended_all/split_origin
"""

from __future__ import annotations

import argparse
import json
import statistics
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd


DEFAULT_INPUT_DIR = Path("eval_data/one_prompt_metrics/combined_intended_all")
SYNTHETIC_EXACT_SOURCES = {"extra_respond_alex"}
EVAL_STATUS_COLUMN = "eval_status"
OUTPUT_RELEVANCY_COLUMN = "eval_output_relevancy_score"
COMPLETENESS_COLUMN = "eval_completeness_score"
DATASET_SOURCE_COLUMN = "dataset_source"


def find_default_input() -> Path | None:
    """Return the newest enriched run under the default one-prompt directory."""
    if not DEFAULT_INPUT_DIR.exists():
        return None

    return find_latest_enriched_file(DEFAULT_INPUT_DIR)


def find_latest_enriched_file(directory: Path) -> Path | None:
    """Return the newest enriched CSV/JSONL file inside a directory."""
    if not directory.exists() or not directory.is_dir():
        return None

    candidates = sorted(directory.glob("*_all_results_enriched.csv"))
    if not candidates:
        candidates = sorted(directory.glob("*_all_results_enriched.jsonl"))
    if not candidates:
        return None

    return max(candidates, key=lambda path: (path.stat().st_mtime, path.name))


def resolve_input_path(input_path: Path) -> Path:
    """Resolve a file or directory input into a concrete enriched file path."""
    if input_path.is_dir():
        resolved = find_latest_enriched_file(input_path)
        if resolved is None:
            raise ValueError(
                f"No enriched eval CSV or JSONL files were found in directory: {input_path}"
            )
        return resolved
    return input_path


def _avg_std(scores: list[float]) -> tuple[float, float]:
    """Return (mean, stdev) of scores, or (nan, 0.0) for an empty list."""
    if not scores:
        return float("nan"), 0.0
    avg = sum(scores) / len(scores)
    return avg, (statistics.stdev(scores) if len(scores) >= 2 else 0.0)


def _compute_score_stats(ok_results: list[dict[str, Any]]) -> dict[str, Any]:
    """Compute per-route, micro, and macro score statistics from OK-status results."""
    route_buckets: dict[str, dict[str, list[float]]] = {}
    for result in ok_results:
        bucket = route_buckets.setdefault(
            result["route"],
            {"output_relevancy_score": [], "completeness_score": []},
        )
        for field in ("output_relevancy_score", "completeness_score"):
            if result.get(field) is not None:
                bucket[field].append(result[field])

    per_route = {
        route: {
            "n": len(bucket["output_relevancy_score"]),
            "output_relevancy": _avg_std(bucket["output_relevancy_score"]),
            "completeness": _avg_std(bucket["completeness_score"]),
        }
        for route, bucket in route_buckets.items()
    }

    all_relevancy = [score for bucket in route_buckets.values() for score in bucket["output_relevancy_score"]]
    all_completeness = [score for bucket in route_buckets.values() for score in bucket["completeness_score"]]
    route_relevancy_avgs = [
        _avg_std(bucket["output_relevancy_score"])[0]
        for bucket in route_buckets.values()
        if bucket["output_relevancy_score"]
    ]
    route_completeness_avgs = [
        _avg_std(bucket["completeness_score"])[0]
        for bucket in route_buckets.values()
        if bucket["completeness_score"]
    ]

    return {
        "per_route": per_route,
        "micro": {
            "n": len(all_relevancy),
            "output_relevancy": _avg_std(all_relevancy),
            "completeness": _avg_std(all_completeness),
        },
        "macro": {
            "n_routes": len(route_relevancy_avgs),
            "output_relevancy": _avg_std(route_relevancy_avgs),
            "completeness": _avg_std(route_completeness_avgs),
        },
    }


def load_enriched_results(input_path: Path) -> pd.DataFrame:
    """Load an enriched eval file from CSV or JSONL."""
    suffix = input_path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(input_path, encoding="utf-8-sig")
    if suffix == ".jsonl":
        return pd.read_json(input_path, lines=True)
    raise ValueError(f"Unsupported input format: {input_path}")


def _first_existing_column(df: pd.DataFrame, candidates: list[str]) -> str | None:
    """Return the first candidate column that exists in the DataFrame."""
    for candidate in candidates:
        if candidate in df.columns:
            return candidate
    return None


def resolve_route_column(df: pd.DataFrame, requested: str, input_path: Path) -> str:
    """Resolve the route column to use for aggregation.

    In auto mode, prefer inferring from the input path:
    - paths containing 'intended' -> route_intended
    - paths containing 'orchestrator' -> route_orchestrator / route_orch
    - otherwise fall back to a generic route column, then legacy columns
    """
    if requested != "auto":
        explicit_column = _first_existing_column(df, [requested])
        if explicit_column is None:
            raise ValueError(f"Requested route column '{requested}' not found in input file")
        return explicit_column

    path_text = str(input_path).lower()
    if "intended" in path_text:
        inferred_column = _first_existing_column(df, ["route_intended"])
        if inferred_column is not None:
            return inferred_column

    if "orchestrator" in path_text:
        inferred_column = _first_existing_column(df, ["route_orchestrator", "route_orch"])
        if inferred_column is not None:
            return inferred_column

    fallback_column = _first_existing_column(
        df,
        ["route", "route_intended", "route_orchestrator", "route_orch"],
    )
    if fallback_column is not None:
        return fallback_column

    raise ValueError(
        "No route column found. Expected one of: route, route_intended, route_orchestrator, route_orch"
    )


def build_synthetic_mask(df: pd.DataFrame) -> pd.Series:
    """Classify rows as synthetic using the same rule as eval_pipeline.py."""
    if DATASET_SOURCE_COLUMN not in df.columns:
        raise ValueError(f"Missing required column: {DATASET_SOURCE_COLUMN}")

    dataset_source = df[DATASET_SOURCE_COLUMN].fillna("").astype(str)
    return dataset_source.str.startswith("Synthetic") | dataset_source.isin(SYNTHETIC_EXACT_SOURCES)


def normalize_eval_frame(df: pd.DataFrame, route_column: str) -> pd.DataFrame:
    """Return a copy with score columns coerced to numeric and route normalized."""
    required_columns = [route_column, EVAL_STATUS_COLUMN, OUTPUT_RELEVANCY_COLUMN, COMPLETENESS_COLUMN]
    missing = [column for column in required_columns if column not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(missing)}")

    normalized = df.copy()
    normalized[route_column] = normalized[route_column].fillna("UNKNOWN").astype(str)
    normalized[EVAL_STATUS_COLUMN] = normalized[EVAL_STATUS_COLUMN].fillna("MISSING").astype(str)
    normalized[OUTPUT_RELEVANCY_COLUMN] = pd.to_numeric(normalized[OUTPUT_RELEVANCY_COLUMN], errors="coerce")
    normalized[COMPLETENESS_COLUMN] = pd.to_numeric(normalized[COMPLETENESS_COLUMN], errors="coerce")
    return normalized


def extract_ok_results(df: pd.DataFrame, route_column: str) -> list[dict[str, Any]]:
    """Convert OK rows with valid scores into the structure expected by the aggregator."""
    ok_mask = df[EVAL_STATUS_COLUMN].str.upper() == "OK"
    valid_score_mask = df[OUTPUT_RELEVANCY_COLUMN].notna() & df[COMPLETENESS_COLUMN].notna()
    filtered = df.loc[ok_mask & valid_score_mask, [route_column, OUTPUT_RELEVANCY_COLUMN, COMPLETENESS_COLUMN]]

    return [
        {
            "route": row[route_column],
            "output_relevancy_score": float(row[OUTPUT_RELEVANCY_COLUMN]),
            "completeness_score": float(row[COMPLETENESS_COLUMN]),
        }
        for _, row in filtered.iterrows()
    ]


def summarize_subset(name: str, df: pd.DataFrame, route_column: str) -> dict[str, Any]:
    """Build counts and score stats for one subset."""
    status_counts = Counter(df[EVAL_STATUS_COLUMN].astype(str)) if EVAL_STATUS_COLUMN in df.columns else Counter()
    route_counts = Counter(df[route_column].astype(str)) if route_column in df.columns else Counter()
    ok_results = extract_ok_results(df, route_column)

    return {
        "name": name,
        "rows": len(df),
        "ok_rows": len(ok_results),
        "status_counts": dict(status_counts),
        "route_counts": dict(route_counts),
        "score_stats": _compute_score_stats(ok_results),
    }


def _stats_to_jsonable(stats: dict[str, Any]) -> dict[str, Any]:
    """Convert tuple-based stats into JSON-friendly dicts."""
    per_route = {}
    for route, route_stats in stats["per_route"].items():
        per_route[route] = {
            "n": route_stats["n"],
            "output_relevancy": {
                "avg": route_stats["output_relevancy"][0],
                "std": route_stats["output_relevancy"][1],
            },
            "completeness": {
                "avg": route_stats["completeness"][0],
                "std": route_stats["completeness"][1],
            },
        }

    return {
        "per_route": per_route,
        "micro": {
            "n": stats["micro"]["n"],
            "output_relevancy": {
                "avg": stats["micro"]["output_relevancy"][0],
                "std": stats["micro"]["output_relevancy"][1],
            },
            "completeness": {
                "avg": stats["micro"]["completeness"][0],
                "std": stats["micro"]["completeness"][1],
            },
        },
        "macro": {
            "n_routes": stats["macro"]["n_routes"],
            "output_relevancy": {
                "avg": stats["macro"]["output_relevancy"][0],
                "std": stats["macro"]["output_relevancy"][1],
            },
            "completeness": {
                "avg": stats["macro"]["completeness"][0],
                "std": stats["macro"]["completeness"][1],
            },
        },
    }


def print_subset_summary(summary: dict[str, Any]) -> None:
    """Print one subset summary in a console-friendly format."""
    print("\n" + "=" * 80)
    print(f"{summary['name'].upper()} DATA")
    print("=" * 80)
    print(f"Rows: {summary['rows']}")
    print(f"OK rows: {summary['ok_rows']}")

    if summary["status_counts"]:
        print("Status breakdown:")
        for status, count in sorted(summary["status_counts"].items()):
            print(f"  {status:20} {count:4}")

    if summary["route_counts"]:
        print("Route distribution:")
        for route, count in sorted(summary["route_counts"].items()):
            print(f"  {route:20} {count:4}")

    stats = summary["score_stats"]
    if not stats["per_route"]:
        print("No OK rows with valid eval scores were found.")
        return

    print("Scores by route:")
    for route in sorted(stats["per_route"]):
        route_stats = stats["per_route"][route]
        relevancy_avg, relevancy_std = route_stats["output_relevancy"]
        completeness_avg, completeness_std = route_stats["completeness"]
        print(
            f"  {route:20} Output Relevancy={relevancy_avg:.3f} (+/-{relevancy_std:.3f})  "
            f"Completeness={completeness_avg:.3f} (+/-{completeness_std:.3f})  n={route_stats['n']}"
        )

    micro = stats["micro"]
    macro = stats["macro"]
    micro_rel_avg, micro_rel_std = micro["output_relevancy"]
    micro_comp_avg, micro_comp_std = micro["completeness"]
    macro_rel_avg, macro_rel_std = macro["output_relevancy"]
    macro_comp_avg, macro_comp_std = macro["completeness"]
    print(
        f"  {'OVERALL (micro)':20} Output Relevancy={micro_rel_avg:.3f} (+/-{micro_rel_std:.3f})  "
        f"Completeness={micro_comp_avg:.3f} (+/-{micro_comp_std:.3f})  n={micro['n']}"
    )
    print(
        f"  {'OVERALL (macro)':20} Output Relevancy={macro_rel_avg:.3f} (+/-{macro_rel_std:.3f})  "
        f"Completeness={macro_comp_avg:.3f} (+/-{macro_comp_std:.3f})  n_routes={macro['n_routes']}"
    )


def maybe_write_outputs(
    output_dir: Path,
    input_path: Path,
    subsets: dict[str, pd.DataFrame],
    summaries: dict[str, dict[str, Any]],
    write_splits: bool,
    save_summary_json: bool,
) -> None:
    """Persist optional split files and summary JSON."""
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = input_path.stem.replace("_all_results_enriched", "")
    suffix = input_path.suffix.lower()

    if write_splits:
        for subset_name, subset_df in subsets.items():
            subset_path = output_dir / f"{stem}_{subset_name}{suffix}"
            if suffix == ".csv":
                subset_df.to_csv(subset_path, index=False, encoding="utf-8")
            else:
                subset_df.to_json(subset_path, orient="records", lines=True, force_ascii=False)
            print(f"Wrote {subset_name} split: {subset_path}")

    if save_summary_json:
        summary_path = output_dir / f"{stem}_natural_synthetic_summary.json"
        json_payload = {
            name: {
                "name": summary["name"],
                "rows": summary["rows"],
                "ok_rows": summary["ok_rows"],
                "status_counts": summary["status_counts"],
                "route_counts": summary["route_counts"],
                "score_stats": _stats_to_jsonable(summary["score_stats"]),
            }
            for name, summary in summaries.items()
        }
        summary_path.write_text(json.dumps(json_payload, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"Wrote summary JSON: {summary_path}")


def build_argument_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser."""
    default_input = find_default_input()
    parser = argparse.ArgumentParser(
        description="Split an enriched eval run into natural and synthetic subsets and recompute scores.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python -m src.scripts.split_natural_synthetic\n"
            "  python -m src.scripts.split_natural_synthetic --input eval_data/one_prompt_metrics/combined_intended_all/eval_20260313_164718_all_results_enriched.csv\n"
            "  python -m src.scripts.split_natural_synthetic --route-column route_intended --save-summary-json\n"
            "  python -m src.scripts.split_natural_synthetic --write-splits --output-dir eval_data/one_prompt_metrics/combined_intended_all/split_origin\n"
        ),
    )
    parser.add_argument(
        "--input",
        default=str(default_input) if default_input else None,
        help="Path to an enriched eval CSV/JSONL file, or a directory containing enriched files.",
    )
    parser.add_argument(
        "--route-column",
        choices=["auto", "route", "route_intended", "route_orchestrator", "route_orch"],
        default="auto",
        help=(
            "Route column used for aggregation. Default: auto, which infers the column "
            "from the input path: folders/files containing 'intended' use route_intended; "
            "folders/files containing 'orchestrator' use route_orchestrator or route_orch."
        ),
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory where optional split files and summary JSON will be written.",
    )
    parser.add_argument(
        "--write-splits",
        action="store_true",
        help=(
            "Write separate natural and synthetic subset files containing the original "
            "rows plus eval columns. Output format matches the input file type."
        ),
    )
    parser.add_argument(
        "--save-summary-json",
        action="store_true",
        help=(
            "Write the computed all/natural/synthetic summary metrics to a JSON file, "
            "including status counts, route counts, and per-route/micro/macro scores."
        ),
    )
    return parser


def main() -> None:
    parser = build_argument_parser()
    args = parser.parse_args()

    if not args.input:
        parser.error("No default input file was found. Pass --input explicitly.")

    input_path = resolve_input_path(Path(args.input))
    if not input_path.exists():
        parser.error(f"Input file not found: {input_path}")

    df = load_enriched_results(input_path)
    route_column = resolve_route_column(df, args.route_column, input_path)
    df = normalize_eval_frame(df, route_column)

    synthetic_mask = build_synthetic_mask(df)
    subsets = {
        "all": df.copy(),
        "natural": df.loc[~synthetic_mask].copy(),
        "synthetic": df.loc[synthetic_mask].copy(),
    }

    summaries = {
        subset_name: summarize_subset(subset_name, subset_df, route_column)
        for subset_name, subset_df in subsets.items()
    }

    print(f"Input file: {input_path}")
    print(f"Route column: {route_column}")
    print_subset_summary(summaries["all"])
    print_subset_summary(summaries["natural"])
    print_subset_summary(summaries["synthetic"])

    if args.write_splits or args.save_summary_json:
        output_dir = Path(args.output_dir) if args.output_dir else input_path.parent / "natural_synthetic_split"
        maybe_write_outputs(
            output_dir=output_dir,
            input_path=input_path,
            subsets=subsets,
            summaries=summaries,
            write_splits=args.write_splits,
            save_summary_json=args.save_summary_json,
        )


if __name__ == "__main__":
    main()