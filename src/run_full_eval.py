"""Run the full evaluation pipeline end-to-end.

Orchestrates these scripts in order:
  1. eval_pipeline.py          — LLM rubrics evaluation
  2. split_natural_synthetic.py — split enriched results by data origin
  3. heuristic_scoring.py       — heuristic scoring for all / natural / synthetic
  4. extract_reasoning_by_score.py — extract per-metric/score reasoning
  5. summarize_reasoning.py     — LLM summarisation of reasoning patterns

Usage:
  python -m src.run_full_eval
  python -m src.run_full_eval --route-column orchestrator
  python -m src.run_full_eval --routes RESEARCH RESPOND --run-name my_experiment
  python -m src.run_full_eval --metrics completeness output_relevancy
"""

from __future__ import annotations

import argparse
import csv
import io
import math
import sys
from datetime import datetime
from pathlib import Path

# Ensure the repo root is on sys.path so `src.*` imports work whether the script
# is invoked as `python src/run_full_eval.py` or `python -m src.run_full_eval`.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def _timestamp() -> str:
    return datetime.now().strftime("%m%d_%H%M")


def _find_file_ending_with(directory: Path, suffix: str) -> Path:
    """Return the newest file in *directory* whose name ends with *suffix*."""
    candidates = sorted(directory.glob(f"*{suffix}"))
    if not candidates:
        raise FileNotFoundError(f"No file ending with '{suffix}' found in {directory}")
    return max(candidates, key=lambda p: (p.stat().st_mtime, p.name))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the full Writing-Coach evaluation pipeline end-to-end.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--input",
        default="final_data/all_results.csv",
        help="Path to input CSV with system outputs (default: final_data/all_results.csv)",
    )
    parser.add_argument(
        "--route-column",
        choices=["intended", "orchestrator"],
        default="intended",
        help=(
            "Which route column to use. "
            "'intended' (default): uses route_intended and stores results under route_intended/. "
            "'orchestrator': uses route_orch and stores results under route_orch/."
        ),
    )
    parser.add_argument(
        "--routes",
        nargs="+",
        default=None,
        help="Only evaluate rows with these routes (e.g. RESEARCH RESPOND)",
    )
    parser.add_argument(
        "--run-name",
        default=None,
        help="Name for output files (default: eval_YYYYMMDD_HHMMSS)",
    )
    parser.add_argument(
        "--metrics",
        "--metric",
        nargs="+",
        default=None,
        choices=["completeness", "output_relevancy", "correctness"],
        metavar="METRIC",
        help=(
            "Subset of metrics to evaluate (default: all). "
            "Choices: completeness, output_relevancy, correctness. "
            "Example: --metrics completeness output_relevancy"
        ),
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Max number of rows to process (for testing)",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Step runners
# ---------------------------------------------------------------------------


def step_eval_pipeline(
    input_path: str,
    output_dir: str,
    route_column: str,
    routes: list[str] | None,
    run_name: str | None,
    metrics: list[str] | None = None,
    limit: int | None = None,
) -> None:
    """Step 1 — run the LLM rubrics evaluation pipeline."""
    import asyncio
    from src.evaluation.eval_pipeline import run_pipeline

    asyncio.run(
        run_pipeline(
            input_path=input_path,
            output_dir=output_dir,
            route_column=route_column,
            routes=routes,
            run_name=run_name,
            save_local=True,
            metrics=metrics,
            limit=limit,
        )
    )


def step_split_natural_synthetic(
    enriched_csv: Path,
    output_dir: Path,
    metrics: list[str] | None = None,
) -> tuple[str, dict]:
    """Step 2 — split enriched results into natural / synthetic and print scores.

    Returns (captured_stdout_text, summaries_dict).
    """
    from src.scripts.split_natural_synthetic import (
        build_synthetic_mask,
        load_enriched_results,
        maybe_write_outputs,
        normalize_eval_frame,
        print_subset_summary,
        resolve_route_column,
        summarize_subset,
    )

    df = load_enriched_results(enriched_csv)
    route_column = resolve_route_column(df, "auto", enriched_csv)
    df = normalize_eval_frame(df, route_column)

    synthetic_mask = build_synthetic_mask(df)
    subsets = {
        "all": df.copy(),
        "natural": df.loc[~synthetic_mask].copy(),
        "synthetic": df.loc[synthetic_mask].copy(),
    }
    summaries = {
        name: summarize_subset(name, subset_df, route_column, metrics=metrics)
        for name, subset_df in subsets.items()
    }

    buf = io.StringIO()
    old_stdout = sys.stdout
    sys.stdout = buf
    try:
        print(f"Input file: {enriched_csv}")
        print(f"Route column: {route_column}")
        print_subset_summary(summaries["all"])
        print_subset_summary(summaries["natural"])
        print_subset_summary(summaries["synthetic"])

        maybe_write_outputs(
            output_dir=output_dir,
            input_path=enriched_csv,
            subsets=subsets,
            summaries=summaries,
            write_splits=True,
            save_summary_json=False,
        )
    finally:
        sys.stdout = old_stdout

    return buf.getvalue(), summaries


def step_heuristic_scoring(
    input_path: Path,
    metrics: list[str] | None = None,
) -> tuple[str, dict]:
    """Run heuristic_scoring on one file and return (captured_stdout, summary_dict)."""
    from src.scripts.heuristic_scoring import (
        analyze_rows,
        build_augmented_output_path,
        load_rows,
        print_summary,
        write_rows,
    )

    buf = io.StringIO()
    old_stdout = sys.stdout
    sys.stdout = buf
    try:
        rows = load_rows(input_path)
        augmented_rows, summary = analyze_rows(rows, metrics=metrics)
        print_summary(summary, input_path)
        output_path = build_augmented_output_path(input_path)
        write_rows(output_path, augmented_rows)
        print()
        print(f"Augmented output written to: {output_path}")
    finally:
        sys.stdout = old_stdout

    return buf.getvalue(), summary


def step_extract_reasoning(
    enriched_csv: Path,
    output_dir: Path,
    metrics: list[str] | None = None,
) -> None:
    """Step 4 — extract per-metric/score reasoning."""
    from src.evaluation.extract_reasoning_by_score import main as extract_main

    extract_main(input_csv=enriched_csv, output_dir=output_dir, metrics=metrics)


def step_summarize_reasoning(
    data_dir: Path,
    metrics: list[str] | None = None,
) -> None:
    """Step 5 — LLM-summarise reasoning patterns."""
    from src.evaluation.summarize_reasoning import main as summarize_main

    summarize_main(data_dir=data_dir, metrics=metrics)


# ---------------------------------------------------------------------------
# Summary CSV builder
# ---------------------------------------------------------------------------

_METRICS = [
    ("output_relevancy", "Output Relevancy"),
    ("completeness", "Completeness"),
    ("correctness", "Correctness"),
]

_LLM_STAT_FIELDS = ("mean", "std", "n")
_HEURISTIC_STAT_FIELDS = ("mean", "std")
_DISTRIBUTION_SCORES = (0, 1, 2)

_SUMMARY_CSV_FIELDNAMES = [
    "data_origin",
    "scope",
    "route",
    # Row counts
    "n_rows",
    "n_ok_rows",
    "n_no_response",
    "n_scored",
    *[
        f"llm_{metric_key}_{stat_name}"
        for metric_key, _ in _METRICS
        for stat_name in _LLM_STAT_FIELDS
    ],
    *[
        f"{metric_key}_score{score}_pct"
        for metric_key, _ in _METRICS
        for score in _DISTRIBUTION_SCORES
    ],
    *[
        f"heuristic_{metric_key}_{stat_name}"
        for metric_key, _ in _METRICS
        for stat_name in _HEURISTIC_STAT_FIELDS
    ],
]


def _nan_to_none(v: object) -> object:
    """Convert float NaN to None so CSV cells are blank rather than 'nan'."""
    if isinstance(v, float) and math.isnan(v):
        return None
    return v


def build_summary_csv_rows(
    summaries: dict[str, dict],
    heuristic_summaries: dict[str, dict],
) -> list[dict]:
    """Build flat CSV rows from LLM (split) summaries and heuristic summaries.

    *summaries* maps data_origin -> summarize_subset() result dict:
        {name, rows, ok_rows, status_counts, route_counts,
         score_stats: {per_route, micro, macro}}

    *heuristic_summaries* maps data_origin -> analyze_rows() summary dict:
        {rows_total, rows_scored, metrics: {metric_name: {...}}}

    Returns a list of flat dicts, one per (data_origin, scope, route) combination.
    """
    rows: list[dict] = []

    def _empty_metric_fields(prefix: str, stat_names: tuple[str, ...]) -> dict[str, None]:
        return {
            f"{prefix}_{metric_key}_{stat_name}": None
            for metric_key, _ in _METRICS
            for stat_name in stat_names
        }

    def _llm_metric_fields(stats: dict, counts: dict | None) -> dict[str, object]:
        values: dict[str, object] = {}
        counts = counts or {}
        for metric_key, _ in _METRICS:
            mean_value, std_value = stats.get(metric_key, (None, None))
            values[f"llm_{metric_key}_mean"] = _nan_to_none(mean_value)
            values[f"llm_{metric_key}_std"] = _nan_to_none(std_value)
            values[f"llm_{metric_key}_n"] = counts.get(f"{metric_key}_score")
        return values

    def _distribution_fields(distributions: dict[str, dict]) -> dict[str, object]:
        values: dict[str, object] = {}
        for metric_key, _ in _METRICS:
            field = f"{metric_key}_score"
            metric_distribution = distributions.get(field, {})
            for score in _DISTRIBUTION_SCORES:
                values[f"{field}{score}_pct"] = _nan_to_none(
                    metric_distribution.get(score, {}).get("pct")
                )
        return values

    def _empty_distribution_fields() -> dict[str, None]:
        return {
            f"{metric_key}_score{score}_pct": None
            for metric_key, _ in _METRICS
            for score in _DISTRIBUTION_SCORES
        }

    def _heuristic_metric_fields(heuristic_metrics: dict[str, dict]) -> dict[str, object]:
        values: dict[str, object] = {}
        for metric_key, display_name in _METRICS:
            metric_stats = heuristic_metrics.get(display_name, {})
            values[f"heuristic_{metric_key}_mean"] = _nan_to_none(
                metric_stats.get("heuristic_score_mean")
            )
            values[f"heuristic_{metric_key}_std"] = _nan_to_none(
                metric_stats.get("heuristic_score_std")
            )
        return values

    for origin in ("all", "natural", "synthetic"):
        subset = summaries.get(origin, {})
        heuristic = heuristic_summaries.get(origin, {})

        score_stats = subset.get("score_stats", {})
        per_route = score_stats.get("per_route", {})
        micro = score_stats.get("micro", {})
        macro = score_stats.get("macro", {})
        distributions = micro.get("distributions", {})
        heuristic_metrics = heuristic.get("metrics", {})

        status_counts = subset.get("status_counts", {})
        n_rows = subset.get("rows", None)
        n_ok = subset.get("ok_rows", None)
        n_no_response = status_counts.get("NO_RESPONSE", 0)

        # ---- per-route rows ----
        for route, rs in per_route.items():
            row: dict = {
                "data_origin": origin,
                "scope": "per_route",
                "route": route,
                "n_rows": rs.get("n"),
                "n_ok_rows": rs.get("n"),
                "n_no_response": None,
                "n_scored": rs.get("n"),
                **_llm_metric_fields(rs, rs.get("counts")),
                **_distribution_fields(rs.get("distributions", {})),
                **_empty_metric_fields("heuristic", _HEURISTIC_STAT_FIELDS),
            }
            rows.append(row)

        # ---- micro row ----
        micro_row: dict = {
            "data_origin": origin,
            "scope": "micro",
            "route": "OVERALL",
            "n_rows": n_rows,
            "n_ok_rows": n_ok,
            "n_no_response": n_no_response,
            "n_scored": micro.get("n"),
            **_llm_metric_fields(micro, micro.get("counts")),
            **_distribution_fields(distributions),
            **_heuristic_metric_fields(heuristic_metrics),
        }
        rows.append(micro_row)

        # ---- macro row ----
        macro_row: dict = {
            "data_origin": origin,
            "scope": "macro",
            "route": "OVERALL",
            "n_rows": n_rows,
            "n_ok_rows": n_ok,
            "n_no_response": n_no_response,
            "n_scored": macro.get("n_routes"),
            **_llm_metric_fields(macro, macro.get("counts")),
            **_empty_distribution_fields(),
            **_empty_metric_fields("heuristic", _HEURISTIC_STAT_FIELDS),
        }
        rows.append(macro_row)

    return rows


def write_summary_csv(output_path: Path, rows: list[dict]) -> None:
    """Write the flat summary rows to a CSV file."""
    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=_SUMMARY_CSV_FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)


# ---------------------------------------------------------------------------
# Main orchestrator
# ---------------------------------------------------------------------------


def main() -> None:
    args = parse_args()

    route_folder = "route_orch" if args.route_column == "orchestrator" else "route_intended"
    ts = _timestamp()
    base_output_dir = Path("eval_data/wcv2_one_prompt") / route_folder / ts

    # ---- Step 1: eval pipeline ----
    print("=" * 80)
    print(f"STEP 1/5  eval_pipeline  ->  {base_output_dir}")
    print("=" * 80)
    step_eval_pipeline(
        input_path=args.input,
        output_dir=str(base_output_dir),
        route_column=args.route_column,
        routes=args.routes,
        run_name=args.run_name,
        metrics=args.metrics,
        limit=args.limit
    )

    # Discover enriched CSV produced by step 1
    enriched_csv = _find_file_ending_with(base_output_dir, "_enriched.csv")
    results_csv = _find_file_ending_with(base_output_dir, "_results.csv")
    print(f"\nEnriched CSV: {enriched_csv}")

    # ---- Step 2: split natural / synthetic ----
    print("\n" + "=" * 80)
    print(f"STEP 2/5  split_natural_synthetic  ->  {base_output_dir}")
    print("=" * 80)
    split_output_dir = base_output_dir / "natural_synthetic_split"
    split_text, summaries = step_split_natural_synthetic(
        enriched_csv,
        split_output_dir,
        metrics=args.metrics,
    )
    print(split_text)

    scores_txt = base_output_dir / "scores_all_natural_synthetic.txt"
    scores_txt.write_text(split_text, encoding="utf-8")
    print(f"Saved scores -> {scores_txt}")

    # ---- Step 3: heuristic scoring (all, natural, synthetic) ----
    print("\n" + "=" * 80)
    print("STEP 3/5  heuristic_scoring  (all / natural / synthetic)")
    print("=" * 80)
    heuristic_parts: list[str] = []
    heuristic_summaries: dict[str, dict] = {}

    # 3a — all data (uses *_results.csv from the base output dir, enriched also works)
    print("\n--- all data ---")
    part, h_summary = step_heuristic_scoring(results_csv, metrics=args.metrics)
    print(part)
    heuristic_parts.append(f"=== ALL DATA ===\n{part}")
    heuristic_summaries["all"] = h_summary

    # 3b — natural
    natural_csv = _find_file_ending_with(split_output_dir, "_natural.csv")
    print("--- natural data ---")
    part, h_summary = step_heuristic_scoring(natural_csv, metrics=args.metrics)
    print(part)
    heuristic_parts.append(f"=== NATURAL DATA ===\n{part}")
    heuristic_summaries["natural"] = h_summary

    # 3c — synthetic
    synthetic_csv = _find_file_ending_with(split_output_dir, "_synthetic.csv")
    print("--- synthetic data ---")
    part, h_summary = step_heuristic_scoring(synthetic_csv, metrics=args.metrics)
    print(part)
    heuristic_parts.append(f"=== SYNTHETIC DATA ===\n{part}")
    heuristic_summaries["synthetic"] = h_summary

    heuristic_txt = base_output_dir / "heuristic_scoring_all_natural_synthetic.txt"
    heuristic_txt.write_text("\n\n".join(heuristic_parts), encoding="utf-8")
    print(f"Saved heuristic scores -> {heuristic_txt}")

    # ---- Build consolidated summary CSV ----
    summary_csv_path = base_output_dir / "summary_scores.csv"
    csv_rows = build_summary_csv_rows(summaries, heuristic_summaries)
    write_summary_csv(summary_csv_path, csv_rows)
    print(f"Saved summary CSV -> {summary_csv_path}")

    # ---- Step 4: extract reasoning by score ----
    print("\n" + "=" * 80)
    print("STEP 4/5  extract_reasoning_by_score")
    print("=" * 80)
    reasoning_dir = base_output_dir / "metrics_score_combinations"
    step_extract_reasoning(enriched_csv, reasoning_dir, metrics=args.metrics)

    # ---- Step 5: summarize reasoning ----
    print("\n" + "=" * 80)
    print("STEP 5/5  summarize_reasoning")
    print("=" * 80)
    item_level_dir = reasoning_dir / "item_level"
    step_summarize_reasoning(item_level_dir, metrics=args.metrics)

    print("\n" + "=" * 80)
    print("FULL PIPELINE COMPLETE")
    print(f"All outputs in: {base_output_dir}")
    print("=" * 80)


if __name__ == "__main__":
    main()
