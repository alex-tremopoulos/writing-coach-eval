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
"""

from __future__ import annotations

import argparse
import io
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
        )
    )


def step_split_natural_synthetic(enriched_csv: Path, output_dir: Path) -> str:
    """Step 2 — split enriched results into natural / synthetic and print scores.

    Returns the captured stdout text.
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
        name: summarize_subset(name, subset_df, route_column)
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

    return buf.getvalue()


def step_heuristic_scoring(input_path: Path) -> str:
    """Run heuristic_scoring on one file and return captured stdout."""
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
        augmented_rows, summary = analyze_rows(rows)
        print_summary(summary, input_path)
        output_path = build_augmented_output_path(input_path)
        write_rows(output_path, augmented_rows)
        print()
        print(f"Augmented output written to: {output_path}")
    finally:
        sys.stdout = old_stdout

    return buf.getvalue()


def step_extract_reasoning(enriched_csv: Path, output_dir: Path) -> None:
    """Step 4 — extract per-metric/score reasoning."""
    from src.evaluation.extract_reasoning_by_score import main as extract_main

    extract_main(input_csv=enriched_csv, output_dir=output_dir)


def step_summarize_reasoning(data_dir: Path) -> None:
    """Step 5 — LLM-summarise reasoning patterns."""
    from src.evaluation.summarize_reasoning import main as summarize_main

    summarize_main(data_dir=data_dir)


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
    split_text = step_split_natural_synthetic(enriched_csv, split_output_dir)
    print(split_text)

    scores_txt = base_output_dir / "scores_all_natural_synthetic.txt"
    scores_txt.write_text(split_text, encoding="utf-8")
    print(f"Saved scores -> {scores_txt}")

    # ---- Step 3: heuristic scoring (all, natural, synthetic) ----
    print("\n" + "=" * 80)
    print("STEP 3/5  heuristic_scoring  (all / natural / synthetic)")
    print("=" * 80)
    heuristic_parts: list[str] = []

    # 3a — all data (uses *_results.csv from the base output dir, enriched also works)
    print("\n--- all data ---")
    part = step_heuristic_scoring(results_csv)
    print(part)
    heuristic_parts.append(f"=== ALL DATA ===\n{part}")

    # 3b — natural
    natural_csv = _find_file_ending_with(split_output_dir, "_natural.csv")
    print("--- natural data ---")
    part = step_heuristic_scoring(natural_csv)
    print(part)
    heuristic_parts.append(f"=== NATURAL DATA ===\n{part}")

    # 3c — synthetic
    synthetic_csv = _find_file_ending_with(split_output_dir, "_synthetic.csv")
    print("--- synthetic data ---")
    part = step_heuristic_scoring(synthetic_csv)
    print(part)
    heuristic_parts.append(f"=== SYNTHETIC DATA ===\n{part}")

    heuristic_txt = base_output_dir / "heuristic_scoring_all_natural_synthetic.txt"
    heuristic_txt.write_text("\n\n".join(heuristic_parts), encoding="utf-8")
    print(f"Saved heuristic scores -> {heuristic_txt}")

    # ---- Step 4: extract reasoning by score ----
    print("\n" + "=" * 80)
    print("STEP 4/5  extract_reasoning_by_score")
    print("=" * 80)
    reasoning_dir = base_output_dir / "metrics_score_combinations"
    step_extract_reasoning(enriched_csv, reasoning_dir)

    # ---- Step 5: summarize reasoning ----
    print("\n" + "=" * 80)
    print("STEP 5/5  summarize_reasoning")
    print("=" * 80)
    item_level_dir = reasoning_dir / "item_level"
    step_summarize_reasoning(item_level_dir)

    print("\n" + "=" * 80)
    print("FULL PIPELINE COMPLETE")
    print(f"All outputs in: {base_output_dir}")
    print("=" * 80)


if __name__ == "__main__":
    main()
