"""Summary score CSV building utilities."""

import csv
import math
from pathlib import Path


_METRICS = [
    ("output_relevancy", "Output Relevancy"),
    ("completeness", "Completeness"),
    ("correctness", "Correctness"),
]

_LLM_STAT_FIELDS = ("mean", "std", "n")
_HEURISTIC_STAT_FIELDS = ("mean", "std")
_DISTRIBUTION_SCORES = (0, 1, 2)

SUMMARY_CSV_FIELDNAMES = [
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


def nan_to_none(v: object) -> object:
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
            values[f"llm_{metric_key}_mean"] = nan_to_none(mean_value)
            values[f"llm_{metric_key}_std"] = nan_to_none(std_value)
            values[f"llm_{metric_key}_n"] = counts.get(f"{metric_key}_score")
        return values

    def _distribution_fields(distributions: dict[str, dict]) -> dict[str, object]:
        values: dict[str, object] = {}
        for metric_key, _ in _METRICS:
            field = f"{metric_key}_score"
            metric_distribution = distributions.get(field, {})
            for score in _DISTRIBUTION_SCORES:
                values[f"{field}{score}_pct"] = nan_to_none(
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
            values[f"heuristic_{metric_key}_mean"] = nan_to_none(
                metric_stats.get("heuristic_score_mean")
            )
            values[f"heuristic_{metric_key}_std"] = nan_to_none(
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
        writer = csv.DictWriter(f, fieldnames=SUMMARY_CSV_FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)

