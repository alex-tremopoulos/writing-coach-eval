"""Giskard scan orchestration and metric routing."""


SCORING_METRIC_NAMES: list[str] = ["completeness", "output_relevancy", "correctness"]


def _run_harmfulness_scan(*args, **kwargs):
    from src.evaluation.giskard_scans.scan_harmfulness import run_harmfulness_scan

    return run_harmfulness_scan(*args, **kwargs)


def _run_stereotypes_scan(*args, **kwargs):
    from src.evaluation.giskard_scans.scan_stereotypes import run_stereotypes_scan

    return run_stereotypes_scan(*args, **kwargs)


def _run_jailbreak_scan(*args, **kwargs):
    from src.evaluation.giskard_scans.scan_jailbreak import run_jailbreak_scan

    return run_jailbreak_scan(*args, **kwargs)


SCAN_METRIC_RUNNERS = {
    "potential_harm": _run_harmfulness_scan,
    "toxicity": _run_stereotypes_scan,
    "resilience": _run_jailbreak_scan,
}
ALL_METRIC_NAMES: list[str] = [*SCORING_METRIC_NAMES, *SCAN_METRIC_RUNNERS.keys()]


def split_selected_metrics(metrics: list[str] | None) -> tuple[list[str] | None, list[str]]:
    """Split selected metrics into row-scoring metrics and Giskard scan metrics.

    Args:
        metrics: List of metric names, or None to select all row-level metrics.

    Returns:
        Tuple of (scoring_metrics, scan_metrics) where scoring_metrics is None
        if all row-level metrics are selected, or a list of the selected ones.
        scan_metrics is always a list (possibly empty).
    """
    if metrics is None:
        return None, []

    normalized: list[str] = []
    for metric in metrics:
        value = metric.lower().replace(" ", "_")
        if value not in normalized:
            normalized.append(value)

    scoring = [metric for metric in normalized if metric in SCORING_METRIC_NAMES]
    scans = [metric for metric in normalized if metric in SCAN_METRIC_RUNNERS]
    return scoring, scans

