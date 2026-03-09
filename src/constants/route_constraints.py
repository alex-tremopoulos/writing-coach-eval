"""Route-specific rubrics constraints for the evaluation pipeline.

Each entry contains additional constraints that apply when generating rubrics for
a given route. These are injected into the rubrics generator prompt under the
``<route_rubrics_constraints>`` section.

Keys must match the route names used in all_results.csv:
    RESEARCH, RESPOND, REVISE_SIMPLE, REVISE_RESEARCH

Values will be filled in as the evaluation framework matures. Empty strings are
treated as "no additional constraints" by the rubrics generator.
"""

ROUTE_CONSTRAINTS: dict[str, str] = {
    "RESEARCH": "",
    "RESPOND": "",
    "REVISE_SIMPLE": "",
    "REVISE_RESEARCH": "",
}
