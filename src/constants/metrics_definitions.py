METRICS_DEFINITION = {
    "Output Relevancy": "Measures whether the edited output text w.r.t. the input text or the provided edit suggestions directly address the user command and demonstrate an understanding of the users’ intent(s).",
    "Completeness": "Assesses the extent to which the system’s output (edited text or edit suggestions) addresses all required aspects of the user’s request without omitting critical elements.",
}

CORRECTNESS_METRIC_NAME = "Correctness"
CORRECTNESS_METRIC_DESCRIPTION = (
    "Assesses whether revision suggestions are structurally sound, usable, and appropriate as writing-coach edits. "
    "This metric applies only to suggestion outputs in REVISE_SIMPLE and REVISE_RESEARCH."
)