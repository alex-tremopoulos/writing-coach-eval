"""Prompt template loader for the evaluation pipeline.

Loads prompt templates from ``src/prompts/`` (one level above this package) and
populates them with data for the rubrics generator and judge LLM calls.

Prompt files use Jinja2-style block markers to separate system and user prompts
inside a single file::

    {% block system %}
    <system prompt content — may contain {slot} placeholders>
    {% endblock %}

    {% block prompt %}
    <user prompt content — may contain {slot} placeholders>
    {% endblock %}

Route prompts and constraints are imported from ``src/constants/`` Python dicts
rather than individual `.txt` files, enabling version control and code-level editing.
"""

from pathlib import Path
from typing import Tuple

from src.constants.metrics_definitions import METRICS_DEFINITION
from src.constants.route_prompts import ROUTE_PROMPTS
from src.constants.route_constraints import ROUTE_CONSTRAINTS

# src/prompts/ — one level above src/evaluation/
PROMPTS_DIR = Path(__file__).parent.parent / "prompts"


# ---------------------------------------------------------------------------
# Low-level file loaders
# ---------------------------------------------------------------------------


def load_prompt(filename: str) -> str:
    """Load a plain prompt/config file from the prompts directory.

    Args:
        filename: Name of the file (e.g., 'meta_rubrics.txt').

    Returns:
        File content as a stripped string.

    Raises:
        FileNotFoundError: If the file does not exist.
    """
    filepath = PROMPTS_DIR / filename
    if not filepath.exists():
        raise FileNotFoundError(f"Prompt file not found: {filepath}")
    return filepath.read_text(encoding="utf-8").strip()


def parse_block_prompt(content: str) -> Tuple[str, str]:
    """Parse a block-formatted prompt file into (system, user) parts.

    Expected format::

        {% block system %}
        <system prompt>
        {% endblock %}

        {% block prompt %}
        <user prompt>
        {% endblock %}

    Args:
        content: Raw file content.

    Returns:
        Tuple of (system_content, user_content), both stripped.

    Raises:
        ValueError: If the expected block markers are not found.
    """
    import re

    system_match = re.search(
        r"\{%\s*block system\s*%\}(.*?)\{%\s*endblock\s*%\}",
        content,
        re.DOTALL,
    )
    prompt_match = re.search(
        r"\{%\s*block prompt\s*%\}(.*?)\{%\s*endblock\s*%\}",
        content,
        re.DOTALL,
    )

    if not system_match:
        raise ValueError(
            "Prompt file is missing '{% block system %}...{% endblock %}' section"
        )
    if not prompt_match:
        raise ValueError(
            "Prompt file is missing '{% block prompt %}...{% endblock %}' section"
        )

    return system_match.group(1).strip(), prompt_match.group(1).strip()


def load_combined_prompt(filename: str) -> Tuple[str, str]:
    """Load a block-formatted prompt file and return (system_template, user_template).

    Args:
        filename: Name of the prompt file in ``src/prompts/``
            (e.g., ``'rubrics_prompt.txt'``).

    Returns:
        Tuple of (system_prompt_template, user_prompt_template) ready for
        ``.format()`` substitution.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the file does not contain the expected block markers.
    """
    filepath = PROMPTS_DIR / filename
    if not filepath.exists():
        raise FileNotFoundError(f"Prompt file not found: {filepath}")
    content = filepath.read_text(encoding="utf-8")
    return parse_block_prompt(content)


# ---------------------------------------------------------------------------
# Constants-backed loaders
# ---------------------------------------------------------------------------


def load_route_prompt(route: str) -> str:
    """Return the orchestrator prompt section for a specific route.

    Loaded from the ``ROUTE_PROMPTS`` dict in ``src/constants/route_prompts.py``.

    Args:
        route: Route name (e.g., 'RESEARCH', 'RESPOND', 'REVISE_SIMPLE', 'REVISE_RESEARCH').

    Returns:
        Route prompt string, or a placeholder if the route is not defined.
    """
    key = route.upper()
    if key not in ROUTE_PROMPTS:
        return f"(No route prompt defined for route: {route})"
    return ROUTE_PROMPTS[key]


def load_route_constraints(route: str) -> str:
    """Return the rubrics constraints for a specific route.

    Loaded from the ``ROUTE_CONSTRAINTS`` dict in ``src/constants/route_constraints.py``.

    Args:
        route: Route name (e.g., 'RESEARCH', 'RESPOND', 'REVISE_SIMPLE', 'REVISE_RESEARCH').

    Returns:
        Route constraints string, or a placeholder if the route is not defined.
    """
    key = route.upper()
    if key not in ROUTE_CONSTRAINTS:
        return f"(No specific constraints defined for route: {route})"
    content = ROUTE_CONSTRAINTS[key]
    return content if content.strip() else "(No specific constraints defined for this route)"


def format_metrics_definition(metrics: dict[str, str] | None = None) -> str:
    """Format the metrics definition dict into a prompt-ready string.

    Uses ``METRICS_DEFINITION`` from ``src/constants/metrics_definitions.py``
    unless an override dict is supplied.

    Args:
        metrics: Optional override dict mapping metric name → description.

    Returns:
        Formatted string with one metric per line: ``**Name**: description``
    """
    source = metrics if metrics is not None else METRICS_DEFINITION
    return "\n".join(
        f"**{name}**: {description}" for name, description in source.items()
    )


# ---------------------------------------------------------------------------
# High-level prompt builders
# ---------------------------------------------------------------------------


def build_generator_prompts(
    user_query: str,
    input_text: str,
    route: str,
    metrics_definition: str | None = None,
    route_prompt: str | None = None,
    route_rubrics_constraints: str | None = None,
) -> Tuple[str, str]:
    """Build system and user prompts for the rubrics generator LLM.

    Loads ``src/prompts/rubrics_prompt.txt`` (block format) and fills in all
    dynamic slots.

    The system block (``{% block system %}``) has no dynamic slots.

    Slots filled in the user block (``{% block prompt %}``):
        - ``{user_command}``: the user's instruction
        - ``{input_text}``: the document text
        - ``{metrics_definition}``: formatted metrics string
        - ``{route_prompt}``: the orchestrator's route-specific prompt section

    Args:
        user_query: The user's instruction/query (mapped to ``{user_command}``).
        input_text: The document text the user is working on.
        route: The intended processing route (e.g., 'RESEARCH').
        metrics_definition: Override string for the ``{metrics_definition}`` slot.
            If None, formatted from ``METRICS_DEFINITION`` in constants.
        route_prompt: Override string for the ``{route_prompt}`` slot.
            If None, loaded from ``ROUTE_PROMPTS[route]`` in constants.
        route_rubrics_constraints: Override for route constraints (loaded from
            ``ROUTE_CONSTRAINTS[route]`` if None). Reserved for future use when
            the rubrics prompt template includes a ``{route_rubrics_constraints}`` slot.

    Returns:
        Tuple of (system_prompt, user_prompt) ready to send to the LLM.
    """
    if metrics_definition is None:
        metrics_definition = format_metrics_definition()
    if route_prompt is None:
        route_prompt = load_route_prompt(route)
    # route_rubrics_constraints is loaded but not yet injected — the current
    # rubrics_prompt.txt template does not include a {route_rubrics_constraints} slot.
    # Kept for forward compatibility.
    if route_rubrics_constraints is None:
        route_rubrics_constraints = load_route_constraints(route)

    system_template, user_template = load_combined_prompt("rubrics_prompt.txt")

    # System block has no dynamic slots
    system_prompt = system_template

    user_prompt = user_template.format(
        user_command=user_query,
        input_text=input_text,
        metrics_definition=metrics_definition,
        route_prompt=route_prompt,
    )

    return system_prompt, user_prompt


def build_judge_prompts(
    user_query: str,
    input_text: str,
    output_text: str,
    rubrics: str,
    meta_rubrics: str | None = None,
) -> Tuple[str, str]:
    """Build system and user prompts for the rubrics judge LLM.

    Loads ``src/prompts/rubrics_judge_prompt.txt`` (block format) and fills in
    all dynamic slots.

    Slots filled in the system block (``{% block system %}``):
        - ``{meta_rubrics}``: meta-level rubrics criteria

    Slots filled in the user block (``{% block prompt %}``):
        - ``{user_query}``: the user's instruction
        - ``{input_text}``: the document text
        - ``{output_text}``: the system's generated output to evaluate
        - ``{rubrics}``: JSON string of rubrics from Stage 1

    Args:
        user_query: The user's instruction/query.
        input_text: The document text the user is working on.
        output_text: The system's generated output to evaluate.
        rubrics: JSON string of generated rubrics from Stage 1.
        meta_rubrics: Override for meta-rubrics content.
            If None, loaded from ``src/prompts/meta_rubrics.txt``.

    Returns:
        Tuple of (system_prompt, user_prompt) ready to send to the LLM.
    """
    if meta_rubrics is None:
        try:
            meta_rubrics = load_prompt("meta_rubrics.txt")
        except FileNotFoundError:
            meta_rubrics = "(No meta-rubrics defined)"

    system_template, user_template = load_combined_prompt("rubrics_judge_prompt.txt")

    system_prompt = system_template.format(
        meta_rubrics=meta_rubrics,
    )
    user_prompt = user_template.format(
        user_query=user_query,
        input_text=input_text,
        output_text=output_text,
        rubrics=rubrics,
    )

    return system_prompt, user_prompt
