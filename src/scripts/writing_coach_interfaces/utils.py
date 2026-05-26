from __future__ import annotations

import sys
import types
from pathlib import Path
from typing import Any


def install_wc_app_path(wc_app_src: str | None) -> None:
    """Add external Writing Coach app source path to import paths when provided.

    Accepts either the app root (containing ``src/``) or the ``src`` directory itself.
    """
    if not wc_app_src:
        return

    candidate = Path(wc_app_src).expanduser().resolve()
    if candidate.name == "src":
        app_root = candidate.parent
        external_src = candidate
    else:
        app_root = candidate
        external_src = candidate / "src"

    app_root_str = str(app_root)
    if app_root_str not in sys.path:
        sys.path.insert(0, app_root_str)

    # The local eval repo already has a `src` package; extend it so
    # imports like `src.graph_builder` can resolve from the external app too.
    if external_src.is_dir():
        try:
            import src as local_src

            external_src_str = str(external_src)
            if external_src_str not in local_src.__path__:
                local_src.__path__.append(external_src_str)
        except Exception:
            # Best effort only; sys.path insertion above still supports many setups.
            pass


def install_learning_formatter_shim() -> None:
    """Provide legacy learning_formatters for Writing Coach V2 nodes."""
    module_name = "src.tools.search.learning_formatters"
    if module_name in sys.modules:
        return

    from src.tools.search.llm_formatters import format_llm_results

    shim = types.ModuleType(module_name)

    def format_documents_for_learnings(results, include_reference_prefix=True):
        return format_llm_results(
            results,
            include_reference_prefix=include_reference_prefix,
        )

    shim.format_documents_for_learnings = format_documents_for_learnings
    shim.__all__ = ["format_documents_for_learnings"]
    sys.modules[module_name] = shim


def coerce_llm_text_result(result: Any) -> Any:
    """Convert LangChain message objects to plain text for legacy consumers."""
    if isinstance(result, (dict, list, str)) or result is None:
        return result

    content = getattr(result, "content", None)
    if content is None:
        return result

    if isinstance(content, list):
        return " ".join(
            part.get("text", "") if isinstance(part, dict) else str(part)
            for part in content
        ).strip()

    if isinstance(content, str):
        return content

    return str(content)


def install_invoke_llm_shim() -> None:
    """Normalize invoke_llm outputs for legacy Writing Coach graph code."""
    import src.utils.llm_utils as llm_utils

    if getattr(llm_utils.invoke_llm, "_store_output_shimmed", False):
        return

    original_invoke_llm = llm_utils.invoke_llm

    def invoke_llm_compat(*args, **kwargs):
        result = original_invoke_llm(*args, **kwargs)
        response_format = kwargs.get("response_format", "text")
        if response_format == "json_object" or (
            isinstance(response_format, dict) and response_format.get("type") == "json_object"
        ):
            return result
        if kwargs.get("stream") and not kwargs.get("writer"):
            return result
        return coerce_llm_text_result(result)

    invoke_llm_compat._store_output_shimmed = True
    llm_utils.invoke_llm = invoke_llm_compat

